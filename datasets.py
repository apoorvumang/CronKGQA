# Copyright (c) Facebook, Inc. and its affiliates.
#

from pathlib import Path
import pkg_resources
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List

from sklearn.metrics import average_precision_score

import numpy as np
import torch
from models import TKBCModel


DATA_PATH = pkg_resources.resource_filename('tkbc', 'data/')

class TemporalDataset(object):
    def __init__(self, name: str):
        self.root = Path(DATA_PATH) / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2
        if maxis.shape[0] > 4:
            self.n_timestamps = max(int(maxis[3] + 1), int(maxis[4] + 1))
        else:
            self.n_timestamps = int(maxis[3] + 1)
        try:
            inp_f = open(str(self.root / f'ts_diffs.pickle'), 'rb')
            self.time_diffs = torch.from_numpy(pickle.load(inp_f)).cuda().float()
            # print("Assume all timestamps are regularly spaced")
            # self.time_diffs = None
            inp_f.close()
        except OSError:
            print("Assume all timestamps are regularly spaced")
            self.time_diffs = None

        try:
            e = open(str(self.root / f'event_list_all.pickle'), 'rb')
            self.events = pickle.load(e)
            e.close()

            f = open(str(self.root / f'ts_id'), 'rb')
            dictionary = pickle.load(f)
            f.close()
            self.timestamps = sorted(dictionary.keys())
        except OSError:
            print("Not using time intervals and events eval")
            self.events = None

        if self.events is None:
            inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
            self.to_skip: Dict[str, Dict[Tuple[int, int, int], List[int]]] = pickle.load(inp_f)
            inp_f.close()


        # If dataset has events, it's wikidata.
        # For any relation that has no beginning & no end:
        # add special beginning = end = no_timestamp, increase n_timestamps by one.

    def has_intervals(self):
        return self.events is not None

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: TKBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        if self.events is not None:
            return self.time_eval(model, split, n_queries, 'rhs', at)
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def time_eval(
            self, model: TKBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        assert missing_eval == 'rhs', "other evals not implemented"
        test = torch.from_numpy(
            self.get_examples(split).astype('int64')
        )
        if n_queries > 0:
            permutation = torch.randperm(len(test))[:n_queries]
            test = test[permutation]

        time_range = test.float()
        sampled_time = (
                torch.rand(time_range.shape[0]) * (time_range[:, 4] - time_range[:, 3]) + time_range[:, 3]
        ).round().long()
        has_end = (time_range[:, 4] != (self.n_timestamps - 1))
        has_start = (time_range[:, 3] > 0)

        masks = {
            'full_time': has_end + has_start,
            'only_begin': has_start * (~has_end),
            'only_end': has_end * (~has_start),
            'no_time': (~has_end) * (~has_start)
        }

        with_time = torch.cat((
            sampled_time.unsqueeze(1),
            time_range[:, 0:3].long(),
            masks['full_time'].long().unsqueeze(1),
            masks['only_begin'].long().unsqueeze(1),
            masks['only_end'].long().unsqueeze(1),
            masks['no_time'].long().unsqueeze(1),
        ), 1)
        # generate events
        eval_events = sorted(with_time.tolist())

        to_filter: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        id_event = 0
        id_timeline = 0
        batch_size = 100
        to_filter_batch = []
        cur_batch = []

        ranks = {
            'full_time': [], 'only_begin': [], 'only_end': [], 'no_time': [],
            'all': []
        }
        while id_event < len(eval_events):
            # Follow timeline to add events to filters
            while id_timeline < len(self.events) and self.events[id_timeline][0] <= eval_events[id_event][3]:
                date, event_type, (lhs, rel, rhs) = self.events[id_timeline]
                if event_type < 0:  # begin
                    to_filter[(lhs, rel)][rhs] += 1
                if event_type > 0:  # end
                    to_filter[(lhs, rel)][rhs] -= 1
                    if to_filter[(lhs, rel)][rhs] == 0:
                        del to_filter[(lhs, rel)][rhs]
                id_timeline += 1
            date, lhs, rel, rhs, full_time, only_begin, only_end, no_time = eval_events[id_event]

            to_filter_batch.append(sorted(to_filter[(lhs, rel)].keys()))
            cur_batch.append((lhs, rel, rhs, date, full_time, only_begin, only_end, no_time))
            # once a batch is ready, call get_ranking and reset
            if len(cur_batch) == batch_size or id_event == len(eval_events) - 1:
                cuda_batch = torch.cuda.LongTensor(cur_batch)
                bbatch = torch.LongTensor(cur_batch)
                batch_ranks = model.get_time_ranking(cuda_batch[:, :4], to_filter_batch, 500000)

                ranks['full_time'].append(batch_ranks[bbatch[:, 4] == 1])
                ranks['only_begin'].append(batch_ranks[bbatch[:, 5] == 1])
                ranks['only_end'].append(batch_ranks[bbatch[:, 6] == 1])
                ranks['no_time'].append(batch_ranks[bbatch[:, 7] == 1])

                ranks['all'].append(batch_ranks)
                cur_batch = []
                to_filter_batch = []
            id_event += 1

        ranks = {x: torch.cat(ranks[x]) for x in ranks if len(ranks[x]) > 0}
        mean_reciprocal_rank = {x: torch.mean(1. / ranks[x]).item() for x in ranks if len(ranks[x]) > 0}
        hits_at = {z: torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks[z] <= x).float()).item(),
            at
        )))) for z in ranks if len(ranks[z]) > 0}

        res = {
            ('MRR_'+x): y for x, y in mean_reciprocal_rank.items()
        }
        res.update({('hits@_'+x): y for x, y in hits_at.items()})
        return res

    def breakdown_time_eval(
            self, model: TKBCModel, split: str, n_queries: int = -1, missing_eval: str = 'rhs',
    ):
        assert missing_eval == 'rhs', "other evals not implemented"
        test = torch.from_numpy(
            self.get_examples(split).astype('int64')
        )
        if n_queries > 0:
            permutation = torch.randperm(len(test))[:n_queries]
            test = test[permutation]

        time_range = test.float()
        sampled_time = (
                torch.rand(time_range.shape[0]) * (time_range[:, 4] - time_range[:, 3]) + time_range[:, 3]
        ).round().long()
        has_end = (time_range[:, 4] != (self.n_timestamps - 1))
        has_start = (time_range[:, 3] > 0)

        masks = {
            'full_time': has_end + has_start,
            'only_begin': has_start * (~has_end),
            'only_end': has_end * (~has_start),
            'no_time': (~has_end) * (~has_start)
        }

        with_time = torch.cat((
            sampled_time.unsqueeze(1),
            time_range[:, 0:3].long(),
            masks['full_time'].long().unsqueeze(1),
            masks['only_begin'].long().unsqueeze(1),
            masks['only_end'].long().unsqueeze(1),
            masks['no_time'].long().unsqueeze(1),
        ), 1)
        # generate events
        eval_events = sorted(with_time.tolist())

        to_filter: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        id_event = 0
        id_timeline = 0
        batch_size = 100
        to_filter_batch = []
        cur_batch = []

        ranks = defaultdict(list)
        while id_event < len(eval_events):
            # Follow timeline to add events to filters
            while id_timeline < len(self.events) and self.events[id_timeline][0] <= eval_events[id_event][3]:
                date, event_type, (lhs, rel, rhs) = self.events[id_timeline]
                if event_type < 0:  # begin
                    to_filter[(lhs, rel)][rhs] += 1
                if event_type > 0:  # end
                    to_filter[(lhs, rel)][rhs] -= 1
                    if to_filter[(lhs, rel)][rhs] == 0:
                        del to_filter[(lhs, rel)][rhs]
                id_timeline += 1
            date, lhs, rel, rhs, full_time, only_begin, only_end, no_time = eval_events[id_event]

            to_filter_batch.append(sorted(to_filter[(lhs, rel)].keys()))
            cur_batch.append((lhs, rel, rhs, date, full_time, only_begin, only_end, no_time))
            # once a batch is ready, call get_ranking and reset
            if len(cur_batch) == batch_size or id_event == len(eval_events) - 1:
                cuda_batch = torch.cuda.LongTensor(cur_batch)
                bbatch = torch.LongTensor(cur_batch)
                batch_ranks = model.get_time_ranking(cuda_batch[:, :4], to_filter_batch, 500000)
                for rank, predicate in zip(batch_ranks, bbatch[:, 1]):
                    ranks[predicate.item()].append(rank.item())
                cur_batch = []
                to_filter_batch = []
            id_event += 1

        ranks = {x: torch.FloatTensor(ranks[x]) for x in ranks}
        sum_reciprocal_rank = {x: torch.sum(1. / ranks[x]).item() for x in ranks}

        return sum_reciprocal_rank

    def time_AUC(self, model: TKBCModel, split: str, n_queries: int = -1):
        test = torch.from_numpy(
            self.get_examples(split).astype('int64')
        )
        if n_queries > 0:
            permutation = torch.randperm(len(test))[:n_queries]
            test = test[permutation]

        truth, scores = model.get_auc(test.cuda())

        return {
            'micro': average_precision_score(truth, scores, average='micro'),
            'macro': average_precision_score(truth, scores, average='macro')
        }


    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps

