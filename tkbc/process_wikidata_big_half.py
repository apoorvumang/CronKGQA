# Copyright (c) Facebook, Inc. and its affiliates.

import pkg_resources
import os
import errno
import math
from pathlib import Path
import pickle
import re
import sys

import numpy as np

from collections import defaultdict


DATA_PATH = 'data/wikidata_big_half/kg/tkbc_processed_data/'

# DATA_PATH = pkg_resources.resource_filename('tkbc', 'data/')

def get_be(begin, end):
    if begin is None:
        begin = (-math.inf, 0, 0)
    else:
        begin = (int(begin), 0, 0)

    if end is None:
        end = (math.inf, 0, 0)
    else:
        end = (int(end), 0, 0)

    return begin, end


def prepare_dataset_rels(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(type)\t(timestamp)\n
    Maps each entity, relation+type and timestamp to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """

    # to get sets of entities, relations, timestamps
    # we need to use full dataset, not half

    path_for_full = path.replace('_half', '')

    files = ['train', 'valid', 'test']

    entities, relations, timestamps = set(), set(), set()

    # using files from full here
    # replaced path with path_for_full
    for f in files:
        file_path = os.path.join(path_for_full, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            v = line.strip().split('\t')
            lhs, rel, rhs, begin, end = v

            begin, end = get_be(begin, end)
            timestamps.add(begin)
            timestamps.add(end)
            entities.add(lhs)
            entities.add(rhs)   
            relations.add(rel)

        to_read.close()
    timestamps_int = set()
    for t in timestamps:
        timestamps_int.add(t[0])
    min_t = min(timestamps_int)
    max_t = max(timestamps_int)
    print('Minimum, maximum times:', min_t, max_t)
    timestamps = [(i, 0, 0) for i in range(min_t, max_t + 1)]
    print(f"{len(timestamps)} timestamps")

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}

    # we need to sort timestamps and associate them to actual dates
    
    # all_ts = sorted(timestamps)[1:-1]
    all_ts = sorted(timestamps)
    timestamps_to_id = {x: i for (i, x) in enumerate(all_ts)}
    # print(timestamps_to_id)

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    try:
        os.makedirs(os.path.join(DATA_PATH, name))
    except OSError as e:
        r = input(f"{e}\nContinue ? [y/n]")
        if r != "y":
            sys.exit()

    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'wb')
        pickle.dump(dic, ff)
        ff.close()

    # dump the time differences between timestamps for continuity regularizer
    # ignores number of days in a month but who cares
    # ts_to_int = [x[0] * 365 + x[1] * 30 + x[2] for x in all_ts]
    ts_to_int = [x[0] for x in all_ts]
    ts = np.array(ts_to_int, dtype='float')
    diffs = ts[1:] - ts[:-1]  # remove last timestamp from time diffs. it's not really a timestamp
    out = open(os.path.join(DATA_PATH, name, 'ts_diffs.pickle'), 'wb')
    pickle.dump(diffs, out)
    out.close()

    # map train/test/valid with the ids
    event_list = {
        'all': [],
    }
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        ignore = 0
        total = 0
        full_intervals = 0
        half_intervals = 0
        point = 0
        for line in to_read.readlines():
            v = line.strip().split('\t')
            lhs, rel, rhs, begin, end = v
            begin_t, end_t = get_be(begin, end)
            total += 1

            begin = begin_t
            end = end_t

            if begin_t[0] == -math.inf:
                begin = all_ts[0]
                if not end_t[0] == math.inf:
                    half_intervals += 1
            if end_t[0] == math.inf:
                end = all_ts[-1]
                if not begin_t[0] == -math.inf:
                    half_intervals += 1

            if begin_t[0] > -math.inf and end_t[0] < math.inf:
                if begin_t[0] == end_t[0]:
                    point += 1
                else:
                    full_intervals += 1

            begin = timestamps_to_id[begin]
            end = timestamps_to_id[end]

            if begin > end:
                ignore += 1
                continue

            lhs = entities_to_id[lhs]
            rel = relations_to_id[rel]
            rhs = entities_to_id[rhs]

            event_list['all'].append((begin, -1, (lhs, rel, rhs)))
            event_list['all'].append((end, +1, (lhs, rel, rhs)))

            try:
                examples.append([lhs, rel, rhs, begin, end])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()
        print(f"Ignored {ignore} events.")
        print(f"Total : {total} // Full : {full_intervals} // Half : {half_intervals} // Point : {point}")

    for k, v in event_list.items():
        out = open(Path(DATA_PATH) / name / ('event_list_' + k + '.pickle'), 'wb')
        print("Dumping all events", len(v))
        pickle.dump(sorted(v), out)
        out.close()


if __name__ == "__main__":
    datasets = ['wikidata_big_half']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            dataset_location = 'data/wikidata_big_half/kg/'
            prepare_dataset_rels(
                dataset_location,
                d
            )
            # prepare_dataset_rels(
            #     os.path.join(
            #         os.path.dirname(os.path.realpath(__file__)), 'src_data', d
            #     ),
            #     d
            # )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise

