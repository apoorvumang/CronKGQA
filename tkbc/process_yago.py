# Copyright (c) Facebook, Inc. and its affiliates.

import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import re
import sys

import numpy as np

from collections import defaultdict

DATA_PATH = pkg_resources.resource_filename('tkbc', 'data/')


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
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), defaultdict(int)
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            v = line.strip().split('\t')
            if len(v) > 4:
                lhs, rel, rhs, type, timestamp = v
                rel += type
                timestamp = int(re.search(r'\d+', timestamp).group())
                timestamps[timestamp] += 1
            elif len(v) == 4:
                # india participated in Korean War without timestamp :|
                continue
            else:
                lhs, rel, rhs = v
                rel += '_notime'

            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)

        to_read.close()

    print(f"{len(timestamps)} timestamps")

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps.keys()))}
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
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # dump the time differences between timestamps for continuity regularizer
    ts = np.array(sorted(timestamps.keys()), dtype='float')
    diffs = ts[1:] - ts[:-1]
    out = open(os.path.join(DATA_PATH, name, 'ts_diffs.pickle'), 'wb')
    pickle.dump(diffs, out)
    out.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            v = line.strip().split('\t')
            ts = False
            if len(v) > 4:
                lhs, rel, rhs, type, timestamp = v
                rel += type
                timestamp = int(re.search(r'\d+', timestamp).group())
                ts = True
            elif len(v) == 4:
                # india participated in Korean War without timestamp :|
                continue
            else:
                lhs, rel, rhs = v
                rel += '_notime'

            try:
                examples.append([
                    entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs],
                    timestamps_to_id[timestamp] if ts else len(timestamps_to_id)
                ])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()


if __name__ == "__main__":
    datasets = ['yago15k']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset_rels(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 'src_data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise

