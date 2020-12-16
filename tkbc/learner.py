# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
import torch
from torch import optim

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx
from regularizers import N3, Lambda3

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--save_to', default='', type=str,
    help="Where to save trained model"
)
parser.add_argument(
    '--load_from', default='', type=str,
    help="Where to load model from"
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)


args = parser.parse_args()

dataset = TemporalDataset(args.dataset)

sizes = dataset.get_shape()
model = {
    'ComplEx': ComplEx(sizes, args.rank),
    'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
    'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
}[args.model]
model = model.cuda()

if args.load_from != '':
    filename = 'models/{dataset_name}/kg_embeddings/{model_file}'.format(
                    dataset_name = args.dataset, model_file=args.load_from
                )
    print('Loading from ' + filename)
    model.load_state_dict(torch.load(filename))
    print('Loaded model.')

opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

emb_reg = N3(args.emb_reg)
time_reg = Lambda3(args.time_reg)

best_valid_score = 0.0

def save_model(model, filename):
    print('Saving model to', filename)
    torch.save(model.state_dict(), filename)
    print('Saved model to ', filename)
    return

for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )

    model.train()
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)

    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)


    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}

    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        if dataset.has_intervals():
            valid, test, train = [
                dataset.eval(model, split, -1 if split != 'train' else 50000)
                for split in ['valid', 'test', 'train']
            ]
            print("valid: ", valid)
            print("test: ", test)
            print("train: ", train)
            valid_score = valid['MRR_full_time']
            if valid_score > best_valid_score:
                best_valid_score = valid_score
                print('Valid score increased, saving model')
                if args.save_to == '':
                    print('No save file specified. Skipping saving.')
                else:
                    filename = 'models/{dataset_name}/kg_embeddings/{model_file}'.format(
                        dataset_name = args.dataset, model_file=args.save_to
                    )
                    save_model(model, filename)

        else:
            valid, test, train = [
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]
            print("valid: ", valid['MRR'])
            print("test: ", test['MRR'])
            print("train: ", train['MRR'])

