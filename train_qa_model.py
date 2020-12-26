import argparse
from typing import Dict
import logging
import torch
from torch import optim
import pickle
import numpy as np

from qa_models import QA_model
from qa_datasets import QA_Dataset, QA_Dataset_model1

import utils
from tqdm import tqdm
from utils import loadTkbcModel
from collections import defaultdict

parser = argparse.ArgumentParser(
    description="Temporal KGQA"
)
parser.add_argument(
    '--tkbc_model_file', default='model_tkbc_60kent.ckpt', type=str,
    help="Pretrained tkbc model checkpoint"
)

parser.add_argument(
    '--model', default='model1', type=str,
    help="Which model to use."
)

parser.add_argument(
    '--mode', default='train', type=str,
    help="Which mode to use."
)


parser.add_argument(
    '--questions_file', default='/scratche/home/apoorv/tempqa/data/questions/questions_position_held_small_1_paraphrases_shuffled.pickle', type=str,
    help="Pickle file containing questions"
)
parser.add_argument(
    '--load_from', default='', type=str,
    help="Pretrained qa model checkpoint"
)
parser.add_argument(
    '--dataset', default='', type=str,
    help="Pretrained qa model checkpoint"
)


parser.add_argument(
    '--save_to', default='', type=str,
    help="Where to save checkpoint."
)

parser.add_argument(
    '--max_epochs', default=100, type=int,
    help="Number of epochs."
)

parser.add_argument(
    '--eval_k', default=1, type=int,
    help="Hits@k used for eval. Default 10."
)

parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)

parser.add_argument(
    '--batch_size', default=512, type=int,
    help="Batch size."
)
parser.add_argument(
    '--frozen', default=0, type=int,
    help="Whether entity/time embeddings are frozen or not. Default unfrozen."
)

parser.add_argument(
    '--lr', default=1e-3, type=float,
    help="Learning rate"
)

args = parser.parse_args()

# todo: this function may not be properly implemented
# might want to compare predicted khot with answers khot
# right now actual answers come from dataset.data[split][i]['answers']
# which works for now
def eval(qa_model, dataset, split='valid', k=10):
    qa_model.eval()
    print('Evaluating with k = %d' % k)
    batch_size = 500
    
    if len(dataset.data[split]) < batch_size:
        batch_size = len(dataset.data[split])
    
    topk_answers = []
    total_loss = 0
    for i in tqdm(range(len(dataset.data[split]) // batch_size)):
        question_text, entities_times_padded, entities_times_padded_mask, answers_khot = dataset.get_batch(
        split=split,
        start_index=i*batch_size,
        batch_size=batch_size) 
        scores = qa_model.forward(question_text, entities_times_padded.cuda(), entities_times_padded_mask.cuda())
        for s in scores:
            pred = dataset.getAnswersFromScores(s, k=k)
            topk_answers.append(pred)
        loss = qa_model.loss(scores, answers_khot)
        total_loss += loss.item()
    print(split, 'loss: ', total_loss)

    hits_at_k = 0
    total = 0
    question_types_count = defaultdict(list)

    for i, question in enumerate(dataset.data[split]):
        actual_answers = question['answers']
        if 'type' in question.keys():
            question_template = question['type']
        else:
            question_template = question['template']
        predicted = topk_answers[i]
        # if question_template == 'Who was the {tail} in {time}?':
        #     print(question)
        #     print(predicted)
        #     exit(0)
        if len(set(actual_answers).intersection(set(predicted))) > 0:
            question_types_count[question_template].append(1)
            hits_at_k += 1
        else:
            question_types_count[question_template].append(0)
        total += 1

    eval_accuracy = hits_at_k/total
    
    print('Hits at %d: ' % k, round(eval_accuracy, 3))
    
    for key, value in question_types_count.items():
        hits_at_k = sum(value)/len(value)
        print(key, round(hits_at_k, 3), 'total questions %d' % len(value))

    return eval_accuracy


def train(qa_model, dataset, args):
    optimizer = torch.optim.Adam(qa_model.parameters(), lr=args.lr)
    optimizer.zero_grad()
    batch_size = args.batch_size
    print('Starting training')
    max_eval_score = 0
    for epoch in range(args.max_epochs):
        qa_model.train()
        epoch_loss = 0
        for i in tqdm(range(len(dataset.train)// batch_size)):
            qa_model.zero_grad()
            question_text, entities_times_padded, entities_times_padded_mask, answers_khot = dataset.get_batch(
                split='train',
                start_index = i*batch_size,
                batch_size=batch_size)
            scores = qa_model.forward(question_text, entities_times_padded.cuda(), entities_times_padded_mask.cuda())
            loss = qa_model.loss(scores, answers_khot.cuda())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(epoch, '/', args.max_epochs, 'epoch loss', epoch_loss)
        if epoch % args.valid_freq == 0 and epoch > 0:
            print('starting eval')
            eval_score = eval(qa_model, dataset, k = args.eval_k)
            if eval_score > max_eval_score:
                print('Valid score increased')
                save_model(qa_model, args.save_to)
                max_eval_score = eval_score


def save_model(qa_model, filename):
    if filename == '':
        filename = 'models/temp.ckpt'
        print('Save filename not specified. Using %s' % filename)
    else:
        filename = 'models/' + filename + '.ckpt'
    print('Saving model to', filename)
    torch.save(qa_model.state_dict(), filename)
    print('Saved model to ', filename)
    return

model_file = 'models/wikidata_small/kg_embeddings/' +  args.tkbc_model_file

tkbc_model = loadTkbcModel(model_file)

if args.model == 'model1':
    qa_model = QA_model(tkbc_model, args)
    dataset = QA_Dataset_model1(args.questions_file)
else:
    print('Model %s not implemented!' % args.model)
    exit(0)

if args.load_from != '':
    filename = 'models/' + args.load_from + '.ckpt'
    print('Loading model from', filename)
    qa_model.load_state_dict(torch.load(filename))
    print('Loaded qa model from ', filename)
else:
    print('Not loading from checkpoint. Starting fresh!')

qa_model = qa_model.cuda()

train(qa_model, dataset, args)

print('Training finished')
