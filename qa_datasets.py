from pathlib import Path
import pkg_resources
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List


import numpy as np
import torch
# from qa_models import QA_model
import utils
from tqdm import tqdm
# warning: padding id 0 is being used, can have issue like in Tucker
# however since so many entities (and timestamps?), it may not pose problem

class QA_Dataset(object):
    def __init__(self, 
                filename='/scratche/home/apoorv/tempqa/data/questions/questions_position_held_small_with_paraphrases_v2_shuffled.pickle'):
        num_valid = 5000
        num_test = 5000
        folder_name = 'data/wikidata_big/questions/'
        
        self.train = pickle.load(open(folder_name + 'train.pickle', 'rb'))
        self.valid = pickle.load(open(folder_name + 'valid.pickle', 'rb'))
        self.test = pickle.load(open(folder_name + 'test.pickle', 'rb'))
        self.all_dicts = utils.getAllDicts()
        # questions = pickle.load(open(filename, 'rb'))
        # self.valid = questions[:num_valid]
        # self.test = questions[num_valid: num_valid + num_test]
        # self.train = questions[num_valid + num_test :]
        # 
        print('Total questions = ', len(self.train + self.valid + self.test ))

        self.data = {}
        self.data['valid'] = self.valid
        self.data['train'] = self.train
        self.data['test'] = self.test

    def getEntitiesLocations(self, question):
        question_text = question['question']
        entities = question['entities']
        ent2id = self.all_dicts['ent2id']
        loc_ent = []
        for e in entities:
            e_id = ent2id[e]
            location = question_text.find(e)
            loc_ent.append((location, e_id))
        return loc_ent

    def getTimesLocations(self, question):
        question_text = question['question']
        times = question['times']
        ts2id = self.all_dicts['ts2id']
        loc_time = []
        for t in times:
            t_id = ts2id[(t,0,0)] + len(self.all_dicts['ent2id']) # add num entities
            location = question_text.find(str(t))
            loc_time.append((location, t_id))
        return loc_time

    def getOrderedEntityTimeIds(self, question):
        loc_ent = self.getEntitiesLocations(question)
        loc_time = self.getTimesLocations(question)
        loc_all = loc_ent + loc_time
        loc_all.sort()
        ordered_ent_time = [x[1] for x in loc_all]
        return ordered_ent_time

    def entitiesToIds(self, entities):
        output = []
        ent2id = self.all_dicts['ent2id']
        for e in entities:
            output.append(ent2id[e])
        return output
    
    def idToEntTime(self, id):
        type = self.getIdType(id)
        if type == 'entity':
            return self.all_dicts['id2ent'][id]
        else:
            return self.all_dicts['id2ts'][id]
        
    def getIdType(self, id):
        if id < len(self.all_dicts['ent2id']):
            return 'entity'
        else:
            return 'time'
    
    def getEntityToText(self, entity_wd_id):
        return self.all_dicts['wd_id_to_text'][entity_wd_id]
    
    def getEntityIdToText(self, id):
        ent = self.all_dicts['id2ent'][id]
        return self.getEntityToText(ent)
    
    def getEntityIdToWdId(self, id):
        return self.all_dicts['id2ent'][id]

    def timesToIds(self, times):
        output = []
        ts2id = self.all_dicts['ts2id']
        for t in times:
            output.append(ts2id[(t, 0, 0)])
        return output

    def getAnswersFromScores(self, scores, largest=True, k=10):
        _, ind = torch.topk(scores, k, largest=largest)
        predict = ind
        answers = []
        for a_id in predict:
            a_id = a_id.item()
            type = self.getIdType(a_id)
            if type == 'entity':
                # answers.append(self.getEntityIdToText(a_id))
                answers.append(self.getEntityIdToWdId(a_id))
            else:
                time_id = a_id - len(self.all_dicts['ent2id'])
                time = self.all_dicts['id2ts'][time_id]
                answers.append(time[0])
        return answers

    # from pytorch Transformer:
    # If a BoolTensor is provided, the positions with the value of True will be ignored 
    # while the position with the value of False will be unchanged.
    # 
    # so we want to pad with True
    def padding_tensor(self, sequences):
        """
        :param sequences: list of tensors
        :return:
        """
        num = len(sequences)
        max_len = max([s.size(0) for s in sequences])
        out_dims = (num, max_len)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        # mask = sequences[0].data.new(*out_dims).fill_(0)
        mask = torch.ones((num, max_len), dtype=torch.bool) # fills with True
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = False # fills good area with False
        return out_tensor, mask
    
    def toOneHot(self, indices, vec_len):
        indices = torch.LongTensor(indices).cuda()
        one_hot = torch.FloatTensor(vec_len).cuda()
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot



class QA_Dataset_model1(QA_Dataset):
    def __init__(self, dataset_file):
        super().__init__(dataset_file)

    def process_data(self, data):
        question_text = []
        entity_time_ids = []
        
        num_total_entities = len(self.all_dicts['ent2id'])
        num_total_times = len(self.all_dicts['ts2id'])
        answers_khot = []

        for question in data:
            question_text.append(question['paraphrases'][0])
            # question_text.append(question['template']) # todo: this is incorrect
            # et_id = []
            # entity_ids = self.entitiesToIds(self.getOrderedEntities(question))
            # time_ids = self.timesToIds(self.getOrderedTimes(question))
            # # adding num_total_entities to each time id
            # for i in range(len(time_ids)):
            #     time_ids[i] += num_total_entities
            # et_id = entity_ids + time_ids # todo: maybe we want ordering as is in question? here entities first, time 2nd
            et_id = self.getOrderedEntityTimeIds(question)
            entity_time_ids.append(torch.tensor(et_id, dtype=torch.long))
            if question['answer_type'] == 'entity':
                answers = self.entitiesToIds(question['answers'])
            else:
                # adding num_total_entities to each time id
                answers = [x + num_total_entities for x in self.timesToIds(question['answers'])]
            answers_khot.append(self.toOneHot(answers, num_total_entities + num_total_times))

        entities_times_padded, entities_times_padded_mask = self.padding_tensor(entity_time_ids)
        answers_khot = torch.stack(answers_khot)
        return question_text, entities_times_padded, entities_times_padded_mask, answers_khot

    def get_batch(self, split='train', start_index=0, batch_size=50):
        # just example
        return self.process_data(self.data[split][start_index: start_index + batch_size])
        
    
class QA_Dataset_EaE(QA_Dataset):
    def __init__(self, dataset_file):
        super().__init__(dataset_file)

    def process_data(self, data):
        # we want tokenized question
        # we also want to know position of entity/time in tokenized question
        # finally, we want khot answer
        return
