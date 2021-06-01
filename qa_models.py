import math
import torch
from torch import nn
import numpy as np
from tkbc.models import TComplEx
from transformers import RobertaModel
from transformers import BertModel
from transformers import DistilBertModel
# from kb.include_all import ModelArchiveFromParams
# from kb.knowbert_utils import KnowBertBatchifier
# from allennlp.common import Params
# from allennlp.nn.util import move_to_device
from torch.nn import LayerNorm


# training data: questions
# model:
# 1. tkbc model embeddings (may or may not be frozen)
# 2. question sentence embeddings (may or may not be frozen)
# 3. linear layer to project question embeddings (unfrozen)
# 4. transformer that takes these embeddings (unfrozen) (cats them along a dimension, also takes a mask)
# 5. average output embeddings of transformer or take last token embedding?
# 6. linear projection of this embedding to tkbc embedding dimension
# 7. score with all possible entities/times and sigmoid
# 8. BCE loss (multiple correct possible)

class QA_model(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768 # hardwired from roberta?
        self.pretrained_weights = 'distilbert-base-uncased'
        self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        # self.pretrained_weights = 'roberta-base'
        # self.roberta_model = RobertaModel.from_pretrained(self.pretrained_weights)
        if args.lm_frozen == 1:
            for param in self.roberta_model.parameters():
                param.requires_grad = False
        # transformer
        self.transformer_dim = self.tkbc_embedding_dim # keeping same so no need to project embeddings
        self.nhead = args.num_transformer_heads
        self.num_layers = args.num_transformer_layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)
        # creating combined embedding of time and entities (entities come first)
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        ent_emb_matrix = tkbc_model.embeddings[0].weight.data
        time_emb_matrix = tkbc_model.embeddings[2].weight.data
        full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
        self.entity_time_embedding = nn.Embedding(num_entities + num_times, self.tkbc_embedding_dim)
        self.entity_time_embedding.weight.data.copy_(full_embed_matrix)
        # print('Random entity embeddings!')
        self.max_seq_length = 100 # randomly defining max length of tokens for question
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)
        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')
        # print('Random starting embedding')
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.layer_norm = nn.LayerNorm(self.transformer_dim)
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        # TODO delete following 3 lines to load previous model
        # self.linear1 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        # self.dropout = torch.nn.Dropout(0.3)
        # self.bn1 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        outputs = self.roberta_model(question_tokenized, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        states = last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(last_hidden_states, dim=1)
        return question_embedding

    # def forward(self, question_tokenized, question_attention_mask, 
    #             entities_times_padded, entities_times_padded_mask, question_text):
    def forward(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        entities_times_padded = a[2].cuda()
        entities_times_padded_mask = a[3].cuda()
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        # question_embedding = torch.from_numpy(self.st_model.encode(question_text)).cuda()
        question_embedding = self.project_sentence_to_transformer_dim(question_embedding)
        question_embedding = question_embedding.unsqueeze(1)
        sequence = torch.cat([question_embedding, entity_time_embedding], dim=1)
        # making position embedding
        sequence_length = sequence.shape[1]
        v = np.arange(0, sequence_length, dtype=np.long)
        indices_for_position_embedding = torch.from_numpy(v).cuda()
        position_embedding = self.position_embedding(indices_for_position_embedding)
        position_embedding = position_embedding.unsqueeze(0).expand(sequence.shape)

        # adding position embedding
        sequence = sequence + position_embedding
        sequence = self.layer_norm(sequence)

        sequence = torch.transpose(sequence, 0, 1)
        batch_size = entity_time_embedding.shape[0]
        true_vector = torch.zeros((batch_size, 1), dtype=torch.bool).cuda() # fills with True
        mask = torch.cat([true_vector, entities_times_padded_mask], dim=1)


        # comment foll 2 lines for returning to normal behaviour
        # layer_norm = nn.LayerNorm(sequence.size()[1:], elementwise_affine=False)
        # sequence = layer_norm(sequence)
        output = self.transformer_encoder(sequence, src_key_padding_mask=mask)
        output = torch.transpose(output, 0, 1)
        # averaging token embeddings
        output = torch.mean(output, dim=1)
        scores = torch.matmul(output, self.entity_time_embedding.weight.data.T)
#         scores = self.final_linear(output)
        # scores = torch.sigmoid(scores)
        return scores
        
class QA_model_EaE(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768 # hardwired from roberta?
        self.pretrained_weights = 'distilbert-base-uncased'
        self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        if args.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.roberta_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')

        # transformer
        self.transformer_dim = self.tkbc_embedding_dim # keeping same so no need to project embeddings
        self.nhead = args.num_transformer_heads
        self.num_layers = args.num_transformer_layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead, dropout=args.transformer_dropout)
        encoder_norm = LayerNorm(self.transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers, norm=encoder_norm)

        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)


        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        ent_emb_matrix = tkbc_model.embeddings[0].weight.data
        time_emb_matrix = tkbc_model.embeddings[2].weight.data
        full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
        # +1 is for padding idx
        self.entity_time_embedding = nn.Embedding(num_entities + num_times + 1,
                                                  self.tkbc_embedding_dim,
                                                  padding_idx=num_entities + num_times)
        self.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)
        # print('Random entity/time embeddings!')
        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')
        # position embedding for transformer
        self.max_seq_length = 100 # randomly defining max length of tokens for question
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)
        # print('Random starting embedding')
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.layer_norm = nn.LayerNorm(self.transformer_dim)
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def forward(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        entities_times_padded = a[2].cuda()
        entities_times_padded_mask = a[3].cuda()
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        outputs = self.roberta_model(question_tokenized, attention_mask=question_attention_mask)
        last_hidden_states = outputs.last_hidden_state
        question_embedding = self.project_sentence_to_transformer_dim(last_hidden_states)

        # we add those 2 now, and do layer norm??
        combined_embed = question_embedding + entity_time_embedding

        # also need to add position embedding
        sequence_length = combined_embed.shape[1]
        v = np.arange(0, sequence_length, dtype=np.long)
        indices_for_position_embedding = torch.from_numpy(v).cuda()
        position_embedding = self.position_embedding(indices_for_position_embedding)
        position_embedding = position_embedding.unsqueeze(0).expand(combined_embed.shape)

        combined_embed = combined_embed + position_embedding

        # layer_norm = nn.LayerNorm(combined_embed.size()[1:], elementwise_affine=False)
        # combined_embed = layer_norm(combined_embed)
        combined_embed = self.layer_norm(combined_embed)
        # need to transpose lol, why is this like this?
        # why is first dimension sequence length and not batch size?
        combined_embed = torch.transpose(combined_embed, 0, 1)
        # question_embedding = torch.from_numpy(self.st_model.encode(question_text)).cuda()
        output = self.transformer_encoder(combined_embed, src_key_padding_mask=entities_times_padded_mask)
        output = output[0] #cls token embedding
        # output = torch.transpose(output, 0, 1)
        # # averaging token embeddings
        # output = torch.mean(output, dim=1)
        scores = torch.matmul(output, self.entity_time_embedding.weight.data[:-1, :].T) # cuz padding idx
#         scores = self.final_linear(output)
        # scores = torch.sigmoid(scores)
        return scores

class QA_model_EaE_replace(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768 # hardwired from roberta?
        self.pretrained_weights = 'distilbert-base-uncased'
        self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        if args.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.roberta_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')

        # transformer
        self.transformer_dim = self.tkbc_embedding_dim # keeping same so no need to project embeddings
        self.nhead = args.num_transformer_heads
        self.num_layers = args.num_transformer_layers
        # self.transformer_dropout = args.transformer_dropout
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead, dropout=args.transformer_dropout)
        encoder_norm = LayerNorm(self.transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers, norm=encoder_norm)

        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)

        self.project_entity = nn.Linear(self.tkbc_embedding_dim, self.transformer_dim)

        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        ent_emb_matrix = tkbc_model.embeddings[0].weight.data
        time_emb_matrix = tkbc_model.embeddings[2].weight.data
        full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
        # +1 is for padding idx
        self.entity_time_embedding = nn.Embedding(num_entities + num_times + 1,
                                                  self.tkbc_embedding_dim,
                                                  padding_idx=num_entities + num_times)
        self.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)
        # print('Random entity/time embeddings!')
        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')
        
        # position embedding for transformer
        self.max_seq_length = 100 # randomly defining max length of tokens for question
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)
        # print('Random starting embedding')
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.layer_norm = nn.LayerNorm(self.transformer_dim)
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def invert_binary_tensor(self, tensor):
        ones_tensor = torch.ones(tensor.shape, dtype=torch.float32).cuda()
        inverted = ones_tensor - tensor
        return inverted

    def forward(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        entities_times_padded = a[2].cuda()
        entity_mask_padded = a[3].cuda()
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        outputs = self.roberta_model(question_tokenized, attention_mask=question_attention_mask)
        last_hidden_states = outputs.last_hidden_state
        question_embedding = self.project_sentence_to_transformer_dim(last_hidden_states)
        entity_mask = entity_mask_padded.unsqueeze(-1).expand(question_embedding.shape)
        masked_question_embedding = question_embedding * entity_mask # set entity positions 0
        # project to get into same space as word vectors
        # E-BERT does pretraining of this kind of projection
        # TODO: do we need such projection training beforehand?
        entity_time_embedding_projected = self.project_entity(entity_time_embedding)
        masked_entity_time_embedding = entity_time_embedding_projected * self.invert_binary_tensor(entity_mask) # invert mask for this
        combined_embed = masked_question_embedding + masked_entity_time_embedding
        # also need to add position embedding
        sequence_length = combined_embed.shape[1]
        v = np.arange(0, sequence_length, dtype=np.long)
        indices_for_position_embedding = torch.from_numpy(v).cuda()
        position_embedding = self.position_embedding(indices_for_position_embedding)
        position_embedding = position_embedding.unsqueeze(0).expand(combined_embed.shape)

        combined_embed = combined_embed + position_embedding

        combined_embed = self.layer_norm(combined_embed)
        combined_embed = torch.transpose(combined_embed, 0, 1)

        mask2 = ~(question_attention_mask.bool()).cuda()
        output = self.transformer_encoder(combined_embed, src_key_padding_mask=mask2)
        output = output[0] #cls token embedding
        scores = torch.matmul(output, self.entity_time_embedding.weight.data[:-1, :].T) # cuz padding idx
        return scores
        

class QA_model_BERT(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        # self.pretrained_weights = 'distilbert-base-uncased'
        # self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        # self.pretrained_weights = 'roberta-base'
        # self.roberta_model = RobertaModel.from_pretrained(self.pretrained_weights)
        self.pretrained_weights = 'bert-base-uncased'
        self.roberta_model = BertModel.from_pretrained(self.pretrained_weights)

        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        self.linear = nn.Linear(768, num_entities + num_times)
        if args.lm_frozen == 1:
            for param in self.roberta_model.parameters():
                param.requires_grad = False
        # transformer
        # print('Random starting embedding')
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding

    def forward(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        entities_times_padded = a[2].cuda()
        entities_times_padded_mask = a[3].cuda()
        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        scores = self.linear(question_embedding)
#         scores = self.final_linear(output)
        # scores = torch.sigmoid(scores)
        return scores

 

class QA_model_Only_Embeddings(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768 # hardwired from roberta?
        self.pretrained_weights = 'distilbert-base-uncased'
        # transformer
        self.transformer_dim = self.tkbc_embedding_dim # keeping same so no need to project embeddings
        self.nhead = args.num_transformer_heads
        self.num_layers = args.num_transformer_layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)
        # creating combined embedding of time and entities (entities come first)
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        ent_emb_matrix = tkbc_model.embeddings[0].weight.data
        time_emb_matrix = tkbc_model.embeddings[2].weight.data
        full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
        self.entity_time_embedding = nn.Embedding(num_entities + num_times, self.tkbc_embedding_dim)
        self.entity_time_embedding.weight.data.copy_(full_embed_matrix)
        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')
        # print('Random starting embedding')
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.layer_norm = nn.LayerNorm(self.transformer_dim)
        self.max_seq_length = 100 # randomly defining max length of tokens for question
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def forward(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        entities_times_padded = a[2].cuda()
        entities_times_padded_mask = a[3].cuda()
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        sequence = entity_time_embedding
        sequence_length = sequence.shape[1]
        v = np.arange(0, sequence_length, dtype=np.long)
        indices_for_position_embedding = torch.from_numpy(v).cuda()
        position_embedding = self.position_embedding(indices_for_position_embedding)
        position_embedding = position_embedding.unsqueeze(0).expand(sequence.shape)

        # adding position embedding
        sequence = sequence + position_embedding
        sequence = self.layer_norm(sequence)

        sequence = torch.transpose(sequence, 0, 1)
        
        mask = entities_times_padded_mask
        output = self.transformer_encoder(sequence, src_key_padding_mask=mask)
        output = torch.transpose(output, 0, 1)
        # averaging token embeddings
        output = torch.mean(output, dim=1)
        scores = torch.matmul(output, self.entity_time_embedding.weight.data.T)
#         scores = self.final_linear(output)
        # scores = torch.sigmoid(scores)
        return scores


class QA_model_EmbedKGQA(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768 # hardwired from roberta?
        self.pretrained_weights = 'distilbert-base-uncased'
        self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)

        if args.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.roberta_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')

        # creating combined embedding of time and entities (entities come first)
        self.tkbc_model = tkbc_model
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        ent_emb_matrix = tkbc_model.embeddings[0].weight.data
        time_emb_matrix = tkbc_model.embeddings[2].weight.data
        full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
        self.entity_time_embedding = nn.Embedding(num_entities + num_times, self.tkbc_embedding_dim)
        self.entity_time_embedding.weight.data.copy_(full_embed_matrix)
        self.num_entities = num_entities
        self.num_times = num_times

        self.answer_type_embedding = nn.Embedding(2, num_entities + num_times)
        x = torch.zeros(num_entities + num_times)
        x[num_entities:] = torch.ones(num_times) * -1e10
        self.answer_type_embedding.weight.data[0].copy_(x)

        x = torch.zeros(num_entities + num_times)
        x[:num_entities] = torch.ones(num_entities) * -1e10
        self.answer_type_embedding.weight.data[1].copy_(x)

        # now freeze this
        self.answer_type_embedding.weight.requires_grad = False


        #Should you combine all entities while entity scoring?
        self.combine_all_entities_bool=True if args.combine_all_ents!="None" else False
        # self.combine_all_entities_func=(lambda x: torch.sum(x,dim=1)) if args.combine_all_ents=="add"\
        #     else (lambda x: torch.prod(x, dim=1)) if args.combine_all_ents == "mult"\
        #     else None
        self.combine_all_entities_func_forReal=nn.Linear(self.tkbc_embedding_dim,self.tkbc_model.rank)
        self.combine_all_entities_func_forCmplx=nn.Linear(self.tkbc_embedding_dim,self.tkbc_model.rank)

        # if self.combine_all_entities_bool:
        #     self.


        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
            for param in self.tkbc_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')
        # print('Random starting embedding')
        self.linear = nn.Linear(768, self.tkbc_embedding_dim) # to project question embedding

        self.linear1 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        # self.linear1.weight.data.copy_(torch.eye(self.tkbc_embedding_dim))
        self.linear2 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        # self.linear2.weight.data.copy_(torch.eye(self.tkbc_embedding_dim))
        # self.loss = nn.BCELoss(reduction='mean')
        self.loss = nn.CrossEntropyLoss(reduction='mean')

        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding

    # scoring function from TComplEx
    def score_time(self, head_embedding, tail_embedding, relation_embedding):
        lhs = head_embedding
        rhs = tail_embedding
        rel = relation_embedding

        time = self.tkbc_model.embeddings[2].weight
        # time = self.entity_time_embedding.weight

        lhs = lhs[:, :self.tkbc_model.rank], lhs[:, self.tkbc_model.rank:]
        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
        rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def score_entity(self, head_embedding, tail_embedding,relation_embedding, time_embedding):
        if self.combine_all_entities_bool:
            lhs=  self.combine_all_entities_func_forReal(torch.cat((head_embedding[:,:self.tkbc_model.rank],
                                                                    tail_embedding[:,:self.tkbc_model.rank]),dim=1))\
                ,self.combine_all_entities_func_forCmplx(torch.cat((head_embedding[:,self.tkbc_model.rank:],
                                                                tail_embedding[:,self.tkbc_model.rank:]),dim=1))

        else:
            lhs = head_embedding[:, :self.tkbc_model.rank], head_embedding[:, self.tkbc_model.rank:]
        rel = relation_embedding
        time = time_embedding

        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        right = self.tkbc_model.embeddings[0].weight
        # right = self.entity_time_embedding.weight
        right = right[:, :self.tkbc_model.rank], right[:, self.tkbc_model.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               )

    def forward(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        heads = a[2].cuda()
        tails = a[3].cuda()
        times = a[4].cuda()

        head_embedding = self.entity_time_embedding(heads)
        tail_embedding = self.entity_time_embedding(tails)
        time_embedding = self.entity_time_embedding(times)
        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        relation_embedding = self.linear(question_embedding)

        relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
        relation_embedding2 = self.dropout(self.bn2(self.linear2(relation_embedding)))
        scores_time = self.score_time(head_embedding, tail_embedding, relation_embedding1)
        scores_entity = self.score_entity(head_embedding, tail_embedding,relation_embedding2, time_embedding)

        scores = torch.cat((scores_entity, scores_time), dim=1)
        return scores



class QA_model_EmbedKGQA_complex(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768 # hardwired from roberta?
        self.pretrained_weights = 'distilbert-base-uncased'
        self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        if args.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.roberta_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')

        # creating combined embedding of time and entities (entities come first)
        self.entity_embedding = tkbc_model.embeddings[0]
        self.time_embedding = tkbc_model.embeddings[2]
        self.rank = tkbc_model.rank
        self.num_entities = tkbc_model.embeddings[0].weight.shape[0]
        self.num_times = tkbc_model.embeddings[2].weight.shape[0]

        if args.frozen == 1:
            print('Freezing entity but not time embeddings')
            self.entity_embedding.weight.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')
        # print('Random starting embedding')
        self.linear = nn.Linear(768, self.tkbc_embedding_dim) # to project question embedding

        self.linear1 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        # self.linear1.weight.data.copy_(torch.eye(self.tkbc_embedding_dim))
        # self.linear2.weight.data.copy_(torch.eye(self.tkbc_embedding_dim))
        # self.loss = nn.BCELoss(reduction='mean')
        self.loss = nn.CrossEntropyLoss(reduction='mean')

        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding


    def score(self, head_embedding, relation_embedding):
        lhs = head_embedding
        rel = relation_embedding

        right = torch.cat((self.entity_embedding.weight, self.time_embedding.weight), dim=0)
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        right = right[:, :self.rank], right[:, self.rank:]

        return (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) + (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
               


    # def forward(self, question_tokenized, question_attention_mask, 
    #             heads, times, question_text):
    def forward(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        heads = a[2].cuda()

        head_embedding = self.entity_embedding(heads)
        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        relation_embedding = self.linear(question_embedding)
        relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
        scores = self.score(head_embedding, relation_embedding1)
        # exit(0)
        # scores = torch.cat((scores_entity, scores_time), dim=1)
        return scores
