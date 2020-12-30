import math
import torch
from torch import nn
import numpy as np
from tkbc.models import TComplEx
from sentence_transformers import SentenceTransformer
from transformers import RobertaModel
from transformers import DistilBertModel

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
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        # transformer
        self.transformer_dim = self.tkbc_embedding_dim # keeping same so no need to project embeddings
        self.nhead = 8
        self.num_layers = 6
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
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding

    def forward(self, question_tokenized, question_attention_mask, 
                entities_times_padded, entities_times_padded_mask):
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        # question_embedding = torch.from_numpy(self.st_model.encode(question_text)).cuda()
        question_embedding = self.project_sentence_to_transformer_dim(question_embedding)
        question_embedding = question_embedding.unsqueeze(1)
        sequence = torch.cat([question_embedding, entity_time_embedding], dim=1)
        sequence = torch.transpose(sequence, 0, 1)
        batch_size = entity_time_embedding.shape[0]
        false_vector = torch.zeros((batch_size, 1), dtype=torch.bool).cuda() # fills with True
        mask = torch.cat([false_vector, entities_times_padded_mask], dim=1)
        output = self.transformer_encoder(sequence, src_key_padding_mask=mask)
        output = torch.transpose(output, 0, 1)
        # averaging token embeddings
        output = torch.mean(output, dim=1)
        scores = torch.matmul(output, self.entity_time_embedding.weight.data.T)
#         scores = self.final_linear(output)
        # scores = torch.sigmoid(scores)
        return scores
        

class QA_model_KnowBERT(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()

        self.pretrained_weights = 'distilbert-base-uncased'
        self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        self.linear = nn.Linear(768, num_entities + num_times)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        # transformer
        # print('Random starting embedding')
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding

    def forward(self, question_tokenized, question_attention_mask, 
                entities_times_padded, entities_times_padded_mask):
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
        self.nhead = 8
        self.num_layers = 6
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
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def forward(self, question_tokenized, question_attention_mask, 
                entities_times_padded, entities_times_padded_mask):
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        sequence = entity_time_embedding
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