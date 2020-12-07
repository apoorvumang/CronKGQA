import math
import torch
from torch import nn
import numpy as np
from models import TComplEx
from sentence_transformers import SentenceTransformer


# training data: questions
# model:
# 1. tkbc model embeddings (may or may not be frozen)
# 2. question sentence embeddings (may or may not be frozen)
# 3. linear layer to project question embeddings (unfrozen)
# 4. transformer that takes these embeddings (unfrozen) (cats them along a dimension, also takes a mask)
# 5. average output embeddings of transformer or take last token embedding?
# 6. linear projection of this embedding to tkbc embedding dimension
# 7. score with all possible entities/times and sigmoid
# or
# 7. directly project to dimension num_entity + num_time and sigmoid
# 8. BCE loss (multiple correct possible)


class QA_model(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.st_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768 # hardwired from sentence_transformers?
        # transformer
        self.transformer_dim = self.tkbc_embedding_dim # keeping same so no need to project embeddings
        self.nhead = 8
        self.num_layers = 6
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)
        # not needed:
        # self.project_tkbc_to_transformer_dim = nn.Linear(self.tkbc_embedding_dim, self.transformer_dim)
        
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

    def forward(self, question_text, entities_times_padded, entities_times_padded_mask):
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        question_embedding = torch.from_numpy(self.st_model.encode(question_text)).cuda()
        question_embedding = self.project_sentence_to_transformer_dim(question_embedding)
        question_embedding = question_embedding.unsqueeze(1)
        sequence = torch.cat([question_embedding, entity_time_embedding], dim=1)
        sequence = torch.transpose(sequence, 0, 1)
        batch_size = len(question_text)
        false_vector = torch.zeros((batch_size, 1), dtype=torch.bool).cuda() # fills with True
        mask = torch.cat([false_vector, entities_times_padded_mask], dim=1)
        output = self.transformer_encoder(sequence, src_key_padding_mask=mask)
        output = torch.transpose(output, 0, 1)
        # summing token embeddings
        output = torch.sum(output, dim=1)
        # now we can either project output to final dim, or we can take dot-product with
        # entity/time embedding weight matrix
        scores = torch.matmul(output, self.entity_time_embedding.weight.data.T)
#         scores = self.final_linear(output)
        # scores = torch.sigmoid(scores)
        return scores
        

class QA_model_EaE(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        # transformer
        self.transformer_dim = self.tkbc_embedding_dim # keeping same so no need to project embeddings
        self.nhead = 8
        self.num_layers = 6
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        # not needed:
        # self.project_tkbc_to_transformer_dim = nn.Linear(self.tkbc_embedding_dim, self.transformer_dim)
        
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
        return

    def forward(self, question_text, entities_times_padded, entities_times_padded_mask):
        entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        question_embedding = torch.from_numpy(self.st_model.encode(question_text)).cuda()
        question_embedding = self.project_sentence_to_transformer_dim(question_embedding)
        question_embedding = question_embedding.unsqueeze(1)
        sequence = torch.cat([question_embedding, entity_time_embedding], dim=1)
        sequence = torch.transpose(sequence, 0, 1)
        batch_size = len(question_text)
        false_vector = torch.zeros((batch_size, 1), dtype=torch.bool).cuda() # fills with True
        mask = torch.cat([false_vector, entities_times_padded_mask], dim=1)
        output = self.transformer_encoder(sequence, src_key_padding_mask=mask)
        output = torch.transpose(output, 0, 1)
        # summing token embeddings
        output = torch.sum(output, dim=1)
        # now we can either project output to final dim, or we can take dot-product with
        # entity/time embedding weight matrix
        scores = torch.matmul(output, self.entity_time_embedding.weight.data.T)
#         scores = self.final_linear(output)
        # scores = torch.sigmoid(scores)
        return scores
