import math
import torch
from torch import nn
import numpy as np
from tkbc.models import TComplEx
from sentence_transformers import SentenceTransformer
from transformers import RobertaModel
from transformers import DistilBertModel
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params
from allennlp.nn.util import move_to_device



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
        self.max_seq_length = 100 # randomly defining max length of tokens for question
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)
        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')
        # print('Random starting embedding')
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.layer_norm = nn.LayerNorm(self.transformer_dim)
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        outputs = self.roberta_model(question_tokenized, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        states = last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(last_hidden_states, dim=1)
        return question_embedding

    def forward(self, question_tokenized, question_attention_mask, 
                entities_times_padded, entities_times_padded_mask, question_text):
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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

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
        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')
        # position embedding for transformer
        self.max_seq_length = 100 # randomly defining max length of tokens for question
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)
        # print('Random starting embedding')
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.layer_norm = nn.LayerNorm(self.transformer_dim)
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def forward(self, question_tokenized, question_attention_mask, 
                entities_times_padded, entities_times_padded_mask, question_text):
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
        

class QA_model_BERT(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.pretrained_weights = 'distilbert-base-uncased'
        self.roberta_model = DistilBertModel.from_pretrained(self.pretrained_weights)
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        self.linear = nn.Linear(768, num_entities + num_times)
        if args.lm_frozen == 1:
            for param in self.roberta_model.parameters():
                param.requires_grad = False
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
                entities_times_padded, entities_times_padded_mask, question_text):
        question_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        scores = self.linear(question_embedding)
#         scores = self.final_linear(output)
        # scores = torch.sigmoid(scores)
        return scores

class QA_model_KnowBERT(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'
        params = Params({"archive_file": archive_file})
        self.kbert_model = ModelArchiveFromParams.from_params(params=params)
        if args.lm_frozen == 1:
            for param in self.kbert_model.parameters():
                param.requires_grad = False
        print('KnowBERT model loaded')
        batch_size = args.batch_size
        self.batcher = KnowBertBatchifier(archive_file, batch_size=batch_size)
        print('KnowBERT batcher loaded')
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        self.linear = nn.Linear(768, num_entities + num_times)
        # transformer
        # print('Random starting embedding')
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        return

    def getQuestionEmbedding(self, question_text):
        for batch in self.batcher.iter_batches(question_text, verbose=False):
            # model_output['contextual_embeddings'] is (batch_size, seq_len, embed_dim) tensor of top layer activations
            batch = move_to_device(batch, 0)
            model_output = self.kbert_model(**batch)
            x = model_output['contextual_embeddings']
            cls_embeddings = x.transpose(0,1)[0]
            return cls_embeddings

    def forward(self, question_tokenized, question_attention_mask, 
                entities_times_padded, entities_times_padded_mask, question_text):
        question_embedding = self.getQuestionEmbedding(question_text)
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
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def forward(self, question_tokenized, question_attention_mask, 
                entities_times_padded, entities_times_padded_mask, question_text):
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


class QA_model_EmbedKGQA(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768 # hardwired from roberta?
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
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
#         self.final_linear = nn.Linear(self.transformer_dim, num_entities + num_times)
        return

    def forward(self, question_tokenized, question_attention_mask, 
                entities_times_padded, entities_times_padded_mask, question_text):
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