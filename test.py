from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params
from allennlp.nn.util import move_to_device

import torch

# a pretrained model, e.g. for Wordnet+Wikipedia
archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'

# load model and batcher
params = Params({"archive_file": archive_file})
model = ModelArchiveFromParams.from_params(params=params)
model.cuda()
print('Model loaded')
batch_size = 128
batcher = KnowBertBatchifier(archive_file, batch_size=batch_size)
print('Batched made')
sentences = ["Paris is located in France.", "KnowBert is a knowledge enhanced BERT"]

while(len(sentences) < batch_size):
    sentences.append("KnowBert is a knowledge enhanced BERT")
# batcher takes raw untokenized sentences
# and yields batches of tensors needed to run KnowBert
for batch in batcher.iter_batches(sentences, verbose=True):
    # model_output['contextual_embeddings'] is (batch_size, seq_len, embed_dim) tensor of top layer activations
    batch = move_to_device(batch, 0)
    model_output = model(**batch)
    x = model_output['contextual_embeddings']
    cls_embeddings = x.transpose(0,1)[0]
    print(cls_embeddings.shape)

    # print(model_output.keys())
    # print(model_output['contextual_embeddings'].shape)
