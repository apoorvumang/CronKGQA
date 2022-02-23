# CronKGQA
This is the code for our ACL 2021 paper [Question Answering over Temporal Knowledge Graphs](https://arxiv.org/abs/2106.01515)

**UPDATE**: There has been a small update to the CronQuestions dataset. There was an error in dataset creation that resulted in '{tail2}' being present in the question string instead of getting
slot filled by the proper entity (https://github.com/apoorvumang/CronKGQA/issues/13). This affected 31951/410000 questions. We have uploaded a fixed version of the dataset. Although the numbers reported in the paper are based on the original version of the dataset, we encourage everyone to used the fixed version. Details of this are in the dataset download section.


## Installation

Clone and create a conda environment
``` 
git clone git@github.com:apoorvumang/CronKGQA.git
cd CronKGQA
conda create --prefix ./cronkgqa_env python=3.8
conda activate ./cronkgqa_env
```
<!-- Make sure ``python`` and ``pip`` commands point to ``./tkgqa_env``. Output of ``which`` should be something like
```
which python
[...]/CronKGQA/cronkgqa_env/bin/python
```
If this is not the case, try replacing ``python`` with ``python3``. If that works, replace ``python`` with ``python3`` in all commands below.
 -->
 
We use TComplEx KG Embeddings as proposed in [Tensor Decompositions for temporal knowledge base completion](https://arxiv.org/abs/2004.04926). We use a slightly modified version of their code from https://github.com/facebookresearch/tkbc

Install tkbc requirements and setup tkbc
```
conda install --file requirements_tkbc.txt -c pytorch
python setup_tkbc.py install
```

Install CronKGQA requirements
```
conda install --file requirements.txt -c conda-forge
```

## Dataset and pretrained models download

Download and unzip ``data_v2.zip`` and ``models.zip`` in the root directory. ``data.zip`` contains the old version without the '{tail2}' fix, please refrain from using it.

Drive: https://drive.google.com/drive/folders/15L4bpGEvCCp7Kuz6xJOFBQFzQGWKJ9rL?usp=sharing, or use ``gdown``
 
```
gdown https://drive.google.com/uc\?id\=1fe7-x7ChszqzczKncoZcpwmWc1PBq1_0
gdown https://drive.google.com/uc\?id\=18w_aPl-oLfWnTLnoMnTU9Pm4El1T9wkB
unzip -q data_v2.zip && unzip -q models.zip
rm data_v2.zip && rm models.zip
```

## Try out pretrained model

Run a jupyter notebook in the root folder. Make sure to activate the correct environment before running the notebook

The notebook ``cronkgqa_testing.ipynb`` can be used to test a model's responses to any textual question, provided you give the list of entities and times in the question as well - this is needed since perfect entity linking is assumed. You can explore the dataset for questions which have entity annotation and modify those questions. You can also make a reverse dict of ``data/wikidata_big/kg/wd_id2entity_text.txt`` and search for wikidata ids of an entity that you want.


## Running the code


Finally you can run training of QA model using these trained tkbc embeddings. embedkgqa model = cronkgqa (will fix naming etc. soon)
```
 CUDA_VISIBLE_DEVICES=1 python -W ignore ./train_qa_model.py --frozen 1 --eval_k 1 --max_epochs 200 \
 --lr 0.00002 --batch_size 250 --mode train --tkbc_model_file tcomplex_17dec.ckpt \
 --dataset wikidata_big --valid_freq 3 --model embedkgqa --valid_batch_size 50  \
 --save_to temp --lm_frozen 1 --eval_split valid
 ```
 
Evaluating the pretrained model (CronKGQA):
 ```
  CUDA_VISIBLE_DEVICES=1 python -W ignore ./train_qa_model.py \
 --mode eval --tkbc_model_file tcomplex_17dec.ckpt \
 --dataset wikidata_big --model embedkgqa --valid_batch_size 50  \
 --load_from cronkgqa_29may --eval_split test
 ```

Please explore the qa_models.py file for other models, you can change the model by providing the --model parameter.

## How to cite
If you used our work or found it helpful, please use the following citation:

```
@inproceedings{saxena2021cronkgqa,
  title={Question Answering over Temporal Knowledge Graphs},
  author={Saxena, Apoorv and Chakrabarti, Soumen and Talukdar, Partha},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```
