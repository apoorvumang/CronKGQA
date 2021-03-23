# Temporal_KGQA
Temporal KGQA

## Installation

Create a conda environment
``` 
conda create --prefix ./tkgqa_env 
conda activate ./tkgqa_env
```
Install tkbc requirements and setup tkbc
```
conda install --file requirements_tkbc.txt -c pytorch
python setup_tkbc.py install
```
Install KnowBERT (https://github.com/allenai/kb). You can follow instructions on their repo if you prefer (for up-to-date instructions), except
don't create a new conda environment.
```
git clone git@github.com:allenai/kb.git
cd kb
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
python -m spacy download en_core_web_sm
pip install --editable .

```
Install Temporal KGQA requirements
```
conda install --file requirements.txt -c conda-forge
```
## Dataset and pretrained models download

Dataset: ``curl http://transfer.sh/6Nm27/data.zip -o data.zip``

Models: ``curl http://transfer.sh/10EKIn/models.zip -o models.zip``

Unzip these in the root directory.

## Running the code

To generate tkbc embeddings, you can run something like

```
CUDA_VISIBLE_DEVICES=5 python tkbc/learner.py --dataset wikidata_small --model \
TComplEx --rank 156 --emb_reg 1e-2 --time_reg 1e-2 \
--save_to tkbc_model_60k_fulltimestamp.ckpt
```

Finally you can run training of QA model using these trained tkbc embeddings
```
CUDA_VISIBLE_DEVICES=5 python ./train_qa_model.py --frozen 1 --eval_k 1 \
--max_epochs 500 --lr 0.00002 --batch_size 100 --save_to output_model \
--tkbc_model_file tkbc_model_60k_fulltimestamp.ckpt
```

Note: If you get an error about not having GPU support, please install pytorch according to the CUDA version installed on the system. For eg. if you have CUDA 9.2
```
conda install pytorch torchvision torchaudio cudatoolkit=9.2 -c pytorch
```

## Investigating results

Use the branch 'investigate_results' and run a jupyter notebook in the root folder. Make sure to activate the correct environment before running the notebook

The notebook 'cronkgqa_testing.ipynb' can be used to test a model's responses to any textual question, provided you give the list of entities and times in the question as well - this is needed since perfect entity linking is assumed. You can explore the dataset for questions which have entity annotation and modify those questions. You can also make a reverse dict of ``data/wikidata_big/kg/wd_id2entity_text.txt`` and search for wikidata ids of an entity that you want.
