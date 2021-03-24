# Temporal_KGQA
Note: If you face any installation issues, please create a new issue or email me at apoorvsaxena@iisc.ac.in . Even if you are able to resolve the issue on your own, it will be helpful for me since I can make the necessary changes so as to make installation easier for any future users.

## Installation

Clone and create a conda environment
``` 
git clone git@github.com:apoorvumang/Temporal_KGQA.git
cd Temporal_KGQA
git checkout investigate_results
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
cd ..
conda install --file requirements.txt -c conda-forge
```

## Dataset and pretrained models download
```
curl http://transfer.sh/6Nm27/data.zip -o data.zip
curl http://transfer.sh/iQxhN/models.zip -o models.zip
unzip -q data.zip && unzip -q models.zip
rm data.zip && rm models.zip
```

## Hack to fix misc errors
```
pip install scikit-learn==0.22.2
mkdir results && mkdir results/wikidata_big
```

## Running the code

To generate tkbc embeddings, you can run something like

```
TODO: add command with right hyperparams for training the temporal KGE
```

Finally you can run training of QA model using these trained tkbc embeddings. embedkgqa model = cronkgqa (will fix naming etc. soon)
```
 CUDA_VISIBLE_DEVICES=1 python -W ignore ./train_qa_model.py --frozen 1 --eval_k 1 --max_epochs 200 \
 --lr 0.00002 --batch_size 250 --mode train --tkbc_model_file tkbc_model_17dec.ckpt \
 --dataset wikidata_big --valid_freq 3 --model embedkgqa --valid_batch_size 50  \
 --save_to temp --lm_frozen 1 --eval_split valid
 ```
 
 Evaluating the pretrained model (CronKGQA):
 ```
  CUDA_VISIBLE_DEVICES=1 python -W ignore ./train_qa_model.py \
 --mode eval --tkbc_model_file tkbc_model_17dec.ckpt \
 --dataset wikidata_big --model embedkgqa --valid_batch_size 50  \
 --load_from embedkgqa_dual_frozen_lm_fix_order_ce --eval_split test
 ```

Please explore the qa_models.py file for other models, you can change the model by providing the --model parameter.

Note: If you get an error about not having GPU support, please install pytorch according to the CUDA version installed on the system. For eg. if you have CUDA 9.2
```
conda install pytorch torchvision torchaudio cudatoolkit=9.2 -c pytorch
```

## Investigating results

Use the branch 'investigate_results' and run a jupyter notebook in the root folder. Make sure to activate the correct environment before running the notebook

The notebook 'cronkgqa_testing.ipynb' can be used to test a model's responses to any textual question, provided you give the list of entities and times in the question as well - this is needed since perfect entity linking is assumed. You can explore the dataset for questions which have entity annotation and modify those questions. You can also make a reverse dict of ``data/wikidata_big/kg/wd_id2entity_text.txt`` and search for wikidata ids of an entity that you want.
