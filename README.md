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
Install Temporal KGQA requirements
```
conda install --file requirements.txt -c conda-forge
```

Download pretrained models and extract in the models folder. You will need at least the pretrained tkbc model.

Finally you can run training of QA model using
```
CUDA_VISIBLE_DEVICES=5 python ./train_qa_model.py --save_to 'mymodel' --frozen 0 --eval_k 1 --max_epochs 500 --lr 0.00002 --batch_size 100 
```

Note: You will need GPU support to load/train the model. If you get an error about not having GPU support, please install pytorch according to the CUDA version installed on the system. For eg. if you have CUDA 9.2
```
conda install pytorch torchvision torchaudio cudatoolkit=9.2 -c pytorch
```
