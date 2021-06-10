"""
Credits: https://github.com/Bjarten/early-stopping-pytorch
The authors did not write the code in this file
"""

import numpy as np
import torch

class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, score_at_min1=0,patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = score_at_min1
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.state_dict_list=[None]*patience
        self.improved=0
        self.stop_update=0
    def __call__(self, score, epoch,model):
        if not self.stop_update:
            if self.verbose:
                self.trace_func(f'\033[91m The val score  of epoch {epoch} is {score:.4f} \033[0m')
            if score <= self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'\033[93m EarlyStopping counter: {self.counter} out of {self.patience} \033[0m')
                if self.counter >= self.patience:
                    self.early_stop = True
                self.improved=0
            else:
                self.save_checkpoint(score, model)
                self.best_score = score
                self.counter = 0
                self.improved=1
        else:
            self.improved=0 #not needed though

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        self.trace_func(f'\033[92m Validation score improved ({self.best_score:.4f} --> {score:.4f}). \033[0m')
        # torch.save(model.state_dict(), self.path)
