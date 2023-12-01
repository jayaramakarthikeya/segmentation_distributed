import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, model, model_type,optimizer, config, checkpoint_dir, mnt_mode = 'max', patience=7, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir
        self.best_score = -np.inf if mnt_mode == 'max' else np.inf
        self.delta = delta
        self.model = model
        self.model_type = model_type
        self.config = config
        self.optimizer = optimizer
    
        self.trace_func = trace_func
    def __call__(self, score, epoch, model):


        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score,epoch, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score,epoch, model)
            self.counter = 0

    def save_checkpoint(self, score,epoch, model):
        '''Saves model when validation metric decrease.'''
        if self.verbose:
            self.trace_func(f'Metric improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        state = {
            'arch': self.model.model_type,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'config': self.config
        }
        checkpoint_path = os.path.join(self.checkpoint_dir,f"checkpoint-epoch{epoch}.pth")
        torch.save(state, checkpoint_path)