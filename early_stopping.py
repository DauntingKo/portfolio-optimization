import numpy as np
import torch
import os

def save_best_loss(best_loss, filepath):
    with open(filepath, "w") as f:
        f.write(str(best_loss.item()))

def load_best_loss(filepath):
    if os.path.exists(filepath):
        print('found best_loss.txt')
        with open(filepath, "r") as f:
            best_loss = float(f.read())
        return best_loss
    else:
        return None

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : Save dir
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = load_best_loss(os.path.join(self.save_path, 'best_loss.txt'))
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = 1

    def __call__(self, val_loss, model, **kwargs):
        
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, kwargs)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, kwargs)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, kwargs):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(kwargs['confusion_matrix'])
        self.best_test_confusion_matrix = kwargs['confusion_matrix']
        self.best_epoch = kwargs['epoch']
        self.best_test_acc = kwargs['test_acc']
        self.best_test_loss = kwargs['test_loss']
        path = os.path.join(self.save_path, 'best_model.pt')
        torch.save(model, path)
        save_best_loss(self.best_test_loss,os.path.join(self.save_path, 'best_loss.txt'))
        self.val_loss_min = val_loss

