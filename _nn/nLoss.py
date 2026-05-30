exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")

import torch

from torch import Tensor

def loss_reduce(loss: Tensor, reduction='mean'):
    """lambda, 'none','mean','sum','min','max','norm','abssum'"""
    if not isinstance(loss,Tensor):return 0
    if len(loss.shape) <= 0:  
        return loss
    if callable(reduction):
        loss = reduction(loss)
    if isinstance(reduction, str):
        assert reduction in ['none', 'mean', 'sum', 'min', 'max', 'norm', 'abssum']
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'abssum':
            return loss.abs().sum()
        elif reduction == 'max':
            return loss.max()
        elif reduction == 'min':
            return loss.min()
        elif reduction == 'norm':
            return torch.norm(loss, 2)
    return loss
