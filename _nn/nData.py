import torch
import torch.nn as nn
import random
import numpy as np
import time
def freeze(x:nn.Parameter):
    """with optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)"""
    x.requires_grad=False
_device= 'cuda' if torch.cuda.is_available() else ( 'mps' if torch.mps.is_available() else 'cpu')
def auto_device(x:torch.Tensor=None): return _device if x is None else x.to(_device)
min_max_var=lambda w:[w.min(),w.max(),torch.var_mean(w)]
def random_seed(seed=None):
    if seed is None:seed=time.time_ns()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
