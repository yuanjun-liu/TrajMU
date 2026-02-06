exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from mu.MU import *
from _nn.nBasic import basic_train
import torch,math
from torch import nn
from torch.nn.modules import Module
from torch.utils.data.dataloader import DataLoader
from typing import Dict,List
import torch.nn.functional as F
import numpy as np
from _tool.mList import topk
from _tool.mData import deepcopy
from _nn.nBasic import to_device


class Retrain(MU):
    """retrain on Dr"""
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        return self.model_retrain()


