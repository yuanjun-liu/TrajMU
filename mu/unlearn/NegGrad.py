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



class NegGrad(MU):
    """-loss on Du"""
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        def lossF(*arg,**kw):
            return -self.model.lossF(*arg,**kw) 
        opt,sch=self.model.call_opt_sch()
        loader=DataLoader(self.du,self.bs,True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        self.model=basic_train(model=self.model,train_loader=loader, opt=opt,call_loss=lossF,epoch_max=self.epoch_tune,device=self.device,sch=sch,epoch_min=1,estop_fn=estop_fn,log=ptloss,log_step=1,save_ep_fn=self.save_epoch)
        return self.model

