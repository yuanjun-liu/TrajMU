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
class _SFRon_AdaptiveLoss(torch.nn.Module):
    def __init__(self, loss_function, lambd=1, reduction='mean'):
        super(_SFRon_AdaptiveLoss, self).__init__()
        self.loss_function = loss_function 
        self.lambd = lambd
        self.reduction = reduction
    def forward(self, *arg):
        ori_loss = self.loss_function(*arg)
        coef = 1 / (torch.pow(ori_loss.detach().clone(), self.lambd) + 1e-15)
        ad_loss = (coef / coef.sum()) * ori_loss 
        if self.reduction == 'mean':
            ad_loss = ad_loss.mean()
        elif self.reduction == 'sum':
            ad_loss = ad_loss.sum()
        return ad_loss
class SFRon(MU):
    """nips24, unlearn sensitive param(du_fisher>dr_fisher) on Du, tune on Dr"""
    def __init__(self,**kw):
        self.slow_update_rate=1 
        self.max_norm = 7.0
        self.th = 1 
        super().__init__(**kw)
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        du_fisher=get_fisher_dict(model=self.model,lossF=self.model.lossF,loader=DataLoader(self.du,self.bs,collate_fn=self.model.get_collate_fn()))
        dr_fisher=get_fisher_dict(model=self.model,lossF=self.model.lossF,loader=DataLoader(self.dr,self.bs,collate_fn=self.model.get_collate_fn()))
        mask={}
        for name in du_fisher.keys():
            if name not in dr_fisher.keys():continue
            mask[name]=((du_fisher[name]+1e-15)/(dr_fisher[name]+1e-15))>=self.th
        dr_loader=DataLoader(self.dr,self.bs,True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        du_loader=DataLoader(self.du,self.bs,True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        opt,sch=self.model.call_opt_sch()
        dr_iter=iter(dr_loader)
        for ep in range(self.epoch_tune):
            for bi,xu in enumerate(du_loader):
                try: xr=next(dr_iter)
                except StopIteration:
                    dr_iter=iter(dr_loader)
                    xr=next(dr_iter)
                xu=to_device(xu,self.device);xr=to_device(xr,self.device)
                yu=self.model(xu)
                loss=-self.model.lossF(yu,xu)*25 
                loss=loss_reduce(loss,'mean') 
                opt.zero_grad()
                loss.backward()
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name].to(param.grad.device)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.max_norm)
                opt.step()
                if sch:sch.step(self.epoch_train)
                yr=self.model(xr)
                loss=self.model.lossF(yr,xr)
                loss=loss_reduce(loss,'mean') 
                opt.zero_grad();loss.backward();opt.step()
                if ptloss:print(f'ep:{ep},step:{bi}, loss:{loss.item()}')
                if sch:sch.step(self.epoch_train)
                if ep>=1 and estop_fn is not None and estop_fn(loss.item()):return self.model
            self.save_epoch(ep+1)
        return self.model