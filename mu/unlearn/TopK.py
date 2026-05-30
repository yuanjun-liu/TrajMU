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
class TopK(MU):
    """modify k% paramter, tune. tkde25 Machine Unlearning Through Fine-Grained Model Parameters Perturbation"""
    def __init__(self,**kw):
        self.k=kw['Top_Rd_k'];self.noise_range=kw['Top_Rd_noise_range']
        super().__init__(**kw)
    def mu_name(self):
        return super().mu_name() +f'{self.k}_{self.noise_range}'
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        ori_model=deepcopy(self.model)
        ori_model.train() ; ori_model.requires_grad_(False)
        opt,sch=self.model.call_opt_sch()
        du_loader=DataLoader(self.du,self.bs,True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        dr_loader=DataLoader(self.dr,self.bs,True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        x=next(iter(du_loader))
        x=to_device(x)
        y=self.model(x)
        loss=self.model.lossF(y,x)
        loss=loss_reduce(loss,'mean') 
        opt.zero_grad()
        loss.backward()
        if sch:sch.step(self.epoch_train)
        grad=get_grad_vec(self.model).detach().cpu().pow(2).contiguous().numpy()
        K=self.k if self.k>1 else int(len(grad)*self.k)
        x=min(topk(grad,K))
        for name,para in self.model.named_parameters():
            mask=para.data>=x
            if torch.any(mask):
                noise=torch.randn_like(para.data)*self.noise_range
                para.data[mask]+=noise[mask]
        opt.zero_grad()
        jsd=lambda a,b :(self.model.lossDistill(a,b)+self.model.lossDistill(b,a))/2
        dr_iter=iter(dr_loader)
        for ep in range(self.epoch_tune):
            for bi,xu in enumerate(du_loader):
                try:
                    xr=next(dr_iter)
                except StopIteration:
                    dr_iter=iter(dr_loader)
                    xr=next(dr_iter)
                xr=to_device(xr,self.device)
                xu=to_device(xu,self.device)
                yr=self.model(xr)
                loss1=self.model.lossF(yr,xr)
                loss1=loss_reduce(loss1,'sum')
                loss1/=len(self.dr)
                y0=ori_model(deepcopy(xu))
                yu=self.model(xu)
                loss2=jsd(yu,y0)
                loss2=loss_reduce(loss2,'sum') 
                loss2/=len(self.du)
                loss=loss1+0.1*loss2
                opt.zero_grad();loss.backward();opt.step()
                if ptloss:print(f'ep:{ep},step:{bi}, loss:{loss.item()}')
                if sch:sch.step(self.epoch_train)
                if ep>=1 and estop_fn is not None and estop_fn(loss.item()):return self.model
            self.save_epoch(ep+1)
        return self.model