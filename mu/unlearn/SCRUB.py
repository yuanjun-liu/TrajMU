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
class SCRUB(MU):
    """min KLD+task in Dr, and max KLD+task on Du"""
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        teacher:Module=deepcopy(self.model) ; student:Module=self.model
        teacher.train() ; teacher.requires_grad_(False) ; student.train() 
        opt,sch=student.call_opt_sch()
        du_laoder=DataLoader(self.du,self.bs,True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        dr_laoder=DataLoader(self.dr,self.bs,True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        for ep in range(self.epoch_tune):
            for bi,xu in enumerate(du_laoder):
                xu=to_device(xu,self.device)
                out_stu=student(deepcopy(xu))
                with torch.no_grad():
                    out_good=teacher(xu)
                loss_task=self.model.lossF(out_stu,xu)
                loss_distill=self.model.lossDistill(out_good,out_stu)
                loss=-(loss_task+0.1*loss_distill) 
                loss=loss_reduce(loss,'mean') 
                opt.zero_grad();loss.backward();opt.step()
                if sch:sch.step(self.epoch_train)
            for xr in dr_laoder:
                xr=to_device(xr,self.device)
                out_stu=student(deepcopy(xr))
                with torch.no_grad():
                    out_good=teacher(xr)
                loss_task=self.model.lossF(out_stu,xr)
                loss_distill=self.model.lossDistill(out_good,out_stu)
                loss=(loss_task+0.1*loss_distill)
                loss=loss_reduce(loss,'mean') 
                opt.zero_grad();loss.backward();opt.step()
                if ptloss:print(f'ep:{ep},step:{bi}, loss:{loss.item()}')
                if sch:sch.step(self.epoch_train)
                if ep>=1 and estop_fn is not None and estop_fn(loss.item()):return self.model
            self.save_epoch(ep+1)
        self.model=student
        return student