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



class BadT(MU):
    """min KLD(student,teacher) & max KLD(student,random) on Dr"""
    def __init__(self, **kw):
        super().__init__(**kw)
        """CUDA out of mem"""
        self.bs=int(self.bs*0.8) 
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        good=deepcopy(self.model) ; bad=deepcopy(self.model) ; bad.call_new_model()
        opt,sch=self.model.call_opt_sch() ; self.model.train(); self.model.model.train() ; bad.eval() ; good.eval() 
        dr_laoder=DataLoader(self.dr,self.bs,True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        for ep in range(self.epoch_tune): 
            for bi,x in enumerate(dr_laoder):
                x=to_device(x,self.device)
                with torch.no_grad():
                    x_good=deepcopy(x)
                    out_good=good(x_good)
                    x_bad=deepcopy(x)
                    out_bad=bad(x_bad)
                out_stu=self.model(x)
                loss_task=self.model.lossF(out_stu,x)
                loss_good=self.model.lossDistill(out_good,out_stu)
                loss_bad=self.model.lossDistill(out_bad,out_stu)
                loss=loss_task+0.1*(loss_good-loss_bad)
                assert not torch.any(torch.isnan(loss))
                loss=loss_reduce(loss,'mean') 
                opt.zero_grad();loss.backward();opt.step()
                if ptloss:print(f'ep:{ep},step:{bi}, loss:{loss.item()}')
                if sch:sch.step(self.epoch_train)
                if ep>=1 and estop_fn is not None and estop_fn(loss.item()):return self.model
            self.save_epoch(ep+1)
        return self.model

