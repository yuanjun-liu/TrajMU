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
class _GDRGMA_MemoryBank:
    def __init__(self, size):
        self.grads = []    
        self.size = size
    def update(self, grads):
        self.grads.append(grads)
        if len(self.grads) > self.size:
            del self.grads[0]
    def get_graident(self, model:nn.Module):
        gradient = []
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.clone().detach()
                gradient.append(grad.view(-1))
        return gradient  
    def mean_grads(self, t_grads):
        grads = []
        for grad in self.grads:
            if torch.cosine_similarity(grad, t_grads, dim=0) < 0:
                grads.append(grad)
        if len(grads) > 0:
            avg_grad = grads[0]
            for grad in grads[1:]:
                avg_grad += grad
            avg_grad = avg_grad/len(grads) 
            return avg_grad
        else:
            return None
def _GDRGMA_rectify_graident(grads_x, grads_y):
    r_grads_x = []
    r_grads_y = []
    for x, y in zip(grads_x, grads_y):
        if torch.cosine_similarity(x, y, dim=0) < 0:
            InP_xy = torch.matmul(y, x) 
            Inp_xx = torch.norm(x, p=2) ** 2
            Inp_yy = torch.norm(y, p=2) ** 2
            x = x - InP_xy/Inp_yy * y
            y = y - InP_xy/Inp_xx * x
        r_grads_x.append(x)
        r_grads_y.append(y)
    return r_grads_x, r_grads_y
class GDRGMA(MU):
    """clear the grad that cos(Dr,Du)<0"""
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        bank = None
        dr_loader=DataLoader(self.dr,self.bs,True,drop_last=False,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        du_loader=DataLoader(self.du,self.bs,True,drop_last=False,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        dr_iter=iter(dr_loader)
        opt,sch=self.model.call_opt_sch()
        for ep in range(self.epoch_tune):
            for bi,xu in enumerate(du_loader):
                if bank is None:
                    bank = _GDRGMA_MemoryBank(size=math.ceil(len(du_loader)/self.bs))
                try:
                    xr=next(dr_iter) 
                except StopIteration:
                    dr_iter=iter(dr_loader)
                    xr=next(dr_iter) 
                xr=to_device(xr,self.device);xu=to_device(xu,self.device)
                yr=self.model(xr) ; lr=self.model.lossF(yr,xr) ; lr=loss_reduce(lr)
                opt.zero_grad() ; lr.backward() ; r_grads=bank.get_graident(self.model)
                yu=self.model(xu) ; lu=self.model.lossF(yu,xu) ; lu=loss_reduce(lu)
                opt.zero_grad() ; lu.backward() ; u_grads=bank.get_graident(self.model)
                bank.update(u_grads[-1]) 
                r_n_grads, r_t_grads = _GDRGMA_rectify_graident(r_grads, u_grads)
                if ep > 0 and bank.mean_grads(r_t_grads[-1]) != None:
                    grads, _ = _GDRGMA_rectify_graident([r_t_grads[-1]], [bank.mean_grads(r_t_grads[-1])])
                    r_t_grads[-1] = grads[-1]
                with torch.no_grad():
                        gamma, epsilon = 100, 0.02
                        lambda_weight = 1/(1+torch.exp(gamma*(lr-epsilon)))
                opt.zero_grad()
                idx=0
                for _, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad =  ((1-lambda_weight)*r_n_grads[idx]+lambda_weight*r_t_grads[idx]).view(param.size())
                        idx+=1
                opt.step()
                if sch:sch.step(self.epoch_train)
            self.save_epoch(ep+1)
        return self.model