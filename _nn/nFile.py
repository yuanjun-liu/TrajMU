import torch
import torch.nn as nn
import os
from _tool.mFile import check_dir
from _tool.mIO import load_th,loadZ_th,saveZ_th,save_th,is_zipfile
def load_weight_mem(model:nn.Module,ckpt:dict,opt:torch.optim.Optimizer=None,sch=None):
    """after model.to(device)"""
    if opt is None and sch is None:
        try:
            model.load_state_dict(ckpt)
            return
        except:pass
    model.load_state_dict(ckpt['model'])
    if 'opt' in ckpt and opt is not None: opt.load_state_dict(ckpt['opt'])
    if 'sch' in ckpt and sch is not None: sch.load_state_dict(ckpt['sch'])
def save_weight_mem(model:nn.Module,opt:torch.optim.Optimizer=None,sch=None)->dict:
    if opt is None and sch is None:
        return model.state_dict()
    else:
        data={'model':model.state_dict()} 
        if opt is not None:data['opt']=opt.state_dict()
        if sch is not None:data['sch']=sch.state_dict()
        return data
def load_weight(model:nn.Module,path:str,opt:torch.optim.Optimizer=None,sch=None,device=None):
    """after model.to(device)"""
    if is_zipfile(path): load_weight_mem(model,loadZ_th(path,device),opt,sch)
    else: load_weight_mem(model,load_th(path,device),opt,sch)
def save_weight(model:nn.Module,path:str,opt:torch.optim.Optimizer=None,sch=None):
    check_dir(path)
    if is_zipfile(path): saveZ_th(path,save_weight_mem(model,opt,sch))
    else:save_th(path,save_weight_mem(model,opt,sch))
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
