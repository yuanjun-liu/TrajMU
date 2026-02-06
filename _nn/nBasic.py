exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from _nn.nData import auto_device
from _nn.nLoss import loss_reduce
from _nn.nFile import save_weight
import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.utils.data.dataloader import DataLoader
import numpy as np
def to_item_th(x):
    if isinstance(x,list) or isinstance(x,np.ndarray) or isinstance(x,torch.Tensor):return [to_item_th(i) for i in x]
    if isinstance(x,tuple):return (to_item_th(i) for i in x)
    if isinstance(x,dict):return {k:to_item_th(x[k]) for k in x}
    try: return x.item()
    except:return x
def to_device(x,device=auto_device()):
    """x=to_device(x)"""
    if isinstance(x,Tensor):return x.to(device,non_blocking=True)
    if isinstance(x,np.ndarray):return torch.tensor(x,device=device)
    if isinstance(x,list):return [to_device(i,device) for i in x]
    if isinstance(x,tuple):return tuple([to_device(i,device) for i in x])
    if isinstance(x,set):return {to_device(i,device) for i in x}
    if isinstance(x,dict):return {k:to_device(x[k],device) for k in x}
    return x
def detach(x):
    if isinstance(x,Tensor):return x.detach()
    if isinstance(x,list):return [detach(i) for i in x]
    if isinstance(x,tuple):return tuple([detach(i) for i in x])
    if isinstance(x,set):return {detach(i) for i in x}
    if isinstance(x,dict):return {k:detach(x[k]) for k in x}
    return x
def basic_train(model:Module, train_loader:DataLoader, opt:torch.optim.Optimizer, call_loss, path=None, epoch_min=1, epoch_max=1, device=auto_device(), log=False, log_step=10, sch=None, train_after_batch_fn=None, train_after_epoch_fn=None, train_before_train_fn=None, estop_fn=None,save_ep_fn=None, **kw):
    """call(y,x)"""
    if train_before_train_fn is not None:train_before_train_fn()
    model.train()
    finish=False
    for ep in range(epoch_max):
        for i,x in enumerate(train_loader):
            x=to_device(x,device)
            y=model(x)
            loss:Tensor=call_loss(y,x)
            loss=loss_reduce(loss,'mean') 
            if log and i%log_step==0: print(f'ep:{ep},step:{i},loss:{loss.item()}')
            if torch.isnan(loss): raise RuntimeError('nan loss')
            if torch.isinf(loss): raise RuntimeError('inf loss')
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sch is not None:sch.step(ep)
            if train_after_batch_fn is not None and train_after_batch_fn():
                finish=True;break
            if ep>=epoch_min and estop_fn is not None and estop_fn(loss.item()):
                finish=True;break
        if save_ep_fn is not None:save_ep_fn(ep+1)
        if finish:break
        if train_after_epoch_fn is not None and train_after_epoch_fn():
            finish=True;break
    if path:save_weight(model,path,opt=opt,sch=sch)
    return model
@torch.no_grad()
def basic_infer(model:Module,data_loader:DataLoader,calls=None,device=auto_device()):
    """calls:[ call(y,x) ]"""
    model.eval();model=model.to(device)
    calls=[lambda y,x:y.detach().cpu()] if calls is None else calls
    res=[[] for i in calls]
    for x in data_loader:
        x=to_device(x,device)
        y=model(x)
        for ci, call in enumerate(calls):
            res[ci].append(call(y,x))
    if len(res)==1:return res[0]
    return res
