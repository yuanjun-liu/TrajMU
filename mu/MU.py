exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from _nn.nFile import load_weight,save_weight
from _nn.nData import auto_device
from _tool.mFile import out_dir,check_dir
from _tool.mIO import load_pk,save_pk
from _tool.mList import deep_flatten2
from _nn.nBasic import to_device,loss_reduce
from _nn.nBasic import basic_train
from _tool.mList import deep_copy as deepcopy
import torch,os
from torch import nn
from torch.nn.modules import Module
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import time
from _tool.SysMonitor import mPrintCapturer
DEBUG=False
num_workers=0
def dict2vec(d):
    res=[]
    for name in d:
        param=d[name]
        res.append(param.reshape(-1))
    return torch.concat(res)
def get_grad_dict(model:Module):
    res={}
    for name, param in model.named_parameters():
        if param.grad is not None:
            res[name]=param.grad.data
    return res
def apply_grad_dict(model:Module,grad:dict,lr):
    for name, param in model.named_parameters():
        if name in grad:
            param.data-=grad[name]*lr
def get_grad_vec(model:Module):
    return dict2vec(get_grad_dict(model))
def zeors_para_dict(model:Module):
    res={}
    for name, param in model.named_parameters():
        if param.requires_grad :
            res[name]=torch.zeros_like(param,requires_grad=False,device=param.device)
    return res
def zeros_para_vec(model:Module):
    return dict2vec(zeors_para_dict(model))
def get_fisher_dict(model:Module,lossF,ds:Dataset=None,loader=None,collate_fn=None,device=auto_device()):
    """fisher(model,lossF,loader) or fisher(model,lossF,ds,collate)"""
    fisher={}
    for name, param in model.named_parameters():
        fisher[name]=torch.zeros_like(param.data)
    if loader is None:
        loader=DataLoader(ds,1,collate_fn=collate_fn)
        num=len(ds)
    else: num=len(loader)
    for x in loader:
        model.zero_grad()
        x=to_device(x,device)
        y=model(x)
        loss=lossF(y,x) ; loss=loss_reduce(loss)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name]+=param.grad.data.pow(2)/num
    return fisher
def get_fisher_vec(model:Module,lossF,ds:Dataset=None,loader=None,collate_fn=None,device=auto_device()):
    return dict2vec(get_fisher_dict(model=model,lossF=lossF,ds=ds,loader=loader,collate_fn=collate_fn,device=device))
def get_hessian_dict(model:Module,lossF,ds:Dataset=None,loader=None,collate_fn=None,device=auto_device()):
    """hessian(model,lossF,loader) or hessian(model,lossF,ds,collate)"""
    hessian={}
    for name, param in model.named_parameters():
        hessian[name]=torch.zeros_like(param.data)
    if loader is None:
        loader=DataLoader(ds,1,collate_fn=collate_fn)
        num=len(ds)
    else: num=len(loader)
    for x in loader:
        model.zero_grad()
        x=to_device(x,device)
        y=model(x)
        loss=lossF(y,x) ; loss=loss_reduce(loss)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None: continue
                grad=param.grad
                for i in range(param.numel()):
                    hessian_ii = torch.autograd.grad(grad.flatten()[i], param, retain_graph=True)[0].flatten()[i]
                    hessian[name][i]=hessian_ii/num
    return hessian
call_opt_Adam=lambda model,**kw:torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),**kw)  if isinstance(model,torch.nn.Module) else torch.optim.Adam(filter(lambda p: p.requires_grad, deep_flatten2(model)),**kw) 
call_opt_AdamW=lambda model,**kw:torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),**kw)  if isinstance(model,torch.nn.Module) else torch.optim.AdamW(filter(lambda p: p.requires_grad, deep_flatten2(model)),**kw) 
call_opt_SGD=lambda model,**kw:torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),**kw)  if isinstance(model,torch.nn.Module) else torch.optim.AdamW(filter(lambda p: p.requires_grad, deep_flatten2(model)),**kw) 
class TaskModel(nn.Module):
    def __init__(self,data_name=None, du_rate=0.2, device=auto_device(), urv='',**kw):
        super().__init__()
        self.device=device ; self.root_data=self.root_model=None;self.data_name=data_name
        self.model:nn.Module=None ; self.du_rate=du_rate ; self.root_map=None
        self.dr=self.du=self.dv=None ; self.bs=None ; self.urv=urv
    def data_init(self,uvr,rt_if_exist=False):
        """{dr,du,dv,dtrain,dtest,dval,dr_raw,du_raw} + (optinal) {dr_eval,du_eval,dr_eval} """
        raise NotImplementedError()
    def get_collate_fn(self): 
        return None
    def get_collate_fn_test(self): 
        return self.get_collate_fn()
    def _call_new_model(self):
        raise NotImplementedError()
    def _call_new_opt(self,**kw):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),**kw)
    def _call_new_sch(self,opt):
        return None
    def model_pretrain(self,path_train):
        """time"""
        return 0
    def train_before_train(self):
        pass
    def forward(self,x):
        raise NotImplementedError()
    def lossF(self,y,x): 
        raise NotImplementedError()
    def lossFdist(self,teacher,student):
        return self._lossF_dist_vec(teacher,student)
    def train_after_batch(self)->bool:
        """return: if exit train"""
        return False
    def train_after_epoch(self)->bool:
        """return: if exit train"""
        return False
    def get_task_metrics(self): 
        return {}
    def save(self,path):
        assert self.model is not None,'can not save None model'
        save_weight(model=self.model,path=path,opt=None,sch=None)
    def load(self,path):
        if self.model is None:self.model=self.call_new_model()
        load_weight(model=self.model,path=path,opt=None,sch=None,device=self.device)
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    def call_new_model(self):
        self.model=self._call_new_model()
        self.model=self.model.to(self.device)
        self.add_module('model',self.model)
        return self.model
    def __repr__(self): 
        return type(self).__name__
    def call_opt_sch(self,**opt_kw):
        opt=self._call_new_opt(**opt_kw)
        return opt,self._call_new_sch(opt)
    def lossDistill(self,teacher,student):
        if isinstance(teacher,dict):
            return sum([self.lossDistill(teacher[k],student[k]) for k in teacher])
        if isinstance(teacher,list) or isinstance(teacher,tuple):
            return sum([self.lossDistill(t,s) for t,s in zip(teacher,student)])
        return self.lossFdist(teacher,student)
    def _lossF_dist_cls(self,t,s): 
        T=1
        return nn.functional.kl_div(torch.log_softmax(s / T, dim=1),torch.softmax(t / T, dim=1),reduction='batchmean')*T*T 
    def _lossF_dist_vec(self,t,s):
        return torch.norm(t-s,dim=1).mean() - nn.functional.cosine_similarity(t, s, dim=1).mean()
class DataInitFinishExp(Exception):
    def __init__(self, message=''):
        super().__init__(message)
        self.message = message
    def __str__(self):
        return self.message
class MU:
    def __init__(self, model:TaskModel=None,bs=256, epoch_tune=1, epoch_train=None, device=auto_device(), data_name='', train_test_step=100, dudvdr_type=None, du_rate=None,save_each_epoch=False, **kw):
        self.kw=kw; self.device=auto_device(); self.time=-1; self.du_rate=du_rate; self.__dict__.update(kw) 
        self.model:TaskModel=model; self.data_name=data_name ;self.urv=dudvdr_type
        self.epoch_tune=epoch_tune; self.epoch_train=epoch_train; self.bs=bs; self.train_test_step=train_test_step
        self.dr=self.du=self.dv=self.dtrain=self.dtest=self.dval=None;self.dudvdr_type=dudvdr_type
        self.dr_raw=self.du_raw=None 
        self.dr_eval=self.du_eval=self.dv_eval=None 
        self.save_each_epoch=save_each_epoch ; self._last_time=0
        if os.path.exists(self.path_origin()[1]):self.time=load_pk(self.path_origin()[1])
        if os.path.exists(self.path_unlearn()[1]):self.time=load_pk(self.path_unlearn()[1])
    def mu_name(self):
        return f'{type(self).__name__}'
    def root_datamap(self): 
        path= os.path.join(out_dir(self.data_name),'map')
        check_dir(path+os.sep)
        return path
    def root_data_task(self): 
        path= os.path.join(out_dir(self.data_name),type(self.model).__name__+str(self.du_rate))
        check_dir(path+os.sep)
        return path
    def root_init(self): 
        path= os.path.join(self.root_data_task(),'init')
        check_dir(path+os.sep)
        return path
    def path_origin(self): 
        """path of model, path of time"""
        root=os.path.join(os.path.join(out_dir(self.data_name),type(self.model).__name__))
        check_dir(root+os.sep)
        return os.path.join(root,f'origin.th.zst'),os.path.join(root,f'origin.time.pk')
    def path_retrain(self): 
        """path of model, path of time"""
        return os.path.join(self.root_init(),'Retrain',f'{self.urv}.th.zst'),os.path.join(self.root_init(),'Retrain',f'{self.urv}.time.pk')
    def root_mu(self): 
        path= os.path.join(self.root_data_task(),self.mu_name())
        check_dir(path+os.sep)
        return path
    def path_unlearn(self,ep=None): 
        """path of model, path of time"""
        str_epoch=f'-{ep}' if self.save_each_epoch and ep is not None else ''
        return os.path.join(self.root_mu(),f'{self.urv}{str_epoch}.th.zst'),os.path.join(self.root_mu(),f'{self.urv}{str_epoch}.time.pk')
    def save_epoch(self,epoch): 
        if not self.save_each_epoch:return
        t1=time.time() ; self.time=t1-self._last_time
        path_model,path_time=self.path_unlearn(epoch)
        self.model.save(path_model)
        save_pk(path_time,self.time)
        t2=time.time()
        self._last_time+=t2-t1
    def data_init(self,rt_if_exist=False):
        """before model ini"""
        self.model.root_data=self.root_init(); self.model.root_model=self.root_mu(); self.model.data_name=self.data_name; self.model.du_rate=self.du_rate; self.model.root_map=self.root_datamap() ; self.model.urv=self.urv
        try:
            x=self.model.data_init(self.dudvdr_type,rt_if_exist=rt_if_exist)
        except DataInitFinishExp as e: 
            x=self.model.data_init(self.dudvdr_type,rt_if_exist=rt_if_exist)
        if rt_if_exist:return x
        x=self.model.data_init(self.dudvdr_type,rt_if_exist=False)
        self.dr=x['dr'];self.du=x['du'];self.dv=x['dv'];self.dtrain=x['dtrain'];self.dval=x['dval'];self.dtest=x['dtest']
        self.du_raw=x['du_raw'];self.dr_raw=x['dr_raw']
        assert len(self.du)==len(self.du_raw) and len(self.dr)==len(self.dr_raw) 
        self.dr_eval=x['dr_eval'] if 'dr_eval' in x else self.dr
        self.du_eval=x['du_eval'] if 'du_eval' in x else self.du
        self.dv_eval=x['dv_eval'] if 'dv_eval' in x else self.dv
        assert len(self.dr)>5 and len(self.du)>5 and len(self.dv)>5 and len(self.dtrain)>5
        self.model.dr=self.dr;self.model.dv=self.dv;self.model.du=self.du ; self.model.bs=self.bs
    def __model_ini(self,type,*args,**kw):
        assert type in ['origin','retrain']
        if type =='origin':
            path_model,path_time=self.path_origin() 
        elif type =='retrain':
            path_model,path_time=self.path_retrain() 
        else:
            print('bad model ini type')
            raise RuntimeError('bad model ini type')
        path_log=path_time[:-7]+'log'
        self.model.call_new_model() 
        opt,sch=self.model.call_opt_sch()
        if os.path.exists(path_model):
            self.model.load(path_model)
            self.time=load_pk(path_time)
            return self.model
        log=mPrintCapturer(path_log) ; log.replace() 
        if type =='origin':
            train_loader=DataLoader(self.dtrain,self.bs,shuffle=True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        elif type =='retrain':
            train_loader=DataLoader(self.dr,self.bs,shuffle=True,num_workers=num_workers,collate_fn=self.model.get_collate_fn())
        self.model.train()
        self.time=self.model.model_pretrain(path_model)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.mps.is_available(): torch.mps.empty_cache()
        t1=time.time()
        self.model=basic_train(model=self.model, train_loader=train_loader, opt=opt, call_loss=self.model.lossF, path= None, epoch_max= self.epoch_train, log_step=self.train_test_step, sch=sch, train_after_batch_fn=self.model.train_after_batch, train_after_epoch_fn=self.model.train_after_epoch) 
        t2=time.time()
        log.restore()
        self.time+=t2-t1;save_pk(path_time,self.time)
        self.model.save(path_model)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.mps.is_available(): torch.mps.empty_cache()
        return self.model
    def model_origin(self,*args,**kw):
        return self.__model_ini('origin',*args,**kw)
    def model_retrain(self,*args,**kw):
        return self.__model_ini('retrain',*args,**kw)
    def _unlearn(self,*args,**kw):
        """return: model"""
        raise NotImplementedError()
    def unlearn(self,*args,rerun=False,ptloss=False,**kw):
        path_model2,path_time2=self.path_unlearn(self.epoch_tune) 
        if not rerun and os.path.exists(path_model2): 
            self.model.load(path_model2)
            if os.path.exists(path_time2): self.time=load_pk(path_time2)
            return self.model
        path_model,path_time=self.path_unlearn()
        if not rerun and os.path.exists(path_model):
            self.model.load(path_model)
            if os.path.exists(path_time): self.time=load_pk(path_time)
            return self.model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.mps.is_available(): torch.mps.empty_cache()
        self.model.train()
        t1=time.time();self._last_time=t1
        self.model=self._unlearn(*args,ptloss=ptloss,**kw)
        t2=time.time()
        if 'Origin' in self.mu_name():self.time=load_pk(self.path_origin()[1])
        elif 'Retrain' in self.mu_name():self.time=load_pk(self.path_retrain()[1])
        elif self.save_each_epoch:
            self.save_epoch(self.epoch_tune) 
        else:
            self.time=t2-t1
            save_pk(path_time,self.time)
            self.model.save(path_model)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.mps.is_available(): torch.mps.empty_cache()
        return self.model
