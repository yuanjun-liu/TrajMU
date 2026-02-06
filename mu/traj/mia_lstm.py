exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import sys,os,time,torch
import torch.utils
from _nn.nData import auto_device
from _nn.nBasic import to_device
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import numpy as np
from traj.GPE.GPE import GPE
device=auto_device()
dim=128;num_layer=3;bidir=True
augment_drop_rate=0.1;augment_noise_size=20e-5
class BaseDataset(torch.utils.data.Dataset): 
    def __init__(self,ts,lb,ts_max_len,train=True):
        super().__init__()
        self.ts=ts; self.lb=lb
        if len(self.ts)<5:self.ts=self.ts[0]
        self.h=ts_max_len
        self.dim=len(self.ts[0][0])
        self.len_mask=[torch.Tensor(np.array([1]*i+[0]*(ts_max_len-i))) for i in range(ts_max_len)]
        self.train=train
    def __len__(self):return len(self.ts)
    def _augment(self,t:torch.Tensor):
        le=len(t)
        drop_num=int(max(0,le-self.h)*augment_drop_rate)
        idx=torch.Tensor(np.random.choice(le,drop_num,replace=False)).long()
        mask=torch.ones(le).bool()
        mask[idx]=False
        t=t[mask]
        shift_xy=torch.rand(len(t),2)*augment_noise_size
        t[:,:2]+=shift_xy
        return t
    def __getitem__(self,i): 
        """return ts,len,mask, lb""" 
        t=self.ts[i]
        le=len(t)
        t=torch.Tensor(t)[:,:2]
        tr=torch.zeros((self.h,2))
        tr[:le]=t
        if not self.train:
            return tr,le,self.len_mask[le-1] ,self.lb[i]
        t2=self._augment(t)
        le2=len(t2)
        tr2=torch.zeros((self.h,2))
        tr2[:le2]=t2
        return tr2,le2,self.len_mask[le2-1],self.lb[i]
def cut_len(ts,ls): return ts[:,:max(ls)]
class MIAModel(nn.Module):
    def __init__(self,dim=dim,num_layer=num_layer,bidir=bidir,device=auto_device()) -> None:
        super(MIAModel,self).__init__()  
        self.ebd = GPE(dim)
        self.dim,self.bidir,self.num_layer,self.device=dim,bidir,num_layer,device
        self.lstm= nn.LSTM(dim,dim,num_layer,bidirectional=bidir) 
        self.dim2=dim*2 if bidir else dim 
        self.l1= nn.Sequential(nn.BatchNorm1d(self.dim2), nn.Linear(self.dim2,self.dim2),nn.ReLU())
        self.l2=nn.Linear(self.dim2,1) 
    def forward(self,x:torch.Tensor,len_mask:torch.Tensor):
        """input: len_seq,batch_size,3 ; return: batch_size,dim_out"""
        bs=len(x) 
        x=self.ebd(x) 
        len_mask=len_mask.unsqueeze(-1).expand(x.shape)
        x=x*len_mask 
        x=x.transpose(0,1).contiguous() 
        h0=torch.zeros((self.num_layer*(2 if self.bidir else 1),bs,self.dim)).to(self.device)
        c0=torch.zeros_like(h0)
        x,(h,c)=self.lstm(x,(h0,c0)) 
        if self.bidir: x=torch.concat([x[-1,:,:self.dim],x[0,:,self.dim:]],dim=-1) 
        else:x=x[-1,:,:self.dim]
        x= self.l2(self.l1(x))
        return x
def train(model,dr:list,dv:list,ts_max_len,epoch=30):
    model.train()
    bs=256;log_int=10;earlystop=30
    lb=torch.cat([torch.ones(len(dr)),torch.zeros(len(dv))]).unsqueeze(1)
    datasets=BaseDataset(ts=dr+dv,lb=lb,ts_max_len=ts_max_len,train=True)
    dataloder=DataLoader(datasets,batch_size=bs,drop_last=True,shuffle=True,num_workers=0)
    lossf=nn.BCEWithLogitsLoss(pos_weight=torch.ones(1,device=device)* len(dv) / len(dr))
    opt=torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    bi,best_bi,best_loss,best_w=0,0,10000000,None
    for ep in range(epoch):
        for x in dataloder:
            ts,les,masks,lb=to_device(x,device)
            ts=cut_len(ts,les);masks=cut_len(masks,les)
            vs=model(ts,masks) 
            loss=lossf(vs,lb) 
            opt.zero_grad();loss.backward();opt.step()
            losv=loss.item()
            if bi % log_int==0:
                if losv<best_loss: 
                    best_bi,best_loss=bi,losv
            if ep>=1 and bi-best_bi>earlystop: 
                return
            bi+=1
def infer(model,du,ts_max_len):
    model.eval()
    bs=256
    res=[]
    data_loader=DataLoader(BaseDataset(du,[0]*len(du),train=False,ts_max_len=ts_max_len),batch_size=bs,shuffle=False,drop_last=False)
    with torch.no_grad():
        for x in data_loader:
            ts,les,masks,lb=to_device(x,device)
            ts,masks=cut_len(ts,les),cut_len(masks,les)
            vs=model(ts,masks).detach().cpu()
            res.append(torch.sigmoid(vs))
    res=torch.concat(res,dim=0).numpy()
    return res
def MIA(du,dr,dv):
    global augment_drop_rate
    ts_max_len=max(max(map(len,dr)),max(map(len,dv)),max(map(len,du)))
    if ts_max_len<30: augment_drop_rate=0
    if ts_max_len>100:augment_drop_rate=0.3
    model:MIAModel=MIAModel()
    model=model.to(device)
    train(model=model,dr=dr,dv=dv,ts_max_len=ts_max_len)
    res=infer(model,du,ts_max_len)
    return res.mean().item()
if __name__=='__main__':  
    pass
