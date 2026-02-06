exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import sys ; sys.setrecursionlimit(200)
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from _nn.nBasic import basic_infer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from _nn.nData import auto_device
import warnings
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.multiclass")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils._classification")
warnings.filterwarnings('ignore',category=FutureWarning,module='sklearn')
warnings.filterwarnings('ignore',category=ConvergenceWarning,module='sklearn')
call_output_flatten=lambda y,x:y.detach().cpu().reshape(-1)
@torch.no_grad()
def CallLambda(model:Module,loader,calls,device=auto_device()):
    """call(model(x),x), batch"""
    assert len(calls)
    res=basic_infer(model,loader,calls,device)[1:]
    if len(calls)==1:return res[0]
    return res
def _CallLambda(infer_res,loader,calls,device=auto_device()):
    """call(model(x),x), batch"""
    assert len(calls)
    RES=[[] for i in calls]
    for i,x in enumerate(loader):
        y=infer_res[i]
        for ci, call in enumerate(calls):
            RES[ci].append(call(y,x))
    if len(calls)==1:return RES[0]
    return RES
@torch.no_grad()
def Accuracy(model:Module,loader,device=auto_device()):
    """x:[data,,label]"""
    call_acc=lambda y,x:(torch.argmax(y,dim=-1)==x[-1]).detach().cpu().numpy()
    res= CallLambda(model,loader,[call_acc],device)
    num_acc,num_all=0,0
    for r in res:
        num_acc+=sum(r)
        num_all+=len(r)
    return num_acc/num_all
def _Accuracy(infer_res,loader,device=auto_device()):
    """x:[data,,label]"""
    call_acc=lambda y,x:(torch.argmax(y,dim=-1)==x[-1]).detach().cpu().numpy()
    res= _CallLambda(infer_res,loader,[call_acc],device)
    num_acc,num_all=0,0
    for r in res:
        num_acc+=sum(r)
        num_all+=len(r)
    return (num_acc/num_all).item()
@torch.no_grad()
def CosSimModel(model1:Module,model2:Module):
    """cosine_similarity of the paramters of two models"""
    param1=torch.cat([m.data.flatten() for m in model1.parameters()])
    param2=torch.cat([m.data.flatten() for m in model2.parameters()])
    return torch.cosine_similarity(param1,param2,dim=0).item()
@torch.no_grad()
def L2Model(model1:Module,model2:Module):
    """cosine_similarity of the paramters of two models"""
    param1=torch.cat([m.data.flatten() for m in model1.parameters()])
    param2=torch.cat([m.data.flatten() for m in model2.parameters()])
    return torch.norm(param1-param2).item()
def _OutputDistance(outs1,outs2):
    distances = []
    for y1,y2 in zip(outs1,outs2):
        diff = torch.sqrt(
            torch.sum(
                torch.square(
                    F.softmax(y1, dim=1) - F.softmax(y2, dim=1)
                ),
                axis=1,
            )
        )
        diff = diff.detach().cpu()
        distances.append(diff)
    distances:Tensor = torch.cat(distances, axis=0)
    return distances.mean().item()
@torch.no_grad()
def OutputDistance(model1:Module,model2:Module,dataloader,device=auto_device()):
    """dis(softmax(y1),softmax(y2))"""
    outs1=basic_infer(model1,dataloader,device=device)
    outs2=basic_infer(model2,dataloader,device=device)
    return _OutputDistance(outs1,outs2)
@torch.no_grad()
def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)
def _MIA(out_dr,out_dv,out_du): 
    """small mia-value, better unlern"""
    out_dr=torch.cat(out_dr) ; out_dv=torch.cat(out_dv) ; out_du=torch.cat(out_du)
    out_dr=F.softmax(out_dr,dim=-1) ; out_dv=F.softmax(out_dv,dim=-1) ; out_du=F.softmax(out_du,dim=-1)
    Xtrain=torch.cat([entropy(out_dr),entropy(out_dv)]).cpu().numpy().reshape(-1,1)
    Ytrain= np.concatenate([np.ones(len(out_dr)), np.zeros(len(out_dv))])
    Xtest=entropy(out_du).cpu().numpy().reshape(-1,1)
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs")
    clf.fit(Xtrain, Ytrain)
    results = clf.predict(Xtest)
    return results.mean().item()
def MIA(Dr,Dv,Du,model:Module,device=auto_device(),bs=32): 
    """remain, valid(not in train all time), forget."""
    """based on nips24 SSD"""
    out_dr=basic_infer(model,DataLoader(Dr,bs),device=device) 
    out_dv=basic_infer(model,DataLoader(Dv,bs),device=device)
    out_du=basic_infer(model,DataLoader(Du,bs),device=device)
    return _MIA(out_dr,out_dv,out_du)
def _MIA2(out_dr,out_dv,out_du): 
    """0.5, random guess. 0->du, 1->dr"""
    device=auto_device()
    out_dr = torch.cat(out_dr)
    out_dv = torch.cat(out_dv)
    out_du = torch.cat(out_du)
    Xtrain = torch.cat([out_dr, out_dv], dim=0)
    Ytrain = torch.cat([
        torch.ones(len(out_dr)),
        torch.zeros(len(out_dv))
    ]).unsqueeze(1)
    Xtest = out_du
    Xtrain, Ytrain, Xtest = Xtrain.to(device), Ytrain.to(device), Xtest.to(device)
    dim=len(out_dr[0])
    class MIANet(nn.Module):
        def __init__(self,):
            super(MIANet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim//2),
                nn.ReLU(),
                nn.Linear(dim//2, 1),
            )
        def forward(self, x):
            return self.net(x)
    model = MIANet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_weight = torch.tensor([len(out_dv) / max(len(out_dr), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(TensorDataset(Xtrain, Ytrain), batch_size=64, shuffle=True)
    model.train()
    for epoch in range(30):
        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(Xtest)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        mia_score = probs.mean()
    return mia_score.item()
def MIA2(Dr,Dv,Du,model:Module,device=auto_device(),bs=32): 
    """remain, valid(not in train all time), forget."""
    """based on nips24 SSD"""
    out_dr=basic_infer(model,DataLoader(Dr,bs),device=device) 
    out_dv=basic_infer(model,DataLoader(Dv,bs),device=device)
    out_du=basic_infer(model,DataLoader(Du,bs),device=device)
    return _MIA2(out_dr,out_dv,out_du)
def MIAloss(Dr,Dv,Du,model:Module,lossF,collate_fn=None,device=auto_device(),bs=32):
    calls=[lambda y,x:lossF(y,x).reshape(-1,1)]
    loss_dr=basic_infer(model,DataLoader(Dr,bs,collate_fn=collate_fn),calls=calls,device=device)
    loss_dv=basic_infer(model,DataLoader(Dv,bs,collate_fn=collate_fn),calls=calls,device=device)
    loss_du=basic_infer(model,DataLoader(Du,bs,collate_fn=collate_fn),calls=calls,device=device)
    return _MIA(loss_dr,loss_dv,loss_du)
def JSDiv(p, q):
    m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)
def _OutEntropy(res):
    return entropy(torch.cat(res)).mean().item()
@torch.no_grad()
def OutEntropy(model:Module,loader,device=auto_device()):
    """x:[data,,label]"""
    res=basic_infer(model,loader,device=device) 
    return _OutEntropy(res)
def all_metrics_result(metrics,Dr=None,Du=None,Dv=None,modelR=None,modelU=None,bs=None,device=auto_device(),collate_fn=None,num_worker=0,**kw):
    kw_loader={'batch_size':bs, 'collate_fn':collate_fn,'num_workers':num_worker}
    oDrMU=None ; oDuMU=None ; oDvMU=None
    res=metrics
    if modelU is not None:modelU.eval()
    if modelR is not None:modelR.eval()
    for x in metrics:
        if 'DrMR' in x: oDrMR=basic_infer(modelR,DataLoader(Dr,**kw_loader),device=device)
        if 'DuMR' in x: oDuMR=basic_infer(modelR,DataLoader(Du,**kw_loader),device=device)
        if 'DvMR' in x: oDvMR=basic_infer(modelR,DataLoader(Dv,**kw_loader),device=device)
        if 'DrMU' in x: oDrMU=basic_infer(modelU,DataLoader(Dr,**kw_loader),device=device)
        if 'DuMU' in x: oDuMU=basic_infer(modelU,DataLoader(Du,**kw_loader),device=device)
        if 'DvMU' in x: oDvMU=basic_infer(modelU,DataLoader(Dv,**kw_loader),device=device)
        if 'OutX' in x:
            kw={}
            if 'DrMR' in x:kw['DrMR']=oDrMR
            if 'DuMR' in x:kw['DuMR']=oDuMR
            if 'DvMR' in x:kw['DvMR']=oDvMR
            if 'DrMU' in x:kw['DrMU']=oDrMU
            if 'DuMU' in x:kw['DuMU']=oDuMU
            if 'DvMU' in x:kw['DvMU']=oDvMU
            res[x]=metrics[x](**kw)
        if 'Call' in x:
            kw={}
            if 'Dv' in x: kw['Dv']=Dv
            if 'Dr' in x: kw['Dr']=Dr
            if 'Du' in x: kw['Du']=Du
            if 'MU' in x: kw['MU']=modelU
            if 'MR' in x: kw['MR']=modelR
            res[x]= metrics[x](**kw)
    if 'oDrMR' in metrics: res['oDrMR']=oDrMR
    if 'oDuMR' in metrics: res['oDuMR']=oDuMR
    if 'oDvMR' in metrics: res['oDvMR']=oDvMR
    if 'oDrMU' in metrics: res['oDrMU']=oDrMU
    if 'oDuMU' in metrics: res['oDuMU']=oDuMU
    if 'oDvMU' in metrics: res['oDvMU']=oDvMU
    if 'AccDrMR' in metrics:res['AccDrMR']=_Accuracy(oDrMR,DataLoader(Dr,**kw_loader),device=device)
    if 'AccDuMR' in metrics:res['AccDuMR']=_Accuracy(oDuMR,DataLoader(Du,**kw_loader),device=device)
    if 'AccDvMR' in metrics:res['AccDvMR']=_Accuracy(oDvMR,DataLoader(Dv,**kw_loader),device=device)
    if 'AccDrMU' in metrics:res['AccDrMU']=_Accuracy(oDrMU,DataLoader(Dr,**kw_loader),device=device)
    if 'AccDuMU' in metrics:res['AccDuMU']=_Accuracy(oDuMU,DataLoader(Du,**kw_loader),device=device)
    if 'AccDvMU' in metrics:res['AccDvMU']=_Accuracy(oDvMU,DataLoader(Dv,**kw_loader),device=device)
    if 'CosSimModel' in metrics:
        res['CosSimModel']=CosSimModel(model1=modelR,model2=modelU)
    if 'L2Model' in metrics:
        res['L2Model']=L2Model(model1=modelR,model2=modelU)
    if "MIA" in metrics:
        if 'mia_call' in metrics:
            calls=[metrics['mia_call']]
            _oDrMU=basic_infer(modelU,DataLoader(Dr,**kw_loader),device=device,calls=calls)
            _oDvMU=basic_infer(modelU,DataLoader(Dv,**kw_loader),device=device,calls=calls)
            _oDuMU=basic_infer(modelU,DataLoader(Du,**kw_loader),device=device,calls=calls)
            res['MIA']=_MIA(_oDrMU,_oDvMU,_oDuMU)
            res['MIA2']=_MIA2(_oDrMU,_oDvMU,_oDuMU)
            del res['mia_call']
        else:
            if oDrMU is None: oDrMU=basic_infer(modelU,DataLoader(Dr,**kw_loader),device=device)
            if oDvMU is None: oDvMU=basic_infer(modelU,DataLoader(Dv,**kw_loader),device=device)
            if oDuMU is None: oDuMU=basic_infer(modelU,DataLoader(Du,**kw_loader),device=device)
            res['MIA']=_MIA(oDrMU,oDvMU,oDuMU)
            res['MIA2']=_MIA2(oDrMU,oDvMU,oDuMU)
    if 'EntDrMR' in metrics: res['EntDrMR']=_OutEntropy(oDrMR)
    if 'EntDuMR' in metrics: res['EntDuMR']=_OutEntropy(oDuMR)
    if 'EntDvMR' in metrics: res['EntDvMR']=_OutEntropy(oDvMR)
    if 'EntDrMU' in metrics: res['EntDrMU']=_OutEntropy(oDrMU)
    if 'EntDuMU' in metrics: res['EntDuMU']=_OutEntropy(oDuMU)
    if 'EntDvMU' in metrics: res['EntDvMU']=_OutEntropy(oDvMU)
    if 'OutDisDr' in metrics:
        if oDrMR is None: oDrMR=basic_infer(modelR,DataLoader(Dr,**kw_loader),device=device)
        if oDrMU is None: oDrMU=basic_infer(modelU,DataLoader(Dr,**kw_loader),device=device)
        res['OutDisDr']=_OutputDistance(oDrMR,oDrMU)
    if 'OutDisDu' in metrics:
        if oDuMR is None: oDuMR=basic_infer(modelR,DataLoader(Du,**kw_loader),device=device)
        if oDuMU is None: oDuMU=basic_infer(modelU,DataLoader(Du,**kw_loader),device=device)
        res['OutDisDu']=_OutputDistance(oDuMR,oDuMU)
    if 'OutDisDv' in metrics:
        if oDvMR is None: oDvMR=basic_infer(modelR,DataLoader(Dv,**kw_loader),device=device)
        if oDvMU is None: oDvMU=basic_infer(modelU,DataLoader(Dv,**kw_loader),device=device)
        res['OutDisDv']=_OutputDistance(oDvMR,oDvMU)
    return res

