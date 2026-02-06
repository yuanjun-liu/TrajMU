exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import math,sys,os
from _tool.mIO import loadZ_pk,saveZ_pk
from _tool.mFile import out_dir
from _tool.mTime import now_date_time
from mu.traj.loaddata import datasets,DuDrDvTypes,DuRates
from mu.traj.model_blue import TrajSim,TaskModel
from mu.traj.model_mlsimp import TrajSimp
from mu.traj.model_trmma import TrajMap,TrajRec
from mu.methods import mu_methods
from mu.MU import MU,DEBUG,num_workers
from mu.metrics import all_metrics_result
from _nn.nData import random_seed
from _tool.SysMonitor import LogToTxt,LogJsonIdxs
import argparse
import torch
from _tool.mData import to_item_np
"""TODO"""
random_seed(42)
print=LogToTxt(now_date_time())
TasksModel={'Sim':TrajSim,'Map':TrajMap,'Rec':TrajRec,'Simp':TrajSimp }
MUs=mu_methods
train_test_step=100
save_each_epoch=True
test_each_epoch=False
tune_epoch={'Origin':0, 'Retrain':0, 'FineTune':4, 'NegGrad':7, 'BadT':10,'SCRUB':10,'GDRGMA':2,'TopK':10,'RandomK':4,'SFRon':10,'SSD':1}
DuRates=[0.1,0.2,0.3]
DuDrDvTypes=['Area','Usr']
datasets=['Porto','Beijing']
path_json='./mu-traj.json'
def one(data,mu,task,urvtype,durate,epoch_tune=None,pre_data=False,pt=True,rerun=False,ptloss=False,test_exist=False):
    batch_size={'Sim':64,'Simp':32,'Map':256,'Rec':256}[task]
    train_epoch={'Sim':30,'Simp':20,'Map':50,'Rec':50}[task]
    if epoch_tune is None:  epoch_tune=tune_epoch[mu] 
    if tune_epoch[mu] ==0: epoch_tune=0
    title=f'one-{data}-{mu}-{task}-{urvtype}-{durate}-{epoch_tune}'
    json_key=tuple(map(str,[data,mu,task,urvtype,durate,epoch_tune]))
    if pt: print(f'{"pre_data" if pre_data else "one"}, data:{data}, task:{task}, urv:{urvtype}, durate:{durate}, mu:{mu}, epoch:{epoch_tune}')
    json=LogJsonIdxs(path_json,refresh=True,mode='wr')
    path_res=os.path.join(out_dir('res'),title+'.pk.zst')
    if not rerun and os.path.exists(path_res):
        res= loadZ_pk(path_res);json[json_key]=to_item_np(res)
        if pt:print(res)
        return res
    if test_exist:return False
    assert task in TasksModel 
    assert urvtype in DuDrDvTypes
    assert durate in DuRates
    random_seed(42)
    task_model:TaskModel=TasksModel[task](bs=batch_size,data_name=data,durate=durate)
    mu_kw={'Top_Rd_k':0.01, 'Top_Rd_noise_range':0.1}
    mu:MU=mu_methods[mu](model=task_model, bs=batch_size,epoch_tune=epoch_tune, epoch_train=train_epoch, dudvdr_type=urvtype, data_name=data, train_test_step=train_test_step,du_rate=durate,save_each_epoch=save_each_epoch, **mu_kw)
    if pre_data:
        mu.data_init(rt_if_exist=True)
        return
    mu.data_init(rt_if_exist=False)
    mu.model_origin() 
    modelU=mu.unlearn(rerun=rerun,ptloss=ptloss,estop_fn=None) 
    metrics=task_model.get_task_metrics()
    res=all_metrics_result(metrics, Dr=mu.dr_eval, Du=mu.du_eval, Dv=mu.dv_eval, modelU=modelU, bs=batch_size, collate_fn=task_model.get_collate_fn_test(), num_worker=num_workers)
    res['time']=mu.time
    saveZ_pk(path_res,res) ; json[json_key]=to_item_np(res)
    if pt:print(res)
    return res
def ini_cpu():
    for data in datasets:
        for urv in DuDrDvTypes:
            for durate in DuRates:
                for task in ['Sim','Map']: 
                    one(data=data,mu='Origin',task=task,urvtype=urv,durate=durate,pre_data=1)
def ini_gpu(pt=True):
    for durate in DuRates:
        for data in datasets:
            for urv in DuDrDvTypes:
                for task in TasksModel:
                    for mu in MUs:
                        if task in ['Rec','Simp'] and mu=='Origin':
                            one(data=data,mu='Origin',task=task,urvtype=urv,durate=durate,pre_data=1,pt=pt)
                        one(data=data,mu=mu,task=task,urvtype=urv,durate=durate,pre_data=0,pt=pt)
                        if test_each_epoch:
                            for epoch in list(range(tune_epoch[mu]))[::-1]:
                                one(data=data,mu=mu,task=task,urvtype=urv,durate=durate,epoch_tune=epoch+1,pre_data=0,pt=pt)
def parse():
    global datasets,TasksModel,DuDrDvTypes,DuRates
    parser = argparse.ArgumentParser(description='mutraj')
    parser.add_argument('--data',type=str,default='')
    parser.add_argument('--task',choices=list(TasksModel.keys()),default='')
    parser.add_argument('--urv',choices=DuDrDvTypes,default='')
    parser.add_argument('--durate',choices=DuRates,type=float,default=0)
    parser.add_argument('--mu',type=str,default='')
    parser.add_argument('--pre_data',action='store_true')
    parser.add_argument('--ini_cpu',action='store_true')
    parser.add_argument('--ini_gpu',action='store_true')
    parser.add_argument('--rerun',action='store_true')
    parser.add_argument('--ptloss',action='store_true')
    args = parser.parse_args()
    if args.data: datasets=[args.data]
    if args.task: TasksModel={args.task:TasksModel[args.task]}
    if args.urv:DuDrDvTypes=[args.urv]
    if args.durate: DuRates=[args.durate]
    if args.ini_cpu: ini_cpu()
    if args.ini_gpu and (torch.cuda.is_available()or torch.mps.is_available()):ini_gpu()
    for mu in MUs:
        if args.mu:mu=args.mu
        one(data=args.data,mu=mu,task=args.task,urvtype=args.urv,durate=args.durate,pre_data=args.pre_data,rerun=args.rerun,ptloss=args.ptloss)
        if args.mu:break

if __name__=='__main__':
    print('mu traj begin')
    parse()
    print('mu traj over')
