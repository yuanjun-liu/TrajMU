exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import math,sys,os
import numpy as np
from copy import deepcopy
from _tool.mIO import loadZ_pk,saveZ_pk
from _tool.mFile import out_dir,is_linux,is_mac,check_dir
from _tool.mThread import plock_default as flock_default
from _tool.mTime import now_date_time
if is_mac:os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
from mu.traj.loaddata import datasets as datasets_all,DuDrDvTypes,DuRates
from mu.traj.model_registry import (
    TASK_DEFAULT_MODEL,
    get_task_model_class,
    get_train_bundle,
    task_model_name,
    task_result_name,
)
import time
from mu.methods import mu_methods
from mu.MU import MU,TaskModel,num_workers
from mu.metrics import all_metrics_result
from _nn.nData import random_seed
from _tool.SysMonitor import LogToTxt,LogJsonIdxs
import argparse
import torch
from _tool.mData import to_item_np
random_seed(42)
print=LogToTxt(now_date_time())
TasksModel={}
MUs=mu_methods
train_test_step=100
save_each_epoch=True
test_each_epoch=False
tune_epoch={'Origin':0, 'Retrain':0, 'FineTune':4, 'NegGrad':7, 'BadT':10,'SCRUB':10,'GDRGMA':2,'TopK':10,'RandomK':4,'SFRon':10,'SSD':1}
DuRates=[0.1,0.2,0.3]
DuDrDvTypes=['Area','Usr']
datasets=['Porto','Beijing','Xian']
def _map_profile_enabled():
    return os.environ.get('MU_MAP_PROFILE', '0').lower() in ['1', 'true', 'yes', 'on']
def _stage_profile_add(stats, enabled, name, cost):
    if enabled:
        stats[name] = stats.get(name, 0.0) + float(cost)
def _stage_profile_report(stats, enabled, title):
    if enabled and stats:
        msg = ', '.join([f'{k}={stats[k]:.3f}s' for k in sorted(stats)])
        print(f'[MAIN-PROFILE] {title}: {msg}')
path_json='./mu-traj.json'
def _stable_unlearn_batch(mu, unlearn_batch=1):
    return 1 if mu in {'Origin','Retrain'} else unlearn_batch
def _run_marks(unlearn_batch=1):
    marks=[]
    if unlearn_batch!=1:
        marks.append(f'u{unlearn_batch}')
    return marks
def run_title(data,mu,task,urvtype,durate,epoch_tune,unlearn_batch=1,model_group=1):
    task_name=task_result_name(task,model_group)
    unlearn_batch=_stable_unlearn_batch(mu,unlearn_batch)
    parts=['one',data,mu,task_name,urvtype,durate,epoch_tune,*_run_marks(unlearn_batch=unlearn_batch)]
    return '-'.join(map(str,parts))
def run_json_key(data,mu,task,urvtype,durate,epoch_tune,unlearn_batch=1,model_group=1):
    task_name=task_result_name(task,model_group)
    unlearn_batch=_stable_unlearn_batch(mu,unlearn_batch)
    parts=[data,mu,task_name,urvtype,durate,epoch_tune,*_run_marks(unlearn_batch=unlearn_batch)]
    return tuple(map(str,parts))
def one(data,mu,task,urvtype,durate,unlearn_batch,model_group,epoch_tune=None,pre_data=False,pt=True,rerun=False,ptloss=False,test_exist=False):
    t_wall=time.time()
    profile_enabled=_map_profile_enabled()
    profile_stats={}
    mus=mu
    stable_unlearn_batch=_stable_unlearn_batch(mu,unlearn_batch)
    model=task_model_name(task,model_group)
    task_model_cls=get_task_model_class(task,model_group)
    task_name=task_result_name(task,model_group)
    train_bundle=get_train_bundle(task,model_group)
    batch_size=train_bundle['batch_size']
    train_epoch=train_bundle['train_epoch']
    if data=='AIS':train_epoch*=2
    if epoch_tune is None:  epoch_tune=tune_epoch[mu]
    if tune_epoch[mu] ==0: epoch_tune=0
    task_model:TaskModel=task_model_cls(bs=batch_size,data_name=data,durate=durate)
    title=run_title(data,mus,task,urvtype,durate,epoch_tune,unlearn_batch=stable_unlearn_batch,model_group=model_group)
    json_key=run_json_key(data,mus,task,urvtype,durate,epoch_tune,unlearn_batch=stable_unlearn_batch,model_group=model_group)
    if pt: print(f'{"pre_data" if pre_data else "one"}, data:{data}, task:{task_name}, model:{model}, urv:{urvtype}, durate:{durate}, mu:{mu}, epoch:{epoch_tune}, unlearn_batch:{unlearn_batch}')
    json=LogJsonIdxs(path_json,refresh=True,mode='wr',_flock=flock_default)
    path_res=os.path.join(out_dir('res'),title+'.pk.zst')
    cached_res=None
    if not rerun and os.path.exists(path_res):
        cached_res=loadZ_pk(path_res)
        json[json_key]=to_item_np(cached_res)
        if pt:print(cached_res)
        return cached_res
    if test_exist:return False
    assert task in TasksModel
    assert urvtype in DuDrDvTypes
    assert durate in DuRates
    random_seed(42)
    mu_kw={'Top_Rd_k':0.01, 'Top_Rd_noise_range':0.1, }
    mu:MU=mu_methods[mu](model=task_model, bs=batch_size,epoch_tune=epoch_tune, epoch_train=train_epoch, dudvdr_type=urvtype, data_name=data, train_test_step=train_test_step,du_rate=durate,save_each_epoch=save_each_epoch, unlearn_batch=unlearn_batch, **mu_kw)
    if pre_data:
        t_stage=time.time()
        mu.data_init(rt_if_exist=True)
        _stage_profile_add(profile_stats, profile_enabled, 'pre_data_init', time.time()-t_stage)
        _stage_profile_add(profile_stats, profile_enabled, 'total_wall', time.time()-t_wall)
        _stage_profile_report(profile_stats, profile_enabled, f'{data}-{task_name}-{mus}-{urvtype}-{durate}')
        return
    t_stage=time.time()
    mu.data_init(rt_if_exist=False)
    _stage_profile_add(profile_stats, profile_enabled, 'data_init', time.time()-t_stage)
    t_stage=time.time()
    mu.model_origin()
    _stage_profile_add(profile_stats, profile_enabled, 'model_origin', time.time()-t_stage)
    t_stage=time.time()
    modelU=mu.unlearn2(rerun=rerun,ptloss=ptloss,estop_fn=None) if unlearn_batch==2 else mu.unlearn(rerun=rerun,ptloss=ptloss,estop_fn=None)
    _stage_profile_add(profile_stats, profile_enabled, 'unlearn', time.time()-t_stage)
    if cached_res is None or rerun:
        t_stage=time.time()
        metrics=task_model.get_task_metrics()
        _stage_profile_add(profile_stats, profile_enabled, 'get_task_metrics', time.time()-t_stage)
        t_stage=time.time()
        res=all_metrics_result(metrics, Dr=mu.dr_eval, Du=mu.du_eval, Dv=mu.dv_eval, modelU=modelU, bs=batch_size, collate_fn=task_model.get_collate_fn_test(), num_worker=num_workers)
        _stage_profile_add(profile_stats, profile_enabled, 'all_metrics_result', time.time()-t_stage)
        res['time']=mu.time
        res['unlearn_batch']=stable_unlearn_batch
        t_stage=time.time()
        saveZ_pk(path_res,res)
        _stage_profile_add(profile_stats, profile_enabled, 'save_result', time.time()-t_stage)
    json[json_key]=to_item_np(res)
    _stage_profile_add(profile_stats, profile_enabled, 'total_wall', time.time()-t_wall)
    _stage_profile_report(profile_stats, profile_enabled, f'{data}-{task_name}-{mus}-{urvtype}-{durate}')
    if pt:print(res)
    return res
def ini_cpu(unlearn_batch,model_group):
    ds,urvs,durates,tasks=datasets,DuDrDvTypes,DuRates,['Sim','Map']
    for data in ds:
        for urv in urvs:
            for durate in durates:
                for task in tasks:
                    one(data=data,mu='Origin',task=task,urvtype=urv,durate=durate,pre_data=1,unlearn_batch=unlearn_batch,model_group=model_group)
def ini_gpu(unlearn_batch,model_group,pt=True):
    durates,datas,urvs,tasks,mus=DuRates,datasets,DuDrDvTypes,list(TasksModel.keys()),MUs
    for durate in durates:
        for data in datas:
            for urv in urvs:
                for task in tasks:
                    for mu in mus:
                        if task in ['Rec','Simp'] and mu=='Origin':
                            one(data=data,mu='Origin',task=task,urvtype=urv,durate=durate,pre_data=1,pt=pt,unlearn_batch=unlearn_batch,model_group=model_group)
                        one(data=data,mu=mu,task=task,urvtype=urv,durate=durate,pre_data=0,pt=pt,unlearn_batch=unlearn_batch,model_group=model_group)
                        if test_each_epoch:
                            for epoch in list(range(tune_epoch[mu]))[::-1]:
                                one(data=data,mu=mu,task=task,urvtype=urv,durate=durate,epoch_tune=epoch+1,pre_data=0,pt=pt,unlearn_batch=unlearn_batch,model_group=model_group)
def parse():
    global datasets,TasksModel,DuDrDvTypes,DuRates,MUs
    parser = argparse.ArgumentParser(description='mutraj')
    parser.add_argument('--data',choices=datasets_all,type=str,default='')
    parser.add_argument('--task',default='')
    parser.add_argument('--model_group',type=int,default=1)
    parser.add_argument('--urv',choices=DuDrDvTypes,default='')
    parser.add_argument('--durate',choices=DuRates,type=float,default=0)
    parser.add_argument('--mu',choices=list(mu_methods.keys()),type=str,default='')
    parser.add_argument('--pre_data',action='store_true')
    parser.add_argument('--ini_cpu',action='store_true')
    parser.add_argument('--ini_gpu',action='store_true')
    parser.add_argument('--rerun',action='store_true')
    parser.add_argument('--ptloss',action='store_true')
    parser.add_argument('--unlearn_batch',choices=[1,2],type=int,default=1)
    args = parser.parse_args()
    if args.mu: MUs=[args.mu]
    if args.data: datasets=[args.data]
    TasksModel={task:get_task_model_class(task,args.model_group) for task in TASK_DEFAULT_MODEL}
    if args.task:
        if args.model_group: task_model_name(args.task,args.model_group)
        TasksModel={args.task:get_task_model_class(args.task,args.model_group)}
    if args.urv:DuDrDvTypes=[args.urv]
    if args.durate: DuRates=[args.durate]
    print('args: ', args._get_kwargs())
    if args.ini_cpu:
        ini_cpu(unlearn_batch=args.unlearn_batch,model_group=args.model_group)
        print('ini_cpu over')
        return
    if args.ini_gpu and (torch.cuda.is_available()or torch.mps.is_available()):
        ini_gpu(unlearn_batch=args.unlearn_batch,model_group=args.model_group)
        print('ini_gpu over')
        return
    run_data = args.data or datasets[0]
    run_task = args.task or next(iter(TasksModel.keys()))
    run_urv = args.urv or DuDrDvTypes[0]
    run_durate = args.durate or DuRates[0]
    for mu in MUs:
        if args.mu:mu=args.mu
        one(data=run_data,mu=mu,task=run_task,urvtype=run_urv,durate=run_durate,pre_data=args.pre_data,rerun=args.rerun,ptloss=args.ptloss,unlearn_batch=args.unlearn_batch,model_group=args.model_group)
        if args.mu:break
    print('mu over')
if __name__=='__main__':
    print('mu traj begin')
    print('cuda',torch.cuda.is_available())
    parse()
    print('mu traj over')
