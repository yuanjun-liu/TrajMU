exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import _tool.mPlot as mplt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import math
from _tool.mPlot import set_figsize
import numpy as np
import os
from _tool.mList import zipxs
from _tool.mFile import out_dir
from _tool.SysMonitor import LogJsonIdxs
from _tool.mData import alpha
from mu.traj.model_registry import get_metric_bundle, task_model_name, task_result_name
MUs = ['Retrain','FineTune', 'NegGrad', 'BadT','SCRUB','GDRGMA', 'TopK','RandomK','SFRon','SSD']
Tasks=['Sim','Simp','Map','Rec']
Datasets=['Porto','Beijing','Xian'] 
DuRates=[0.1,0.2,0.3]
Urvs=['Usr','Area']
model1_metrics1={task:get_metric_bundle(task,model_group=1)['primary'] for task in Tasks}
model1_metrics2={task:get_metric_bundle(task,model_group=1)['secondary'] for task in Tasks}
model2_metrics1={task:get_metric_bundle(task,model_group=2)['primary'] for task in Tasks}
model2_metrics2={task:get_metric_bundle(task,model_group=2)['secondary'] for task in Tasks}
metricses=[[model1_metrics1,model1_metrics2],[model2_metrics1,model2_metrics2]]
path_res='./mu-traj.json'
tune_epoch={'Origin':0, 'Retrain':0, 'FineTune':4, 'NegGrad':7, 'BadT':10,'SCRUB':10,'GDRGMA':2,'TopK':10,'RandomK':4,'SFRon':10,'SSD':1}
_json=LogJsonIdxs(path_res,refresh=False,mode='r')
mu2=[x for x in MUs if x != 'Retrain']
def tbf(x,f,b=3):
    x=float(x)
    a=f':.{b}f'
    x=eval("f'{x"+a+"}'")
    if f==0: return '\\textbf{'+x+'}'
    if f==1: return '\\underline{'+x+'}'
    return x
def _stable_unlearn_batch(mu, unlearn_batch=1):
    return 1 if mu in {'Origin','Retrain'} else unlearn_batch
def _task_metric(task, model, metrics_map):
    if isinstance(metrics_map, dict):
        if task in metrics_map and not isinstance(metrics_map[task], dict):
            return metrics_map[task]
        task_key=task_result_name(task, model)
        if task_key in metrics_map:
            return metrics_map[task_key]
    return get_metric_bundle(task, model)['primary']
def _metric_key_name(task, model, metric):
    return get_metric_bundle(task, model)['metric_keys'][metric]
_abs_compare_metrics={'MIA','HR','HR1','HR5','HR10','F1','Acc','AVGTC'}
def _is_abs_compare_metric(metric):
    return metric in _abs_compare_metrics or (isinstance(metric,str) and metric.startswith('TC') and metric[2:].isdigit())
def _comparison_metric(task, model, metrics_map, key):
    if isinstance(task,list):
        ms=[_comparison_metric(t,model,metrics_map,key) for t in task]
        return ms[0] if all(m==ms[0] for m in ms) else None
    if isinstance(key,list):
        ms=[_comparison_metric(task,model,metrics_map,k) for k in key]
        return ms[0] if all(m==ms[0] for m in ms) else None
    if key=='MIA':
        return 'MIA'
    if key=='time':
        return 'time'
    return _task_metric(task,model,metrics_map)
def _similarity_score(x,x_gt,metric):
    if _is_abs_compare_metric(metric):
        return 1-abs(x-x_gt)
    x_max=max(x,x_gt)
    if x_max==0:
        return 1
    return min(x,x_gt)/x_max
def json_item(data=Datasets,mu=mu2,task=Tasks,urv=Urvs,key='',durate=DuRates,metrics=model1_metrics1,sims=False,unlearn_batch=1,model_group=1):
    """a col of SimScore/Rank of all MUs"""
    if not isinstance(data,list):data=[data]
    if not isinstance(task,list):task=[task]
    if not isinstance(durate,list):durate=[durate]
    if not isinstance(urv,list):urv=[urv]
    if not isinstance(key,list):key=[key]
    if not isinstance(mu,list):mu=[mu]
    res=[] 
    gt='Retrain'
    for u in mu:
        ep=tune_epoch[u]
        ep_gt=tune_epoch[gt]
        stable_unlearn_batch=_stable_unlearn_batch(u,unlearn_batch)
        stable_unlearn_batch_gt=_stable_unlearn_batch(gt,unlearn_batch)
        _res=0 ; _count=0
        for _data in data:
            for _task in task:
                if _data=='AIS':
                    if _task=='Rec':model_group=2 
                    else:model_group=1
                _task_name=task_result_name(_task,model_group)
                for _rate in durate:
                    for _urv in urv:
                        for _key in key:
                            metric=_task_metric(_task,model_group,metrics)
                            comparison_metric=_comparison_metric(_task,model_group,metrics,_key)
                            k_du=_metric_key_name(_task,model_group,metric)+'Du'
                            k_dr=_metric_key_name(_task,model_group,metric)+'Dr'
                            k_dv=_metric_key_name(_task,model_group,metric)+'Dv'
                            k_mia=_metric_key_name(_task,model_group,'MIA')
                            _key={'Du':k_du,'Dr':k_dr,'Dv':k_dv,'MIA':k_mia,'time':'time'}[_key]
                            i=[_data,u,_task_name,_urv,_rate,ep]
                            if stable_unlearn_batch!=1:
                                i.append(f'u{stable_unlearn_batch}')
                            i.append(_key)
                            x=_json[tuple(map(str,i))]
                            if x is None: raise RuntimeError()
                            if sims:
                                i_gt=[_data,gt,_task_name,_urv,_rate,ep_gt]
                                if stable_unlearn_batch_gt!=1:
                                    i_gt.append(f'u{stable_unlearn_batch_gt}')
                                i_gt.append(_key)
                                x_gt=_json[tuple(map(str,i_gt))]
                                if x_gt is None:raise RuntimeError()
                                x=_similarity_score(x,x_gt,comparison_metric)
                            _res+=x ; _count+=1
        res.append(_res/_count)
    res=np.array(res) if len(res)>1 else res[0]
    return res
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from traj.data.load_trajs import load_ts_box 
from traj.data.process_ts import t2ps_steplen_jit as t2ps_steplen
from _tool.mIO import loadZ_pk,saveZ_pk
from mu.traj.loaddata import ts_split,fix_trajs_num
def plt_trajs(e=1e-4, path_figs=''):
    color_red,color_blue=[0.9,0,0,0.8],[0,0,0.9]
    cmap = ListedColormap([(1,1,1), color_blue, color_red])
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N) 
    set_figsize(width=1.1*3,height=1.42*2,dpi=100)
    fig, axs = plt.subplots(nrows=2,ncols=3)
    for di,data in enumerate(['Porto','Beijing','Xian']):
        x=load_ts_box(data)
        ts,uid,bbox=x[0],x[1],x[-1]
        [xmin,xmax],[ymin,ymax] = bbox
        gwidth,gheight=int((xmax-xmin)/e)+3,int((ymax-ymin)/e)+3
        def t2gxgy(T:np.ndarray):
            T=t2ps_steplen(T,e)
            x,y=T[:,0],T[:,1]
            xi,yi=((x-xmin)/e+0.5).astype(int),((y-ymin)/e+0.5).astype(int)
            return xi,yi
        durate=0.1
        for ri,urv in enumerate(['Usr','Area']):
            ax:Axes=axs[ri,di]
            path=os.path.join(out_dir('cache'),f'plt-ts-{data}-{urv}-{durate}.pk.zst')
            if os.path.exists(path):
                print('load',path)
                dr,du=loadZ_pk(path)
            else:
                train, val, test, du, dr, dv=ts_split(ts,uid,urv,float(durate))
                dr_idx,du_idx=fix_trajs_num({'dr':dr,'du':du},float(durate)).values()
                dr,du=ts[dr_idx],ts[du_idx]
                saveZ_pk(path,[dr,du])
            label_map = np.zeros((gwidth,gheight), dtype=int)
            for T in dr:
                if T.shape[0] < 2: continue
                gx,gy=t2gxgy(T)
                label_map[gx-1,gy]=1
                label_map[gx-1,gy-1]=1
                label_map[gx-1,gy+1]=1
                label_map[gx,gy]=1
                label_map[gx,gy-1]=1
                label_map[gx,gy+1]=1
                label_map[gx+1,gy]=1
                label_map[gx+1,gy-1]=1
                label_map[gx+1,gy+1]=1
            for T in du:
                if T.shape[0] < 2:continue
                gx,gy=t2gxgy(T)
                label_map[gx+1,gy+1]=2
            ax.imshow(label_map, cmap=cmap, norm=norm, origin='lower',aspect='auto')
            ax.set_ylim(0, gwidth)
            ax.set_xlim(0, gheight)
            ax.set_xticks([]);ax.set_yticks([])
            sdata='BJ' if data =='Beijing' else data
            if urv=='Usr':urv='User'
            ax.set_xlabel(f'({alpha[ri*3+di]}) {int(float(durate)*100)}% {sdata} {urv}')
    del du,dr
    legend_elements = [
        Patch(facecolor=color_blue, edgecolor=None, label='Dr (the remaining set)',color=None),
        Patch(facecolor=color_red, edgecolor=None, label='Du (the unlearning set)',color=None),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.998), labelspacing=0.,ncol=2)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.16)
    if path_figs:
        plt.savefig(path_figs, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)
def task_metric_data_task__same_model_group(model_group=1,keys=['Du','Dr','Dv'],pt=True): 
    num_col=len(Datasets)*len(Tasks)*2
    raw=np.zeros((len(MUs),num_col))
    sim=np.zeros((len(mu2),num_col))
    rank=np.zeros((len(mu2),num_col))
    rks=[]
    for col,(metric_group, data,task) in enumerate(zipxs([1,2],Datasets,Tasks)):
        raw[:,col]=json_item(data=data,mu=MUs,task=task,key=keys,metrics=metricses[model_group-1][metric_group-1],model_group=model_group,urv=Urvs,durate=DuRates)
        comparison_metric=_comparison_metric(task,model_group,metricses[model_group-1][metric_group-1],keys)
        for mi,mu in enumerate(mu2):sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        if col%len(Tasks)==len(Tasks)-1: rks.append(np.argsort(np.argsort(-sim[:,col-len(Tasks)+1:col+1].sum(axis=1))))
    if not pt: return raw,sim,rank,rks
    print('Retrain',end='')
    for col,(model_group, data,task) in enumerate(zipxs([1,2],Datasets,Tasks)):
        if model_group ==1:
            if task=='Rec':  print(f'&{tbf(raw[0][col],-1,1)}',end='')
            else: print(f'&{tbf(raw[0][col],-1,3)}',end='')
        if col%len(Tasks)==len(Tasks)-1:print('&-',end='')
    print('\\\\ \\midrule')
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        for i1 in range(2):
            for i2,data in enumerate(Datasets):
                for i3,task in enumerate(Tasks):
                    col=i1*len(Datasets)*len(Tasks)+i2*len(Tasks)+i3
                    if i1==0:
                        if task=='Rec':  print(f'&{tbf(raw[1+mi][col],rank[mi][col],1)}',end='')
                        else:print(f'&{tbf(raw[1+mi][col],rank[mi][col],3)}',end='')
                    rk=rks[col//len(Tasks)]
                    if col%len(Tasks)==len(Tasks)-1:
                        print(f'&{tbf(rk[mi]+1,rk[mi],0)}',end='  ')
        print('\\\\')
def task_model_data_task__same_metric_group(metric_group=1,keys=['Du','Dr','Dv'],pt=True):
    num_col=len(Datasets)*len(Tasks)*2
    raw=np.zeros((len(MUs),num_col))
    sim=np.zeros((len(mu2),num_col))
    rank=np.zeros((len(mu2),num_col))
    rks=[]
    for col,(model_group, data,task) in enumerate(zipxs([1,2],Datasets,Tasks)):
        raw[:,col]=json_item(data=data,mu=MUs,task=task,key=keys,metrics=metricses[model_group-1][metric_group-1],model_group=model_group,urv=Urvs,durate=DuRates)
        comparison_metric=_comparison_metric(task,model_group,metricses[model_group-1][metric_group-1],keys)
        for mi,mu in enumerate(mu2):sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        if col%len(Tasks)==len(Tasks)-1: rks.append(np.argsort(np.argsort(-sim[:,col-len(Tasks)+1:col+1].sum(axis=1))))
    if not pt: return raw,sim,rank,rks
    print('Retrain',end='')
    for col,(model_group, data,task) in enumerate(zipxs([1,2],Datasets,Tasks)):
        if model_group ==1:
            if task=='Rec':  print(f'&{tbf(raw[0][col],-1,1)}',end='')
            else: print(f'&{tbf(raw[0][col],-1,3)}',end='')
        if col%len(Tasks)==len(Tasks)-1:print('&-',end='')
    print('\\\\ \\midrule')
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        for i1 in range(2):
            for i2,data in enumerate(Datasets):
                for i3,task in enumerate(Tasks):
                    col=i1*len(Datasets)*len(Tasks)+i2*len(Tasks)+i3
                    if i1==0:
                        if task=='Rec':  print(f'&{tbf(raw[1+mi][col],rank[mi][col],1)}',end='')
                        else:print(f'&{tbf(raw[1+mi][col],rank[mi][col],3)}',end='')
                    rk=rks[col//len(Tasks)]
                    if col%len(Tasks)==len(Tasks)-1:
                        print(f'&{tbf(rk[mi]+1,rk[mi],0)}',end='  ')
        print('\\\\')
def task_model_metric_data_rank(keys=['Du','Dr','Dv'],pt=True): 
    num_col=len(Datasets)*len(Tasks)*2*2
    raw=np.zeros((len(MUs),num_col))
    sim=np.zeros((len(mu2),num_col))
    rank=np.zeros((len(mu2),num_col))
    rks=[]
    for col,(model_group,metric_group, data,task) in enumerate(zipxs([1,2],[1,2],Datasets,Tasks)):
        metrics=metricses[model_group-1][metric_group-1]
        raw[:,col]=json_item(data=data,mu=MUs,task=task,key=keys,metrics=metrics,model_group=model_group,urv=Urvs,durate=DuRates)
        comparison_metric=_comparison_metric(task,model_group,metrics,keys)
        for mi,mu in enumerate(mu2):sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        if col%len(Tasks)==len(Tasks)-1: rks.append(np.argsort(np.argsort(-sim[:,col-len(Tasks)+1:col+1].sum(axis=1))))
    if not pt: return raw,sim,rank,rks
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        for rk in rks:
            print(f'&{tbf(rk[mi]+1,rk[mi],0)}',end='  ')
        print('\\\\')
def task__data_task__metric_group__model_group(keys=['Du','Dr','Dv'],pt=True): 
    num_col=len(Datasets)*len(Tasks)*3
    raw=np.zeros((len(MUs),num_col))
    sim=np.zeros((len(mu2),num_col))
    rank=np.zeros((len(mu2),num_col))
    rks=[]
    col=0
    model_group=1
    for col,(metric_group, data,task) in enumerate(zipxs([1,2],Datasets,Tasks)):
        raw[:,col]=json_item(data=data,mu=MUs,task=task,key=keys,metrics=metricses[model_group-1][metric_group-1],model_group=model_group,urv=Urvs,durate=DuRates)
        comparison_metric=_comparison_metric(task,model_group,metricses[model_group-1][metric_group-1],keys)
        for mi,mu in enumerate(mu2):sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        if col%len(Tasks)==len(Tasks)-1: rks.append(np.argsort(np.argsort(-sim[:,col-len(Tasks)+1:col+1].sum(axis=1))))
    col=len(Datasets)*len(Tasks)*2
    model_group=2;metric_group=1
    for _,(data,task) in enumerate(zipxs(Datasets,Tasks)):
        raw[:,col]=json_item(data=data,mu=MUs,task=task,key=keys,metrics=metricses[model_group-1][metric_group-1],model_group=model_group,urv=Urvs,durate=DuRates)
        comparison_metric=_comparison_metric(task,model_group,metricses[model_group-1][metric_group-1],keys)
        for mi,mu in enumerate(mu2):sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        if col%len(Tasks)==len(Tasks)-1: rks.append(np.argsort(np.argsort(-sim[:,col-len(Tasks)+1:col+1].sum(axis=1))))
        col+=1
    if not pt: return raw,sim,rank,rks
    print('Retrain',end='')
    for col,(model_group, data,task) in enumerate(zipxs([1,2],Datasets,Tasks)):
        if model_group ==1:
            if task=='Rec':  print(f'&{tbf(raw[0][col],-1,1)}',end='')
            else: print(f'&{tbf(raw[0][col],-1,3)}',end='')
        if col%len(Tasks)==len(Tasks)-1:print('&-',end='')
    col=len(Datasets)*len(Tasks)*2
    for (data,task) in enumerate(zipxs(Datasets,Tasks)):
        if col%len(Tasks)==len(Tasks)-1:print('&-',end='')
        col+=1
    print('\\\\ \\midrule')
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        col=0
        for i1 in range(2):
            for i2,data in enumerate(Datasets):
                for i3,task in enumerate(Tasks):
                    if i1==0:
                        if task=='Rec':  print(f'&{tbf(raw[1+mi][col],rank[mi][col],1)}',end='')
                        else:print(f'&{tbf(raw[1+mi][col],rank[mi][col],3)}',end='')
                    rk=rks[col//len(Tasks)]
                    if col%len(Tasks)==len(Tasks)-1:
                        print(f'&{tbf(rk[mi]+1,rk[mi],0)}',end='  ')
                    col+=1
        for i2,data in enumerate(Datasets):
            for i3,task in enumerate(Tasks):
                rk=rks[col//len(Tasks)]
                if col%len(Tasks)==len(Tasks)-1:
                    print(f'&{tbf(rk[mi]+1,rk[mi],0)}',end='  ')
                col+=1
        print('\\\\')
def views_task_data_urv(keys=['Du','Dr','Dv'],pt=True):
    num_col=len(Datasets)+len(Tasks)+len(Urvs)
    raw=np.zeros((len(MUs),num_col))
    sim=np.zeros((len(mu2),num_col))
    rank=np.zeros((len(mu2),num_col))
    model_group,metric_group=1,1
    col=0
    for data in Datasets:
        raw[:,col]=json_item(data=data,mu=MUs,key=keys,metrics=metricses[model_group-1][metric_group-1],model_group=model_group)
        comparison_metric=_comparison_metric(Tasks,model_group,metricses[model_group-1][metric_group-1],keys)
        for mi,mu in enumerate(mu2):sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        col+=1
    for task in Tasks:
        raw[:,col]=json_item(mu=MUs,task=task,key=keys,metrics=metricses[model_group-1][metric_group-1],model_group=model_group)
        comparison_metric=_comparison_metric(task,model_group,metricses[model_group-1][metric_group-1],keys)
        for mi,mu in enumerate(mu2):sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        col+=1
    for urv in Urvs:
        raw[:,col]=json_item(mu=MUs,key=keys,metrics=metricses[model_group-1][metric_group-1],model_group=model_group,urv=urv)
        comparison_metric=_comparison_metric(Tasks,model_group,metricses[model_group-1][metric_group-1],keys)
        for mi,mu in enumerate(mu2):sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        col+=1
    assert col==num_col
    if not pt: return raw,sim,rank
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        for col in range(num_col):
            print(f'&{tbf(raw[1+mi][col],rank[mi][col],3)}',end='')
            print(f'&{tbf(rank[mi][col]+1,rank[mi][col],0)}',end='  ')
        print('\\\\')
def mia_model_scenario_task(keys=['MIA'],pt=True): 
    num_col=len(Urvs)*len(Tasks)*2
    raw=np.zeros((len(MUs),num_col))
    sim=np.zeros((len(mu2),num_col))
    rank=np.zeros((len(mu2),num_col))
    rks=[]
    metric_group=1
    for col,(model_group,urv,task) in enumerate(zipxs([1,2],Urvs,Tasks)):
        raw[:,col]=json_item(data=Datasets,mu=MUs,task=task,key=keys,metrics=metricses[model_group-1][metric_group-1],model_group=model_group,urv=urv,durate=DuRates)
        comparison_metric=_comparison_metric(task,model_group,metricses[model_group-1][metric_group-1],keys)
        for mi,mu in enumerate(mu2): sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        if col%len(Tasks)==len(Tasks)-1: rks.append(np.argsort(np.argsort(-sim[:,col-len(Tasks)+1:col+1].sum(axis=1))))
    if not pt: return raw,sim,rank,rks
    print('Retrain',end='')
    for col,(model_group,urv,task) in enumerate(zipxs([1,2],Urvs,Tasks)):
        print(f'&{tbf(raw[0][col],-1,3)}',end='')   
        if col%len(Tasks)==len(Tasks)-1:print('&-',end='')
    print('\\\\ \\midrule')
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        for i1 in range(2):
            for i2,urv in enumerate(Urvs):
                for i3,task in enumerate(Tasks):
                    col=i1*len(Urvs)*len(Tasks)+i2*len(Tasks)+i3
                    print(f'&{tbf(raw[1+mi][col],rank[mi][col],3)}',end='')
                    rk=rks[col//len(Tasks)]
                    if col%len(Tasks)==len(Tasks)-1:
                        print(f'&{tbf(rk[mi]+1,rk[mi],0)}',end='  ')
        print('\\\\')
def unlearn_batch(key_task=['Du','Dr','Dv'],key_mia=['MIA'],pt=True):
    metric_group,model_group=1,1
    num_col=len(Tasks)*4
    raw=np.zeros((len(MUs),num_col))
    sim=np.zeros((len(mu2),num_col))
    rank=np.zeros((len(mu2),num_col))
    rks=[]
    for col,(unlearn_num,is_mia,task) in  enumerate(zipxs([1,2],[0,1],Tasks)):
        _key=[key_task,key_mia][is_mia]
        raw[:,col]=json_item(mu=MUs,task=task,key=_key,metrics=metricses[model_group-1][metric_group-1],model_group=model_group,unlearn_batch=unlearn_num)
        comparison_metric=_comparison_metric(task,model_group,metricses[model_group-1][metric_group-1],_key)
        for mi,mu in enumerate(mu2): sim[mi,col]=_similarity_score(raw[1+mi,col],raw[0,col],comparison_metric)
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
        if col%len(Tasks)==len(Tasks)-1: rks.append(np.argsort(np.argsort(-sim[:,col-len(Tasks)+1:col+1].sum(axis=1))))
    if not pt: return raw,sim,rank,rks
    print('Retrain',end='')
    for col,(unlearn_num,is_mia,task) in  enumerate(zipxs([1,2],[0,1],Tasks)):
        if task=='Rec' and not is_mia:  print(f'&{tbf(raw[0][col],-1,1)}',end='')
        else: print(f'&{tbf(raw[0][col],-1,3)}',end='')  
        if col%len(Tasks)==len(Tasks)-1:print('&-',end='')
    print('\\\\ \\midrule')
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        for i1,unlearn_num in enumerate([1,2]):
            for i2,is_mia in enumerate([0,1]):
                for i3,task in enumerate(Tasks):
                    col=i1*2*len(Tasks)+i2*len(Tasks)+i3
                    if task=='Rec' and not is_mia:  print(f'&{tbf(raw[1+mi][col],rank[mi][col],1)}',end='')
                    else:print(f'&{tbf(raw[1+mi][col],rank[mi][col],3)}',end='')
                    rk=rks[col//len(Tasks)]
                    if col%len(Tasks)==len(Tasks)-1:
                        print(f'&{tbf(rk[mi]+1,rk[mi],0)}',end='  ')
        print('\\\\')
def speed_up(pt=True): 
    sp_time=np.zeros((len(mu2),4))
    rk_time=np.zeros((len(mu2),4),dtype=int)
    for ti,task in enumerate(Tasks):
        gt_time=json_item(mu='Retrain',task=task,key='time')
        time=json_item(task=task,key='time')
        sp_time[:,ti]=gt_time/time
        rk_time[:,ti]=np.argsort(np.argsort(-sp_time[:,ti]))
    sum_rk_time=rk_time.sum(axis=1)
    sum_rk_time=np.argsort(np.argsort(sum_rk_time))
    for mi,mu in enumerate(mu2):
        print(f'{mu}&',end='')
        for ti,task in enumerate(Tasks): print(f'{tbf(sp_time[mi,ti],rk_time[mi,ti],0)}&',end='')
        print(f'{tbf(sum_rk_time[mi]+1,sum_rk_time[mi],0)}\\\\')
if __name__=='__main__':
    pass
    task_metric_data_task__same_model_group(model_group=1); print('\n') 
    task_metric_data_task__same_model_group(model_group=2); print('\n') 
