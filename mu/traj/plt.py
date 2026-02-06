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
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from traj.data.load_trajs import load_ts_box 
from traj.data.process_ts import t2ps_steplen_jit as t2ps_steplen
from _tool.mIO import loadZ_pk,saveZ_pk
from mu.traj.loaddata import ts_split,fix_trajs_num

MUs = ['Retrain', 'FineTune', 'NegGrad', 'BadT','SCRUB','GDRGMA','TopK','RandomK','SFRon','SSD']
Tasks=['Sim','Simp','Map','Rec']
Datasets=['Porto','Beijing'] 
DuRates=[0.1,0.2,0.3]
Urvs=['Usr','Area']

metrics={'Sim':'MR','Simp':'SED','Map':'F1','Rec':'MAE'}
metrics2={'Sim':'HR10','Simp':'F1','Map':'Acc','Rec':'Acc'}
metric_keys={
    'Sim':{'MR':'MR_','MRR':'MRR_','MIA':'MIA2','HR10':'HR10_','HR5':'HR5_','HR1':'HR1_'},
    'Simp':{'SED':'SEDwQ_','MIA':'MIA3Q','F1':'RangeF1'},
    'Map':{'Acc':'Acc_','MIA':'MIA5','F1':'F1_'},
    'Rec':{'MAE':'MAE_','MIA':'MIA3','RMSE':'RMSE_','Acc':'Acc_','F1':'F1_'},
    }

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
def json_item(data=Datasets,mu=mu2,task=Tasks,urv=Urvs,key='',durate=DuRates,metrics=metrics):
    """a col of SimScore/Rank of all MUs"""
    if not isinstance(data,list):data=[data]
    if not isinstance(task,list):task=[task]
    if not isinstance(durate,list):durate=[durate]
    if not isinstance(urv,list):urv=[urv]
    if not isinstance(key,list):key=[key]
    if not isinstance(mu,list):mu=[mu]
    res=[] 
    for u in mu:
        ep=tune_epoch[u]
        _res=0 ; _count=0
        for _data in data:
            for _task in task:
                for _rate in durate:
                    for _urv in urv:
                        for _key in key:
                            metric=metrics[_task]
                            k_du=metric_keys[_task][metric]+'Du'
                            k_dr=metric_keys[_task][metric]+'Dr'
                            k_dv=metric_keys[_task][metric]+'Dv'
                            k_mia=metric_keys[_task]['MIA']
                            _key={'Du':k_du,'Dr':k_dr,'Dv':k_dv,'MIA':k_mia,'time':'time'}[_key]
                            i=[_data,u,_task,_urv,_rate,ep,_key]
                            x=_json[tuple(map(str,i))]
                            
                            _res+=x ; _count+=1
        res.append(_res/_count)
    res=np.array(res) if len(res)>1 else res[0]
    return res

def task_cityurv_raw_simsort(task,keys=['Du','Dr','Dv','MIA'],pt=True,durate=DuRates):
    """ method | task result of {data}-{urv} (x4) | rank of SimScore  """
    raw=np.zeros((len(MUs),len(Datasets)*len(Urvs)*len(keys)))
    sim=np.zeros((len(mu2),len(Datasets)*len(Urvs)*len(keys)))
    rank=np.zeros((len(mu2),len(Datasets)*len(Urvs)*len(keys)))
    for col,(data,urv,key) in enumerate(zipxs(Datasets,Urvs,keys)):
        raw[:,col]=json_item(data=data,mu=MUs,task=task,urv=urv,key=key,durate=durate)
        for mi,mu in enumerate(mu2):sim[mi,col]=min(raw[0,col],raw[1+mi,col])/max(raw[0,col],raw[1+mi,col])
        
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
    sim_sum=sim.sum(axis=1)
    rk_sum=np.argsort(np.argsort(-sim_sum))
    
    if not pt:return rk_sum
    print(task)
    
    print('Retrain&',end='')
    for col,(data,urv,key) in enumerate(zipxs(Datasets,Urvs,keys)):
        print(f'{tbf(raw[0][col],-1,3)}&',end='')
    print('-\\\\ \\midrule')
    
    for mi,mu in enumerate(mu2):
        print(f'{mu}&',end='')
        for col,(data,urv,key) in enumerate(zipxs(Datasets,Urvs,keys)):
            print(f'{tbf(raw[1+mi][col],rank[mi][col],3)}&',end='')
        print(f'{tbf(rk_sum[mi]+1,rk_sum[mi],0)} \\\\'+('\\midrule' if mu=='SSD' else ''))
    return rk_sum
def all_tasks_rank_times_rank(task_keys=['Du','Dr','Dv','MIA']):
    """ task ranks (x4) | rank | Time speed up (x4) | rank  """
    rk_tasks=np.zeros((len(mu2),4),dtype=int)
    sp_time=np.zeros((len(mu2),4))
    rk_time=np.zeros((len(mu2),4),dtype=int)
    for ti,task in enumerate(Tasks):
        rk_tasks[:,ti]=task_cityurv_raw_simsort(task=task,pt=False,keys=task_keys)
        gt_time=json_item(mu='Retrain',task=task,key='time')
        time=json_item(task=task,key='time')
        sp_time[:,ti]=gt_time/time
        rk_time[:,ti]=np.argsort(np.argsort(-sp_time[:,ti]))
    sum_rk_task=rk_tasks.sum(axis=1)
    sum_rk_time=rk_time.sum(axis=1)
    sum_rk_task=np.argsort(np.argsort(sum_rk_task))
    sum_rk_time=np.argsort(np.argsort(sum_rk_time))
    
    for mi,mu in enumerate(mu2):
        print(f'{mu}&',end='')
        for ti,task in enumerate(Tasks): print(f'{tbf(rk_tasks[mi,ti]+1,rk_tasks[mi,ti],0)}&',end='')
        print(f'{tbf(sum_rk_task[mi]+1,sum_rk_task[mi],0)}&',end='  ')
        for ti,task in enumerate(Tasks): print(f'{tbf(sp_time[mi,ti],rk_time[mi,ti],0)}&',end='')
        print(f'{tbf(sum_rk_time[mi]+1,sum_rk_time[mi],0)}\\\\' +('\\midrule' if mu=='SSD' else ''))


def json_item(data=Datasets,mu=mu2,task=Tasks,urv=Urvs,key='',durate=DuRates):
    """a col of SimScore/Rank of all MUs"""
    if not isinstance(data,list):data=[data]
    if not isinstance(task,list):task=[task]
    if not isinstance(durate,list):durate=[durate]
    if not isinstance(urv,list):urv=[urv]
    if not isinstance(key,list):key=[key]
    if not isinstance(mu,list):mu=[mu]
    res=[] 
    for u in mu:
        ep=tune_epoch[u]
        _res=0 ; _count=0
        for _data in data:
            for _task in task:
                for _rate in durate:
                    for _urv in urv:
                        for _key in key:
                            metric=metrics[_task]
                            k_du=metric_keys[_task][metric]+'Du'
                            k_dr=metric_keys[_task][metric]+'Dr'
                            k_dv=metric_keys[_task][metric]+'Dv'
                            k_mia=metric_keys[_task]['MIA']
                            _key={'Du':k_du,'Dr':k_dr,'Dv':k_dv,'MIA':k_mia,'time':'time'}[_key]
                            i=[_data,u,_task,_urv,_rate,ep,_key]
                            x=_json[tuple(map(str,i))]
                            
                            _res+=x ; _count+=1
        res.append(_res/_count)
    res=np.array(res) if len(res)>1 else res[0]
    return res

def plt_trajs(e=1e-4, path_figs='',debug=False):
    color_red,color_blue=[0.9,0,0,0.8],[0,0,0.9]
    cmap = ListedColormap([(1,1,1), color_blue, color_red])
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N) 
    set_figsize(width=1.1*4,height=1.42*1,dpi=100)
    fig, axs = plt.subplots(nrows=1,ncols=4)
    for di,data in enumerate(['Porto','Beijing']):
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
            
            col=di*2+ri
            ax:Axes=axs[col]
            
            path=os.path.join(out_dir('cache'),f'plt-ts-{data}-{urv}-{durate}.pk.zst')
            if os.path.exists(path):
                print('load',path)
                dr,du=loadZ_pk(path)
            else:
                train, val, test, du, dr, dv=ts_split(ts,uid,urv,float(durate))
                dr_idx,du_idx=fix_trajs_num({'dr':dr,'du':du},float(durate)).values()
                dr,du=ts[dr_idx],ts[du_idx]
                saveZ_pk(path,[dr,du])
            if debug: du,dr=du[:100],dr[:1000]
            
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
            ax.imshow(label_map, cmap=cmap, norm=norm, origin='lower')
            ax.set_ylim(0, gwidth)
            ax.set_xlim(0, gheight)
            ax.set_xticks([]);ax.set_yticks([])
            sdata='BJ' if data =='Beijing' else data
            if urv=='Usr':urv='User'
            ax.set_xlabel(f'({alpha[col]}) {int(float(durate)*100)}% {sdata} {urv}')
            
            
    del du,dr
    legend_elements = [
        Patch(facecolor=color_blue, edgecolor=None, label='Dr (the remaining set)',color=None),
        Patch(facecolor=color_red, edgecolor=None, label='Du (the unlearning set)',color=None),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.996), labelspacing=0.,ncol=2)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0.06)
    if path_figs:
        plt.savefig(path_figs, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)


def ana_city_task(keys=['Du','Dr','Dv'],metrics=metrics):
    """ method | task result on {city1} (x4 task) | task result on {city2} (x4 task)"""
    raw=np.zeros((len(MUs),8))
    sim=np.zeros((len(mu2),8))
    rank=np.zeros((len(mu2),8))
    for col,(data,task) in enumerate(zipxs(Datasets,Tasks)):
        raw[:,col]=json_item(data=data,mu=MUs,task=task,key=keys,metrics=metrics)
        for mi,mu in enumerate(mu2):sim[mi,col]=min(raw[0,col],raw[1+mi,col])/max(raw[0,col],raw[1+mi,col])
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
    rk1=np.argsort(np.argsort(-sim[:,:4].sum(axis=1)))
    rk2=np.argsort(np.argsort(-sim[:,4:].sum(axis=1)))

    print('task on city')
    
    print('Retrain',end='')
    for col,(data,task) in enumerate(zipxs(Datasets,Tasks)):
        print(f'&{tbf(raw[0][col],-1,3)}',end='')
        if task==Tasks[-1]: print('&-',end='')            
    print('\\\\ \\midrule')
    
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        for i1,data in enumerate(Datasets):
            for i2,task in enumerate(Tasks):
                col=i1*len(Tasks)+i2
                print(f'&{tbf(raw[1+mi][col],rank[mi][col],3)}',end='')
                if i1==0 and i2==3:print(f'&{tbf(rk1[mi]+1,rk1[mi],0)}',end='  ')
                if i1==1 and i2==3:print(f'&{tbf(rk2[mi]+1,rk2[mi],0)}',end='  ')
        print(f'\\\\')
    return rk1,rk2


def ana_urv_mia():
    """ method | mia result on Usr (x4 task) | task result on Area (x4 task)"""
    raw=np.zeros((len(MUs),8))
    sim=np.zeros((len(mu2),8))
    rank=np.zeros((len(mu2),8))
    for col,(urv,task) in enumerate(zipxs(Urvs,Tasks)):
        raw[:,col]=json_item(urv=urv,mu=MUs,task=task,key='MIA')
        for mi,mu in enumerate(mu2):sim[mi,col]=min(raw[0,col],raw[1+mi,col])/max(raw[0,col],raw[1+mi,col])
        rank[:,col]=np.argsort(np.argsort(-sim[:,col]))
    rk1=np.argsort(np.argsort(-sim[:,:4].sum(axis=1)))
    rk2=np.argsort(np.argsort(-sim[:,4:].sum(axis=1)))

    print('mia on Scenarios')
    
    print('Retrain',end='')
    for col,(urv,task) in enumerate(zipxs(Urvs,Tasks)):
        print(f'&{tbf(raw[0][col],-1,3)}',end='')
        if task==Tasks[-1]: print('&-',end='')            
    print('\\\\ \\midrule')
    
    for mi,mu in enumerate(mu2):
        print(f'{mu}',end='')
        for i1,urv in enumerate(Urvs):
            for i2,task in enumerate(Tasks):
                col=i1*len(Tasks)+i2
                print(f'&{tbf(raw[1+mi][col],rank[mi][col],3)}',end='')
                if i1==0 and i2==3:print(f'&{tbf(rk1[mi]+1,rk1[mi],0)}',end='  ')
                if i1==1 and i2==3:print(f'&{tbf(rk2[mi]+1,rk2[mi],0)}',end='  ')
        print(f'\\\\')
    return rk1,rk2

if __name__=='__main__':
    task_cityurv_raw_simsort('Sim'); print('\n')
    task_cityurv_raw_simsort('Simp'); print('\n')
    task_cityurv_raw_simsort('Map'); print('\n')
    task_cityurv_raw_simsort('Rec'); print('\n')
    all_tasks_rank_times_rank()

    ana_city_task(metrics=metrics); print('\n')
    ana_city_task(metrics=metrics2); print('\n')
    ana_urv_mia(); print('\n')

