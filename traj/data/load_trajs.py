exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import os,sys
import numpy as np
from _tool.mData import dicted_idx
from _tool.mFile import path_traj, list_dir, read_lines_iter, parse_line,out_base
from _tool.mIO import mcache
from traj.data.process_ts import ids_shrink,ts_bound,ts_shrink,ts_bbox_tim,ts_remove_bad_speed,t_spare
from _tool.mTime import tim2sec
traj_data_path = {
    'Beijing':os.path.join(path_traj, 'beijing'),
    'Porto': os.path.join(path_traj, 'Porto/train.csv'),
    'Xian':os.path.join(path_traj, 'xian'),
}
ts_data_name=list(traj_data_path.keys())
filter_len=10 
def load_beijing(name='Beijing',filter_len=filter_len):
    """TS,UIDS"""
    TS=[] ; UIDS=[] 
    assert name in traj_data_path
    for file in list_dir(traj_data_path[name])[3]:
        print('process',file)
        lines=open(file,'r',encoding='utf-8').readlines()
        for line in lines:
            line=line.removesuffix('\n').removesuffix('\t')
            if len(line)<5:continue
            T=[]
            for utxy in line.split(';'):
                uid,tim,lon,lat=parse_line(utxy,',',str,int,float,float)
                if tim< 1227764824 or tim >1247764824:continue
                uid=dicted_idx('uid_didi',uid)
                T.append([lon,lat,tim])
            if len(T)<filter_len:continue
            TS.append(np.array(T))
            UIDS.append(uid)
    assert len(TS)==len(UIDS)
    return np.array([np.array(TS,dtype=object),np.array(UIDS,dtype=int)],dtype=object)


def load_Porto(name='Porto',filter_len=filter_len):
    """TS,UIDS,Type"""
    assert name in traj_data_path
    data = traj_data_path[name]
    _tim = 0
    trajs = [] ; UIDS=[];Types=[]
    for line in read_lines_iter(data, 1):
        T = []
        xxx=line.split('","')
        uid=dicted_idx('uid_porto',xxx[4])
        call_type=dicted_idx('call_type_porto',xxx[1])
        tim=int(xxx[5])
        points=xxx[-1][2:-4]
        if points == "": continue
        for point in points.split('],['):
            pp = point.split(',')
            lat, lon = float(pp[0]), float(pp[1])
            T.append([lon ,lat , tim])
            tim+=15
        if len(T)>filter_len: 
            trajs.append(np.array(T))
            UIDS.append(uid)
            Types.append(call_type)
    assert len(trajs)==len(UIDS)
    return np.array([np.array(trajs,dtype=object),np.array(UIDS,dtype=int),np.array(Types,dtype=int)],dtype=object)
def load_xian(name='Xian',time_int=5,dis_int=1e-5,time_span='30m',filter_len=filter_len):
    """TS,UIDS"""
    TS=[] ; UIDS=[] ;_uid=None ; uts={}
    time_span = tim2sec(time_span)
    def append_filter_len(T):
        if len(T)>=filter_len:
            TS.append(T)
            UIDS.append(_uid)
    def process_u():
        for tid in uts:
            T:list=uts[tid]
            T.sort(key=lambda x:x[2]);T=np.array(T)
            T=t_spare(np.array(T),tim_int=time_int,dis_int=dis_int)
            j=0
            for i in range(1,len(T)):
                p,q=T[i-1],T[i]
                if q[2]-p[2]>time_span:
                    t=T[j:i];j=i
                    append_filter_len(t)
            append_filter_len(T[j:])
    assert name in traj_data_path
    for file in list_dir(traj_data_path[name])[3]:
        # lines=open(file,'r',encoding='utf-8').readlines()
        print('process',file)
        for line in read_lines_iter(file,block='100m'):
            line=line.removesuffix('\n').removesuffix('\t')
            if len(line)<5:continue
            uid,tid,tim,lon,lat=parse_line(line,',',str,str,int,float,float)
            uid=dicted_idx('uid_didi',uid)
            tid=dicted_idx('tid_didi',tid)
            if uid!=_uid:
                process_u()
                _uid=uid 
                uts={}
            if tid not in uts:uts[tid]=[]
            uts[tid].append([lon,lat,tim])
    process_u()
    assert len(TS)==len(UIDS)
    return np.array([np.array(TS,dtype=object),np.array(UIDS,dtype=int)],dtype=object)



_load_funs={
    'Porto':load_Porto,
    'Beijing':load_beijing,
    'Xian':load_xian,
}
def load(data):
    """TS,Uids"""
    redir=os.path.join(out_base, "TrajData")
    assert data in _load_funs
    fun=_load_funs[data]
    cache_name=f'load({"_".join([str(x) for x in fun.__defaults__])})'
    X= mcache(cache_name,redir=redir,ftype='npy.gz')
    if X is not None:
        if len(X)==1:X=X[0] 
        if len(X)==1:X=X[0] 
        if len(X)==1:X=X[0] 
        return X
    X=fun()
    if len(X)==1:X=X[0]
    mcache(cache_name,X,redir=redir,ftype='npy.gz')
    return X
traj_bbox={ 
    'Beijing':[[116.1994,116.5452],[39.7547,40.0244]],
    'Porto':[[41.086125252, 41.255937432],[-8.69043114799999, -8.51008941766], ],
    'Xian':[[108.9058,109.0049],[34.2060,34.2825]],
}
traj_tbox={
    'Porto':[1372636853, 1404172787],  
    'Beijing':[1235952614, 1237994859],
    'Xian':[1475251332, 1477929749],
}
@mcache(name='loadB',redir=os.path.join(out_base, "TrajData"),ftype='npy.gz')
def load_ts_box(data):
    """TS,UIDS,...,BBOX"""
    assert data in traj_bbox
    x=load(data)
    if len(x)==1:x=x[0]
    if len(x)==1:x=x[0]
    if len(x)==1:x=x[0]
    TS,others=x[0],x[1:]
    TS=ts_remove_bad_speed(TS) 
    if traj_bbox[data] is None: 
        TS,others,bbox=ts_shrink(TS,0.01,others=others)
    else:
        bbox=traj_bbox[data]
        TS,idx=ts_bound(TS,traj_bbox[data])
        others=[ ids_shrink(o[idx],str(i)) for i,o in enumerate(others)]
    return np.array([TS,*others,bbox],dtype=object)
def traj_bbox_tim(name):
    if name in traj_tbox:return traj_tbox[name]
    x=load_ts_box(name)
    if len(x)==1:x=x[0]
    if len(x)==1:x=x[0]
    if len(x)==1:x=x[0]
    TS=x[0]
    return ts_bbox_tim(TS)
