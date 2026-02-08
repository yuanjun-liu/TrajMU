exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import random
import numpy as np
from typing import List
from _tool.mData import ids_shrink
from numba import njit

def t_spare(t:np.ndarray,tim_int=None,dis_int=None):
    """t is sorted"""
    if len(t)==0 or tim_int is None and dis_int is None:return t
    T=[t[0]]
    for p in t[1:]:
        if tim_int is not None and p[2]-T[-1][2]<tim_int:continue
        if dis_int is not None and np.linalg.norm(T[-1][:2]-p[:2])<dis_int:continue
        T.append(p)
    return np.array(T)
def ps_bbox(ps: np.ndarray) -> List[List[float]]:
    """[min_lat,max_lat],[min_lon,max_lon],[min_tim,max_tim],..."""
    if len(ps)==0:return None
    ps=np.array(ps)
    return [[ps[:, x].min(), ps[:, x].max()] for x in range(len(ps[0]))]
def ts_bbox(ts) -> List[List[float]]:
    """[min_x1,max_x1],[min_x2,max_x2],..."""
    res = None
    i=0
    while res is None:
        res = ps_bbox(ts[i])
        i+=1
    for t in ts[i:]:
        box = ps_bbox(t)
        if box is None:continue
        for i, b in enumerate(box):
            res[i][0] = min(res[i][0], b[0])
            res[i][1] = max(res[i][1], b[1])
    return res
def ts_bbox_tim(ts):
    tmin=tmax=ts[0][0][2]
    for T in ts:
        tmin=min(tmin,min(T[:,2]))
        if tmin < 1202493:raise RuntimeError()
        tmax=max(tmax,max(T[:,2]))
    return int(tmin),int(tmax)
def t_len(t: np.ndarray) -> float: return sum([np.linalg.norm(t[i, :2] - t[i + 1, :2]) for i in range(len(t) - 1)])
def t2ps_its(T: np.ndarray, h: int):
    """list of p:x,y,t"""
    l = t_len(T)
    vec = np.array([T[0]] * h)
    vec[-1] = T[-1]
    if l < 1e-12: return vec
    e = l / (h - 1)
    a, b = 0, 0
    j = 0
    for i in range(1, len(T)):
        p1, p2 = T[i - 1], T[i]
        s = np.linalg.norm(p1[:2] - p2[:2], 2)
        if s<1e-12:continue
        b += s
        while b >= a:
            if j >= h: break
            vec[j] = (p2 - p1) * (1 - (b - a) / s) + p1
            j += 1
            a += e
    return vec
def its(T1,T2,h):return np.linalg.norm(t2ps_its(T1,h)-t2ps_its(T2,h))/np.sqrt(h)
def t2ps_steplen(T: np.ndarray, e: float):
    """list of p:x,y,t"""
    l = t_len(T)
    vec = []
    if l < e: return np.array([T[0]])
    a, b = 0, 0
    j = 0
    for i in range(1, len(T)):
        p1, p2 = T[i - 1], T[i]
        s = np.linalg.norm(p1[:2] - p2[:2])
        if s<1e-12:continue
        b += s
        while b >= a:
            vec.append((p2 - p1) * (1 - (b - a) / s) + p1)
            j += 1
            a += e
    vec.append(T[-1])
    return np.array(vec)
@njit
def __t2ps_steplen(vec:np.ndarray,T:np.ndarray, e: float):
    """list of p:x,y,t"""
    a, b = 0, 0
    j = 0
    num=0
    for i in range(1, len(T)):
        x1,y1=T[i-1][0],T[i-1][1]
        x2,y2=T[i][0],T[i][1]
        s = ((x1-x2)**2+(y1-y2)**2)**0.5
        if s<1e-12:continue
        b += s
        while b >= a:
            vec[num][0]=(x2 - x1) * (1 - (b - a) / s) + x1
            vec[num][1]=(y2 - y1) * (1 - (b - a) / s) + y1
            if len(vec[0])==3:
                t1=T[i-1][2];t2=T[i][2]
                vec[num][2]=(t2 - t1) * (1 - (b - a) / s) + t1
            j += 1
            a += e
            num+=1
    return num
def t2ps_steplen_jit(T: np.ndarray, e: float):
    """list of p:x,y,t"""
    l = t_len(T)
    vec=np.zeros((int(l/e)+1,len(T[0])),dtype=np.float32)
    num=__t2ps_steplen(vec,T,e)
    return vec[:num]
def t2gs(T: np.ndarray, xa, xb, ya, yb, xh, yh):
    """:return [g:[gxi,gyi,tim], p1, p2]"""
    res = []
    dx, dy = (xb - xa) / xh, (yb - ya) / yh
    ps = t2ps_steplen(T, min(dx, dy)/3.0)  
    p2g = lambda p: (int((p[0] - xa) / dx), int((p[1] - ya) / dy),p[2])
    if len(ps.shape) == 1: return [[p2g(ps), ps, ps]]
    g, p = p2g(ps[0]), ps[0]
    for i in range(1, len(ps)):
        g2 = p2g(ps[i])
        if g2 != g:
            res.append([g, p, ps[i]])
            g, p = g2, ps[i]
    return res
def ts_len_info(trajs) -> dict:
    len_min, len_max = float('inf'), float('-inf')
    len_sum, len_avg = 0, 0
    num_traj = len(trajs)
    for T in trajs:
        l = len(T)
        len_min = min(l, len_min)
        len_max = max(l, len_max)
        len_sum += l
    len_avg = len_sum / num_traj
    return {"num": num_traj, "len_max": len_max, "len_min": len_min, "len_avg": int(len_avg)}
def ts_filter_len(trajs, len_min=30,uids=None):
    """return trajs, uids"""
    uid_none=uids is None
    if uid_none:
        uids=[0]*len(trajs)
    else:
        assert len(trajs)==len(uids)
    TS,UIDS=[],[]
    for t,u in zip(trajs,uids):
        if len(t)>len_min:
            TS.append(t)
            UIDS.append(u)
    TS,UIDS=np.array(TS,dtype=object),np.array(UIDS,dtype=int)
    if uid_none:return TS,None
    return TS,ids_shrink(UIDS)
def ts_bound(ts, xx_y_bound,tlen_min=1e-6,pnum_min=10, ):
    [xmin, xmax], [ymin, ymax] = xx_y_bound[:2]
    idx=[]
    TS = []
    for i,t in enumerate(ts):
        mask=(xmin<=t[:,0])*(t[:,0]<=xmax)*(ymin<=t[:,1])*(t[:,1]<=ymax)
        T=t[mask]
        if len(T) >= pnum_min and t_len(T)>tlen_min:
            TS.append(T)
            idx.append(i)
        ts[i]=None
    return np.array(TS, dtype=object),np.array(idx,dtype=int)
def t_has_bad_speed(T:np.ndarray,max_speed=50):
    """mark as bad if speed is more that 50m/s, i.e., 180km/h"""
    dis_adj=np.linalg.norm(T[1:,:2]-T[:-1,:2],axis=1)
    tim_adj=T[1:,2]-T[:-1,2]
    speed=dis_adj/tim_adj
    if any(np.isnan(speed)) or any(np.isinf(speed)) or any(speed>max_speed):
        return True
    return False
def ts_remove_bad_speed(TS):
    res=[]
    for T in TS:
        if t_has_bad_speed(T):
            continue
        res.append(T)
    return np.array(res,dtype=object)
def ts_shrink(TS, del_rate,others=[]):
    """:return ts, uids, bbox"""
    assert 0 < del_rate < 1
    x = ts_bbox(TS)
    [xa, xb], [ya, yb]=x[0],x[1]
    h = 10000
    dx, dy = (xb - xa) / h, (yb - ya) / h
    xns = np.zeros(h + 1, dtype=int)
    yns = np.zeros(h + 1, dtype=int)
    p2i = lambda p: [int((p[0] - xa) / dx), int((p[1] - ya) / dy)]
    for t in TS:
        for p in t:
            xi, yi = p2i(p)
            xns[xi] += 1
            yns[yi] += 1
    num = sum(xns)
    for i in range(h):
        if sum(xns[:i]) / num <= del_rate:
            x_min = xa + i * dx
        if sum(xns[i:]) / num >= del_rate:
            x_max = xa + i * dx
        if sum(yns[:i]) / num <= del_rate:
            y_min = ya + i * dy
        if sum(yns[i:]) / num >= del_rate:
            y_max = ya + i * dy
    bbox = [[x_min, x_max], [y_min, y_max]]
    ts,idx=ts_bound(TS, bbox) 
    others=[ ids_shrink(o[idx],str(i)) for i,o in enumerate(others)]
    return ts,others, bbox
def reverse(T):
    dt=[T[i+1][2]-T[i][2] for i in range(len(T)-1)]
    dt=dt[::-1]
    t0=T[0][2]
    T=T[::-1]
    T[0][2]=t0
    for i in range(1,len(T)):
        T[i][2]=T[i-1][2]+dt[i-1]
    return T
