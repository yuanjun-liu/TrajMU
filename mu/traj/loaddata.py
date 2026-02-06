exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import numpy as np
from traj.data.load_trajs import load_ts_box as load_traj_raw
DuDrDvTypes=['Usr','Area']
traj_len_min=20
datasets=['Porto','Beijing'] 
''' split of data
train=du+dr, test+val=dv
[6 2 2] of [train test val], [5 1 4] of [dr du dv]
[8 1 1] of [train test val], [7 1 2] of [dr du dv]
'''
train_rate=0.6 
DuRates=[0.2,0.1] 
max_num_trajs=100_000 
def fix_trajs_num(dic,du_rate,slice_fn=None):
    """{val:[], test:[], train:[], du:[], dv:[], dr:[], ... }"""
    if slice_fn is None:
        slice_fn=lambda ts,num: ts[: min(len(ts),num)]
    for k in dic:
        if 'val' in k:     dic[k]=slice_fn(dic[k],int(max_num_trajs*0.2+0.5))
        elif 'test' in k:  dic[k]=slice_fn(dic[k],int(max_num_trajs*0.2+0.5))
        elif 'train' in k: dic[k]=slice_fn(dic[k],int(max_num_trajs*0.6+0.5))
        elif 'dv' in k:    dic[k]=slice_fn(dic[k],int(max_num_trajs*0.4+0.5))
        elif 'du' in k:    dic[k]=slice_fn(dic[k],int((max_num_trajs*0.6+0.5)*du_rate+0.5))
        elif 'dr' in k:    dic[k]=slice_fn(dic[k],int((max_num_trajs*0.6+0.5)*(1-du_rate)+0.5))
    return dic
def _split_tvt(N,train_rate,train_num=None):
    """train,val,test"""
    if train_num is None: train_num=int(N*train_rate+0.5)
    train_idx = np.random.choice(N, train_num, replace=False)
    mask = np.ones(N, dtype=bool)
    mask[train_idx] = False
    val_test_idx = np.flatnonzero(mask)
    if len(val_test_idx) > train_num:
        val_test_idx = val_test_idx[:train_num]
    n_val_test = len(val_test_idx)
    val_size = n_val_test // 2
    perm = np.random.permutation(n_val_test)
    val_idx = val_test_idx[perm[:val_size]]
    test_idx = val_test_idx[perm[val_size:]]
    return train_idx,val_idx,test_idx
def _split_cls(train_idx,cls_train,K,du_rate):
    """train_idx:|Nt|. cls_train:|Nt|,max=K """
    du_cls_num=int(K*du_rate+0.5) 
    du_cls_idx=np.random.choice(list(set(cls_train)),du_cls_num,False)
    mask = np.isin(cls_train, du_cls_idx)
    du_idx = train_idx[mask]
    dr_idx = train_idx[~mask]
    assert len(du_idx)>5
    return du_idx,dr_idx
def ts_split_area(ts,du_rate,train_rate,train_num=None):
    train_idx,val_idx,test_idx=_split_tvt(N=len(ts),train_rate=train_rate,train_num=train_num)
    anchor=np.concatenate(ts[train_idx[-100:]])[:,:2].mean().astype(float)
    ps=np.array([np.mean(t[:,:2],axis=0) for t in ts[train_idx]]).astype(float) 
    D=np.linalg.norm(ps-anchor,axis=1) 
    idx=train_idx[np.argsort(D)]
    du_num=int(len(ps)*du_rate+0.5)
    du_idx,dr_idx=idx[:du_num],idx[du_num:]
    np.random.shuffle(du_idx) 
    np.random.shuffle(dr_idx)
    assert len(du_idx) and len(dr_idx)
    dv_idx=np.concatenate([val_idx,test_idx])
    return train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx
def ts_split_usr(ts,uid,du_rate,train_rate,train_num=None):
    train_idx,val_idx,test_idx=_split_tvt(N=len(ts),train_rate=train_rate,train_num=train_num)
    du_idx,dr_idx=_split_cls(train_idx=train_idx,cls_train=uid[train_idx],K=len(set(uid[train_idx])),du_rate=du_rate)
    dv_idx=np.concatenate([val_idx,test_idx])
    assert len(du_idx)>5
    return train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx
def ts_split(ts,uid,urv,du_rate,train_num=None):
    """train, val, test, du, dr, dv"""
    urv=urv.lower()
    if urv =='usr':  tr,val,te,du,dr,dv= ts_split_usr(ts=ts,du_rate=du_rate,train_rate=train_rate,train_num=train_num,uid=uid)
    if urv =='area': tr,val,te,du,dr,dv=ts_split_area(ts=ts,du_rate=du_rate,train_rate=train_rate,train_num=train_num)
    assert len(du)>5
    return tr,val,te,du,dr,dv
augment_drop_rate = 0.1
augment_noise_size = 0.0002
def t_add_noise(t: np.ndarray):
    le = len(t)
    drop_num = int(max(0, le - traj_len_min) * augment_drop_rate)
    drop_idx = np.random.choice(le, drop_num, replace=False)
    preserve_mask = np.ones(le,dtype=bool)
    preserve_mask[drop_idx] = False  
    preserve_mask[0]=preserve_mask[-1]=True 
    t = t[preserve_mask]
    shift_xy = np.random.randn(len(t), 2) * augment_noise_size
    t[:, :2] += shift_xy
    return t
