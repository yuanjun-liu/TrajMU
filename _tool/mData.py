import numpy as np
import sys
import _pickle
import zlib
import random
import time
import copy as copy
def to_item_np(x):
    if isinstance(x,list) or isinstance(x,np.ndarray):return [to_item_np(i) for i in x]
    if isinstance(x,tuple):return (to_item_np(i) for i in x)
    if isinstance(x,dict):return {k:to_item_np(x[k]) for k in x}
    try: return x.item()
    except:return x
def random_seed(seed=None):
    if seed is None:seed=time.time_ns()
    random.seed(seed)
    np.random.seed(seed)
def deepcopy(x):return copy.deepcopy(x)
alpha = 'abcdefghijklmnopqrstuvwxyz'
ALPHA=alpha.upper()
__dicted_idx ={}
__global_dict={}
def dicted_idx(type, x=None):
    """
    dicted_idx('type','a') -> 0
    dicted_idx('type','b') -> 1
    dicted_idx('type') -> ['a','b']
    """
    if type not in __dicted_idx: __dicted_idx[type] = {}
    if x is None: return list(__dicted_idx[type].keys())
    if x in __dicted_idx[type]: return __dicted_idx[type][x]
    i=len(__dicted_idx[type])
    __dicted_idx[type][x]=i
    return
is_debug = True if sys.gettrace() else False
def isviadecorator():
    """
    @
    __spec__._initializingimportTrue
    :return:
    """
    import inspect
    for fram in inspect.stack():
        if fram.code_context is not None:
            try:
                if fram.code_context[0].lstrip(' ')[0] == '@':
                    return True
            except Exception as e:
                print('error mData.isviadecorator', e)
    return False
def str2int_float_str(s):
    if isinstance(s, int) or isinstance(s, float): return s
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s
def serialize(data): return _pickle.dumps(data)
def deserialize(data): return _pickle.loads(data)
def int2b(x, len=None):
    len = int(np.ceil(np.log(x) / np.log(255))) if len is None else len
    return int(x).to_bytes(length=len, byteorder='big', signed=True)
def b2int(x): return int().from_bytes(x, byteorder='big', signed=True)
def str_zip(s: str): return zlib.compress(str.encode(s), zlib.Z_BEST_COMPRESSION)
def str_unzip(x): return zlib.decompress(x).decode('utf-8')
def zhifang(x):
    a, b = int(min(x)), int(max(x))
    zf = np.zeros(b - a + 1)
    for i in x:
        zf[int(i - a)] += 1
    return zf
def hashable(x):
    try:
        hash(x)
        return True
    except:
        return False
def ids_shrink(uids:np.ndarray,name='sk_uid'):
    res=[]
    for i in uids:
        j=dicted_idx(name,int(i))
        res.append(j)
    return np.array(res,dtype=int)
    vmax=max(uids)+1
    vc=np.zeros(vmax,dtype=int)
    for v in uids:
        vc[v]=1
    j=0
    for v in range(vmax):
        if vc[v]==1:
            vc[v]=j
            j+=1
    res=np.zeros(len(uids),dtype=int)
    for i,v in enumerate(uids):
        res[i]=vc[v]
    return res
if __name__ == '__main__':
    pass
