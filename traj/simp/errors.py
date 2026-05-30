exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import numpy as np
from numba import njit
@njit
def _sed_t2ps(T:np.ndarray,h:int,vec:np.ndarray):
    vec[h-1,0]=T[len(T)-1,0]
    vec[h-1,1]=T[len(T)-1,1]
    if h<=1:
        for i in range(h):
            vec[i,0]=T[0][0]
            vec[i,1]=T[0][1]
        return vec
    k=0
    for i in range(1,len(T)):
        p1,p2=T[i-1],T[i]
        s=int(p2[2])-int(p1[2])
        for j in range(s):
            vec[k,0] = (p2[0] - p1[0]) * (j / s) + p1[0]
            vec[k,1] = (p2[1] - p1[1]) * (j / s) + p1[1]
            k+=1
            if k==h:return vec
    return vec
def SED_fast(t1:np.ndarray,t2:np.ndarray):
    if len(t1)<=0 or len(t2)<=0: 
        return np.inf
    h=max(int(t1[-1][2]-t1[0][2]),int(t2[-1][2]-t2[0][2]))
    if h<=0:
        h=max(len(t1),len(t2))
    v1,v2=np.zeros((h,2),dtype=np.float32),np.zeros((h,2),np.float32)
    t1=_sed_t2ps(t1[:,:2],h,v1)
    t2=_sed_t2ps(t2[:,:2],h,v2)
    return np.linalg.norm(t1-t2,axis=1).max()