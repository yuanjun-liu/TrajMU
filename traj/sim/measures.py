import numpy as np
exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from traj.data.process_ts import t2ps_its
dis_eucid = lambda x, y: np.linalg.norm(x - y)
def ITS(T1: np.ndarray, T2: np.ndarray, h):
    """dis"""
    return np.linalg.norm(t2ps_its(T1, h)- t2ps_its(T2, h)) / np.sqrt(h)
