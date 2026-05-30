exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from mu.MU import *
from mu.unlearn.BadT import BadT
from mu.unlearn.Retrain import Retrain
from mu.unlearn.FineTune import FineTune
from mu.unlearn.TopK import TopK
from mu.unlearn.RandomK import RandomK
from mu.unlearn.SFRon import SFRon
from mu.unlearn.NegGrad import NegGrad
from mu.unlearn.SCRUB import SCRUB
from mu.unlearn.GDRGMA import GDRGMA
from mu.unlearn.SSD import SSD

class Origin(MU):
    """no unlearn"""
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        return self.model_origin()

mu_methods={
    'Retrain':Retrain,
    'FineTune':FineTune,
    'TopK':TopK,
    'RandomK':RandomK,
    'SFRon':SFRon,
    'BadT':BadT,
    'NegGrad':NegGrad,
    'SCRUB':SCRUB,
    'GDRGMA':GDRGMA,
    'SSD':SSD,
}