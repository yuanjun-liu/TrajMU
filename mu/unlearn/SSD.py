exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from mu.MU import *
from _nn.nBasic import basic_train
import torch,math
from torch import nn
from torch.nn.modules import Module
from torch.utils.data.dataloader import DataLoader
from typing import Dict,List
import torch.nn.functional as F
import numpy as np
from _tool.mList import topk
from _tool.mData import deepcopy
from _nn.nBasic import to_device
class SSD(MU):
    """AAAI24 no train"""
    def __init__(self, **kw):
        super().__init__(**kw)
        parameters = {
                "lower_bound": 1,
                "exponent": 1,
                "magnitude_diff": None,
                "min_layer": -1,
                "max_layer": -1,
                "forget_threshold": 1,
                "dampening_constant": 1,
                "selection_weighting": 10,
        }
        self.__dict__.update(parameters)
    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        importances=zeors_para_dict(self.model)
        opt,sch=self.model.call_opt_sch() 
        for x in dataloader:
            x=to_device(x,self.device)
            opt.zero_grad()
            out = self.model(x)
            loss = self.model.lossF(out,x)
            loss=loss_reduce(loss)
            loss.backward()
            modelps=self.model.named_parameters()
            for k,p in modelps:
                if k not in importances:continue
                imp=importances[k]
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances
    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.
        Returns:
        None
        """
        with torch.no_grad():
            for k,p in self.model.named_parameters():
                if k not in original_importance:continue
                if k not in forget_importance:continue
                oimp,fimp=original_importance[k],forget_importance[k]
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm,)
                if locations[0].numel() == 0:continue
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)
    def _unlearn(self,*arg,ptloss=False,estop_fn=None,**kw):
        imp_ds=self.calc_importance(DataLoader(self.dtrain,self.bs,num_workers=num_workers,collate_fn=self.model.get_collate_fn()))
        imp_du=self.calc_importance(DataLoader(self.du,self.bs,num_workers=num_workers,collate_fn=self.model.get_collate_fn()))
        self.modify_weight(imp_ds,imp_du)
        return self.model