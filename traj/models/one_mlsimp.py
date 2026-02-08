exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import numpy as np
import datetime
import torch,math,os,tqdm
import torch.nn as nn
import networkx as nx
from torch.utils.data import Dataset
from _nn.nData import random_seed,auto_device
from _nn.nBasic import to_device
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as pygData
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import average_precision_score, roc_auc_score
from _tool.mFile import is_linux
from sklearn.utils import shuffle
from itertools import zip_longest
from _tool.mIO import loadZ_pk,saveZ_pk,loadZ_th,saveZ_th,save_th,load_th
import time
import enum
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader
from itertools import tee
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from abc import ABC, abstractmethod
from torch.optim import AdamW
import copy
from typing import Tuple
from rtree import index as rtreeIndex
import shutil
from collections import deque, defaultdict
import pickle,numba
from traj.simp.errors import SED_fast
device=auto_device()
random_seed(42)
num_workers=0 
max_length=500 ; overlap=100 ; dim=128
class PositionalEncodingDiff(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len + 1, 1, d_model)
        pe[1:, 0, 0::2] = torch.sin(position * div_term)
        pe[1:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        pe = self.pe[:x.shape[0], 0, :] * 1e-2
        x += pe.unsqueeze(1)
        return self.dropout(x).permute(1, 0, 2)
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr.astype('float32')).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__( self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False, training_mode='e2e',):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.training_mode = training_mode
        print('training mode is ', training_mode)
        self.mapping_func = None
    def training_losses(self, model, *args, **kwargs):
        if self.training_mode == 'e2e':
            return self.training_losses_e2e(model, *args, **kwargs)
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def p_mean_variance(
        self, model, x, mask, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        model_output = model.model(x, mask, self._scale_timesteps(t))
        model_variance, model_log_variance = (self.posterior_variance, self.posterior_log_variance_clipped)
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        pred_xstart = model_output
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return {"mean": model_mean,"variance": model_variance,"log_variance": model_log_variance,"pred_xstart": pred_xstart,"pred_xstart_mean": model_output,}
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    def p_sample(
        self, model, x, mask, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(model,x,mask,t,clip_denoised=clip_denoised,denoised_fn=denoised_fn,model_kwargs=model_kwargs,)
        if top_p is not None and top_p > 0:
            noise =torch.randn_like(x)
            replace_mask =torch.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] =torch.randn_like(noise[replace_mask])
                replace_mask =torch.abs(noise) > top_p
            assert (torch.abs(noise) <= top_p).all()
        else:
            noise =torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  
        sample = out["mean"] + nonzero_mask *torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'greedy_mean':out["mean"], 'out':out, 'pred_xstart_mean': out["pred_xstart_mean"]}
    def p_sample_loop(self,model,noise=None,clip_denoised=True,denoised_fn=None,model_kwargs=None,device=None,progress=False,top_p=None,):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        pred_xstarts = []
        final = None
        for sample in (self.p_sample_loop_progressive( model, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs, device=device, progress=progress, top_p=top_p,
        )): final = sample
        return final
    def p_sample_loop_progressive( self, model, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None, progress=False, top_p=None, ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        diffusion_steps = model_kwargs['diffusion_steps']
        batch = model_kwargs['batch']
        amplify_len = model_kwargs['amplify_len']
        trajs_padding, padding_mask, simp_trajs_padding, simp_padding_mask, labels, labels_mask = to_device(batch)
        x_start_mean = model.model.get_embeds(trajs_padding, padding_mask)
        simp_padding_mask = torch.zeros(padding_mask.size(0),amplify_len).to(device)
        pad_mask = torch.cat([padding_mask,simp_padding_mask],dim=-1)
        shape = (x_start_mean.shape[0], x_start_mean.shape[1] + amplify_len, x_start_mean.shape[2])
        input = torch.randn(*shape).to(device)
        input[:, :trajs_padding.shape[1], :] = x_start_mean
        indices = list(range(diffusion_steps))[::-1]
        for i in indices:
            t =torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample( model, input, pad_mask, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, top_p=top_p,)
                x_gen = out['sample']
                x_gen[:,:trajs_padding.shape[1],:] = x_start_mean
                out['sample'] = x_gen
                yield out
    def get_x_start(self, x_start_mean, std):
        '''
        Using the interpolating policy OR using the convolution policy...
        :param x_start_mean:
        :return:
        '''
        noise =torch.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return ( x_start_mean + std * noise)
    def token_discrete_loss(self, x_t, doc_emb, doc_mask, get_logits, input_ids, model_kwargs=None):
        if self.model_arch == 'conv-unet' or  self.model_arch == '1d-unet':
            reshaped_x_t = x_t.view(x_t.size(0), x_t.size(1), -1).permute(0, 2, 1)
        else:
            reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t, doc_emb, doc_mask) 
        if model_kwargs['output_sen']:
            tokenizer = model_kwargs['tokenizer']
            t0_mask = model_kwargs['t0_mask']
            cands =torch.topk(torch.softmax(logits, dim=-1), k=1, dim=-1)
            for i, seq in enumerate(cands.indices):
                if isinstance(tokenizer, dict):
                    if t0_mask[i]:
                        tokens = " ".join([tokenizer[x[0].item()] for x in seq if x[0].item() > 3])
                else:
                    tokens = tokenizer.decode(seq.squeeze(-1))
        loss_fct =torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.shape[-1]), input_ids.view(-1)).view(logits.shape[0], -1)
        decoder_nll = decoder_nll.mean(dim=-1)
        return decoder_nll
    def x0_helper(self, model_output, x, t):
        pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
        pred_prev = model_output
        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}
    def training_losses_e2e(self, model, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        trajs_padding = model_kwargs['trajs_padding']
        padding_mask = model_kwargs['padding_mask']
        simp_trajs_padding = model_kwargs['simp_trajs_padding']
        simp_padding_mask = model_kwargs['simp_padding_mask']
        labels = model_kwargs['labels']
        labels_mask = model_kwargs['labels_mask']
        pad_mask = torch.cat([padding_mask,simp_padding_mask],dim=-1)
        x_start_mean = torch.cat([trajs_padding, simp_trajs_padding], dim=1)
        x_start_mean = model.model.get_embeds(x_start_mean,pad_mask)
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                  torch.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        x_start = self.get_x_start(x_start_mean, std)
        if noise is None:
            noise =torch.randn_like(x_start)  
        x_t = self.q_sample(x_start, t, noise=noise)  
        x_t[:, :trajs_padding.shape[1], :] = x_start_mean[:, :trajs_padding.shape[1], :]
        model_output = model.model(x_t, pad_mask, self._scale_timesteps(t))
        terms = {}
        target = x_start
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((x_start_mean[:, trajs_padding.shape[1]:, : ] * 10 - model_output[:, trajs_padding.shape[1]:, : ] * 10) ** 2).mean()
        decoder_logits = torch.bmm(x_start_mean[:, trajs_padding.shape[1]:, : ], x_start_mean[:, :trajs_padding.shape[1], : ].permute(0, 2, 1))
        ce_loss =torch.nn.CrossEntropyLoss()
        terms['decoder_nll'] = ce_loss(decoder_logits.view(-1, decoder_logits.shape[-1]),labels.view(-1))
        out_mean, _, _ = self.q_mean_variance(x_start,torch.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        terms['tT_loss'] = mean_flat(out_mean ** 2).mean()
        terms["loss"] = terms["mse"] + terms['tT_loss'] +0.1*terms['decoder_nll']
        return terms
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)
class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps
    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = [] 
        self.original_num_steps = len(kwargs["betas"])
        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)
    def p_mean_variance(
        self, model, *args, **kwargs
    ):  
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)
    def training_losses(
        self, model, *args, **kwargs
    ):  
        return super().training_losses(self._wrap_model(model), *args, **kwargs)
    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )
    def p_sample_loop(self, model,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None):
        return super(SpacedDiffusion, self).p_sample_loop(self._wrap_model(model),
        noise=noise,
        clip_denoised=clip_denoised,
        denoised_fn=denoised_fn,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
        top_p=top_p)
    def _scale_timesteps(self, t):
        return t
def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
class LossType(enum.Enum):
    MSE = enum.auto()  
    RESCALED_MSE = (
        enum.auto()
    )  
    KL = enum.auto()  
    RESCALED_KL = enum.auto()  
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()
    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL
class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()  
    START_X = enum.auto()  
    EPSILON = enum.auto()  
class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))
def lonlat2meters_np(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = np.sin(north)
    return semimajoraxis * east, 3189068.5 * np.log((1 + t) / (1 - t))
def points2meter(points):
    rtn = []
    for p in points:
        lon_meter, lat_meter = lonlat2meters(lon=p[1], lat=p[0])
        rtn.append([lat_meter, lon_meter, p[2]])
    return rtn
def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().sum(dim=1,keepdims=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X
class Grid(object):
    def __init__(self,bbox,gsize, ts, minfreq=500, max_grid_size=10000, k=1, grid_start=6):
        [self.minlon,self.maxlon] ,[self.minlat,self.maxlat] = bbox[:2]
        self.xstep =self.ystep = gsize
        self.minx, self.miny = lonlat2meters(self.minlon, self.minlat)
        self.maxx, self.maxy = lonlat2meters(self.maxlon, self.maxlat)
        numx = round(self.maxx - self.minx, 6) / gsize
        self.numx = int(math.ceil(numx))
        numy = round(self.maxy - self.miny, 6) / gsize
        self.numy = int(math.ceil(numy))
        grid_size, gridmap = self.get_gridmap(minfreq,max_grid_size,k,grid_start,ts)
        self.grid_size = grid_size
        self.gridmap = gridmap
    def get_gridmap(self, minfreq, max_grid_size, k, grid_start,ts):
        grids = dict()
        hotgrid = []
        num_out_range= 0
        for t in ts:
            linecnt =0
            for p in t:
                linecnt+=1
                lon,lat=p[:2]
                if self.out_of_range(lon,lat):
                    num_out_range  += 1
                else:
                    grid = self.gps2grid(lon,lat)
                    if grid in grids.keys():
                        grids[grid]+=1
                    else:
                        grids[grid]=1
                if linecnt>1000:
                    break
        max_grid_size = min(max_grid_size, len(grids.keys()))
        cnt = 0
        grids=sorted(grids.items(), key=lambda d: d[1],reverse=True)
        for grid in grids:
            grid_idx=grid[0]
            freq = grid[1]
            if cnt >=max_grid_size:
                break
            elif freq> minfreq:
                cnt +=1
                hotgrid.append(grid_idx)
        hotgrid2idx = dict([(grid, i - 1 + grid_start)  for (i, grid) in enumerate(hotgrid)])
        grid_size = grid_start+len(hotgrid)
        data = np.zeros([len(hotgrid), 2])
        i = 0
        for grid_ in hotgrid:
            x, y = self.grid2coord(grid_)
            data[i, :] = [x, y]
            i += 1
        hotgrid_kdtree = KDTree(data)
        def knearestHotgrids(grid, k):
            coord = self.grid2coord(grid)
            dists, idxs = hotgrid_kdtree.query(np.array([[coord[0], coord[1]]]), k)
            res = hotgrid[idxs[0].tolist()[0]]
            return res, dists
        def nearestHotgrid(grid):
            hotgrid, _ = knearestHotgrids(grid, 1)
            return hotgrid
        gridmap=[]
        for i in range(self.numx*self.numy):
            if i in hotgrid:
                gridmap.append(hotgrid2idx[i])
            else:
                i_hotgrid = nearestHotgrid(i)
                gridmap.append(hotgrid2idx[i_hotgrid])
        return grid_size, gridmap
    def out_of_range(self,lon, lat):
        return not (self.minlon <= lon < self.maxlon and self.minlat <= lat < self.maxlat)
    def coord2grid(self,x,y):
        xoffset = round(x - self.minx, 6) / self.xstep
        yoffset = round(y - self.miny, 6) / self.ystep
        xoffset = int(math.floor(xoffset))
        yoffset = int(math.floor(yoffset))
        return yoffset * self.numx + xoffset
    def gps2grid(self,lon,lat):
        x,y = lonlat2meters(lon,lat)
        return self.coord2grid(x,y)
    def grid2coord(self,grid):
        yoffset = grid // self.numx
        xoffset = grid % self.numx
        y = self.miny + (yoffset + 0.5) * self.ystep
        x = self.minx + (xoffset + 0.5) * self.xstep
        return x, y
    def gps2idx(self,lon, lat):
        if self.out_of_range(lon, lat):
            return "UNK"
        return self.grid2idx(self.gps2grid(lon, lat))
    def grid2idx(self, grid):
        return self.gridmap[grid]
    def traj2idxseq(self,traj:np.array):
        x, y = lonlat2meters_np(traj[:,:,0], traj[:,:,1])
        xoffset = np.round(x - self.minx, 6) / self.xstep
        yoffset = np.round(y - self.miny, 6) / self.ystep
        xoffset = xoffset.astype(int)
        yoffset = yoffset.astype(int)
        grids = yoffset * self.numx + xoffset
        grid_id = np.array(self.gridmap)
        return grid_id[grids]
def graph_constructor(grid:Grid,ts):
    G =nx.Graph()
    grid_size = grid.grid_size
    G.add_nodes_from(range(grid_size))
    for t in ts:
        pre=None
        linecnt=0
        for p in t:
            linecnt +=1
            lon,lat=p[:2]
            gridID = grid.gps2idx(lon, lat)
            if gridID != 'UNK':
                if pre != None:
                    if G.has_edge(pre, gridID):
                        G[pre][gridID]['weight'] += 1
                    else:
                        G.add_weighted_edges_from([(pre, gridID, 1)])
                else:
                    pre = gridID
            if linecnt>1000:
                break
    return G
def load_data_from_G(G):
    features = torch.eye(G.number_of_nodes())
    adj = nx.adjacency_matrix(G)
    sparse_mx = adj.tocoo().astype(np.float32)
    edge_index = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    data= pygData(edge_index=edge_index,edge_attr=values)
    return data
def node2vec_pretrain(G,embedding_dim,walk_length,context_size,walks_per_node,gridemb_lr,gridemb_epochs):
    data = load_data_from_G(G)
    num_nodes = G.number_of_nodes()
    edge_index = data.edge_index
    data =   train_test_split_edges(data)
    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_nodes = num_nodes
    ).to(device)
    max_test_ap=0
    patient = 200
    cnt = 0
    last_loss = 0
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=gridemb_lr)
    for epoch in range(gridemb_epochs):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(loader)
        model.eval()
        with torch.no_grad():
            z = model()
            pos_y = z.new_ones(data.test_pos_edge_index.size(1))
            neg_y = z.new_zeros(data.test_neg_edge_index.size(1))
            y = torch.cat([pos_y, neg_y],dim=0)
            pos_pred = torch.sigmoid((z[data.test_pos_edge_index[0]] * z[data.test_pos_edge_index[1]]).sum(dim=1))
            neg_pred = torch.sigmoid((z[data.test_neg_edge_index[0]] * z[data.test_neg_edge_index[1]]).sum(dim=1))
            pred = torch.cat([pos_pred, neg_pred], dim=0)
            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
            roc = roc_auc_score(y, pred)
            ap = average_precision_score(y, pred)
            if ap > max_test_ap:
                max_test_ap = ap
            if abs(loss  - last_loss) <1e-3 :
                cnt = cnt+1
                last_loss = loss
                if cnt == patient:
                    break
            else:
                cnt = 0
                last_loss = loss
            print (f'patient:{cnt}')
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Roc: {roc:.4f}, Ap: {ap:.4f}, Max_ap:{max_test_ap:.4f}')
    emb = z.detach().cpu()
    return emb
class BertDataset(Dataset):
    def __init__(self,data,grid:Grid):
        data_lonlat = np.array(data)[:,:,:2]
        data_time = np.array(data)[:,:,2]
        self.data_time = data_time
        self.data = grid.traj2idxseq(data_lonlat)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        seg = []
        segvalid = []
        data_index = []
        seg_time = []
        start = 0
        end=-1
        data= self.data[index]
        time = self.data_time[index]
        while end!=len(data):
            end = min(start + max_length, len(data))
            if start == 0:
                valid_start = 0
            else:
                valid_start = overlap
            if end == len(data):
                valid_end = end - start
            else:
                valid_end = end - overlap
            segvalid.append([valid_start,valid_end])
            seg.append(data[start:end])
            seg_time.append(time[start:end])
            data_index.append(index)
            start += max_length - 2 * overlap
        return seg,segvalid,seg_time, data_index
class TBERTDataset:
    def __init__(self, bertdataset):
        self.loc_index = []
        self.ts = []
        for i in range(len(bertdataset)):
            [seg,_,seg_time,_]= bertdataset[i]
            self.loc_index+=seg
            self.ts+=seg_time
    def gen_sequence(self,min_len=0,select_days=None, include_delta=False):
        seq_set = []
        for i in range(len(self.loc_index)):
            one_set = [self.loc_index[i], self.ts[i], len(self.ts[i])]
            seq_set.append(one_set)
        return seq_set
class TemporalEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)
    def forward(self, x, **kwargs):
        timestamp = kwargs['timestamp']  
        time_encode = timestamp.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        time_encode = torch.cos(time_encode)
        return self.div_term * time_encode
class MaskedLM(nn.Module):
    def __init__(self, input_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(input_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size
    def forward(self, x, **kwargs):
        """
        :param x: input sequence (batch, seq_len, embed_size).
        :param origin_tokens: original tokens, shape (batch, seq_len)
        :return: the loss value of MLM objective.
        """
        origin_tokens = kwargs['origin_tokens']
        origin_tokens = origin_tokens.reshape(-1)
        lm_pre = self.linear(self.dropout(x))  
        lm_pre = lm_pre.reshape(-1, self.vocab_size)  
        return self.loss_func(lm_pre, origin_tokens)
class TBERTEmbedding(nn.Module):
    def __init__(self, encoding_layer, embed_size, num_vocab,grid_emb_matrix=None):
        super().__init__()
        self.embed_size = embed_size
        self.num_vocab = num_vocab
        self.encoding_layer = encoding_layer
        self.add_module('encoding', self.encoding_layer)
        self.token_embed = nn.Embedding(num_vocab, embed_size, padding_idx=0)
        if grid_emb_matrix!=None:
            self.token_embed.weight = nn.Parameter(grid_emb_matrix)
    def forward(self, x, **kwargs):
        token_embed = self.token_embed(x)
        pos_embed = self.encoding_layer(x, **kwargs)
        return token_embed + pos_embed
class TBERT(nn.Module):
    def __init__(self, embed, hidden_size, num_layers, num_heads,  detach=True):
        super().__init__()
        self.embed_size = embed.embed_size
        self.num_vocab = embed.num_vocab
        self.embed = embed
        self.add_module('embed', embed)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(self.embed_size, eps=1e-6))
        self.detach = detach
    def forward(self, x, **kwargs):
        """
        @param x: sequence of tokens, shape (batch, seq_len).
        """
        src_key_padding_mask = (x == 0)
        token_embed = self.embed(x, **kwargs)  
        src_mask = None
        encoder_out = self.encoder(token_embed.transpose(0, 1), mask=src_mask, src_key_padding_mask=src_key_padding_mask).transpose(0, 1)  
        if self.detach: encoder_out = encoder_out.detach()
        return encoder_out
    def static_embed(self):
        return self.embed.token_embed.weight[:self.num_vocab].detach().cpu().numpy()
def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]
def gen_random_mask(src_valid_lens, src_len, mask_prop):
    """
    @param src_valid_lens: valid length of sequence, shape (batch_size)
    """
    index_list = []
    for batch, l in enumerate(src_valid_lens):
        mask_count = torch.ceil(mask_prop * l).int()
        masked_index = torch.randperm(l)[:mask_count]
        masked_index += src_len * batch
        index_list.append(masked_index)
    return torch.cat(index_list).long().to(src_valid_lens.device)
def train_tbert(dataset, tbert_model:TBERT, obj_models, mask_prop, num_epoch, batch_size):
    tbert_model = tbert_model.to(device)
    obj_models = obj_models.to(device)
    src_tokens, src_ts, src_lens = zip(*dataset.gen_sequence(select_days=0))
    optimizer = torch.optim.Adam(list(tbert_model.parameters()) + list(obj_models.parameters()), lr=1e-4)
    cnt = 0
    for epoch in range(num_epoch):
        for batch in next_batch(shuffle(list(zip(src_tokens, src_ts, src_lens))), batch_size=batch_size):
            src_batch, src_t_batch, src_len_batch = zip(*batch)
            src_batch = np.transpose(np.array(list(zip_longest(*src_batch, fillvalue=0))))
            src_t_batch = np.transpose(np.array(list(zip_longest(*src_t_batch, fillvalue=0))))
            src_batch = torch.tensor(src_batch).long().to(device)
            src_t_batch = torch.tensor(src_t_batch).float().to(device)
            hour_batch = (src_t_batch % (24 * 60 * 60) / 60 / 60).long()
            batch_len, src_len = src_batch.size(0), src_batch.size(1)
            src_valid_len = torch.tensor(src_len_batch).long().to(device)
            mask_index = gen_random_mask(src_valid_len, src_len, mask_prop=mask_prop)
            src_batch = src_batch.reshape(-1)
            hour_batch = hour_batch.reshape(-1)
            origin_tokens = src_batch[mask_index]  
            origin_hour = hour_batch[mask_index]
            masked_tokens = src_batch.index_fill(0, mask_index, 1).reshape(batch_len, -1)  
            tbert_out = tbert_model(masked_tokens, timestamp=src_t_batch)  
            masked_out = tbert_out.reshape(-1, tbert_model.embed_size)[mask_index]  
            loss = 0.
            for obj_model in obj_models:
                loss += obj_model(masked_out, origin_tokens=origin_tokens, origin_hour=origin_hour)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'T-Bert training loss:{loss.item()}')
            cnt+=1
    return tbert_model
def _tbert_ini(ts_train,bbox,gsize):
    grid=Grid(bbox=bbox,gsize=gsize,ts=ts_train,minfreq=50,max_grid_size=20000,k=1,grid_start=4)
    G=graph_constructor(grid,ts_train)
    grid_ebd=node2vec_pretrain(G=G,embedding_dim=128,walk_length=80,context_size=10,walks_per_node=10,gridemb_lr=0.01,gridemb_epochs=1000)
    return grid,G,grid_ebd
def _tbert_pretrain(ts_train,grid,grid_ebd,bs=64):
    pretrain_dataset = TBERTDataset(BertDataset(ts_train,grid))
    encoding_layer = TemporalEncoding(dim)
    obj_models = nn.ModuleList([MaskedLM(dim, grid.grid_size)])
    tbert_embedding = TBERTEmbedding(encoding_layer, dim, grid.grid_size, grid_ebd)
    tbert_model = TBERT(tbert_embedding, dim*4, num_layers=4, num_heads=8, detach=False)
    embed_layer = train_tbert(pretrain_dataset, tbert_model, obj_models, mask_prop=0.2, num_epoch=2000, batch_size=64)
    return embed_layer,obj_models
def api_tbert_pretrain(root_data,ts_train=None,bbox=None,gsize=100,bs=64): 
    """grid,G,grid_ebd,tbert_model,obj_model"""
    path1=os.path.join(root_data,f'pre-grid.pk.zst')
    path2=os.path.join(root_data,f'pre-tbert.th.zst')
    if os.path.exists(path1) and os.path.exists(path2):
        grid,G,grid_ebd= loadZ_pk(path1)
        param_tbert_model,param_obj_model= loadZ_th(path2)
        encoding_layer = TemporalEncoding(dim)
        obj_models = nn.ModuleList([MaskedLM(dim, grid.grid_size)])
        tbert_embedding = TBERTEmbedding(encoding_layer, dim, grid.grid_size, grid_ebd)
        tbert_model = TBERT(tbert_embedding, dim*4, num_layers=4, num_heads=8, detach=False)
        tbert_model.load_state_dict(param_tbert_model) ; obj_models.load_state_dict(param_obj_model)
        return grid,G,grid_ebd,tbert_model,obj_models
    if os.path.exists(path1):
        grid,G,grid_ebd=loadZ_pk(path1)
    else:
        grid,G,grid_ebd=_tbert_ini(ts_train=ts_train,bbox=bbox,gsize=gsize) 
        saveZ_pk(path1,[grid,G,grid_ebd])
    tbert_model,obj_model=_tbert_pretrain(ts_train=ts_train,grid=grid,grid_ebd=grid_ebd,bs=bs)
    saveZ_th(path2,[tbert_model.state_dict(),obj_model.state_dict()])
    return grid,G,grid_ebd,tbert_model,obj_model
class GraphSimpDataset(Dataset): 
    def __init__(self, ts, grid:Grid,tbert_pretrained:TBERT,idx=None):
        self.device = device
        data_lonlat = np.array(ts)[:, :, :2]
        data_time = np.array(ts)[:, :, 2]
        self.data_time = data_time
        self.data = grid.traj2idxseq(data_lonlat)
        self.max_length = max_length
        self.overlap =  overlap
        self.pretrain(tbert_pretrained)
        del self.data, self.data_time, self.max_length, self.overlap
        self.idx=np.arange(len(self)) 
        self.n1=torch.zeros(1,device=device)-1
    def __len__(self):
        return len(self.trajs_feature)
    def pretrain(self,tbert_model:TBERT):
        embed_size = dim
        tbert_model = tbert_model.to(device)
        self.trajs_feature = []
        self.trajs_edge_index = []
        self.trajs_point_node_index = []
        self.trajs_seg_node_index = []
        self.trajs_emb = []
        self.trajs_neighbor=[]
        self.amplify_labels = None
        self.pretrain_time = 0
        for i in range(len(self.data)):
            point_emb_list = []
            seg_emb_list = []
            [segs, segs_valid, segs_time, _] = self.segment(i)
            segs = np.transpose(np.array(list(zip_longest(*segs, fillvalue=0))))
            segs_time = np.transpose(np.array(list(zip_longest(*segs_time, fillvalue=0))))
            segs = torch.tensor(segs).long().to(device)
            segs_time = torch.tensor(segs_time).float().to(device)
            start_time = time.time()
            tbert_out = tbert_model(segs, timestamp=segs_time)
            end_time = time.time()
            self.pretrain_time += (end_time - start_time)
            for j in range(segs.size(0)):
                point_emb = tbert_out[j][segs_valid[j][0]:segs_valid[j][1]]
                seg_emb = torch.mean(point_emb, dim=0)
                point_emb_list.append(point_emb)
                seg_emb_list.append(seg_emb)
            traj_point_emb = torch.cat(point_emb_list, dim=0)
            traj_seg_emb = torch.cat(seg_emb_list, dim=0).view(len(seg_emb_list),embed_size)
            traj_emb = torch.mean(traj_point_emb,dim=0)
            sim = F.cosine_similarity(traj_point_emb.unsqueeze(1), traj_point_emb.unsqueeze(0), dim=2)
            value,indices = torch.topk(sim,11,dim=-1)
            feature = torch.cat((traj_point_emb, traj_seg_emb), dim=0)
            point_node_index = torch.tensor(range(traj_point_emb.size(0))).long().to(device)
            seg_node_index = torch.tensor(range(traj_point_emb.size(0), traj_point_emb.size(0) + traj_seg_emb.size(0))).long().to(device)
            edge_index = torch.cartesian_prod(point_node_index, seg_node_index).t().contiguous().to(device)
            self.trajs_neighbor.append(indices[:,1:].T)
            self.trajs_feature.append(feature)
            self.trajs_edge_index.append(edge_index)
            self.trajs_point_node_index.append(point_node_index)
            self.trajs_seg_node_index.append(seg_node_index)
            self.trajs_emb.append(traj_emb)
    def update_simp(self,amplify_labels):
        self.amplify_labels = torch.zeros((len(self.trajs_point_node_index),(self.trajs_point_node_index[0]).size(0))).to(self.device)
        for i,amplify_label in enumerate(amplify_labels):
           for index in amplify_label:
               self.amplify_labels[i][index]=1
    def segment(self, index):
        seg = []
        segvalid = []
        data_index = []
        seg_time = []
        start = 0
        end = -1
        data = self.data[index]
        time = self.data_time[index]
        while end != len(data):
            end = min(start + self.max_length, len(data))
            if start == 0:
                valid_start = 0
            else:
                valid_start = self.overlap
            if end == len(data):
                valid_end = end - start
            else:
                valid_end = end - start - self.overlap
            segvalid.append([valid_start, valid_end])
            seg.append(data[start:end])
            seg_time.append(time[start:end])
            data_index.append(index)
            start += self.max_length - 2 * self.overlap
        return seg, segvalid, seg_time, data_index
    def __getitem__(self, index):
        index=self.idx[index] 
        if 'n1' not in self.__dict__:self.n1=torch.zeros(1,device=device)-1
        if self.amplify_labels==None: return  self.trajs_feature[index], self.trajs_edge_index[index], self.trajs_point_node_index[index], self.trajs_seg_node_index[index], self.trajs_emb[index], self.trajs_neighbor[index], self.n1
        else: return self.trajs_feature[index], self.trajs_edge_index[index], self.trajs_point_node_index[index], self.trajs_seg_node_index[index], self.trajs_emb[index], self.trajs_neighbor[index], self.amplify_labels[index]
class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        in_features = 128
        out_features = 32
        hidden_features = out_features
        num_heads = 4
        self.conv1 = GATConv(in_features, hidden_features, heads=num_heads)
        self.conv2 = GATConv(hidden_features * num_heads, out_features, heads=1, concat=False)
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        norm = torch.norm(x,p=2,dim=1).unsqueeze(-1)
        return x/norm
    def simp_trajs(self,z,neighbor,rate=0.4):
        align_loss = 0
        for i in range(neighbor.size(0)):
            pos = z[neighbor[i]]
            align_loss += (z - pos).norm(dim=1).pow(2)
        align_loss = align_loss / neighbor.size(0)
        sim = torch.cdist(z, z).pow(2)
        uniform_loss = sim.mul(-2).exp().mean(dim=1).log()
        important = (torch.softmax(align_loss* uniform_loss,dim=0))[1:-1]
        _, top_k_indices = torch.sort(important, descending=True)
        K = int(rate * z.size(0))
        top_k_indices = top_k_indices[:K]
        trajs,_ = torch.sort(top_k_indices)
        return trajs
    def important_sigmoid(self,z,neighbor):
        align_loss = 0
        for i in range(neighbor.size(0)):
            pos = z[neighbor[i]]
            align_loss += (z-pos).norm(dim=1).pow(2)
        align_loss = align_loss/neighbor.size(0)
        sim = torch.cdist(z,z).pow(2)
        uniform_loss = sim.mul(-2).exp().mean(dim=1).log()
        important = torch.sigmoid(align_loss* uniform_loss)
        return important
    def loss(self,z,neighbor,amply_labels=None):
        align_loss = 0
        for i in range(neighbor.size(0)):
            pos = z[neighbor[i]]
            align_loss += (z-pos).norm(dim=1).pow(2)
        align_loss = align_loss/neighbor.size(0)
        sim = torch.cdist(z,z).pow(2)
        uniform_loss = sim.mul(-2).exp().mean(dim=1).log()
        important = (torch.softmax(align_loss* uniform_loss,dim=0))
        if amply_labels!=None:
            bce_loss = nn.BCELoss(reduction='mean')
            mutual_loss = bce_loss(important,amply_labels)
        else:
            mutual_loss=0
        _, top_k_indices = torch.sort(important[1:-1], descending=True)
        K=3
        top_k_indices,_ = torch.sort(top_k_indices[:K])
        important_simp = ((z[top_k_indices].sum(dim=0)+z[0]+z[-1]))/(K+2)
        return align_loss,uniform_loss,important_simp,mutual_loss
def GraphSimpcollate(batch):
    batch = list(zip(*batch))
    return batch
def train_graphsimp(train_dataset:GraphSimpDataset, gnn_path,simp_trajs_idx=None,load_model=False,DEBUG=False):
    model = GAT().to(device)
    batch_size = 64
    if simp_trajs_idx!=None:
        train_dataset.update_simp(simp_trajs_idx)
    if load_model and os.path.exists(gnn_path):
        model.load_state_dict(loadZ_th(gnn_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=0)
    dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,collate_fn=GraphSimpcollate)
    cnt = 0
    for epoch in range( 1 if DEBUG else 20 ):
        for batch in dataloader:
            model.train()
            optimizer.zero_grad()
            trajs_feature, trajs_edge_index, trajs_point_node_index, trajs_seg_node_index, trajs_emb, trajs_neighbor, amply_labels = to_device(batch)
            align_losses = 0
            uniform_losses=0
            mutual_losses=0
            simp_trajs=[]
            for i in range(len(trajs_feature)):
                traj_feature = trajs_feature[i]
                traj_edge_index = trajs_edge_index[i]
                traj_point_node_index = trajs_point_node_index[i]
                traj_neighbor = trajs_neighbor[i]
                if simp_trajs_idx!=None:
                    amply_label = amply_labels[i]
                else:
                    amply_label=None
                traj_point_emb = model(traj_feature,traj_edge_index)
                align_loss, uniform_loss, important_simp,mutual_loss = model.loss(traj_point_emb[traj_point_node_index],traj_neighbor,amply_label )
                align_losses+=align_loss
                uniform_losses+=uniform_loss
                mutual_losses += mutual_loss
                simp_trajs.append(important_simp)
            simp_trajs = torch.stack(simp_trajs)
            simp_trajs_norm = torch.norm(simp_trajs, p=2, dim=1).unsqueeze(-1)
            simp_trajs = simp_trajs/simp_trajs_norm
            trajs_emb = torch.stack(trajs_emb)
            trajs_emb_norm = torch.norm(trajs_emb,p=2,dim=1).unsqueeze(-1)
            trajs_emb = trajs_emb/trajs_emb_norm
            batch_losses = F.mse_loss(simp_trajs@simp_trajs.T,trajs_emb@trajs_emb.T)
            align_losses = (align_losses/len(trajs_feature)).mean()
            uniform_losses = (uniform_losses/len(trajs_feature)).mean()
            if mutual_losses!=0:
                mutual_losses = mutual_losses/len(trajs_feature)
            losses = align_losses + 0.3 * uniform_losses + 0.5 * batch_losses + 1 * mutual_losses
            losses.backward()
            optimizer.step()
            if mutual_losses != 0: print(f'epoch:{epoch} | loss:{losses.item():.4f} ')
            else: print(  f'epoch:{epoch} | loss:{losses.item():.4f} ')
    saveZ_th(gnn_path,model.state_dict()) 
    model.eval()
    simp_trajs = []
    for i in range(len(train_dataset)):
        traj_feature, traj_edge_index, traj_point_node_index, traj_seg_node_index, traj_emb, traj_neighbor,_ = to_device(train_dataset[i])
        traj_point_emb = model(traj_feature, traj_edge_index)
        important_simp = model.simp_trajs(traj_point_emb[traj_point_node_index],traj_neighbor)
        simp_trajs.append(important_simp)
    simp_trajs = torch.stack(simp_trajs).detach()
    return simp_trajs
class DiffuSimpDataset(Dataset):
    def __init__(self,trajs,simp_trajs_idx):
        self.trajs = trajs
        self.simp_trajs_idx = simp_trajs_idx
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, item):
        return self.trajs[item], self.simp_trajs_idx[item]
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
def l2_distance(lon1, lat1, lon2, lat2):
    return math.sqrt( (lon2 - lon1) ** 2 + (lat2 - lat1) ** 2 )
def generate_original_features(src, x_max, y_max, x_min, y_min):
    src = np.array(src)[:,:2]
    tgt = []
    lens = []
    for p1, p2 in pairwise(src):
        lens.append(l2_distance(p1[0], p1[1], p2[0], p2[1]))
    lens = np.array(lens)
    for i in range(1, len(src) - 1):
        dist = (lens[i - 1] + lens[i]) / 2
        radian = math.pi - math.atan2(src[i - 1][0] - src[i][0], src[i - 1][1] - src[i][1]) \
                 + math.atan2(src[i + 1][0] - src[i][0], src[i + 1][1] - src[i][1])
        radian = 1 - abs(radian) / math.pi
        x = (src[i][0] - x_min) / (x_max - x_min)
        y = (src[i][1] - y_min) / (y_max - y_min)
        tgt.append([x, y, dist, radian])
    x = (src[0][0] - x_min) / (x_max - x_min)
    y = (src[0][1] - y_min) / (y_max - y_min)
    tgt.insert(0, [x, y, 0.0, 0.0])
    x = (src[-1][0] - x_min) / (x_max - x_min)
    y = (src[-1][1] - y_min) / (y_max - y_min)
    tgt.append([x, y, 0.0, 0.0])
    return tgt
def DiffuSimpcollate(batch,x_max, y_max, x_min, y_min,device):
    trajs, simp_trajs_idx = list(zip(*batch))
    trajs_emb = []
    for traj in trajs:
        traj_emb = generate_original_features(traj,x_max, y_max, x_min, y_min)
        trajs_emb.append(traj_emb)
    trajs_emb = torch.Tensor(trajs_emb).to(device)
    trajs_padding = pad_sequence(trajs_emb, batch_first=True).to(device)
    trajs_len = torch.LongTensor(list(map(len, trajs))).to(device)
    max_trajs_len = trajs_len.max().item()
    padding_mask = torch.arange(max_trajs_len).to(device)[None, :] >= trajs_len[:, None]
    simp_trajs=[]
    for i in range(len(trajs)):
        simp_traj_idx = np.concatenate([[0],simp_trajs_idx[i].cpu().numpy(),[-1]])
        simp_traj = trajs_emb[i][simp_traj_idx]
        simp_trajs.append(simp_traj)
    simp_trajs_padding = pad_sequence(simp_trajs, batch_first=True).to(device)
    simp_trajs_len = torch.LongTensor(list(map(len, simp_trajs))).to(device)
    max_simp_trajs_len = simp_trajs_len.max().item()
    simp_padding_mask = torch.arange(max_simp_trajs_len).to(device)[None, :] >= simp_trajs_len[:, None]
    labels = []
    labels_mask = []
    for i in range(len(trajs)):
        simp_traj_idx = np.concatenate([[0], simp_trajs_idx[i].cpu().numpy(), [len(trajs[i])-1]])
        labels.append(torch.LongTensor(simp_traj_idx).to(device) )
    labels = torch.stack(labels,dim=0)
    return trajs_padding, padding_mask, simp_trajs_padding, simp_padding_mask,labels,labels_mask
class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """
    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """
    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices =torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights =torch.from_numpy(weights_np).float().to(device)
        return indices, weights
class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])
    def weights(self):
        return self._weights
class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.
        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.
        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)
        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        torch.distributed.all_gather(timestep_batches, local_ts)
        torch.distributed.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)
    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.
        Sub-classes should override this method to update the reweighting
        using losses from the model.
        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.
        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """
class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)
    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights
    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1
    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.
    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
class SimpTransformer(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """
    def __init__(self,in_dim,encoder_n_head,encoder_hidden_dim,encoder_n_layer,in_channels,hidden_channels,out_channels,n_head,n_layer,trans_hidden_channels,attn_dropout,dropout):
        super().__init__()
        self.model_channels = model_channels = hidden_channels
        encoder_layer = nn.TransformerEncoderLayer(in_dim, encoder_n_head, encoder_hidden_dim, batch_first=True)
        self.traj_embed = nn.TransformerEncoder(encoder_layer, encoder_n_layer)
        self.traj_transform = nn.Sequential(
            nn.Linear(in_dim, encoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(encoder_hidden_dim, in_channels),
        )
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_channels),
        )
        self.time_embed_transform = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.input_up_proj = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                           nn.Tanh(),
                                           nn.Linear(hidden_channels, hidden_channels)) 
        enc_layer = nn.TransformerEncoderLayer(hidden_channels, n_head, trans_hidden_channels,attn_dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layer)
        self.position_embeddings  = PositionalEncodingDiff(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.output_down_proj = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                             nn.Tanh() , nn.Linear(hidden_channels, out_channels)) 
    def get_embeds(self, traj_input,  padding_mask=None):
        x = self.traj_embed(traj_input, src_key_padding_mask=padding_mask)
        x = F.normalize(x)
        x= self.traj_transform(x)
        x = F.normalize(x)
        return x
    def get_logits(self, hidden_repr, doc_embed, test=False):
        if self.logits_mode == 1:
            if test:
                return torch.bmm(hidden_repr, doc_embed.permute(0, 2, 1)) 
            return self.lm_head(hidden_repr) 
        else:
            raise NotImplementedError
    def forward(self, x, x_t_mask, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb = self.time_embed_transform(emb)
        emb = self.dropout(F.normalize(emb))
        emb_x = self.input_up_proj(x)
        emb_x = F.normalize(emb_x)
        seq_length = x.size(1)
        emb_x += emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = emb_x
        emb_inputs = F.normalize(emb_inputs)
        emb_inputs = self.position_embeddings(emb_inputs)
        input_trans_hidden_states = self.encoder(emb_inputs,src_key_padding_mask = x_t_mask)
        input_trans_hidden_states = F.normalize(input_trans_hidden_states)
        h = self.output_down_proj(input_trans_hidden_states)
        h = F.normalize(h)
        h = h.type(x.dtype)
        return h
def create_gaussian_diffusion( *, steps=1000, learn_sigma=False, sigma_small=False, noise_schedule="linear", use_kl=False, predict_xstart=False, rescale_timesteps=False, rescale_learned_sigmas=False, timestep_respacing="", training_mode='emb',
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if training_mode == 'e2e':
        if use_kl: loss_type = LossType.E2E_KL
        else: loss_type = LossType.E2E_MSE 
    elif training_mode == 'e2e-simple':
        if use_kl: loss_type = LossType.E2E_Simple_KL
        else: loss_type = LossType.E2E_Simple_MSE
    else:
        if use_kl: loss_type = LossType.RESCALED_KL
        elif rescale_learned_sigmas: loss_type = LossType.RESCALED_MSE
        else:  loss_type = LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    print(loss_type, learn_sigma)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type= ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X,
        model_var_type=(
            (
                ModelVarType.FIXED_LARGE
                if not sigma_small
                else ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        training_mode=training_mode,
    )
def create_model_and_diffusion()->Tuple[SimpTransformer,SpacedDiffusion]:
    model = SimpTransformer( in_dim=4, encoder_n_head=2, encoder_hidden_dim=128, encoder_n_layer=3, in_channels=128, hidden_channels=256, out_channels=128, n_head=2, n_layer=2, trans_hidden_channels=256, attn_dropout=0.1, dropout=0.1
    )
    diffusion = create_gaussian_diffusion( steps=2000, learn_sigma=False, sigma_small=False, noise_schedule='linear', use_kl= False, predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True, timestep_respacing='', training_mode= 'e2e'
    )
    return model, diffusion
INITIAL_LOG_LOSS_SCALE = 20.0
def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        print(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            print(f"{key}_q{quartile}", sub_loss)
class TrainLoop:
    def __init__( self, *, model, diffusion, dataloader, lr, ema_rate, log_interval, save_interval, resume_checkpoint, use_fp16=False, fp16_scale_growth=1e-3, schedule_sampler=None, weight_decay=0.0, lr_anneal_steps=0, checkpoint_path='', gradient_clipping=-1., eval_dataloader=None, eval_interval=-1, epochs = 5,):
        self.model:SimpTransformer = model
        self.diffusion:SpacedDiffusion = diffusion
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.gradient_clipping = gradient_clipping
        self.epochs = epochs
        self.step = 0
        self.resume_step = 0
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = torch.cuda.is_available()
        self.checkpoint_path = checkpoint_path 
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.ema_params = [
            copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
        ]
        self.ddp_model = self.model
    def save(self):
        if self.checkpoint_path[0]:saveZ_th(self.checkpoint_path[0],self.model.state_dict())
        if self.checkpoint_path[1]:saveZ_th(self.checkpoint_path[1],self.diffusion.state_dict())
    def run_step(self, batch):
        losses = self.forward_backward(batch)
        return losses
    def forward_backward(self, batch):
        zero_grad(self.model_params)
        trajs_padding, padding_mask, simp_trajs_padding, simp_padding_mask, labels,labels_mask = to_device(batch)
        cond = {
            "trajs_padding": trajs_padding,
            "padding_mask": padding_mask,
            "simp_trajs_padding": simp_trajs_padding,
            "simp_padding_mask": simp_padding_mask,
            "labels":labels,
            "labels_mask":labels_mask
        }
        t, weights = self.schedule_sampler.sample(trajs_padding.size(0), device= device)  
        compute_losses = partial(
            self.diffusion.training_losses,
            self.ddp_model,
            t,
            model_kwargs=cond,
        )
        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        loss = (losses["loss"] * weights).mean()
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.ddp_model.parameters(),3)
            self.opt.step()
            return losses
    def run_loop(self):
        for e in range(self.epochs):
            for batch in self.dataloader:
                self.ddp_model.train()
                losses = self.run_step(batch)
                print(f'Epoches:{e}/{self.epochs}, loss:{losses["loss"].cpu().item()}')
                self.step += 1
            self.save()
def get_model_diffusimp(simp_path='',diff_path='', load_model=True)->Tuple[SimpTransformer,SpacedDiffusion]:
    model, diffusion = create_model_and_diffusion() 
    model = model.to(device)
    if load_model: 
        if simp_path and os.path.exists(simp_path): model.load_state_dict(loadZ_th(simp_path))
        if diff_path and os.path.exists(diff_path): diffusion.load_state_dict(loadZ_th(diff_path))
    return model,diffusion
def train_diffusimp(trajs,bbox, simp_trajs_idx, model:SimpTransformer, diffusion:SpacedDiffusion, simp_path='',diff_path='',DEBUG=False):
    [x_min,x_max],[y_min,y_max]=bbox[:2]
    batch_size = 64 ; amplify_len  = 20
    train_set = DiffuSimpDataset(trajs,simp_trajs_idx)
    dataloader = DataLoader(train_set, batch_size=batch_size,shuffle=True,collate_fn=partial(DiffuSimpcollate, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min,device=device))
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    TrainLoop( model=model, diffusion=diffusion, dataloader=dataloader, lr=1e-4, ema_rate= "0.9990", log_interval= 50, save_interval= 5000, resume_checkpoint= "", use_fp16= False, fp16_scale_growth= 1e-3, schedule_sampler= schedule_sampler, weight_decay= 0, lr_anneal_steps= 0, checkpoint_path=[simp_path,diff_path], gradient_clipping=-1.0, eval_interval=2000, epochs=(1 if DEBUG else 20), 
    ).run_loop()
    model.eval()
    dataloader_eval = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=partial(DiffuSimpcollate, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min, device=device))
    results= []
    with torch.no_grad():
        for batch in dataloader_eval:
            cond = {"diffusion_steps": 500,"batch": batch,"amplify_len": amplify_len}
            sample_fn = (diffusion.p_sample_loop)
            out = sample_fn(model,device=device,clip_denoised=False,denoised_fn=None,model_kwargs=cond,top_p=-1,)
            results.append(out['sample'])
    results = torch.cat(results,dim=0)
    x_sum = results[:,-amplify_len:,:]
    x_original = results[:,:results.shape[1]-amplify_len,:]
    logits = torch.softmax(torch.bmm(x_sum,x_original.permute(0,2,1)),dim=-1)
    simp_trajs = torch.topk(logits,k=amplify_len,dim=-1)
    simp_trajs_candidate = simp_trajs.indices
    simp_trajs_idx=[]
    for i in range(simp_trajs_candidate.size(0)):
        simp_traj_idx=[]
        simp_traj_candidate = simp_trajs_candidate[i]
        for j in range(amplify_len):
            j_candidate_list = simp_traj_candidate[j].cpu().numpy()
            for j_candidate in j_candidate_list:
                if j_candidate not in simp_traj_idx:
                     simp_traj_idx.append(j_candidate)
                     break
        simp_trajs_idx.append(simp_traj_idx)
    return simp_trajs_idx
def api_pre_gnndata(root_data,name,ts=None,bbox=None,bs=64):
    path_old=os.path.join(root_data,f'pre-{name}-gnndata.pk.zst')
    path_new=os.path.join(root_data,f'pre-{name}-gnndata.th')
    if os.path.exists(path_new):return load_th(path_new)
    if os.path.exists(path_old):
        x= loadZ_pk(path_old)
        save_th(path_new,x)
        return x
    grid,G,grid_ebd,tbert_model,obj_model=api_tbert_pretrain(root_data=root_data,ts_train=ts,bbox=bbox,bs=bs)
    graph_train_dataset = GraphSimpDataset(ts=ts,grid=grid,tbert_pretrained=tbert_model)
    save_th(path_new,graph_train_dataset)
    return graph_train_dataset
def api_mlsimp_train(root_model,root_data=None,ts_name=None,ts_train=None,bbox=None,gsize=None):
    """note |ts|=1000"""
    gnn_path= os.path.join(root_model,f'gnn.zst')
    simp_path=os.path.join(root_model,f'simp.zst')
    diff_path=os.path.join(root_model,f'diff.zst') ; diff_path='' 
    if os.path.exists(gnn_path) and os.path.exists(simp_path):
        gnn=GAT() ; gnn.load_state_dict(loadZ_th(gnn_path))
        simp,diff=get_model_diffusimp(simp_path,diff_path,True)
        return gnn,simp,diff
    graph_train_dataset = api_pre_gnndata(root_data,ts_name)
    simp_trajs_idx=None
    diff_trajs_idx=None
    load_model=False
    for i in range(10):
        simp_trajs_idx = train_graphsimp(graph_train_dataset, gnn_path, diff_trajs_idx,load_model)
        simp,diff=get_model_diffusimp(simp_path,diff_path,load_model)
        diff_trajs_idx = train_diffusimp(ts_train,bbox, simp_trajs_idx,simp,diff,simp_path,diff_path)
        load_model=True
    gnn=GAT() ; gnn.load_state_dict(loadZ_th(gnn_path))
    return gnn,simp,diff
def init_query_param(dataset, q_type,distri): 
    t={'Porto':60*60*24*7,'Beijing':1237764824,'Xian':60*60*6,} 
    return 0.02,0.02, t[dataset]

class Rtree():
    def __init__(self, *args):
        self.p = rtreeIndex.Property()
        self.p.dimension = 3
        if len(args) == 0:
            self.idx = rtreeIndex.Index(properties=self.p)
        else:
            self.idx = rtreeIndex.Index(args[0],properties=self.p)
    def insert(self, id, data, obj):  
        self.idx.insert(id, data, obj=obj)
    def delete(self, id, data):  
        self.idx.delete(id, data)
    def knn(self, width, num=1, objects=True):  
        res = list(self.idx.nearest(width, num, objects=objects))
        return res
    def range_query(self, width, objects=True):  
        res = list(self.idx.intersection(width, objects=objects))
        return res
def build_or_load_Rtree(ts, rtree_path):  
    if os.path.exists(rtree_path + '.dat'):
        Rtree_ = Rtree(rtree_path)
    else:
        Rtree_ = Rtree(rtree_path)
        c = 0
        delete_rec = {}
        for tid,t in enumerate(ts):
            for pid,p in enumerate(t):
                Rtree_.insert(c, (p[0], p[1], p[2], p[0], p[1], p[2]), [tid,pid])
                delete_rec[(tid, pid)] = c
                c += 1
    return Rtree_
def simp(model:GAT,test_dataset):
    batch_size = 64
    dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,collate_fn=GraphSimpcollate)
    trajs_score = []
    for batch in dataloader:
        with torch.no_grad():
            trajs_feature, trajs_edge_index, trajs_point_node_index, trajs_seg_node_index, trajs_emb, trajs_neighbor, amply_labels = to_device(batch)
            for i in range(len(trajs_feature)):
                traj_feature = trajs_feature[i]
                traj_edge_index = trajs_edge_index[i]
                traj_point_node_index = trajs_point_node_index[i]
                traj_neighbor = trajs_neighbor[i]
                traj_point_emb = model(traj_feature,traj_edge_index)
                important = model.important_sigmoid(traj_point_emb[traj_point_node_index],traj_neighbor)
                trajs_score.append(important)
    trajs_score = torch.stack(trajs_score)
    trajs_score_norm = torch.norm(trajs_score, p=2, dim=1).unsqueeze(-1)
    trajs_score = trajs_score/trajs_score_norm
    return trajs_score
def get_distribution_feature_data(db,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range'):
    DB_DISTRI, ID2Grid, DB_DISTRI_trajID = {}, {}, {}
    x_step, y_step, t_step = init_query_param(dataset,q_type,'data')
    thre =  1
    for trajID in range(len(db)):
        for pointID in range(len(db[trajID])):
            if pointID == 0 or pointID == len(db[trajID]) - 1:
                continue
            point = db[trajID][pointID]
            [x, y, t] = point
            key = tuple([int((x - Xmin) / x_step), int((y - Ymin) / y_step), int((t - Tmin) / t_step)])
            ID2Grid[(trajID, pointID)] = key
            if key in DB_DISTRI_trajID:
                DB_DISTRI_trajID[key].add(trajID)
            else:
                DB_DISTRI_trajID[key] = set([trajID])
    for key in DB_DISTRI_trajID:
        if len(DB_DISTRI_trajID[key]) > thre:
            DB_DISTRI[key] = len(DB_DISTRI_trajID[key])
    return DB_DISTRI, ID2Grid, DB_DISTRI_trajID
def get_distribution_feature_gau(db,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range'):
    DB_DISTRI, ID2Grid, Grid2ID,DB_DISTRI_trajID= {}, {}, {},{}
    x_step, y_step, t_step = init_query_param(dataset,q_type,'gau')
    X, Y, T = [], [], []
    thre =10
    for trajID in range(len(db)):
        for pointID in range(len(db[trajID])):
            if pointID == 0 or pointID == len(db[trajID]) - 1:
                continue
            point = db[trajID][pointID]
            [x, y, t] = point
            key = tuple([int((x - Xmin) / x_step), int((y - Ymin) / y_step), int((t - Tmin) / t_step)])
            ID2Grid[(trajID, pointID)] = key
            if key in DB_DISTRI_trajID:
                DB_DISTRI_trajID[key].add(trajID)
            else:
                DB_DISTRI_trajID[key] = set([trajID])
            if key in Grid2ID:
                Grid2ID[key].add(trajID)
            else:
                Grid2ID[key] = set([trajID])
                X.append(key[0])
                Y.append(key[1])
                T.append(key[2])
    for key in DB_DISTRI_trajID:
        if len(DB_DISTRI_trajID[key]) > thre:
            DB_DISTRI[key] = len(DB_DISTRI_trajID[key])
    X.sort()
    Y.sort()
    T.sort()
    X_map, Y_map, T_map = {}, {}, {}
    for i in range(len(Grid2ID)):
        X_map[i] = X[i]
        Y_map[i] = Y[i]
        T_map[i] = T[i]
    mu, alpha = (1 + len(Grid2ID)) / 2, (len(Grid2ID) - 1) / 4
    for cnt in range(10000):
        [x, y, t] = [np.random.normal(loc=mu, scale=alpha, size=None),
                     np.random.normal(loc=mu, scale=alpha, size=None),
                     np.random.normal(loc=mu, scale=alpha, size=None)]
        if (int(x) in X_map) and (int(y) in Y_map) and (int(t) in T_map):
            key = tuple([X_map[int(x)], Y_map[int(y)], T_map[int(t)]])
            if key in Grid2ID:
                if key in DB_DISTRI:
                    DB_DISTRI[key] += 1
    return DB_DISTRI
def get_query_workload_data(DB_DISTRI, num=100):
    K, V = list(DB_DISTRI.keys()), list(DB_DISTRI.values())
    np.random.seed(1)
    query_workload = []
    sample_value = np.array(V)
    sample_value = sample_value / np.sum(sample_value)
    while len(query_workload) < num:
        index = int(np.random.choice(len(sample_value), 1, p=sample_value))
        query_workload.append(K[index])
    return DB_DISTRI, query_workload[:int(num / 2)], query_workload[int(num / 2):]
def get_query_workload_gau(DB_DISTRI, num=100):
    K, V = list(DB_DISTRI.keys()), list(DB_DISTRI.values())
    np.random.seed(1)
    query_workload = []
    sample_value = np.array(V)
    sample_value = sample_value / np.sum(sample_value)
    while len(query_workload) < num:
        index = int(np.random.choice(len(sample_value), 1, p=sample_value))
        query_workload.append(K[index])
    return DB_DISTRI, query_workload[:int(num / 2)], query_workload[int(num / 2):]
def range_query_adjust(Rtree,QUERY,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range',distri='data'):
    x_length, y_length, t_length = init_query_param(dataset,q_type,distri)
    As=[]
    for i in range(len(QUERY)):
        (x_idx, y_idx, t_idx) = QUERY[i]
        x_center, y_center, t_center = Xmin + x_length * (0.5 + x_idx), Ymin + y_length * (
                    0.5 + y_idx), Tmin + t_length * (0.5 + t_idx)
        ref_R = Rtree.range_query((x_center - x_length / 2,
                                       y_center - y_length / 2,
                                       t_center - t_length / 2,
                                       x_center + x_length / 2,
                                       y_center + y_length / 2,
                                       t_center + t_length / 2))
        A = [item.object for item in ref_R]
        As.append(A)
    return As
def save_rtree(filename1, filename2):
    shutil.copyfile(filename1 + '.idx', filename2 + '.idx')
    shutil.copyfile(filename1 + '.dat', filename2 + '.dat')
def build_Rtree(DB, filename=''):  
    if os.path.exists(filename + '.dat') and os.path.exists(filename + '.idx'):
        os.remove(filename + '.dat')
        os.remove(filename + '.idx')
    if os.path.exists(filename+'_persisted' + '.dat') and os.path.exists(filename+'_persisted' + '.idx'):
        save_rtree(filename+'_persisted',filename)
        Rtree_ = Rtree(filename)
    else:
        if filename == '':
            Rtree_ = Rtree()
        else:
            Rtree_ = Rtree(filename)
        c = 0
        delete_rec = {}
        for trajID in range(len(DB)):
            for pointID in range(len(DB[trajID])):
                point = DB[trajID][pointID]
                Rtree_.insert(c, (point[0], point[1], point[2], point[0], point[1], point[2]), trajID) 
                delete_rec[(trajID, pointID)] = c
                c += 1
    return Rtree_
def get_block_trajs(DB, A, xmin, ymin, tmin, xmax, ymax, tmax):
    ref_DB = []
    for a in A:
        traj = DB[a]
        ref_db = []
        for pts in traj:
            if pts[2] >= tmin and pts[2] <= tmax:
                ref_db.append(pts)
        ref_DB.append(ref_db)
    return ref_DB
@numba.jit(nopython=True, fastmath=True) 
def edr(ts_a:np.ndarray, ts_b:np.ndarray, eps:float):
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N))
    cost[0, 0] = 0
    for i in range(1, M):
        cost[i, 0] = i
    for j in range(1, N):
        cost[0, j] = j
    for i in range(1, M):
        for j in range(1, N):
            if np.linalg.norm(ts_a[i][0:2] - ts_b[j][0:2]) < eps:
                choices = 0
            else:
                choices = 1
            cost[i, j] = min(cost[i - 1, j - 1] + choices, cost[i, j - 1] + 1, cost[i - 1, j] + 1)
    return cost[-1, -1]
def knn_edr_query_offline(DB, Rtree_ref, test_query,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range',distri='data'):
    x_length, y_length, t_length = init_query_param(dataset,q_type,distri)
    GroundQuerySet, interval, record = [], [], {}
    repeat = {}
    for i in range(len(test_query)):
        (x_idx, y_idx, t_idx) = test_query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        repeat[(x_idx, y_idx, t_idx)] = 1
        x_center, y_center, t_center = Xmin + x_length * (0.5 + x_idx), Ymin + y_length * (
                    0.5 + y_idx), Tmin + t_length * (0.5 + t_idx)
        ref_R = Rtree_ref.range_query((x_center - x_length / 2,
                                       y_center - y_length / 2,
                                       t_center - t_length / 2,
                                       x_center + x_length / 2,
                                       y_center + y_length / 2,
                                       t_center + t_length / 2))
        A = set([item.object for item in ref_R])
        A = list(A)
        if len(A) > 1 and len(A) < 50:
            interval.append((A, test_query[i]))
            ref_DB = get_block_trajs(DB, A, x_center - x_length / 2, y_center - y_length / 2, t_center - t_length / 2,
                                     x_center + x_length / 2, y_center + y_length / 2, t_center + t_length / 2)
            GroundSet, QuerySet = [], []
            for q_ in range(len(ref_DB)):
                query = ref_DB[q_]
                ground = []
                for c_ in range(len(ref_DB)):
                    data = ref_DB[c_]
                    if (A[q_], A[c_]) in record:
                        ground.append([record[(A[q_], A[c_])], A[c_]])
                    else:
                        tmp = edr(query, data, eps=0.02)
                        ground.append([tmp, A[c_]])
                        record[(A[q_], A[c_])] = tmp
                ground.sort(key=lambda s: (s[0]))
                GroundSet.append(ground)
                QuerySet.append(query)
            GroundQuerySet.append((GroundSet, QuerySet))
    return GroundQuerySet, interval
def knn_edr_query_online(GroundQuerySet, interval, Rtree_sim, sim_DB, k=3,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range',distri='data'):
    x_length, y_length, t_length = init_query_param(dataset,q_type,distri)
    result = []
    for (A, test_query), (GroundSet, QuerySet) in zip(interval, GroundQuerySet):
        (x_idx, y_idx, t_idx) = test_query
        x_center = Xmin + x_length * (0.5 + x_idx)
        y_center = Ymin + y_length * (0.5 + y_idx)
        t_center = Tmin + t_length * (0.5 + t_idx)
        sim_R = Rtree_sim.range_query((x_center - x_length / 2,
                                       y_center - y_length / 2,
                                       t_center - t_length / 2,
                                       x_center + x_length / 2,
                                       y_center + y_length / 2,
                                       t_center + t_length / 2))
        B = set([item.object for item in sim_R])
        B = list(B)
        if len(set(A)) == 0 or len(set(B)) == 0:
            continue
        win_sim_DB = get_block_trajs(sim_DB, B, x_center - x_length / 2, y_center - y_length / 2,
                                     t_center - t_length / 2, x_center + x_length / 2, y_center + y_length / 2,
                                     t_center + t_length / 2)
        cnt =0
        for ground, query in zip(GroundSet, QuerySet):
            predict = []
            query_num = A[cnt]
            cnt += 1
            if query_num not in B:
                continue
            query = win_sim_DB[B.index(query_num)]
            if len(query) <= 1:
                continue
            for j in range(len(win_sim_DB)):
                predict.append([edr(query, win_sim_DB[j], eps=0.02), B[j]])
            predict.sort(key=lambda s: (s[0]))
            predict_tmp, ground_tmp = [], []
            for predict_i in range(0, min(k, len(predict))):
                predict_tmp.append(predict[predict_i][1])
            for ground_i in range(0, min(k, len(ground))):
                ground_tmp.append(ground[ground_i][1])
            result.append(len(set(predict_tmp) & set(ground_tmp)) / min(k, len(ground)))
    return sum(result) / len(result)
def range_query_operator(Rtree_ref, Rtree_sim, QUERY, verbose=False,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range',distri='data'):
    F1 = []
    x_length, y_length, t_length = init_query_param(dataset,q_type,distri)
    ee, ef, fe, ff = 0, 0, 0, 0
    for i in range(len(QUERY)):
        (x_idx, y_idx, t_idx) = QUERY[i]
        x_center, y_center, t_center = Xmin + x_length * (0.5 + x_idx), Ymin + y_length * (
                    0.5 + y_idx), Tmin + t_length * (0.5 + t_idx)
        ref_R = Rtree_ref.range_query((x_center - x_length / 2,
                                       y_center - y_length / 2,
                                       t_center - t_length / 2,
                                       x_center + x_length / 2,
                                       y_center + y_length / 2,
                                       t_center + t_length / 2))
        sim_R = Rtree_sim.range_query((x_center - x_length / 2,
                                       y_center - y_length / 2,
                                       t_center - t_length / 2,
                                       x_center + x_length / 2,
                                       y_center + y_length / 2,
                                       t_center + t_length / 2))
        A = set([item.object for item in ref_R])
        B = set([item.object for item in sim_R])
        if verbose:
            if A != B:
                print('A & B', A, B)
        if A == set() and B == set():
            ee += 1
            F1.append(1.0)
        if A == set() and B != set():
            ef += 1
            F1.append(0.0)
        if A != set() and B == set():
            fe += 1
            F1.append(0.0)
        if A != set() and B != set():
            ff += 1
            P = len(A & B) / len(B)
            R = len(A & B) / len(A)
            if (P + R) == 0:
                F1.append(0.0)
            else:
                F1.append((2 * P * R) / (P + R))
    return sum(F1) / len(F1)
class Point(object):
    def __init__(self, x, y, traj_id=None):
        self.trajectory_id = traj_id
        self.x = x
        self.y = y
    def __repr__(self):
        return "{0:.8f},{1:.8f}".format(self.x, self.y)
    def get_point(self):
        return self.x, self.y
    def __add__(self, other: 'Point'):
        if not isinstance(other, Point):
            raise TypeError("The other type is not 'Point' type.")
        _add_x = self.x + other.x
        _add_y = self.y + other.y
        return Point(_add_x, _add_y, traj_id=self.trajectory_id)
    def __sub__(self, other: 'Point'):
        if not isinstance(other, Point):
            raise TypeError("The other type is not 'Point' type.")
        _sub_x = self.x - other.x
        _sub_y = self.y - other.y
        return Point(_sub_x, _sub_y, traj_id=self.trajectory_id)
    def __mul__(self, x: float):
        if isinstance(x, float):
            return Point(self.x*x, self.y*x, traj_id=self.trajectory_id)
        else:
            raise TypeError("The other object must 'float' type.")
    def __truediv__(self, x: float):
        if isinstance(x, float):
            return Point(self.x / x, self.y / x, traj_id=self.trajectory_id)
        else:
            raise TypeError("The other object must 'float' type.")
    def distance(self, other: 'Point'):
        return math.sqrt(math.pow(self.x-other.x, 2) + math.pow(self.y-other.y, 2))
    def dot(self, other: 'Point'):
        return self.x * other.x + self.y * other.y
    def as_array(self):
        return np.array((self.x, self.y))
eps = 1e-12
def _point2line_distance(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)
    return np.divide(np.abs(np.linalg.norm(np.cross(end - start, start - point))), np.linalg.norm(end - start))
class Segment(object):
    def __init__(self, start_point: Point, end_point: Point, traj_id: int = None, cluster_id: int = -1):
        self.start = start_point
        self.end = end_point
        self.traj_id = traj_id
        self.cluster_id = cluster_id
    def set_cluster(self, cluster_id: int):
        self.cluster_id = cluster_id
    def pair(self) -> Tuple[Point, Point]:
        return self.start, self.end
    @property
    def length(self):
        return self.end.distance(self.start)
    def perpendicular_distance(self, other: 'Segment'):
        l1 = other.start.distance(self._projection_point(other, typed="start"))
        l2 = other.end.distance(self._projection_point(other, typed="end"))
        if l1 < self.eps and l2 < self.eps:
            return 0
        else:
            return (math.pow(l1, 2) + math.pow(l2, 2)) / (l1 + l2)
    def parallel_distance(self, other: 'Segment'):
        l1 = self.start.distance(self._projection_point(other, typed='start'))
        l2 = self.end.distance(self._projection_point(other, typed='end'))
        return min(l1, l2)
    def angle_distance(self, other: 'Segment'):
        self_vector = self.end - self.start
        self_dist, other_dist = self.end.distance(self.start), other.end.distance(other.start)
        if self_dist < self.eps:
            return _point2line_distance(self.start.as_array(), other.start.as_array(), other.end.as_array())
        elif other_dist < self.eps:
            return _point2line_distance(other.start.as_array(), self.start.as_array(), self.end.as_array())
        cos_theta = self_vector.dot(other.end - other.start) / (
                    self.end.distance(self.start) * other.end.distance(other.start))
        if cos_theta > self.eps:
            if cos_theta >= 1:
                cos_theta = 1.0
            return other.length * math.sqrt(1 - math.pow(cos_theta, 2))
        else:
            return other.length
    def _projection_point(self, other: 'Segment', typed="e"):
        if typed == 's' or typed == 'start':
            tmp = other.start - self.start
        else:
            tmp = other.end - self.start
        u = tmp.dot(self.end - self.start) / max(math.pow(self.end.distance(self.start), 2), 0.000001)
        return self.start + (self.end - self.start) * u
    def get_all_distance(self, seg: 'Segment'):
        res = self.angle_distance(seg)
        if str(self.start) != str(self.end):
            res += self.parallel_distance(seg)
        if self.traj_id != seg.traj_id:
            res += self.perpendicular_distance(seg)
        return res
def segment_mdl_comp(traj, start_index, current_index, typed='par'):
    length_hypothesis = 0
    length_data_hypothesis_perpend = 0
    length_data_hypothesis_angle = 0
    seg = Segment(traj[start_index], traj[current_index])
    if typed == "par" or typed == "PAR":
        if seg.length < eps:
            length_hypothesis = 0
        else:
            length_hypothesis = math.log2(seg.length)
    for i in range(start_index, current_index, 1):
        sub_seg = Segment(traj[i], traj[i+1])
        if typed == 'par' or typed == 'PAR':
            length_data_hypothesis_perpend += seg.perpendicular_distance(sub_seg)
            length_data_hypothesis_angle += seg.angle_distance(sub_seg)
        elif typed == "nopar" or typed == "NOPAR":
            length_hypothesis += sub_seg.length
    if typed == 'par' or typed == 'PAR':
        if length_data_hypothesis_perpend > eps:
            length_hypothesis += math.log2(length_data_hypothesis_perpend)
        if length_data_hypothesis_angle > eps:
            length_hypothesis += math.log2(length_data_hypothesis_angle)
        return length_hypothesis
    elif typed == "nopar" or typed == "NOPAR":
        if length_hypothesis < eps:
            return 0
        else:
            return math.log2(length_hypothesis)  
    else:
        raise ValueError("The parameter 'typed' given value has error!")
def approximate_trajectory_partitioning(traj, traj_id=None, theta=5.0):
    size = len(traj)
    start_index: int = 0; length: int = 1
    partition_trajectory = []
    while (start_index + length) < size:
        curr_index = start_index + length
        cost_par = segment_mdl_comp(traj, start_index, curr_index, typed='par')
        cost_nopar = segment_mdl_comp(traj, start_index, curr_index, typed='nopar')
        if cost_par > (cost_nopar+theta):
            seg = Segment(traj[start_index], traj[curr_index-1], traj_id=traj_id)
            partition_trajectory.append(seg)
            start_index = curr_index - 1
            length = 1
        else:
            length += 1
    seg = Segment(traj[start_index], traj[size-1], traj_id=traj_id, cluster_id=-1)
    partition_trajectory.append(seg)
    return partition_trajectory
def compare(segment_a: Segment, segment_b: Segment) -> Tuple[Segment, Segment]:
    return (segment_a, segment_b) if segment_a.length > segment_b.length else (segment_b, segment_a)
def neighborhood(seg, segs, epsilon=2.0):
    segment_set = []
    for segment_tmp in segs:
        seg_long, seg_short = compare(seg, segment_tmp)  
        if seg_long.get_all_distance(seg_short) <= epsilon:
            segment_set.append(segment_tmp)
    return segment_set
def expand_cluster(segs, queue: deque, cluster_id: int, epsilon: float, min_lines: int):
    while len(queue) != 0:
        curr_seg = queue.popleft()
        curr_num_neighborhood = neighborhood(curr_seg, segs, epsilon=epsilon)
        if len(curr_num_neighborhood) >= min_lines:
            for m in curr_num_neighborhood:
                if m.cluster_id == -1:
                    queue.append(m)
                    m.cluster_id = cluster_id
        else:
            pass
min_traj_cluster = 2
def line_segment_clustering(traj_segments, epsilon: float = 2.0, min_lines: int = 5):
    cluster_id = 0
    cluster_dict = defaultdict(list)
    for seg in traj_segments:
        _queue = deque(list(), maxlen=50)
        if seg.cluster_id == -1:
            seg_num_neighbor_set = neighborhood(seg, traj_segments, epsilon=epsilon)
            if len(seg_num_neighbor_set) >= min_lines:
                seg.cluster_id = cluster_id
                for sub_seg in seg_num_neighbor_set:
                    sub_seg.cluster_id = cluster_id  
                    _queue.append(sub_seg)  
                expand_cluster(traj_segments, _queue, cluster_id, epsilon, min_lines)
                cluster_id += 1
            else:
                seg.cluster_id = -1
        if seg.cluster_id != -1:
            cluster_dict[seg.cluster_id].append(seg)
    remove_cluster = dict()
    cluster_number = len(cluster_dict)
    for i in range(0, cluster_number):
        traj_num = len(set(map(lambda s: s.traj_id, cluster_dict[i])))
        
        if traj_num < min_traj_cluster:
            remove_cluster[i] = cluster_dict.pop(i)
    return cluster_dict, remove_cluster
def call_traclus(trajs, A):
    traj_set = []
    for ts in trajs:
        traj_set.append([Point(ts[i:i + 2][0], ts[i:i + 2][1]) for i in range(0, len(ts), 2)])
    all_segs = approximate_trajectory_partitioning(traj_set[0], theta=5.0, traj_id=A[0])
    for i in range(1, len(traj_set)):
        part = approximate_trajectory_partitioning(traj_set[i], theta=5.0, traj_id=A[i])
        all_segs += part
    norm_cluster, remove_cluster = line_segment_clustering(all_segs, min_lines=3, epsilon=0.03)
    return norm_cluster
def get_clusters(norm_cluster):
    clusters = []
    traj_cluster_dict = {}
    for nc in range(len(norm_cluster)):
        cluster = []
        for segment in norm_cluster[nc]:
            cluster.append(segment.traj_id)
            if segment.traj_id in traj_cluster_dict:
                traj_cluster_dict[segment.traj_id].add(nc)
            else:
                traj_cluster_dict[segment.traj_id] = set()
                traj_cluster_dict[segment.traj_id].add(nc)
        clusters.append(set(cluster))
    return clusters, traj_cluster_dict
def get_input(traj_db):
    ts = []
    for traj in traj_db:
        ts.append(np.array(traj)[:, 0:2].reshape(1, -1).tolist()[0])
    return ts
def clustering_offline(DB, DB_TREE, query,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range',distri='data'):
    Ts_DB, ID = [], []
    x_length, y_length, t_length = init_query_param(dataset,q_type,distri)
    repeat = {}
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        x_center, y_center, t_center = Xmin + x_length * (0.5 + x_idx), Ymin + y_length * (
                    0.5 + y_idx), Tmin + t_length * (0.5 + t_idx)
        ref_R = DB_TREE.range_query((x_center - x_length / 2,
                                     y_center - y_length / 2,
                                     t_center - t_length / 2,
                                     x_center + x_length / 2,
                                     y_center + y_length / 2,
                                     t_center + t_length / 2))
        A = set([item.object for item in ref_R])
        A = list(A)
        if len(A) > 0:
            ref_DB = get_block_trajs(DB, A, x_center - x_length / 2, y_center - y_length / 2, t_center - t_length / 2,
                                     x_center + x_length / 2, y_center + y_length / 2, t_center + t_length / 2)
            ts_DB = get_input(ref_DB)
            Ts_DB += ts_DB
            ID += A
    norm_cluster_DB = call_traclus(Ts_DB, ID)
    clusters_DB, traj_cluster_dict_DB = get_clusters(norm_cluster_DB)
    return traj_cluster_dict_DB
def clustering_online(traj_cluster_dict_DB, sim_DB, SIMDB_TREE, query,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range',distri='data'):
    F1ALL, Ts_SIMDB, ID = [], [], []
    x_length, y_length, t_length = init_query_param(dataset,q_type,distri)
    repeat = {}
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        if (x_idx, y_idx, t_idx) in repeat:
            continue
        x_center, y_center, t_center = Xmin + x_length * (0.5 + x_idx), Ymin + y_length * (
                    0.5 + y_idx), Tmin + t_length * (0.5 + t_idx)
        sim_R = SIMDB_TREE.range_query((x_center - x_length / 2,
                                        y_center - y_length / 2,
                                        t_center - t_length / 2,
                                        x_center + x_length / 2,
                                        y_center + y_length / 2,
                                        t_center + t_length / 2))
        B = set([item.object for item in sim_R])
        B = list(B)
        if len(B) > 0:
            simDB = get_block_trajs(sim_DB, B, x_center - x_length / 2, y_center - y_length / 2,
                                    t_center - t_length / 2, x_center + x_length / 2, y_center + y_length / 2,
                                    t_center + t_length / 2)
            ts_SIMDB = get_input(simDB)
            Ts_SIMDB += ts_SIMDB
            ID += B
    norm_cluster_simDB = call_traclus(Ts_SIMDB, ID)
    clusters_simDB, traj_cluster_dict_simDB = get_clusters(norm_cluster_simDB)
    CO, CS, COS = 0, 0, 0
    for i in range(len(sim_DB) - 1):
        for j in range(i + 1, len(sim_DB)):
            refind, simind = 0, 0
            if (i in traj_cluster_dict_DB) and (j in traj_cluster_dict_DB):
                if len(traj_cluster_dict_DB[i] & traj_cluster_dict_DB[j]) != 0:
                    refind = 1
            if (i in traj_cluster_dict_simDB) and (j in traj_cluster_dict_simDB):
                if len(traj_cluster_dict_simDB[i] & traj_cluster_dict_simDB[j]) != 0:
                    simind = 1
            CO += refind
            CS += simind
            if refind == 1 and simind == 1:
                COS += 1
    if CS == 0 or CO == 0 or COS==0:
        return 0
    P = COS / CS
    R = COS / CO
    F1 = (2 * P * R) / (P + R)
    F1ALL.append(F1)
    return sum(F1ALL) / len(F1ALL)
def join(Q_sync, D_sync, Q_start, Q_end, eps=0.01):
    for i in range(int(Q_start), int(Q_end)):
        x1 = lonlat2meters(Q_sync[i][0],Q_sync[i][1])
        x2 =  lonlat2meters(D_sync[i][0],D_sync[i][1])
        eul = np.linalg.norm(np.array(x1) - np.array(x2))
        if eul < eps:
            continue
        else:
            return False
    return True
def sync(traj):
    dict_sync = {}
    for i in range(len(traj) - 1):
        ps = traj[i]
        pe = traj[i + 1]
        if pe[2] - ps[2] <= 1:
            dict_sync[ps[2]] = ps[0:2]
            dict_sync[pe[2]] = pe[0:2]
            continue
        else:
            dict_sync[ps[2]] = ps[0:2]
            for i in range(int(ps[2]) + 1, int(pe[2])):
                syn_time = i
                time_ratio = 1 if (pe[2] - ps[2]) == 0 else (syn_time - ps[2]) / (pe[2] - ps[2])
                syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
                syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
                dict_sync[i] = [syn_x, syn_y]
            dict_sync[pe[2]] = pe[0:2]
    return dict_sync
def join_query_operator(ref_DB, sim_DB, Rtree_ref, Rtree_sim, query,Xmin =39.477849, Ymin=115.7097866, Tmin=1176587085,dataset='', q_type='range',distri='data'):
    x_length, y_length, t_length = init_query_param(dataset,q_type,distri)
    F1 = []
    for i in range(len(query)):
        (x_idx, y_idx, t_idx) = query[i]
        x_center, y_center, t_center = Xmin + x_length * (0.5 + x_idx), Ymin + y_length * (
                    0.5 + y_idx), Tmin + t_length * (0.5 + t_idx)
        ref_R = Rtree_ref.range_query((x_center - x_length / 2,
                                       y_center - y_length / 2,
                                       t_center - t_length / 2,
                                       x_center + x_length / 2,
                                       y_center + y_length / 2,
                                       t_center + t_length / 2))
        A = set([item.object for item in ref_R])
        sim_R = Rtree_sim.range_query((x_center - x_length / 2,
                                       y_center - y_length / 2,
                                       t_center - t_length / 2,
                                       x_center + x_length / 2,
                                       y_center + y_length / 2,
                                       t_center + t_length / 2))
        B = set([item.object for item in sim_R])
        A = list(A)
        B = list(B)
        win_ref_DB = get_block_trajs(ref_DB, A, x_center - x_length / 2, y_center - y_length / 2,
                                     t_center - t_length / 2, x_center + x_length / 2, y_center + y_length / 2,
                                     t_center + t_length / 2)
        win_sim_DB = get_block_trajs(sim_DB, B, x_center - x_length / 2, y_center - y_length / 2,
                                     t_center - t_length / 2, x_center + x_length / 2, y_center + y_length / 2,
                                     t_center + t_length / 2)
        ground = set()
        for q_ in range(len(win_ref_DB)):
            Query = win_ref_DB[q_]
            Q_sync = sync(Query)
            Q_start, Q_end = Query[0][2], Query[-1][2]
            for c_ in range(len(win_ref_DB)):
                D1_sync = sync(win_ref_DB[c_])
                D_start, D_end = win_ref_DB[c_][0][2], win_ref_DB[c_][-1][2]
                if (D_start >= Q_start and D_start <= Q_end) or (D_end >= Q_start and D_end <= Q_end):
                    if join(Q_sync, D1_sync, max(Q_start, D_start), min(Q_end, D_end), eps=5000):
                        ground.add(A[c_])
        predict = set()
        for q_ in range(len(win_sim_DB)):
            Query = win_sim_DB[q_]
            Q_sync = sync(Query)
            Q_start, Q_end = Query[0][2], Query[-1][2]
            for c_ in range(len(win_sim_DB)):
                D2_sync = sync(win_sim_DB[c_])
                D_start, D_end = win_sim_DB[c_][0][2], win_sim_DB[c_][-1][2]
                if (D_start >= Q_start and D_start <= Q_end) or (D_end >= Q_start and D_end <= Q_end):
                    if join(Q_sync, D2_sync, max(Q_start, D_start), min(Q_end, D_end), eps=5000):
                        predict.add(B[c_])
        if ground == set() and predict == set():
            F1.append(1.0)
        if ground == set() and predict != set():
            F1.append(0.0)
        if ground != set() and predict == set():
            F1.append(0.0)
        if ground != set() and predict != set():
            P = len(ground & predict) / len(predict)
            R = len(ground & predict) / len(ground)
            if (P + R) == 0:
                F1.append(0.0)
            else:
                F1.append((2 * P * R) / (P + R))
    return sum(F1) / len(F1)
def range_query(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path,  Xmin,Ymin, Tmin,dataset,q_type,distri):
    RES = range_query_operator(DB_TREE, simDB_TREE, query1 + query2, False, Xmin, Ymin, Tmin,dataset,q_type,distri)
    return RES
def knn_edr(DB, DB_TREE, sim_DB, simDB_TREE, query1,query2,gt_path,Xmin,Ymin,Tmin,dataset,q_type,distri):
    edr_name_data = f'_knn_query_edr_{distri}'
    if os.path.exists(gt_path + edr_name_data):
        [GroundQuerySet, interval] = pickle.load(open(gt_path + edr_name_data, 'rb'), encoding='bytes')
    else:
        GroundQuerySet, interval = knn_edr_query_offline(DB, DB_TREE, query1 + query2, Xmin, Ymin, Tmin)
        pickle.dump([GroundQuerySet, interval], open(gt_path + edr_name_data, 'wb'), protocol=2)
    RES = knn_edr_query_online(GroundQuerySet, interval, simDB_TREE, sim_DB, Xmin=Xmin, Ymin=Ymin, Tmin=Tmin,dataset=dataset,q_type=q_type,distri=distri)
    print(f'knn edr query effectiveness ({distri} distribution) f1 = {RES}')
    return RES
def cluster(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path,Xmin,Ymin,Tmin,dataset,q_type,distri):
    clu_name = f'_cluster_query_{distri}'
    if os.path.exists(gt_path + clu_name):
        [traj_clus, query1, query2] = pickle.load(open(gt_path + clu_name, 'rb'), encoding='bytes')
    else:
        traj_clus = clustering_offline(DB, DB_TREE, query1 + query2, Xmin, Ymin, Tmin)
        pickle.dump([traj_clus, query1, query2], open(gt_path + clu_name, 'wb'), protocol=2)
    RES = clustering_online(traj_clus, SimpDB, simDB_TREE, query1 + query2, Xmin, Ymin, Tmin,dataset,q_type,distri)
    print(f'clustering effectiveness ({distri} distribution) f1 = {RES}')
    return RES
def join_query(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, gt_path,  Xmin,Ymin, Tmin,dataset,q_type,distri):
    RES = join_query_operator(DB, SimpDB, DB_TREE, simDB_TREE, query1 + query2, Xmin, Ymin, Tmin,dataset=dataset,q_type=q_type,distri=distri)
    print(f'join(similarity) query effectiveness ({distri} distribution) f1 = {RES}')
    return RES
def api_pre_test(root_data,name,bbox=None,ts_test=None,city=None,q_type='range',distri='data'):
    path_adjust=os.path.join(root_data,name+'-adjust')
    if os.path.exists(path_adjust):return loadZ_pk(path_adjust)
    trajs=ts_test
    Xmin,Ymin,Tmin = bbox[0][0],bbox[1][0],bbox[2][0]
    path_DB_TREE=os.path.join(root_data,name+'-rtree1')
    DB_TREE = build_or_load_Rtree(trajs,path_DB_TREE )
    if distri =='data':
        DB_DISTRI, ID2Grid, DB_DISTRI_trajID = get_distribution_feature_data(trajs,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type)
        gen_distri, query1, query2 = get_query_workload_data(DB_DISTRI)
    if distri =='gau':
        DB_DISTRI, ID2Grid, DB_DISTRI_trajID = get_distribution_feature_gau(trajs,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type)
        gen_distri, query1, query2 = get_query_workload_gau(DB_DISTRI)
    adjust = range_query_adjust(DB_TREE,query1+query2,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type,distri=distri)
    adjust_tensor = torch.zeros((len(trajs),len(trajs[0]))).to(device)
    for i in adjust:
        for j in i:
            adjust_tensor[j[0],j[1]]= 1
    adjust_tensor /= adjust_tensor.sum()
    res=[adjust_tensor,query1,query2]
    saveZ_pk(path_adjust,res)
    return res
def api_pre_test2(root_data,nameQ,nameD,bbox=None,tsQ=None,tsD=None,city=None,q_type='range',distri='data'):
    if nameQ==nameD:return api_pre_test(root_data,nameQ,bbox,tsQ,city,q_type,distri)
    path_adjust=os.path.join(root_data,f'{nameQ}-{nameD}-adjust')
    if os.path.exists(path_adjust):return loadZ_pk(path_adjust)
    Xmin,Ymin,Tmin = bbox[0][0],bbox[1][0],bbox[2][0]
    path_DB_TREE=os.path.join(root_data,f'{nameD}-rtree2')
    if os.path.exists(path_DB_TREE+'.dat'): DB_TREE = Rtree(path_DB_TREE)
    else:DB_TREE=build_or_load_Rtree(tsD,path_DB_TREE )
    if distri =='data':
        DB_DISTRI, ID2Grid, DB_DISTRI_trajID = get_distribution_feature_data(tsQ,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type)
        gen_distri, query1, query2 = get_query_workload_data(DB_DISTRI)
    if distri =='gau':
        DB_DISTRI, ID2Grid, DB_DISTRI_trajID = get_distribution_feature_gau(tsQ,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type)
        gen_distri, query1, query2 = get_query_workload_gau(DB_DISTRI)
    adjust = range_query_adjust(DB_TREE,query1+query2,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type,distri=distri)
    adjust_tensor = torch.zeros((len(tsQ),len(tsQ[0]))).to(device)
    for i in adjust:
        for j in i:
            adjust_tensor[j[0],j[1]]= 1
    adjust_tensor /= adjust_tensor.sum()
    res=[adjust_tensor,query1,query2]
    saveZ_pk(path_adjust,res)
    return res
def validation(gnn:GAT,root_data,nameQ,nameD,tsQ,tsD,city,bbox,cr=0.0025,q_type='range',distri='data'):
    trajs=tsQ
    adjust_tensor,query1,query2=api_pre_test(root_data,nameQ,bbox=bbox,ts_test=tsQ,city=city)
    graph_test_dataset=api_pre_gnndata(root_data,nameQ)
    simptime_start = time.time()
    trajs_score = simp(gnn,graph_test_dataset)
    trajs_score /=trajs_score.sum()
    simptime_end = time.time()
    simptime = simptime_end-simptime_start
    ad_param =  0.5
    final_score = (1-ad_param) * trajs_score+ ad_param * adjust_tensor
    final_score = final_score.cpu().detach().numpy()
    sampletime_start = time.time()
    sum = int(cr * (final_score.shape[0] * final_score.shape[1]))
    mask = np.zeros((final_score.shape[0], final_score.shape[1]), dtype=bool)
    sample_indices = np.random.choice(final_score.shape[0] * final_score.shape[1], size=sum, p=final_score.flatten(),replace=False)
    mask.flat[sample_indices] = True
    mask = mask.reshape((final_score.shape[0], final_score.shape[1]))
    mask[:, 0] = True
    mask[:, -1] = True
    simp_trajs = []
    for i in range(trajs.shape[0]):
        simp_traj = trajs[i][mask[i]].tolist()
        for point in simp_traj:
            point[2] = int(point[2])
        simp_trajs.append(simp_traj)
    sampletime_end = time.time()
    path_gt        =os.path.join(root_data,'gt-'+q_type)
    path_DB_TREE   =''
    path_simDB_TREE=''
    DB,SimpDB=trajs,simp_trajs
    DB_TREE = build_Rtree(DB, path_DB_TREE)
    simDB_TREE = build_Rtree(SimpDB,path_simDB_TREE)
    Xmin,Ymin,Tmin = bbox[0][0],bbox[1][0],bbox[2][0] 
    if q_type == 'range':
        res=range_query(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, path_gt,  Xmin=Xmin,Ymin=Ymin, Tmin=Tmin,dataset=city,q_type=q_type,distri=distri)
    elif q_type == 'knn':
        res=knn_edr(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, path_gt, Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type,distri=distri)
    elif q_type == 'cluster':
        res=cluster(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, path_gt,Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type,distri=distri)
    elif q_type == 'join':
        res=join_query(DB, DB_TREE, SimpDB, simDB_TREE, query1, query2, path_gt, Xmin=Xmin, Ymin=Ymin,Tmin=Tmin,dataset=city,q_type=q_type,distri=distri)
    del simDB_TREE
    del DB_TREE
    return res
def val_sed_wQ(gnn:GAT,root_data,nameQ,nameD,tsQ,tsD,city,bbox,cr=0.0025):
    trajs=tsQ
    adjust_tensor,query1,query2=api_pre_test(root_data,nameQ,bbox,ts_test=tsQ,city=city)
    graph_test_dataset=api_pre_gnndata(root_data,nameQ)
    simptime_start = time.time()
    trajs_score = simp(gnn,graph_test_dataset)
    trajs_score /=trajs_score.sum()
    simptime_end = time.time()
    simptime = simptime_end-simptime_start
    ad_param =  0.5
    final_score = (1-ad_param) * trajs_score+ ad_param * adjust_tensor
    final_score = final_score.cpu().detach().numpy()
    sum = int(cr * (final_score.shape[0] * final_score.shape[1]))
    mask = np.zeros((final_score.shape[0], final_score.shape[1]), dtype=bool)
    sample_indices = np.random.choice(final_score.shape[0] * final_score.shape[1], size=sum, p=final_score.flatten(),replace=False)
    mask.flat[sample_indices] = True
    mask = mask.reshape((final_score.shape[0], final_score.shape[1]))
    mask[:, 0] = True
    mask[:, -1] = True
    simp_trajs = []
    for i in range(trajs.shape[0]):
        simp_traj = trajs[i][mask[i]].tolist()
        for point in simp_traj:
            point[2] = int(point[2])
        simp_trajs.append(simp_traj)
    errs=[]
    for t1,t2 in zip(trajs,simp_trajs):
        err=SED_fast(np.array(t1),np.array(t2))
        errs.append(err)
    res=np.array(errs).mean()
    return res.item()
def _val_sed(gnn:GAT,root_data,name,ts_test,cr=0.0025):
    trajs=ts_test
    graph_test_dataset=api_pre_gnndata(root_data,name)
    final_score = simp(gnn,graph_test_dataset)
    final_score /=final_score.sum()
    final_score = final_score.cpu().detach().numpy()
    sum = int(cr * (final_score.shape[0] * final_score.shape[1]))
    mask = np.zeros((final_score.shape[0], final_score.shape[1]), dtype=bool)
    sample_indices = np.random.choice(final_score.shape[0] * final_score.shape[1], size=sum, p=final_score.flatten(),replace=False)
    mask.flat[sample_indices] = True
    mask = mask.reshape((final_score.shape[0], final_score.shape[1]))
    mask[:, 0] = True
    mask[:, -1] = True
    simp_trajs = []
    for i in range(trajs.shape[0]):
        simp_traj = trajs[i][mask[i]].tolist()
        for point in simp_traj:
            point[2] = int(point[2])
        simp_trajs.append(simp_traj)
    errs=[]
    for t1,t2 in zip(trajs,simp_trajs):
        err=SED_fast(np.array(t1),np.array(t2))
        errs.append(err)
    res=np.array(errs).mean()
    return res
