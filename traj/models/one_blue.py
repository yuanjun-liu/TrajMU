exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import numpy as np
import datetime
import torch,math,os
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from _nn.nData import random_seed,auto_device
from _tool.mIO import loadZ_pk,saveZ_pk,loadZ_th,saveZ_th
import time
import pyproj
from typing import List,Tuple
import pandas as pd
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, f1_score, accuracy_score, precision_score
device=auto_device()
random_seed(42)
num_workers=0 
def time_features(time_seq):
    """
    Parameters:
    time_seq: array-like, 
    Returns:
    numpy.ndarray:  (6, len(time_seq)) 
        - day_of_year:  [-0.5, 0.5]
        - day_of_month:  [-0.5, 0.5]  
        - day_of_week:  [-0.5, 0.5]
        - hour_of_day:  [-0.5, 0.5]
        - minute_of_hour:  [-0.5, 0.5]
        - second_of_minute:  [-0.5, 0.5]
    """
    dates = pd.to_datetime(time_seq, unit='s')
    day_of_year = (dates.dayofyear - 1) / 365.0 - 0.5
    day_of_month = (dates.day - 1) / 30.0 - 0.5
    day_of_week = dates.dayofweek / 6.0 - 0.5
    hour_of_day = dates.hour / 23.0 - 0.5
    minute_of_hour = dates.minute / 59.0 - 0.5
    second_of_minute = dates.second / 59.0 - 0.5
    return np.vstack([
        day_of_year,
        day_of_month, 
        day_of_week,
        hour_of_day,
        minute_of_hour,
        second_of_minute
])
def _get_points_feature(gps_seq, time_seq, geodesic):
    assert len(gps_seq) == len(time_seq)
    lons = gps_seq[:, 0]
    lats = gps_seq[:, 1]
    az1, az2, dist = geodesic.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    fwd_az = np.concatenate([az1, np.array([0])])
    bk_az = np.concatenate([np.array([0]), az2])
    fwd_dist = np.concatenate([dist, np.array([0])])
    bk_dist = np.concatenate([np.array([0]), dist])
    return fwd_dist, bk_dist, fwd_az, bk_az
def _get_multi_scale_traj(gps_seq, scale=3):
    gps_seq = np.round(gps_seq, decimals=scale)
    diff_mask = np.any(gps_seq[1:] != gps_seq[:-1], axis=1)
    start_index = np.where(diff_mask)[0] + 1
    patch_lens = np.diff(np.concatenate(([0], start_index, [len(gps_seq)])))
    unique_index = np.insert(diff_mask, 0, True)
    raw_index = np.where(unique_index)[0]
    gps_seq = gps_seq[raw_index]
    return gps_seq, patch_lens
def process_one(t,lb,geodesic):
    time_seq = t[:,2]
    gps_seq = t[:,:2]
    fwd_dist, bk_dist, fwd_az, bk_az = _get_points_feature(gps_seq, time_seq, geodesic)
    gps_seq_s3, patch_lens_s3 = _get_multi_scale_traj(gps_seq, scale=3)
    gps_seq_s2, patch_lens_s2 = _get_multi_scale_traj(gps_seq_s3, scale=2)
    return {
        'class_type': lb,
        'traj_len_s5': len(t),
        'gps_seq': gps_seq,
        'time_seq': time_seq,
        'time_fea': time_features(time_seq).T,
        'fwd_dist': fwd_dist,
        'bk_dist': bk_dist,
        'fwd_az': fwd_az,
        'bk_az': bk_az,
        'traj_len_s2': gps_seq_s2.shape[0],
        'patch_len_s2': patch_lens_s2,
        'traj_len_s3': gps_seq_s3.shape[0],
        'patch_len_s3': patch_lens_s3
    }
def preprocess(ts,lb): 
    geodesic = pyproj.Geod(ellps='WGS84')
    total_data = []
    for ti,t in enumerate(ts):
        total_data.append(process_one(t,lb[ti],geodesic))
    return total_data    
def api_prerpocess(root_data,tss=None,lbs=None,bbox=None):
    """tss[train,test,val,test_noise], mbr"""
    path=os.path.join(root_data,f'pre.pk.zst')
    if os.path.exists(path):
        return loadZ_pk(path)
    assert tss is not None and lbs is not None and bbox is not None
    tss=[preprocess(ts,lb) for ts,lb in zip(tss,lbs)]
    mbr={'min_lon':bbox[0][0],'max_lon':bbox[0][1],'min_lat':bbox[1][0],'max_lat':bbox[1][1]}
    saveZ_pk(path,[tss,mbr])
    return tss,mbr
class TrajDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]
def collate_fn_pretrain(batch_data_list, mbr):
    batch_len = len(batch_data_list)
    traj_len_s5 = [data['traj_len_s5'] for data in batch_data_list]
    max_len_s5 = max(traj_len_s5)
    traj_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    mask_y = torch.zeros((batch_len, max_len_s5), dtype=torch.float32)
    time_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    patch_len_s2 = []
    patch_len_s3 = []
    traj_len_s2 = []
    traj_len_s3 = []
    for k, data in enumerate(batch_data_list):
        traj_len_s5_k = data['traj_len_s5']
        gps_seq = torch.as_tensor(data['gps_seq'], dtype=torch.float32)
        gps_seq[:, 0] = (gps_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        gps_seq[:, 1] = (gps_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])
        time_fea = torch.as_tensor(data['time_fea'], dtype=torch.float32)
        fwd_dist = torch.as_tensor(data['fwd_dist'] / 1000, dtype=torch.float32)
        bk_dist = torch.as_tensor(data['bk_dist'] / 1000, dtype=torch.float32)
        fwd_az = torch.as_tensor(data['fwd_az'] / 180, dtype=torch.float32)
        bk_az = torch.as_tensor(data['bk_az'] / 180, dtype=torch.float32)
        traj_k = torch.cat([gps_seq, fwd_dist.unsqueeze(-1), bk_dist.unsqueeze(-1), fwd_az.unsqueeze(-1), bk_az.unsqueeze(-1)], axis=1)
        traj_x[k, :traj_len_s5_k] = traj_k
        mask_y[k, :traj_len_s5_k] = 1.0
        time_x[k, :traj_len_s5_k] = time_fea
        patch_len_s2_k = data['patch_len_s2']
        traj_len_s2.append(len(patch_len_s2_k))
        patch_len_s3_k = data['patch_len_s3']
        traj_len_s3.append(len(patch_len_s3_k))
        patch_len_s2.extend(patch_len_s2_k)
        patch_len_s3.extend(patch_len_s3_k)
    data = {
        'x': traj_x,
        'mask_y': mask_y,
        'time_x': time_x,
        'traj_len_s5': torch.as_tensor(traj_len_s5, dtype=torch.long),
        'patch_len_s3': torch.as_tensor(patch_len_s3, dtype=torch.long),
        'traj_len_s3': torch.as_tensor(traj_len_s3, dtype=torch.long),
        'patch_len_s2': torch.as_tensor(patch_len_s2, dtype=torch.long),
        'traj_len_s2': torch.as_tensor(traj_len_s2, dtype=torch.long),
    }
    return data
def collate_fn_inference(batch_data_list, mbr):
    batch_len = len(batch_data_list)
    traj_len_s5 = [data['traj_len_s5'] for data in batch_data_list]
    max_len_s5 = max(traj_len_s5)
    traj_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    time_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    patch_len_s2 = []
    patch_len_s3 = []
    traj_len_s2 = []
    traj_len_s3 = []
    for k, data in enumerate(batch_data_list):
        traj_len_s5_k = data['traj_len_s5']
        gps_seq = torch.as_tensor(data['gps_seq'], dtype=torch.float32)
        gps_seq[:, 0] = (gps_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        gps_seq[:, 1] = (gps_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])
        time_fea = torch.as_tensor(data['time_fea'], dtype=torch.float32)
        fwd_dist = torch.as_tensor(data['fwd_dist'] / 1000, dtype=torch.float32)
        bk_dist = torch.as_tensor(data['bk_dist'] / 1000, dtype=torch.float32)
        fwd_az = torch.as_tensor(data['fwd_az'] / 180, dtype=torch.float32)
        bk_az = torch.as_tensor(data['bk_az'] / 180, dtype=torch.float32)
        traj_k = torch.cat([gps_seq, fwd_dist.unsqueeze(-1), bk_dist.unsqueeze(-1), fwd_az.unsqueeze(-1), bk_az.unsqueeze(-1)], axis=1)
        traj_x[k, :traj_len_s5_k] = traj_k
        time_x[k, :traj_len_s5_k] = time_fea
        patch_len_s2_k = data['patch_len_s2']
        traj_len_s2.append(len(patch_len_s2_k))
        patch_len_s3_k = data['patch_len_s3']
        traj_len_s3.append(len(patch_len_s3_k))
        patch_len_s2.extend(patch_len_s2_k)
        patch_len_s3.extend(patch_len_s3_k)
    data = {
        'x': traj_x,
        'time_x': time_x,
        'traj_len_s5': torch.as_tensor(traj_len_s5, dtype=torch.long),
        'patch_len_s3': torch.as_tensor(patch_len_s3, dtype=torch.long),
        'traj_len_s3': torch.as_tensor(traj_len_s3, dtype=torch.long),
        'patch_len_s2': torch.as_tensor(patch_len_s2, dtype=torch.long),
        'traj_len_s2': torch.as_tensor(traj_len_s2, dtype=torch.long),
    }
    return data
def collate_fn_cls(batch_data_list, mbr):
    batch_len = len(batch_data_list)
    traj_len_s5 = [data['traj_len_s5'] for data in batch_data_list]
    max_len_s5 = max(traj_len_s5)
    traj_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    time_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    patch_len_s2 = []
    patch_len_s3 = []
    traj_len_s2 = []
    traj_len_s3 = []
    labels = []
    for k, data in enumerate(batch_data_list):
        traj_len_s5_k = data['traj_len_s5']
        gps_seq = torch.as_tensor(data['gps_seq'], dtype=torch.float32)
        gps_seq[:, 0] = (gps_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        gps_seq[:, 1] = (gps_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])
        time_fea = torch.as_tensor(data['time_fea'], dtype=torch.float32)
        fwd_dist = torch.as_tensor(data['fwd_dist'] / 1000, dtype=torch.float32)
        bk_dist = torch.as_tensor(data['bk_dist'] / 1000, dtype=torch.float32)
        fwd_az = torch.as_tensor(data['fwd_az'] / 180, dtype=torch.float32)
        bk_az = torch.as_tensor(data['bk_az'] / 180, dtype=torch.float32)
        traj_k = torch.cat([gps_seq, fwd_dist.unsqueeze(-1), bk_dist.unsqueeze(-1), fwd_az.unsqueeze(-1), bk_az.unsqueeze(-1)], axis=1)
        traj_x[k, :traj_len_s5_k] = traj_k
        time_x[k, :traj_len_s5_k] = time_fea
        patch_len_s2_k = data['patch_len_s2']
        traj_len_s2.append(len(patch_len_s2_k))
        patch_len_s3_k = data['patch_len_s3']
        traj_len_s3.append(len(patch_len_s3_k))
        patch_len_s2.extend(patch_len_s2_k)
        patch_len_s3.extend(patch_len_s3_k)
        labels.append(data['class_type'])
    data = {
        'x': traj_x,
        'time_x': time_x,
        'traj_len_s5': torch.as_tensor(traj_len_s5, dtype=torch.long),
        'patch_len_s3': torch.as_tensor(patch_len_s3, dtype=torch.long),
        'traj_len_s3': torch.as_tensor(traj_len_s3, dtype=torch.long),
        'patch_len_s2': torch.as_tensor(patch_len_s2, dtype=torch.long),
        'traj_len_s2': torch.as_tensor(traj_len_s2, dtype=torch.long),
    }
    return data, torch.as_tensor(labels, dtype=torch.long)
def collate_fn_tte(batch_data_list, mbr):
    batch_len = len(batch_data_list)
    traj_len_s5 = [data['traj_len_s5'] for data in batch_data_list]
    max_len_s5 = max(traj_len_s5)
    traj_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    time_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    patch_len_s2 = []
    patch_len_s3 = []
    traj_len_s2 = []
    traj_len_s3 = []
    labels = []
    for k, data in enumerate(batch_data_list):
        traj_len_s5_k = data['traj_len_s5']
        gps_seq = torch.as_tensor(data['gps_seq'], dtype=torch.float32)
        gps_seq[:, 0] = (gps_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        gps_seq[:, 1] = (gps_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])
        time_seq = data['time_seq']
        labels.append(time_seq[-1] - time_seq[0])
        time_seq = [datetime.fromtimestamp(t) for t in data['time_seq']]
        time_fea = torch.as_tensor(data['time_fea'], dtype=torch.float32)
        fwd_dist = torch.as_tensor(data['fwd_dist'] / 1000, dtype=torch.float32)
        bk_dist = torch.as_tensor(data['bk_dist'] / 1000, dtype=torch.float32)
        fwd_az = torch.as_tensor(data['fwd_az'] / 180, dtype=torch.float32)
        bk_az = torch.as_tensor(data['bk_az'] / 180, dtype=torch.float32)
        traj_k = torch.cat([gps_seq, fwd_dist.unsqueeze(-1), bk_dist.unsqueeze(-1), fwd_az.unsqueeze(-1), bk_az.unsqueeze(-1)], axis=1)
        traj_x[k, :traj_len_s5_k] = traj_k
        time_x[k, 0] = time_fea[0]
        time_x[k, 1:traj_len_s5_k] = 0.5  
        patch_len_s2_k = data['patch_len_s2']
        traj_len_s2.append(len(patch_len_s2_k))
        patch_len_s3_k = data['patch_len_s3']
        traj_len_s3.append(len(patch_len_s3_k))
        patch_len_s2.extend(patch_len_s2_k)
        patch_len_s3.extend(patch_len_s3_k)
    data = {
        'x': traj_x,
        'time_x': time_x,
        'traj_len_s5': torch.as_tensor(traj_len_s5, dtype=torch.long),
        'patch_len_s3': torch.as_tensor(patch_len_s3, dtype=torch.long),
        'traj_len_s3': torch.as_tensor(traj_len_s3, dtype=torch.long),
        'patch_len_s2': torch.as_tensor(patch_len_s2, dtype=torch.long),
        'traj_len_s2': torch.as_tensor(traj_len_s2, dtype=torch.long),
    }
    return data, torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1) / 60
def get_dataloader(task,ts_processed,mbr,is_train): 
    """ts must preprocess advance -> DataLoader"""
    dataset=TrajDataset(ts_processed)
    fn={'pretrain':collate_fn_pretrain,'tte':collate_fn_tte,'cls':collate_fn_cls,'sim':collate_fn_inference}[task]
    return DataLoader(dataset, batch_size=256, shuffle=is_train, drop_last=is_train, pin_memory=True, num_workers=num_workers, collate_fn=lambda x: fn(x,mbr))
class PosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.shape[-2]]
class SpaEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.linear = nn.Linear(c_in, d_model)
    def forward(self, x):
        return self.linear(x)
class TemporalEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.te_scale = nn.Linear(c_in, d_model // 2)
        self.te_periodic = nn.Linear(c_in, d_model // 2)
    def forward(self, t):
        out1 = self.te_scale(t)
        out2 = torch.sin(self.te_periodic(t))
        return torch.cat([out1, out2], -1)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        d_model = 128
        n_heads = 4
        d_ff = 512
        dropout = 0.1
        n_layers_s2 = 2
        n_layers_s3 = 4
        n_layers_s5 = 2
        layer_s2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_s2 = nn.TransformerEncoder(
            encoder_layer=layer_s2,
            num_layers=n_layers_s2
        )
        self.decoder_up_s3_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_up_s3_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_s3 = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers_s3
        )
        self.decoder_up_s5_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_up_s5_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_s5 = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=n_layers_s5
        )
    def forward(self, x_s5, x_s3, x_s2, traj_len_s5, traj_len_s3, traj_len_s2):
        max_len = traj_len_s2.max().item()
        padding_mask_s2 = torch.arange(max_len, device=x_s2.device)[None, :] >= traj_len_s2[:, None]
        max_len = traj_len_s3.max().item()
        padding_mask_s3 = torch.arange(max_len, device=x_s2.device)[None, :] >= traj_len_s3[:, None]
        max_len = traj_len_s5.max().item()
        padding_mask_s5 = torch.arange(max_len, device=x_s2.device)[None, :] >= traj_len_s5[:, None]
        x_s2 = self.decoder_s2(x_s2, src_key_padding_mask=padding_mask_s2)
        x_s3_up = self.decoder_up_s3_cross_attn(query=x_s3, key=x_s2, value=x_s2, key_padding_mask=padding_mask_s2)[0]
        x_s3_up = self.decoder_up_s3_attn(query=x_s3_up, key=x_s3_up, value=x_s3_up, key_padding_mask=padding_mask_s3)[0]
        x_s3 = self.decoder_s3(x_s3_up + x_s3, src_key_padding_mask=padding_mask_s3)
        x_s5_up = self.decoder_up_s5_cross_attn(query=x_s5, key=x_s3, value=x_s3, key_padding_mask=padding_mask_s3)[0]
        x_s5_up = self.decoder_up_s5_attn(query=x_s5_up, key=x_s5_up, value=x_s5_up, key_padding_mask=padding_mask_s5)[0]
        x_s5 = self.decoder_s5(x_s5_up + x_s5, src_key_padding_mask=padding_mask_s5)
        return x_s5
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        d_model = 128
        n_heads = 4
        d_ff = 512
        dropout = 0.1
        n_layers_s5 = 2
        n_layers_s3 = 4
        n_layers_s2 = 2
        self.cls = True
        self.cls_token = None
        if self.cls is True:
            self.cls_token = nn.Parameter(torch.randn(d_model))
        self.pe = PosEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
        layer_s5 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_s5 = nn.TransformerEncoder(
            encoder_layer=layer_s5,
            num_layers=n_layers_s5
        )
        layer_s3 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_s3 = nn.TransformerEncoder(
            encoder_layer=layer_s3,
            num_layers=n_layers_s3
        )
        layer_s2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_s2 = nn.TransformerEncoder(encoder_layer=layer_s2, num_layers=n_layers_s2)
        self.patch_attn_s3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )
        self.patch_attn_s2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )
    def forward(self, x, traj_len_s5, traj_len_s3, patch_len_s3, traj_len_s2, patch_len_s2):
        _, _, d_model = x.shape
        if self.cls_token is not None:
            cls_token = self.cls_token.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([cls_token, x], dim=1)
            traj_len_s5 += 1
        max_len = traj_len_s5.max().item()
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= traj_len_s5[:, None]
        x = x + self.pe(x)
        x_s5 = self.encoder_s5(x, src_key_padding_mask=padding_mask)
        if self.cls is True:
            x_s5_wo_cls = x_s5[:, 1:]
            padding_mask = padding_mask[:, 1:]
            cls_token = x_s5[:, 0]
        else:
            x_s5_wo_cls = x_s5
            cls_token = None
        padding_mask = padding_mask.reshape(-1)
        x_s5_flatten = x_s5_wo_cls.reshape(-1, x_s5_wo_cls.size(-1))
        x_s5_flatten = x_s5_flatten[~padding_mask]
        patch_offsets = torch.cumsum(torch.tensor([0] + patch_len_s3.tolist()[:-1], device=x.device), dim=0)
        patch_indices = torch.repeat_interleave(torch.arange(patch_len_s3.shape[0], device=x.device), patch_len_s3)
        max_patch_len_s3 = patch_len_s3.max().item()
        x_s3_padded = x.new_zeros(patch_len_s3.shape[0], max_patch_len_s3, d_model)
        x_s3_padded[patch_indices, torch.arange(x_s5_flatten.size(0), device=x.device) - patch_offsets[patch_indices]] = x_s5_flatten
        attn_mask = torch.arange(max_patch_len_s3, device=x.device)[None, :] >= patch_len_s3[:, None]
        w = self.patch_attn_s3(x_s3_padded)
        w[attn_mask] = float('-inf')
        w = torch.softmax(w, dim=1)
        x_s3 = torch.sum(w * x_s3_padded, dim=1)
        del patch_offsets, patch_indices
        traj_cumsum = torch.cumsum(torch.tensor([0] + traj_len_s3[:-1].tolist(), device=x.device), dim=0)
        traj_indices = torch.repeat_interleave(torch.arange(traj_len_s3.shape[0], device=x.device), traj_len_s3)
        max_len = traj_len_s3.max().item()
        x_s3_aligned = x.new_zeros(traj_len_s3.shape[0], traj_len_s3.max().item(), d_model)
        x_s3_aligned[traj_indices, torch.arange(x_s3.shape[0], device=x.device) - traj_cumsum[traj_indices]] = x_s3
        del traj_cumsum, traj_indices
        if cls_token is not None:
            max_len += 1
            traj_len_s3 += 1
            x_s3 = torch.cat([cls_token.unsqueeze(1), x_s3_aligned], dim=1)
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= traj_len_s3[:, None]
        x_s3 = x_s3 + self.pe(x_s3)
        x_s3 = self.encoder_s3(x_s3, src_key_padding_mask=padding_mask)
        if self.cls is True:
            x_s3_wo_cls = x_s3[:, 1:]
            padding_mask = padding_mask[:, 1:]
            cls_token = x_s3[:, 0]
        else:
            x_s3_wo_cls = x_s3
            cls_token = None
        padding_mask = padding_mask.reshape(-1)
        x_s3_flatten = x_s3_wo_cls.reshape(-1, x_s3_wo_cls.size(-1))
        x_s3_flatten = x_s3_flatten[~padding_mask]
        patch_offsets = torch.cumsum(torch.tensor([0] + patch_len_s2.tolist()[:-1], device=x.device), dim=0)
        patch_indices = torch.repeat_interleave(torch.arange(patch_len_s2.shape[0], device=x.device), patch_len_s2)
        max_patch_len_s2 = patch_len_s2.max().item()
        x_s2_padded = x.new_zeros(patch_len_s2.shape[0], max_patch_len_s2, d_model)
        x_s2_padded[patch_indices, torch.arange(x_s3_flatten.size(0), device=x.device) - patch_offsets[patch_indices]] = x_s3_flatten
        attn_mask = torch.arange(max_patch_len_s2, device=x.device)[None, :] >= patch_len_s2[:, None]
        w = self.patch_attn_s2(x_s2_padded)
        w[attn_mask] = float('-inf')
        w = torch.softmax(w, dim=1)
        x_s2 = torch.sum(w * x_s2_padded, dim=1)
        del patch_offsets, patch_indices
        traj_cumsum = torch.cumsum(torch.tensor([0] + traj_len_s2[:-1].tolist(), device=x.device), dim=0)
        traj_indices = torch.repeat_interleave(torch.arange(traj_len_s2.shape[0], device=x.device), traj_len_s2)
        max_len = traj_len_s2.max().item()
        x_s2_aligned = x.new_zeros(traj_len_s2.shape[0], max_len, d_model)
        x_s2_aligned[traj_indices, torch.arange(x_s2.shape[0], device=x.device) - traj_cumsum[traj_indices]] = x_s2
        del traj_cumsum, traj_indices
        if cls_token is not None:
            max_len += 1
            traj_len_s2 += 1
            x_s2 = torch.cat([cls_token.unsqueeze(1), x_s2_aligned], dim=1)
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= traj_len_s2[:, None]
        assert ((~padding_mask).sum(dim=1) == traj_len_s2).all().item() is True
        x_s2 = x_s2 + self.pe(x_s2)
        x_s2 = self.encoder_s2(x_s2, src_key_padding_mask=padding_mask)
        return x_s5, x_s3, x_s2
class Net(nn.Module):
    def __init__(self,):
        super().__init__()
        self.cls = True
        self.spa_emb = SpaEmbedding(6, 128)
        self.time_emb = TemporalEmbedding(6, 128)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.predictor_spa = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )
        self.predictor_time = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )
    def forward_encoder(self, data):
        x = data['x']
        time_x = data['time_x']
        traj_len_s5 = data['traj_len_s5']
        patch_len_s2 = data['patch_len_s2']
        traj_len_s2 = data['traj_len_s2']
        patch_len_s3 = data['patch_len_s3']
        traj_len_s3 = data['traj_len_s3']
        x = self.spa_emb(x)
        tx = self.time_emb(time_x)
        x = x + tx
        x_s5, x_s3, x_s2 = self.encoder(x, traj_len_s5, traj_len_s3, patch_len_s3, traj_len_s2, patch_len_s2)
        return x_s2[:, 0]
    def forward_loss(self, data):
        x = data['x']
        mask_y = data['mask_y']
        time_x = data['time_x']
        traj_len_s5 = data['traj_len_s5']
        patch_len_s2 = data['patch_len_s2']
        traj_len_s2 = data['traj_len_s2']
        patch_len_s3 = data['patch_len_s3']
        traj_len_s3 = data['traj_len_s3']
        x = self.spa_emb(x)
        tx = self.time_emb(time_x)
        x = x + tx
        x_s5, x_s3, x_s2 = self.encoder(x, traj_len_s5, traj_len_s3, patch_len_s3, traj_len_s2, patch_len_s2)
        x_s5_out = self.decoder(x_s5, x_s3, x_s2, traj_len_s5, traj_len_s3, traj_len_s2)
        x_s5_out = x_s5_out[:, 1:]
        mask_y = mask_y.reshape(-1).bool()
        x_s5_out = x_s5_out.reshape(-1, x_s5_out.shape[-1])
        x_s5_out = x_s5_out[mask_y]
        spa_y = data['x']
        spa_y = spa_y.reshape(-1, spa_y.shape[-1])
        spa_y = spa_y[mask_y]
        spa_y_hat = self.predictor_spa(x_s5_out)
        spa_loss = F.mse_loss(spa_y_hat, spa_y)
        time_y = data['time_x']
        time_y = time_y.reshape(-1, time_y.shape[-1])
        time_y = time_y[mask_y]
        time_y_hat = self.predictor_time(x_s5_out)
        time_loss = F.mse_loss(time_y_hat, time_y)
        return spa_loss, time_loss
@torch.no_grad()
def eval_train( ep, model, data_loader):
    model.eval()
    epoch_loss = []
    epoch_spa_loss = []
    epoch_time_loss = []
    for iter_idx, batch_data in enumerate(data_loader):
        for k, v in batch_data.items():
            batch_data[k] = v.cuda(non_blocking=True)
        spa_loss, time_loss = model.forward_loss(batch_data)
        loss = spa_loss + time_loss
        print(f"Epoch: {ep} | loss: {loss:.8f}")
        epoch_loss.append(loss.item())
        epoch_spa_loss.append(spa_loss.item())
        epoch_time_loss.append(time_loss.item())
    epoch_loss = np.mean(np.array(epoch_loss))
    epoch_spa_loss = np.mean(np.array(epoch_spa_loss))
    epoch_time_loss = np.mean(np.array(epoch_time_loss))
    return epoch_loss, epoch_spa_loss, epoch_time_loss
def get_optimizer(parameters):
    return torch.optim.Adam(params=parameters, lr=1e-4)
def get_scheduler(optimizer,t=5):
    return CosineLRScheduler(optimizer=optimizer, t_initial=30, warmup_t=t, warmup_lr_init=1e-6, lr_min=1e-6)
def save_checkpoint(save_path, model, optim, sched): 
    checkpoint = {
        'model': model.state_dict(),
    }
    saveZ_th(save_path,checkpoint)
def load_checkpoint(load_path, model, optim, sched):
    checkpoint=loadZ_th(load_path)
    model.load_state_dict(checkpoint['model'])
    if optim: optim.load_state_dict(checkpoint['optim'])
    if sched: sched.load_state_dict(checkpoint['sched'])
def api_pretrain(root_data,root_model):
    model = Net().to(device)
    optim = get_optimizer(model.parameters())
    sched = get_scheduler(optimizer=optim)
    path=os.path.join(root_model,f'pretrain.th.zst')
    if os.path.exists(path):
        load_checkpoint(path, model, optim, sched)
        model.to(device=device)
    tss,mbr=api_prerpocess(root_data)
    train_loader, eval_loader = get_dataloader('pretrain',tss[0],mbr,is_train=True),get_dataloader('pretrain',tss[2],mbr,is_train=True) 
    for ep in range(30):
        epoch_loss = []
        epoch_spa_loss = []
        epoch_time_loss = []
        model.train()
        sched.step(ep)
        for iter_idx, batch_data in enumerate(train_loader):
            for k, v in batch_data.items():
                batch_data[k] = v.to(device,non_blocking=True)
            optim.zero_grad()
            spa_loss, time_loss = model.forward_loss(batch_data)
            loss = spa_loss + time_loss
            loss.backward()
            optim.step()
            print(f"Epoch: {ep} | loss: {loss:.8f}")
            epoch_loss.append(loss.item())
            epoch_spa_loss.append(spa_loss.item())
            epoch_time_loss.append(time_loss.item())
            del batch_data
            torch.cuda.empty_cache();torch.mps.empty_cache()
        eval_loss, eval_spa_loss, eval_time_loss = eval(ep, model, eval_loader)
        epoch_lr = optim.state_dict()['param_groups'][0]['lr']
        epoch_loss = np.mean(np.array(epoch_loss))
        epoch_spa_loss = np.mean(np.array(epoch_spa_loss))
        epoch_time_loss = np.mean(np.array(epoch_time_loss))
        print(f"Train epoch: {ep:<2} | lr: {epoch_lr:.8f}")
        print(f"Train loss: {epoch_loss:>11.8f} | spa_loss: {epoch_spa_loss:>11.8f} | time_loss: {epoch_time_loss:>11.8f}")
        print(f"Eval  loss: {eval_loss:>11.8f} | spa_loss: {eval_spa_loss:>11.8f} | time_loss: {eval_time_loss:>11.8f}")
        save_checkpoint(path, model, optim, sched)
@torch.no_grad()
def inference_fn(model, data_loader):
    model.eval()
    embeddings = []
    for batch_data in data_loader:
        for k, v in batch_data.items():
            batch_data[k] = v.to(device,non_blocking=True)
        emb = model.forward_encoder(batch_data)
        embeddings.append(emb.detach().cpu().numpy())
    return np.vstack(embeddings)
def mr(dists):
    targets = np.diag(dists)
    result = np.sum(np.greater_equal(dists.T, targets)) / dists.shape[0]
    return round(result, 5)
def hit_ratio(truth, pred, Ks):
    hit_K = {}
    for K in Ks:
        top_K_pred = pred[:, :K]
        hit = 0
        for i in range(len(pred)): 
            for j, pred_j in enumerate(top_K_pred[i]):
                if truth[j] in pred_j:
                    hit += 1
        hit_K[K] = round(hit / pred.shape[0], 5)
    return hit_K
def travel_time_evaluation(preds, labels):
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = preds * 60
    labels = labels * 60
    mae = mean_absolute_error(labels, preds)
    mape = mean_absolute_percentage_error(labels, preds)
    rmse = mean_squared_error(labels, preds) ** 0.5
    return {'MAE': round(mae, 5), 'RMSE': round(rmse, 5), 'MAPE': round(mape, 5)}
def multi_cls_evaluation(preds, truths, n_classes):
    preds = np.vstack(preds)
    truths = np.concatenate(truths)
    preds_label = np.argmax(preds, axis=-1)
    micro_f1 = f1_score(truths, preds_label, average='micro', labels=np.arange(n_classes).tolist())
    macro_f1 = f1_score(truths, preds_label, average='macro', labels=np.arange(n_classes).tolist())
    return {'Mi-F1': round(micro_f1, 5), 'Ma-F1': round(macro_f1, 5)}
def binary_cls_evaluation(preds, truths):
    preds = np.vstack(preds)
    truths = np.concatenate(truths)
    preds_label = np.argmax(preds, axis=-1)
    f1 = f1_score(truths, preds_label)
    accuracy = accuracy_score(truths, preds_label)
    precision = precision_score(truths, preds_label)
    return {'F1': round(f1, 5), 'Accuracy': round(accuracy, 5), 'Precision': round(precision, 5)}
@torch.no_grad()
def test_cls(model, data_loader, n_cls):
    model.eval()
    preds_list = []
    labels_list = []
    for batch_data, batch_label in data_loader:
        for k, v in batch_data.items():
            batch_data[k] = v.to(device,non_blocking=True)
        preds = model(batch_data)
        preds_list.append(F.log_softmax(preds, dim=-1).detach().cpu().numpy())
        labels_list.append(batch_label.numpy())
    if n_cls>2: results = multi_cls_evaluation(preds_list, labels_list, n_cls)
    else: results = binary_cls_evaluation(preds_list, labels_list)
    return results
@torch.no_grad()
def eval_cls(ep, model, data_loader,n_cls):
    model.eval()
    eval_epoch_loss = []
    eval_preds_list = []
    eval_labels_list = []
    for batch_data, labels in data_loader:
        for k, v in batch_data.items():
            batch_data[k] = v.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(batch_data)
        loss = F.cross_entropy(preds, labels)
        eval_epoch_loss.append(loss.item())
        eval_preds_list.append(F.log_softmax(preds, dim=-1).detach().cpu().numpy())
        eval_labels_list.append(labels.detach().cpu().numpy())
        print(f"Epoch: {ep} | Eval  loss: {loss:.8f}")
    if n_cls>2: eval_results = multi_cls_evaluation(eval_preds_list, eval_labels_list, n_cls)
    else:  eval_results = binary_cls_evaluation(eval_preds_list, eval_labels_list)
    return np.mean(np.array(eval_epoch_loss)), eval_results
def api_test_sim(root_data,root_model):
    model = Net()
    path_model=os.path.join(root_model,f'pretrain.th.zst')
    assert os.path.exists(path_model)
    model.load_state_dict(torch.load(path_model, map_location='cpu', weights_only=True)['model'], strict=False)
    model.to(device=device, non_blocking=True)
    tss,mbr=api_prerpocess(root_data)
    test_loader=get_dataloader('sim',tss[1],mbr,False)
    test_noise_loader=get_dataloader('sim',tss[-1],mbr,False)
    full_databse_emb=inference_fn(model=model, data_loader=test_loader)
    query_emb=inference_fn(model=model, data_loader=test_noise_loader)
    dists = query_emb @ full_databse_emb.T
    scores = np.argsort(dists, axis=-1)[:, ::-1][:, :10]
    truth=list(range(len(test_loader)))
    mr_res = mr(dists)
    hr_res = hit_ratio(truth, scores, [1, 5, 10])
    print('MR',mr_res)
    print('HR',hr_res)
    return {'MR':mr_res,'HR':hr_res}
class Classifier(nn.Module):
    def __init__(self, model,n_cls):
        super(Classifier, self).__init__()
        self.model :Net= model
        self.mlp = nn.Sequential(nn.Linear(128,128),nn.ReLU(inplace=True),nn.Linear(128, n_cls))
    def forward(self, data):
        x = self.model.forward_encoder(data)
        return self.mlp(x)
def api_train_cls(root_data,root_model,n_cls):
    model = Net()
    path_pre=os.path.join(root_model,f'pretrain.th.zst')
    path_cls=os.path.join(root_model,f'cls.th.zst')
    model.load_state_dict(torch.load(path_pre, map_location='cpu', weights_only=True)['model'])
    cls_model = Classifier(model,n_cls).to(device=device, non_blocking=True)
    tss,mbr=api_prerpocess(root_data)
    train_loader, eval_loader = get_dataloader('cls',tss[0],mbr,True),get_dataloader('cls',tss[2],mbr,False)
    optim = get_optimizer(model.parameters())
    sched = get_scheduler(optimizer=optim,t=10)
    best_epoch = -1
    best_loss = 1e10
    for ep in range(30):
        cls_model.train()
        sched.step(ep)
        train_epoch_loss = []
        train_preds_list = []
        train_labels_list = []
        for batch_data, batch_label in train_loader:
            for k, v in batch_data.items():
                batch_data[k] = v.to(device,non_blocking=True)
            batch_label = batch_label.to(device,non_blocking=True)
            optim.zero_grad()
            preds = cls_model(batch_data)
            loss = F.cross_entropy(preds, batch_label)
            loss.backward()
            optim.step()
            with torch.no_grad():
                train_epoch_loss.append(loss.item())
                train_preds_list.append(F.log_softmax(preds, dim=-1).detach().cpu().numpy())
                train_labels_list.append(batch_label.detach().cpu().numpy())
            print(f"Epoch: {ep} | Train loss: {loss:.8f}")
        train_epoch_loss = np.mean(np.array(train_epoch_loss))
        if n_cls>2: train_results = multi_cls_evaluation(train_preds_list, train_labels_list, n_cls)
        else: train_results = binary_cls_evaluation(train_preds_list, train_labels_list)
        eval_epoch_loss, eval_results = eval_cls(ep, cls_model, eval_loader, n_cls)
        epoch_lr = optim.state_dict()['param_groups'][0]['lr']
        print(f'Epoch: {ep:<2} | lr: {epoch_lr:8f}')
        print(f"Train loss: {train_epoch_loss:.8f} | train results: {train_results}")
        print(f"Eval  loss: { eval_epoch_loss:.8f} | eval  results: { eval_results}")
        save_checkpoint(path_cls, cls_model, optim, sched)
        if eval_epoch_loss <= best_loss:
            best_loss = eval_epoch_loss
            best_epoch = ep
            patience = 0
        else:
            patience += 1
        if patience == 7:
            print(f"Early stopping at epoch {ep} with loss {eval_epoch_loss}")
            break
def api_test_cls(root_data,root_model,n_cls):
    path_cls=os.path.join(root_model,f'cls.th.zst')
    cls_model = Classifier(Net(),n_cls).to(device=device, non_blocking=True)
    cls_model.load_state_dict(torch.load(path_cls, map_location='cpu', weights_only=True)['model'])
    cls_model.to(device=device, non_blocking=True)
    tss,mbr=api_prerpocess(root_data)
    test_loader=get_dataloader('cls',tss[1],mbr,False)
    test_results = test_cls(cls_model, test_loader, n_cls)
    return test_results
class TravelTimeEvaluator(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(nn.Linear(128,128),nn.ReLU(inplace=True),nn.Linear(128, 1))
    def forward(self, data):
        traj_emb = self.encoder.forward_encoder(data)
        pred = self.mlp(traj_emb)
        return pred
@torch.no_grad()
def test_tte(model, data_loader):
    model.eval()
    preds_list = []
    label_list = []
    for batch_data, batch_label in data_loader:
        for k, v in batch_data.items():
            batch_data[k] = v.to(device,non_blocking=True)
        preds = model(batch_data)
        preds_list.append(preds.detach().cpu().numpy())
        label_list.append(batch_label.detach().cpu().numpy())
    results = travel_time_evaluation(preds_list, label_list)
    return results
@torch.no_grad()
def eval_tte(ep, model, data_loader):
    model.eval()
    eval_epoch_loss = []
    eval_preds_list = []
    eval_labels_list = []
    for batch_data, batch_label in data_loader:
        for k, v in batch_data.items():
            batch_data[k] = v.to(device,non_blocking=True)
        batch_label = batch_label.to(device,non_blocking=True)
        preds = model(batch_data)
        loss = F.mse_loss(preds, batch_label)
        eval_epoch_loss.append(loss.item())
        eval_preds_list.append(preds.detach().cpu().numpy())
        eval_labels_list.append(batch_label.detach().cpu().numpy())
        print(f"Epoch: {ep} | Eval  loss: {loss:.8f}")
    eval_epoch_loss = np.mean(np.array(eval_epoch_loss))
    eval_results = travel_time_evaluation(eval_preds_list, eval_labels_list)
    return eval_epoch_loss, eval_results
def api_train_tte(root_data,root_model):
    model = Net()
    path_pre=os.path.join(root_model,f'pretrain.th.zst')
    path_tte=os.path.join(root_model,f'tte.th.zst')
    model.load_state_dict(torch.load(path_pre, map_location='cpu', weights_only=True)['model'])
    tss,mbr=api_prerpocess(root_data)
    train_loader, eval_loader = get_dataloader('cls',tss[0],mbr,True),get_dataloader('cls',tss[2],mbr,False)
    tte_model = TravelTimeEvaluator(model)
    tte_model = tte_model.to(device, non_blocking=True)
    optim = get_optimizer(tte_model.parameters())
    sched = get_scheduler(optimizer=optim, t=10)
    best_epoch = -1
    best_loss = 1e10
    patience = 0
    for ep in range(30):
        model.train()
        sched.step(ep)
        train_epoch_loss = []
        train_preds_list = []
        train_labels_list = []
        for batch_data, batch_label in train_loader:
            for k, v in batch_data.items():
                batch_data[k] = v.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)
            optim.zero_grad()
            preds = tte_model(batch_data)
            loss = F.mse_loss(preds, batch_label)
            loss.backward()
            optim.step()
            with torch.no_grad():
                train_epoch_loss.append(loss.item())
                train_preds_list.append(preds.detach().cpu().numpy())
                train_labels_list.append(batch_label.detach().cpu().numpy())
            print(f"Epoch: {ep} | Train loss: {loss:.8f}")
        train_epoch_loss = np.mean(np.array(train_epoch_loss))
        train_results = travel_time_evaluation(train_preds_list, train_labels_list)
        eval_epoch_loss, eval_results = eval_tte(ep, tte_model, eval_loader)
        epoch_lr = optim.state_dict()['param_groups'][0]['lr']
        print(f"Epoch: {ep:<2} | lr: {epoch_lr:>11.8f}")
        print(f"Train loss: {train_epoch_loss:>12.8f} | train results: {train_results}")
        print(f"Eval  loss: { eval_epoch_loss:>12.8f} | eval  results: { eval_results}")
        save_checkpoint(path_tte, tte_model, optim, sched)
        if eval_epoch_loss <= best_loss:
            best_loss = eval_epoch_loss
            best_epoch = ep
            patience = 0
        else:
            patience += 1
        if patience == 7:
            print(f"Early stopping at epoch {ep} with loss {eval_epoch_loss}")
            break
def api_test_tte(root_data,root_model):
    tte_model = TravelTimeEvaluator(Net())
    path_tte=os.path.join(root_model,f'tte.th.zst')
    tte_model.load_state_dict(torch.load(path_tte, map_location='cpu', weights_only=True)['model'])
    tte_model.to(device=device, non_blocking=True)
    tss,mbr=api_prerpocess(root_data)
    train_loader, eval_loader, test_loader = get_dataloader('cls',tss[0],mbr,True),get_dataloader('cls',tss[2],mbr,False),get_dataloader('cls',tss[1],mbr,False)
    test_results = test_tte(tte_model, test_loader)
    print(f"Test results: {test_results}")
    return test_results
