exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))
class CellSpace:
    def __init__(self, x_unit, y_unit, x_min, y_min, x_max, y_max):
        self.x_unit = float(x_unit)
        self.y_unit = float(y_unit)
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.x_size = int(math.ceil((self.x_max - self.x_min) / self.x_unit))
        self.y_size = int(math.ceil((self.y_max - self.y_min) / self.y_unit))
    def get_xyidx_by_point(self, x, y):
        eps = 1e-6
        x = min(max(x, self.x_min), self.x_max - eps)
        y = min(max(y, self.y_min), self.y_max - eps)
        i_x = int((x - self.x_min) // self.x_unit)
        i_y = int((y - self.y_min) // self.y_unit)
        i_x = min(max(i_x, 0), self.x_size - 1)
        i_y = min(max(i_y, 0), self.y_size - 1)
        return i_x, i_y
    def get_cellid_by_point(self, x, y):
        i_x, i_y = self.get_xyidx_by_point(x, y)
        return i_x * self.y_size + i_y
    @property
    def num_cells(self):
        return self.x_size * self.y_size
def build_trajcl_cellspace(mbr, cell_size=100.0, buffer_size=500.0):
    x1, y1 = lonlat2meters(mbr['min_lon'], mbr['min_lat'])
    x2, y2 = lonlat2meters(mbr['max_lon'], mbr['max_lat'])
    return CellSpace(
        cell_size,
        cell_size,
        min(x1, x2) - buffer_size,
        min(y1, y2) - buffer_size,
        max(x1, x2) + buffer_size,
        max(y1, y2) + buffer_size,
    )
def _traj_to_mercator(gps_seq):
    return [lonlat2meters(float(lon), float(lat)) for lon, lat in gps_seq]
def _mask(traj, ratio=0.3):
    if len(traj) <= 2:
        return list(traj)
    keep = max(2, int(round(len(traj) * (1.0 - ratio))))
    idx = sorted(np.random.choice(len(traj), size=keep, replace=False).tolist())
    return [traj[i] for i in idx]
def _subset(traj, ratio=0.7):
    if len(traj) <= 2:
        return list(traj)
    keep = max(2, int(round(len(traj) * ratio)))
    if keep >= len(traj):
        return list(traj)
    start = random.randint(0, len(traj) - keep)
    return list(traj[start : start + keep])
def _shift(traj, sigma=15.0):
    if len(traj) <= 1:
        return list(traj)
    return [(x + random.gauss(0, sigma), y + random.gauss(0, sigma)) for x, y in traj]
def _augment_pair(traj):
    return _mask(traj), _subset(_shift(traj))
def merc2cell(traj_merc, cellspace):
    seq = []
    last = None
    for pt in traj_merc:
        cell_id = cellspace.get_cellid_by_point(*pt)
        if last is not None and cell_id == last:
            continue
        seq.append((cell_id, pt))
        last = cell_id
    if not seq:
        pt = traj_merc[0]
        seq = [(cellspace.get_cellid_by_point(*pt), pt)]
    cells, pts = zip(*seq)
    return list(cells), list(pts)
def generate_spatial_features(src, cellspace):
    if len(src) == 1:
        x = (src[0][0] - cellspace.x_min) / (cellspace.x_max - cellspace.x_min)
        y = (src[0][1] - cellspace.y_min) / (cellspace.y_max - cellspace.y_min)
        return [[x, y, 0.0, 0.0]]
    lens = []
    for (x1, y1), (x2, y2) in zip(src[:-1], src[1:]):
        lens.append(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    tgt = []
    for i in range(1, len(src) - 1):
        dist = (lens[i - 1] + lens[i]) / 2.0
        dist = dist / max(cellspace.x_unit * 11.0 / 1.414, 1.0)
        radian = math.pi - math.atan2(src[i - 1][0] - src[i][0], src[i - 1][1] - src[i][1]) + math.atan2(
            src[i + 1][0] - src[i][0], src[i + 1][1] - src[i][1]
        )
        radian = 1.0 - abs(radian) / math.pi
        x = (src[i][0] - cellspace.x_min) / (cellspace.x_max - cellspace.x_min)
        y = (src[i][1] - cellspace.y_min) / (cellspace.y_max - cellspace.y_min)
        tgt.append([x, y, dist, radian])
    x = (src[0][0] - cellspace.x_min) / (cellspace.x_max - cellspace.x_min)
    y = (src[0][1] - cellspace.y_min) / (cellspace.y_max - cellspace.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0])
    x = (src[-1][0] - cellspace.x_min) / (cellspace.x_max - cellspace.x_min)
    y = (src[-1][1] - cellspace.y_min) / (cellspace.y_max - cellspace.y_min)
    tgt.append([x, y, 0.0, 0.0])
    return tgt
def _pad_cells(cell_ids):
    return pad_sequence([torch.tensor(ids, dtype=torch.long) + 1 for ids in cell_ids], batch_first=False, padding_value=0)
def _pad_spatial(points, cellspace):
    spatial = [torch.tensor(generate_spatial_features(pts, cellspace), dtype=torch.float32) for pts in points]
    return pad_sequence(spatial, batch_first=False)
def _pack_view(trajs, cellspace):
    cells, points = zip(*[merc2cell(t, cellspace) for t in trajs])
    return {
        'cells': _pad_cells(cells),
        'spatial': _pad_spatial(points, cellspace),
        'lengths': torch.tensor([len(x) for x in cells], dtype=torch.long),
    }
def trajcl_train_collate(batch_data_list, cellspace):
    base = [_traj_to_mercator(item['gps_seq']) for item in batch_data_list]
    q_trajs, k_trajs = zip(*[_augment_pair(t) for t in base])
    q_view = _pack_view(q_trajs, cellspace)
    k_view = _pack_view(k_trajs, cellspace)
    return {
        'cells_q': q_view['cells'],
        'spatial_q': q_view['spatial'],
        'lengths_q': q_view['lengths'],
        'cells_k': k_view['cells'],
        'spatial_k': k_view['spatial'],
        'lengths_k': k_view['lengths'],
    }
def trajcl_test_collate(batch_data_list, cellspace):
    base = [_traj_to_mercator(item['gps_seq']) for item in batch_data_list]
    view = _pack_view(base, cellspace)
    return {
        'cells': view['cells'],
        'spatial': view['spatial'],
        'lengths': view['lengths'],
    }
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=2048):
        super().__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])
class SpatialMSMLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src, attn
class SpatialMSMEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.modules.transformer._get_clones(encoder_layer, num_layers)
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn = None
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output, attn
class SpatialMSM(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_dropout):
        super().__init__()
        self.pos_encoder = PositionalEncoding(ninput, pos_dropout)
        self.trans_encoder = SpatialMSMEncoder(SpatialMSMLayer(ninput, nhead, nhidden, attn_dropout), nlayer)
    def forward(self, src, attn_mask, src_padding_mask, src_len):
        src = self.pos_encoder(src)
        rtn, attn = self.trans_encoder(src, attn_mask, src_padding_mask)
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)
        return rtn, attn
class DualSTB(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_dropout):
        super().__init__()
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(ninput, pos_dropout)
        structural_attn_layers = nn.TransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout)
        self.structural_attn = nn.TransformerEncoder(structural_attn_layers, nlayer)
        self.spatial_attn = SpatialMSM(4, 32, 1, 3, attn_dropout, pos_dropout)
        self.gamma_param = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    def forward(self, src, attn_mask, src_padding_mask, src_len, srcspatial):
        if srcspatial is not None:
            _, attn_spatial = self.spatial_attn(srcspatial, attn_mask, src_padding_mask, src_len)
            attn_spatial = attn_spatial.repeat(self.nhead, 1, 1)
            attn_spatial = torch.sigmoid(self.gamma_param) * 10.0 * attn_spatial
        else:
            attn_spatial = None
        src = self.pos_encoder(src)
        rtn = self.structural_attn(src, attn_spatial, src_padding_mask)
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)
        return rtn
class Projector(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nin, nin), nn.ReLU(), nn.Linear(nin, nout))
        self.reset_parameter()
    def reset_parameter(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.414)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.mlp(x)
class MoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k, nemb, nout, queue_size, mmt=0.999, temperature=0.07):
        super().__init__()
        self.queue_size = queue_size
        self.mmt = mmt
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.mlp_q = Projector(nemb, nout)
        self.mlp_k = Projector(nemb, nout)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', F.normalize(torch.randn(nout, queue_size), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1.0 - self.mmt)
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1.0 - self.mmt)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr : self.queue_size] = keys.T[:, : self.queue_size - ptr]
            self.queue[:, : batch_size - self.queue_size + ptr] = keys.T[:, self.queue_size - ptr :]
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
    def forward(self, kwargs_q, kwargs_k):
        q = F.normalize(self.mlp_q(self.encoder_q(**kwargs_q)), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.mlp_k(self.encoder_k(**kwargs_k)), dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels
    def loss(self, logits, targets):
        return self.criterion(logits, targets)
class TrajCLMoCoModel(nn.Module):
    def __init__(
        self,
        num_cells,
        cell_embedding_dim=256,
        seq_embedding_dim=256,
        proj_dim=128,
        queue_size=2048,
        temperature=0.05,
        trans_hidden_dim=2048,
        trans_attention_head=4,
        trans_attention_layer=2,
        dropout=0.1,
    ):
        super().__init__()
        encoder_q = DualSTB(seq_embedding_dim, trans_hidden_dim, trans_attention_head, trans_attention_layer, dropout, dropout)
        encoder_k = DualSTB(seq_embedding_dim, trans_hidden_dim, trans_attention_head, trans_attention_layer, dropout, dropout)
        self.cell_embedding = nn.Embedding(num_cells + 1, cell_embedding_dim, padding_idx=0)
        if cell_embedding_dim != seq_embedding_dim:
            self.input_proj = nn.Linear(cell_embedding_dim, seq_embedding_dim)
        else:
            self.input_proj = nn.Identity()
        self.moco = MoCo(encoder_q, encoder_k, seq_embedding_dim, proj_dim, queue_size, temperature=temperature)
    def _encode_view(self, cells, spatial, lengths, encoder):
        cell_emb = self.input_proj(self.cell_embedding(cells))
        max_len = lengths.max().item()
        padding_mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
        return encoder(src=cell_emb, attn_mask=None, src_padding_mask=padding_mask, src_len=lengths, srcspatial=spatial)
    def forward(self, batch):
        if 'cells_k' in batch:
            if not self.training:
                return self._encode_view(batch['cells_q'], batch['spatial_q'], batch['lengths_q'], self.moco.encoder_q)
            q_cells = batch['cells_q']
            q_spatial = batch['spatial_q']
            q_lengths = batch['lengths_q']
            k_cells = batch['cells_k']
            k_spatial = batch['spatial_k']
            k_lengths = batch['lengths_k']
            q_cell_emb = self.input_proj(self.cell_embedding(q_cells))
            with torch.no_grad():
                k_cell_emb = self.input_proj(self.cell_embedding(k_cells))
            max_q = q_lengths.max().item()
            max_k = k_lengths.max().item()
            q_padding_mask = torch.arange(max_q, device=q_lengths.device)[None, :] >= q_lengths[:, None]
            k_padding_mask = torch.arange(max_k, device=k_lengths.device)[None, :] >= k_lengths[:, None]
            return self.moco(
                {
                    'src': q_cell_emb,
                    'attn_mask': None,
                    'src_padding_mask': q_padding_mask,
                    'src_len': q_lengths,
                    'srcspatial': q_spatial,
                },
                {
                    'src': k_cell_emb,
                    'attn_mask': None,
                    'src_padding_mask': k_padding_mask,
                    'src_len': k_lengths,
                    'srcspatial': k_spatial,
                },
            )
        emb = self._encode_view(batch['cells'], batch['spatial'], batch['lengths'], self.moco.encoder_q)
        rt= F.normalize(emb, dim=-1)
        return rt
    def loss(self, logits, targets):
        return self.moco.loss(logits, targets)
