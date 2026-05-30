exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import heapq
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from traj.simp.errors import SED_fast
def _safe_div(a, b):
    return a / b if abs(b) > 1e-12 else 0.0
def angle(v1):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    angle1 = math.atan2(dy1, dx1)
    return angle1 if angle1 >= 0 else 2 * math.pi + angle1
def get_ped_point(p, start, end):
    x, y = p[:2]
    st_x, st_y = start[:2]
    en_x, en_y = end[:2]
    if abs(en_x - st_x) < 1e-12 and abs(en_y - st_y) < 1e-12:
        return math.sqrt((x - st_x) ** 2 + (y - st_y) ** 2)
    if abs(en_x - st_x) < 1e-12:
        return abs(x - en_x)
    a = (en_y - st_y) / (en_x - st_x)
    b = -1.0
    c = st_y - st_x * a
    return abs(a * x + b * y + c) / math.sqrt(a * a + b * b)
def get_sed_point(p, start, end):
    (x, y), syn_time = p[:2], p[2]
    (st_x, st_y), st_time = start[:2], start[2]
    (en_x, en_y), en_time = end[:2], end[2]
    time_ratio = 1.0 if abs(st_time - en_time) < 1e-12 else (syn_time - st_time) / (en_time - st_time)
    syn_x = st_x + (en_x - st_x) * time_ratio
    syn_y = st_y + (en_y - st_y) * time_ratio
    dx = x - syn_x
    dy = y - syn_y
    return math.sqrt(dx * dx + dy * dy)
def get_dad_segment_error(segment):
    if len(segment) <= 2:
        return -1, 0.0
    mid = -1
    ps = segment[0]
    pe = segment[-1]
    e = 0.0
    theta_0 = angle([ps[0], ps[1], pe[0], pe[1]])
    for i in range(0, len(segment) - 1):
        pm_0 = segment[i]
        pm_1 = segment[i + 1]
        theta_1 = angle([pm_0[0], pm_0[1], pm_1[0], pm_1[1]])
        tmp = min(abs(theta_0 - theta_1), 2 * math.pi - abs(theta_0 - theta_1))
        if tmp > e:
            e = tmp
            mid = i
    return mid, e
def s3_max_error(traj, st, en, mode='sed'):
    if en - st <= 1:
        return st, 0.0
    if mode == 'dad':
        mid, err = get_dad_segment_error(traj[st : en + 1])
        return st + max(mid, 0), err
    max_err = -1.0
    idx = st + 1
    start = traj[st]
    end = traj[en]
    for i in range(st + 1, en):
        if mode == 'ped':
            err = get_ped_point(traj[i], start, end)
        else:
            err = get_sed_point(traj[i], start, end)
        if err > max_err:
            max_err = err
            idx = i
    return idx, max_err
def s3_select_indices(traj, keep_num, mode='sed'):
    n = len(traj)
    if n <= keep_num:
        return list(range(n))
    keep_num = max(2, min(int(keep_num), n))
    selected = {0, n - 1}
    heap = []
    def push_segment(st, en):
        if en - st <= 1:
            return
        idx, err = s3_max_error(traj, st, en, mode)
        heapq.heappush(heap, (-float(err), int(st), int(en), int(idx)))
    push_segment(0, n - 1)
    while len(selected) < keep_num and heap:
        neg_err, st, en, idx = heapq.heappop(heap)
        if idx in selected:
            continue
        selected.add(idx)
        push_segment(st, idx)
        push_segment(idx, en)
    return sorted(selected)
def s3_metric(traj, simp_idx, mode='sed'):
    if len(simp_idx) >= len(traj):
        return 0.0
    max_err = 0.0
    for st, en in zip(simp_idx[:-1], simp_idx[1:]):
        _, err = s3_max_error(traj, st, en, mode)
        max_err = max(max_err, float(err))
    return max_err
def normalize_traj_features(traj, bbox, tbox):
    traj = np.asarray(traj, dtype=np.float32)
    lon_min, lon_max = bbox[0]
    lat_min, lat_max = bbox[1]
    t_min, t_max = tbox
    feat = np.zeros((len(traj), 8), dtype=np.float32)
    lon = traj[:, 0]
    lat = traj[:, 1]
    tim = traj[:, 2]
    feat[:, 0] = _safe_div(lon - lon_min, lon_max - lon_min)
    feat[:, 1] = _safe_div(lat - lat_min, lat_max - lat_min)
    feat[:, 2] = _safe_div(tim - t_min, t_max - t_min)
    if len(traj) > 1:
        dxy = traj[1:, :2] - traj[:-1, :2]
        dt = np.maximum(traj[1:, 2] - traj[:-1, 2], 1.0)
        feat[1:, 3:5] = dxy
        feat[1:, 5] = np.linalg.norm(dxy, axis=1)
        feat[1:, 6] = feat[1:, 5] / dt
        heading = np.arctan2(dxy[:, 1], dxy[:, 0] + 1e-12)
        feat[1:, 7] = heading / math.pi
    return feat
def keep_ratio_to_count(length, ratio):
    return max(2, min(length, int(round(length * ratio))))
def build_keep_label(traj, keep_ratio, mode='sed'):
    keep_num = keep_ratio_to_count(len(traj), keep_ratio)
    keep_idx = s3_select_indices(traj, keep_num, mode=mode)
    label = np.zeros(len(traj), dtype=np.float32)
    label[keep_idx] = 1.0
    return label, keep_num, keep_idx
def s3_collate_train(batch, bbox, tbox, keep_ratio, mode='sed'):
    lengths = torch.tensor([len(traj) for traj in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = 8
    seq = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    labels = torch.zeros((len(batch), max_len), dtype=torch.float32)
    keep_nums = torch.zeros(len(batch), dtype=torch.long)
    for i, traj in enumerate(batch):
        traj = np.asarray(traj, dtype=np.float32)
        le = len(traj)
        seq[i, :le] = torch.from_numpy(normalize_traj_features(traj, bbox, tbox))
        label, keep_num, _ = build_keep_label(traj, keep_ratio, mode=mode)
        labels[i, :le] = torch.from_numpy(label)
        mask[i, :le] = True
        keep_nums[i] = keep_num
    return {'seq': seq, 'mask': mask, 'labels': labels, 'lengths': lengths, 'keep_nums': keep_nums}
def s3_collate_test(batch, bbox, tbox):
    lengths = torch.tensor([len(traj) for traj in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = 8
    seq = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, traj in enumerate(batch):
        traj = np.asarray(traj, dtype=np.float32)
        le = len(traj)
        seq[i, :le] = torch.from_numpy(normalize_traj_features(traj, bbox, tbox))
        mask[i, :le] = True
    return {'seq': seq, 'mask': mask, 'lengths': lengths}
class S3PointSelector(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, batch):
        x = self.in_proj(batch['seq'])
        lengths = batch['lengths'].to('cpu')
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=batch['seq'].size(1))
        logits = self.scorer(out).squeeze(-1)
        logits = logits.masked_fill(~batch['mask'], -1e9)
        return logits
def s3_loss(logits, batch):
    labels = batch['labels']
    mask = batch['mask']
    pos = labels[mask].sum().item()
    neg = mask.sum().item() - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=logits.device, dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(logits[mask], labels[mask], pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    smooth = 0.0
    if probs.size(1) > 1:
        smooth = (probs[:, 1:] - probs[:, :-1]).abs()
        smooth = smooth[batch['mask'][:, 1:] & batch['mask'][:, :-1]].mean() if smooth.numel() else 0.0
    return bce + 0.05 * smooth
def s3_pick_indices_from_logits(logits, length, keep_num):
    logits = logits[:length].copy()
    if length <= keep_num:
        return list(range(length))
    keep_num = max(2, min(int(keep_num), length))
    logits[0] = logits[-1] = 1e9
    idx = np.argpartition(-logits, keep_num - 1)[:keep_num]
    idx = np.unique(np.concatenate([idx, np.array([0, length - 1])]))
    if len(idx) > keep_num:
        scores = logits[idx]
        order = np.argsort(-scores)
        idx = idx[order[:keep_num]]
    return sorted(idx.tolist())
def simplify_by_model(model, trajs, collate_fn, keep_ratio, device):
    loader = torch.utils.data.DataLoader(trajs, batch_size=64, shuffle=False, collate_fn=collate_fn)
    outputs = []
    model.eval()
    with torch.no_grad():
        offset = 0
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch).detach().cpu().numpy()
            lengths = batch['lengths'].detach().cpu().numpy()
            for bi, le in enumerate(lengths):
                traj = np.asarray(trajs[offset + bi], dtype=np.float32)
                keep_num = keep_ratio_to_count(len(traj), keep_ratio)
                keep_idx = s3_pick_indices_from_logits(logits[bi], int(le), keep_num)
                outputs.append(traj[keep_idx].copy())
            offset += len(lengths)
    return outputs
def metric_batch(trajs, simp_trajs, mode='sed'):
    vals = []
    for traj, simp in zip(trajs, simp_trajs):
        traj = np.asarray(traj, dtype=np.float32)
        simp = np.asarray(simp, dtype=np.float32)
        if len(simp) >= len(traj):
            vals.append(0.0)
            continue
        lookup = {tuple(np.asarray(p).tolist()): i for i, p in enumerate(traj)}
        simp_idx = []
        for p in simp:
            simp_idx.append(lookup.get(tuple(np.asarray(p).tolist()), 0))
        simp_idx = sorted(set([0, len(traj) - 1] + simp_idx))
        vals.append(s3_metric(traj, simp_idx, mode=mode))
    return float(np.mean(vals)) if vals else float('inf')
def sed_metric(trajs, simp_trajs):
    vals = []
    for traj, simp in zip(trajs, simp_trajs):
        vals.append(SED_fast(np.asarray(traj, dtype=np.float32), np.asarray(simp, dtype=np.float32)))
    vals = np.asarray(vals, dtype=np.float32)
    vals = vals[~np.isinf(vals)]
    return float(vals.mean()) if len(vals) else float('inf')
