exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def sequence_mask(X, valid_len, value=0.0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
def sequence_mask3d(X, valid_len, valid_len2, value=0.0):
    maxlen = X.size(1)
    maxlen2 = X.size(2)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    mask2 = torch.arange((maxlen2), dtype=torch.float32, device=X.device)[None, :] < valid_len2[:, None]
    mask_fin = torch.bmm(mask.float().unsqueeze(-1), mask2.float().unsqueeze(-2)).bool()
    X[~mask_fin] = value
    return X
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=500):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(x.device)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = self.dropout(scores)
        return torch.matmul(scores, v)
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(concat)
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.norm = Norm(d_model)
    def forward(self, x):
        residual = x
        x = self.linear_2(F.relu(self.linear_1(x)))
        x = self.dropout(x)
        x += residual
        return self.norm(x)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)
    def forward(self, x, mask):
        residual = x
        x = self.dropout_1(self.attn(x, x, x, mask))
        x2 = self.norm_1(residual + x)
        return self.ff(x2)
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(N)])
        self.norm = Norm(d_model)
    def forward(self, src, mask3d=None):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
        return self.norm(x)
class PointEncoder(nn.Module):
    def __init__(self, hid_dim, transformer_layers):
        super().__init__()
        self.hid_dim = hid_dim
        self.fc_point = nn.Linear(3, hid_dim)
        self.transformer = TransformerEncoder(hid_dim, transformer_layers, heads=4)
    def forward(self, src, src_len):
        max_src_len = src.size(1)
        batch_size = src.size(0)
        src_len = torch.tensor(src_len, device=src.device)
        mask3d = torch.ones(batch_size, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(batch_size, max_src_len, device=src.device)
        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        src = self.fc_point(src)
        outputs = self.transformer(src, mask3d)
        return outputs * mask2d
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)
    def forward(self, query, key, value, attn_mask):
        batch_size, src_len = query.shape[0], query.shape[1]
        seg_num = key.shape[-2]
        query = query.unsqueeze(-2).repeat(1, 1, seg_num, 1)
        energy = torch.tanh(self.attn(torch.cat((query, key), dim=-1)))
        attention = self.v(energy).squeeze(-1)
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        scores = F.softmax(attention, dim=-1)
        weighted = torch.bmm(
            scores.reshape(batch_size * src_len, seg_num).unsqueeze(-2),
            value.reshape(batch_size * src_len, seg_num, -1),
        ).squeeze(-2)
        weighted = weighted.reshape(batch_size, src_len, -1)
        return scores, weighted
class TrajEncoder(nn.Module):
    def __init__(self, id_size, hid_dim, transformer_layers, dropout=0.1):
        super().__init__()
        self.id_size = id_size
        self.hid_dim = hid_dim
        self.id_emb_dim = hid_dim
        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))
        self.road_emb = nn.Sequential(
            nn.Linear(self.id_emb_dim + 9, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            Norm(self.hid_dim),
        )
        self.point_encoder = PointEncoder(hid_dim, transformer_layers)
        self.attn = Attention(self.hid_dim)
        self.output = nn.Linear(2 * self.hid_dim, self.hid_dim)
    def forward(self, src, src_len, src_segs, segs_feat, segs_mask):
        src_id_emb = self.emb_id[src_segs]
        src_road_emb = torch.cat((src_id_emb, segs_feat), dim=-1)
        road_emb = self.road_emb(src_road_emb)
        point_encoder_output = self.point_encoder(src, src_len)
        _, attention = self.attn(point_encoder_output, road_emb, road_emb, segs_mask)
        return torch.cat((point_encoder_output, attention), dim=-1)
def init_weights(module):
    ih = (param.data for name, param in module.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in module.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in module.named_parameters() if 'bias' in name)
    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
def get_targets(model, inputs, cond, denoise_steps, device, segs_mask, bootstrap_every=8, force_t=-1, force_dt=-1):
    model.eval()
    batch_size = inputs.shape[0]
    bootstrap_batchsize = batch_size // bootstrap_every
    log2_sections = int(math.log2(denoise_steps))
    dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batchsize // log2_sections)
    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0])])
    force_dt_vec = torch.ones(bootstrap_batchsize) * force_dt
    dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(device)
    dt = 1 / (2 ** dt_base)
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2
    dt_sections = 2 ** dt_base
    t = torch.cat([torch.randint(low=0, high=int(val.item()), size=(1,)).float() for val in dt_sections]).to(device)
    t = t / dt_sections
    force_t_vec = torch.ones(bootstrap_batchsize, dtype=torch.float32).to(device) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
    t_full = t[:, None, None]
    x_1 = inputs[:bootstrap_batchsize]
    cond_bst = cond[:bootstrap_batchsize]
    segs_mask_bst = segs_mask[:bootstrap_batchsize]
    x_0 = torch.randn_like(x_1).masked_fill(segs_mask_bst == 0, 0)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    with torch.no_grad():
        v_b1 = model(x_t, t, dt_base_bootstrap, cond_bst, segs_mask_bst)
    t2 = t + dt_bootstrap
    x_t2 = torch.clip(x_t + dt_bootstrap[:, None, None] * v_b1, -4, 4)
    with torch.no_grad():
        v_b2 = model(x_t2, t2, dt_base_bootstrap, cond_bst, segs_mask_bst)
    v_target = torch.clip((v_b1 + v_b2) / 2, -4, 4).masked_fill(segs_mask_bst == 0, 0)
    bst_v, bst_dt, bst_t, bst_xt = v_target, dt_base, t, x_t
    t = torch.randint(low=0, high=denoise_steps, size=(inputs.shape[0],), dtype=torch.float32)
    t /= denoise_steps
    force_t_vec = torch.ones(inputs.shape[0]) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
    t_full = t[:, None, None]
    x_0 = torch.randn_like(inputs).masked_fill(segs_mask == 0, 0)
    x_1 = inputs
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = (x_1 - (1 - 1e-5) * x_0).masked_fill(segs_mask == 0, 0)
    dt_flow = int(math.log2(denoise_steps))
    dt_base = (torch.ones(inputs.shape[0], dtype=torch.int32) * dt_flow).to(device)
    bst_size = batch_size // bootstrap_every
    bst_size_data = batch_size - bst_size
    x_t = torch.cat([bst_xt, x_t[-bst_size_data:]], dim=0)
    t = torch.cat([bst_t, t[-bst_size_data:]], dim=0)
    dt_base = torch.cat([bst_dt, dt_base[-bst_size_data:]], dim=0)
    v_t = torch.cat([bst_v, v_t[-bst_size_data:]], dim=0)
    return x_t, v_t, t, dt_base
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
class DiTBlock(nn.Module):
    def __init__(self, hid_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cond_linear = nn.Sequential(nn.SiLU(), nn.Linear(hid_dim, 6 * hid_dim))
        self.norm1 = Norm(hid_dim)
        self.norm2 = Norm(hid_dim)
        self.attn = MultiHeadAttention(num_heads, hid_dim, dropout)
        self.ff = FeedForward(hid_dim, d_ff=hid_dim * 2)
    def forward(self, x, c):
        cond = self.cond_linear(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond, 6, dim=-1)
        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_x = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + (gate_msa * attn_x)
        x_modulated2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_x = self.ff(x_modulated2)
        return x + (gate_mlp * mlp_x)
class OutputLayer(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.cond_linear = nn.Sequential(nn.SiLU(), nn.Linear(hid_dim, 2 * hid_dim))
        self.norm = Norm(hid_dim)
        self.output_linear = nn.Linear(hid_dim, out_dim)
    def forward(self, x, c):
        cond = self.cond_linear(c)
        shift, scale = torch.chunk(cond, 2, dim=-1)
        x_modulated = modulate(self.norm(x), shift, scale)
        return self.output_linear(x_modulated)
class DiT(nn.Module):
    def __init__(self, out_dim, hid_dim, depth, cond_dim):
        super().__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        sinu_pos_emb = SinusoidalPosEmb(hid_dim)
        fourier_dim = hid_dim
        time_dim = hid_dim
        self.pe = PositionalEncoder(hid_dim, max_seq_len=2000)
        self.time_embedder = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.timestep_embedder = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.cond_linear = nn.Linear(cond_dim, hid_dim)
        self.DiTBlocks = nn.ModuleList([DiTBlock(hid_dim) for _ in range(depth)])
        self.noise_linear = nn.Sequential(nn.Linear(out_dim, hid_dim), nn.ReLU())
        self.output = OutputLayer(hid_dim, out_dim)
    def forward(self, x, t, dt, cond, segs_mask):
        x = self.noise_linear(x)
        x = self.pe(x)
        c = self.cond_linear(cond)
        te = self.time_embedder(t)
        dte = self.timestep_embedder(dt)
        c = c + te[:, None] + dte[:, None]
        for block in self.DiTBlocks:
            x = block(x, c)
        x = self.output(x, cond)
        return x.masked_fill(segs_mask == 0, 0)
class ShortCut(nn.Module):
    def __init__(self, model, infer_steps, seq_length, bootstrap_every=8):
        super().__init__()
        self.model = model
        self.infer_steps = infer_steps
        self.seq_length = seq_length
        self.bootstrap_every = bootstrap_every
    def forward(self, x_t, v_t, t, dt_base, cond, x_1, segs_mask):
        v_pred = self.model(x_t, t, dt_base, cond, segs_mask)
        x_pred = x_t + v_pred
        mse_loss = F.mse_loss(v_pred, v_t)
        bce_loss = F.binary_cross_entropy(F.softmax(x_pred.masked_fill(segs_mask == 0, -1e9), dim=-1), x_1, reduction='mean')
        return mse_loss + bce_loss
    @torch.no_grad()
    def inference(self, batch_size, cond, segs_mask):
        device = cond.device
        eps = torch.randn((batch_size, 1, self.seq_length), device=device)
        delta_t = 1.0 / self.infer_steps
        x = eps.masked_fill(segs_mask == 0, 0)
        for ti in range(self.infer_steps):
            t = ti / self.infer_steps
            t_vector = torch.full((eps.shape[0],), t, device=device)
            dt_base = torch.ones_like(t_vector, device=device) * math.log2(self.infer_steps)
            v = self.model(x, t_vector, dt_base, cond, segs_mask)
            x = x + v * delta_t
        return F.softmax(x.masked_fill(segs_mask == 0, -1e9), dim=-1)
def batch_to_diffmm_inputs(batch, encoder: TrajEncoder, device, id_size):
    src_seqs, lengths, trg_rids, _candi_onehots, candi_ids, candi_feats, candi_masks = batch
    src_seqs = src_seqs.to(device, non_blocking=True).float()
    candi_ids = candi_ids.to(device, non_blocking=True).long()
    candi_feats = candi_feats.to(device, non_blocking=True).float()
    candi_masks = candi_masks.to(device, non_blocking=True).float()
    enc_out = encoder(src_seqs, lengths, candi_ids, candi_feats, candi_masks)
    traj_cond = []
    trg_rid_diff = []
    src_segs_id = []
    src_segs_mask = []
    for index in range(enc_out.shape[0]):
        length = int(lengths[index])
        if length > 0:
            traj_cond += [i.unsqueeze(0) for i in enc_out[index][:length]]
            trg_rid_diff += [torch.tensor(int(i), device=device).view(1, 1) for i in trg_rids[index][:length]]
            src_segs_id += [i.unsqueeze(0) for i in candi_ids[index][:length]]
            src_segs_mask += [i.unsqueeze(0) for i in candi_masks[index][:length]]
    if len(traj_cond) == 0:
        raise RuntimeError('empty batch for DiffMM')
    traj_cond = torch.cat(traj_cond, dim=0).reshape(-1, 1, enc_out.shape[-1])
    trg_rid_diff = torch.cat(trg_rid_diff, dim=0).reshape(-1, 1, 1)
    src_segs_id = torch.cat(src_segs_id, dim=0).reshape(trg_rid_diff.shape[0], 1, -1)
    src_segs_mask = torch.cat(src_segs_mask, dim=0).reshape(trg_rid_diff.shape[0], 1, -1)
    trg_onehot_diff = torch.zeros((trg_rid_diff.shape[0], 1, id_size - 1), device=device)
    trg_onehot_diff.scatter_(2, trg_rid_diff.long(), 1.0)
    diff_mask = torch.zeros((trg_rid_diff.shape[0], 1, id_size - 1), device=device)
    for i, src_segs in enumerate(src_segs_id):
        seg_num = int(src_segs_mask[i, 0].sum().item())
        if seg_num > 0:
            diff_mask[i, 0, src_segs[0, :seg_num] - 1] = 1
    return traj_cond, trg_rid_diff, trg_onehot_diff, lengths, src_segs_id, src_segs_mask, diff_mask
class DiffMMModel(nn.Module):
    def __init__(
        self,
        id_size,
        hid_dim=256,
        transformer_layers=2,
        num_units=512,
        depth=2,
        timesteps=2,
        samplingsteps=1,
        dropout=0.1,
        bootstrap_every=8,
    ):
        super().__init__()
        self.id_size = id_size
        self.timesteps = timesteps
        self.samplingsteps = samplingsteps
        self.bootstrap_every = bootstrap_every
        self.encoder = TrajEncoder(id_size=id_size, hid_dim=hid_dim, transformer_layers=transformer_layers, dropout=dropout)
        dit = DiT(id_size - 1, num_units, depth=depth, cond_dim=2 * hid_dim)
        self.shortcut = ShortCut(dit, infer_steps=samplingsteps, seq_length=id_size - 1, bootstrap_every=bootstrap_every)
        self.apply(init_weights)
    def forward(self, batch, device):
        traj_cond, trg_rid, trg_onehot, lengths, src_segs_id, src_segs_mask, diff_mask = batch_to_diffmm_inputs(
            batch, self.encoder, device, self.id_size
        )
        x_t, v_t, t, dt_base = get_targets(
            self.shortcut.model,
            trg_onehot,
            traj_cond,
            self.timesteps,
            device,
            diff_mask,
            bootstrap_every=self.bootstrap_every,
        )
        loss = self.shortcut(x_t, v_t, t, dt_base, traj_cond, trg_onehot, diff_mask)
        return {'loss': loss}
    @torch.no_grad()
    def infer(self, batch, device):
        traj_cond, trg_rid, trg_onehot, lengths, src_segs_id, src_segs_mask, diff_mask = batch_to_diffmm_inputs(
            batch, self.encoder, device, self.id_size
        )
        sampled_seq = self.shortcut.inference(batch_size=traj_cond.shape[0], cond=traj_cond, segs_mask=diff_mask)
        max_len = max(int(l) for l in lengths)
        candi_size = src_segs_id.shape[-1]
        pred_rids = torch.zeros((len(lengths), max_len), dtype=torch.long)
        point_score = torch.zeros((len(lengths), max_len, candi_size), dtype=sampled_seq.dtype)
        cur_len = 0
        for bi, length in enumerate(lengths):
            length = int(length)
            logits = sampled_seq[cur_len:cur_len + length, 0]
            candi_ids = src_segs_id[cur_len:cur_len + length, 0].long()
            candi_mask = src_segs_mask[cur_len:cur_len + length, 0].float()
            gather_index = (candi_ids.clamp_min(1) - 1)
            candi_probs = logits.gather(1, gather_index) * candi_mask
            point_score[bi, :length] = candi_probs.detach().cpu()
            pred_local = candi_probs.argmax(dim=-1)
            point_ids = candi_ids[torch.arange(length, device=device), pred_local]
            pred_rids[bi, :length] = (point_ids - 1).detach().cpu()
            cur_len += length
        return {
            'point_score': point_score,
            'pred_rids': pred_rids,
            'lengths': lengths,
        }
def diffmm_loss(output: dict):
    return output['loss']
def diffmm_results_from_batch(pred_rids: torch.Tensor, trg_rids, lengths):
    results = []
    for bi, length in enumerate(lengths):
        length = int(length)
        pred = pred_rids[bi, :length].detach().cpu().tolist()
        trg = [int(x) for x in trg_rids[bi][:length]]
        results.append((pred, trg))
    return results
def acc_f1_from_results(results, cal_id_acc):
    acc, recall, precision, f1 = [], [], [], []
    for pred, trg in results:
        a, r, p, f = cal_id_acc(pred, trg)
        acc.append(a)
        recall.append(r)
        precision.append(p)
        f1.append(f)
    return {
        'acc': float(np.mean(acc)),
        'recall': float(np.mean(recall)),
        'precision': float(np.mean(precision)),
        'f1': float(np.mean(f1)),
    }
