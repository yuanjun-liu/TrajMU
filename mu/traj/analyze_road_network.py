exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import argparse
import csv
import json
import os
from ast import literal_eval
from collections import defaultdict, deque
import sys
import numpy as np
from scipy.stats import pearsonr
from rtree import Rtree
from _tool.mList import iterable
from _nn.nData import random_seed
from _tool.mFile import check_dir, out_dir
from _tool.mIO import loadZ_pk, saveZ_pk
from mu.traj.loaddata import train_rate,_split_tvt
from mu.traj.model_trmma import _map_pre, call_tvt_urv
from traj.models.one_trmma import SPoint
from mu.traj.plt import *
MATCHING_VERSION=1
def _root_map(dataset):
    return os.path.join(out_dir(dataset), 'map')
def _cache_path(dataset, matching_version=None):
    if matching_version is None:
        matching_version = MATCHING_VERSION
    suffix = {
        0: 'traj-light',
        1: 'traj-nearest',
    }.get(matching_version)
    if suffix is None:
        raise ValueError(f'unsupported MATCHING_VERSION={matching_version}')
    return os.path.join(out_dir('cache'), 'road-network-analysis', f'{dataset}-{suffix}.pk.zst')
def _round6(value):
    return round(float(value), 6)
def _parse_path_eids(path_text):
    path = literal_eval(path_text)
    if not path:
        return set()
    return {int(item[0]) for item in path}
def _parse_raw_summary(raw_text):
    raw = literal_eval(raw_text)
    coord_sum = np.zeros(2, dtype=np.float64)
    for attrs in raw:
        coord_sum[0] += float(attrs[3])
        coord_sum[1] += float(attrs[4])
    return coord_sum, len(raw)
def _parse_raw_points(raw_text):
    raw = literal_eval(raw_text)
    points = []
    for attrs in raw:
        if len(attrs) < 3:
            continue
        points.append((float(attrs[0]), float(attrs[1]), float(attrs[2])))
    return points
def _node_path(root_map):
    candidates = [
        os.path.join(root_map, 'map', 'nodeOSM.txt'),
        os.path.join(root_map, 'nodeOSM.txt'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'missing nodeOSM.txt: checked {candidates}')
def _build_node_rtree(root_map):
    idx = Rtree()
    nodes = {}
    with open(_node_path(root_map), newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 3:
                continue
            nid = int(row[0])
            lat = float(row[1])
            lon = float(row[2])
            nodes[nid] = (lon, lat)
            idx.insert(nid, (lon, lat, lon, lat))
    if not nodes:
        raise RuntimeError(f'empty nodeOSM.txt under {root_map}')
    return idx, nodes
def _nearest_node(node_index, lon, lat):
    return next(node_index.nearest((lon, lat, lon, lat), 1))
def _nearest_match_summary(rn, node_index, raw_points):
    coord_sum = np.zeros(2, dtype=np.float64)
    edge_set = set()
    node_set = set()
    coord_count = 0
    for lon, lat, _ in raw_points:
        cand = rn.nearest_query(SPoint(lat, lon), return_type='candidate')
        edge_set.add(int(cand.eid))
        node_set.add(int(_nearest_node(node_index, lon, lat)))
        coord_sum[0] += float(cand.lng)
        coord_sum[1] += float(cand.lat)
        coord_count += 1
    return coord_sum, coord_count, edge_set, node_set
def _load_traj_light(dataset, rebuild=False):
    path = _cache_path(dataset, 0)
    if os.path.exists(path) and not rebuild:
        return loadZ_pk(path)
    root_map = _root_map(dataset)
    csv_paths = [
        os.path.join(root_map, 'traj_train.csv'),
        os.path.join(root_map, 'traj_valid.csv'),
        os.path.join(root_map, 'traj_test.csv'),
    ]
    missing = [p for p in csv_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'missing trajectory csv for {dataset}: {missing}')
    uid_map = {}
    uids = []
    means = []
    coord_sums = []
    coord_counts = []
    edge_sets = []
    for csv_path in csv_paths:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                oid = row[0]
                if oid not in uid_map:
                    uid_map[oid] = len(uid_map)
                coord_sum, coord_count = _parse_raw_summary(row[4])
                if coord_count <= 2:
                    continue
                uids.append(uid_map[oid])
                means.append(coord_sum / coord_count)
                coord_sums.append(coord_sum)
                coord_counts.append(coord_count)
                edge_sets.append(_parse_path_eids(row[3]))
    data = {
        'match_backend': 'legacy',
        'matching_version': 0,
        'uids': np.asarray(uids, dtype=np.int64),
        'means': np.asarray(means, dtype=np.float64),
        'coord_sums': np.asarray(coord_sums, dtype=np.float64),
        'coord_counts': np.asarray(coord_counts, dtype=np.int64),
        'edge_sets': edge_sets,
    }
    check_dir(path)
    saveZ_pk(path, data)
    return data
def _load_traj_nearest(dataset, rn, rebuild=False):
    path = _cache_path(dataset, 1)
    if os.path.exists(path) and not rebuild:
        data = loadZ_pk(path)
        if isinstance(data, dict) and data.get('match_backend') == 'nearest':
            return data
    root_map = _root_map(dataset)
    csv_paths = [
        os.path.join(root_map, 'traj_train.csv'),
        os.path.join(root_map, 'traj_valid.csv'),
        os.path.join(root_map, 'traj_test.csv'),
    ]
    missing = [p for p in csv_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'missing trajectory csv for {dataset}: {missing}')
    uid_map = {}
    uids = []
    means = []
    coord_sums = []
    coord_counts = []
    edge_sets = []
    node_sets = []
    node_index, _ = _build_node_rtree(root_map)
    for csv_path in csv_paths:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                oid = row[0]
                if oid not in uid_map:
                    uid_map[oid] = len(uid_map)
                raw_points = _parse_raw_points(row[4])
                if len(raw_points) <= 2:
                    continue
                coord_sum, coord_count, edge_set, node_set = _nearest_match_summary(rn, node_index, raw_points)
                if coord_count <= 2:
                    continue
                uids.append(uid_map[oid])
                means.append(coord_sum / coord_count)
                coord_sums.append(coord_sum)
                coord_counts.append(coord_count)
                edge_sets.append(edge_set)
                node_sets.append(node_set)
    data = {
        'match_backend': 'nearest',
        'matching_version': 1,
        'uids': np.asarray(uids, dtype=np.int64),
        'means': np.asarray(means, dtype=np.float64),
        'coord_sums': np.asarray(coord_sums, dtype=np.float64),
        'coord_counts': np.asarray(coord_counts, dtype=np.int64),
        'edge_sets': edge_sets,
        'node_sets': node_sets,
    }
    check_dir(path)
    saveZ_pk(path, data)
    return data
def _split_tvt_from_means(num_trajs):
    train_num = int(num_trajs * train_rate + 0.5)
    train_idx = np.random.choice(num_trajs, train_num, replace=False)
    mask = np.ones(num_trajs, dtype=bool)
    mask[train_idx] = False
    val_test_idx = np.flatnonzero(mask)
    if len(val_test_idx) > train_num:
        val_test_idx = val_test_idx[:train_num]
    n_val_test = len(val_test_idx)
    val_size = n_val_test // 2
    perm = np.random.permutation(n_val_test)
    val_idx = val_test_idx[perm[:val_size]]
    test_idx = val_test_idx[perm[val_size:]]
    return train_idx, val_idx, test_idx
def _split_indices(traj_data, dudrdvtype, durate):
    random_seed(42)
    if dudrdvtype.lower() == 'area':
        train_idx, val_idx, test_idx = _split_tvt_from_means(len(traj_data['uids']))
        anchor_idx = train_idx[-100:]
        anchor = (
            traj_data['coord_sums'][anchor_idx].sum(axis=0)
            / traj_data['coord_counts'][anchor_idx].sum()
        ).astype(float)
        ps = traj_data['means'][train_idx].astype(float)
        distances = np.linalg.norm(ps - anchor, axis=1)
        idx = train_idx[np.argsort(distances)]
        du_num = int(len(ps) * durate + 0.5)
        du_idx, dr_idx = idx[:du_num], idx[du_num:]
        np.random.shuffle(du_idx)
        np.random.shuffle(dr_idx)
        dv_idx = np.concatenate([val_idx, test_idx])
        return train_idx, val_idx, test_idx, du_idx, dr_idx, dv_idx
    lightweight_ts = traj_data['means']
    return call_tvt_urv(dudrdvtype, durate)(lightweight_ts, traj_data['uids'])
def _covered_edges(edge_sets, indices=None):
    if indices is None:
        iterable = edge_sets
    else:
        iterable = (edge_sets[int(i)] for i in indices)
    covered = set()
    for edge_set in iterable:
        covered.update(edge_set)
    return covered
def _covered_nodes(node_sets, indices=None):
    if not node_sets:
        return set()
    if indices is None:
        iterable = node_sets
    else:
        iterable = (node_sets[int(i)] for i in indices)
    covered = set()
    for node_set in iterable:
        covered.update(node_set)
    return covered
def _weak_lcc_ratio(edge_nodes, covered_edges, covered_nodes=None):
    if not covered_edges and not covered_nodes:
        return 0.0
    adj = defaultdict(set)
    nodes = set() if covered_nodes is None else set(covered_nodes)
    for eid in covered_edges:
        u, v = edge_nodes[eid]
        nodes.add(u)
        nodes.add(v)
        adj[u].add(v)
        adj[v].add(u)
    seen = set()
    largest = 0
    for node in nodes:
        if node in seen:
            continue
        size = 0
        queue = deque([node])
        seen.add(node)
        while queue:
            cur = queue.popleft()
            size += 1
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        largest = max(largest, size)
    return largest / len(nodes)
def _network_stats(rn):
    area_km2 = rn.mbr.get_h() * rn.mbr.get_w() / 1_000_000
    edges = set(range(rn.edgeNum))
    return {
        'area_km2': area_km2,
        'node_num': rn.nodeNum,
        'edge_num': rn.edgeNum,
        'valid_edge_num': rn.valid_edge_cnt,
        'node_density_per_km2': rn.nodeNum / area_km2,
        'edge_density_per_km2': rn.edgeNum / area_km2,
        'graph_density_directed': rn.edgeNum / (rn.nodeNum * (rn.nodeNum - 1)) if rn.nodeNum > 1 else 0.0,
        'avg_in_degree': rn.edgeNum / rn.nodeNum if rn.nodeNum else 0.0,
        'avg_out_degree': rn.edgeNum / rn.nodeNum if rn.nodeNum else 0.0,
        'avg_total_degree': 2 * rn.edgeNum / rn.nodeNum if rn.nodeNum else 0.0,
        'weak_lcc_ratio': _weak_lcc_ratio(rn.edgeNode, edges),
    }
def _coverage_stats(rn, edge_sets, name, indices, area_km2, node_sets=None):
    covered = _covered_edges(edge_sets, indices)
    nodes = _covered_nodes(node_sets, indices)
    total_length_m = 0.0
    for eid in covered:
        u, v = rn.edgeNode[eid]
        nodes.add(u)
        nodes.add(v)
        total_length_m += float(rn.edgeDis[eid])
    node_num = len(nodes)
    edge_num = len(covered)
    return {
        'split': name,
        'traj_num': len(edge_sets) if indices is None else int(len(indices)),
        'covered_node_num': node_num,
        'covered_edge_num': edge_num,
        'covered_edge_length_km': total_length_m / 1000,
        'node_coverage': node_num / rn.nodeNum if rn.nodeNum else 0.0,
        'edge_coverage': edge_num / rn.edgeNum if rn.edgeNum else 0.0,
        'node_density_per_km2': node_num / area_km2,
        'edge_density_per_km2': edge_num / area_km2,
        'graph_density_directed': edge_num / (node_num * (node_num - 1)) if node_num > 1 else 0.0,
        'avg_in_degree': edge_num / node_num if node_num else 0.0,
        'avg_out_degree': edge_num / node_num if node_num else 0.0,
        'avg_total_degree': 2 * edge_num / node_num if node_num else 0.0,
        'weak_lcc_ratio': _weak_lcc_ratio(rn.edgeNode, covered, nodes),
    }
def _round_stats(stats):
    rounded = {}
    for key, value in stats.items():
        if isinstance(value, (float, np.floating)):
            rounded[key] = _round6(value)
        elif isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        else:
            rounded[key] = value
    return rounded
def analyze_road_one_setting(dataset, dudrdvtype, durate, rebuild_cache=False):
    root_map = _root_map(dataset)
    rn = _map_pre(dataset, root_map)
    if MATCHING_VERSION == 0:
        traj_data = _load_traj_light(dataset, rebuild=rebuild_cache)
    elif MATCHING_VERSION == 1:
        traj_data = _load_traj_nearest(dataset, rn, rebuild=rebuild_cache)
    else:
        raise ValueError(f'unsupported MATCHING_VERSION={MATCHING_VERSION}')
    train_idx, val_idx, test_idx, du_idx, dr_idx, dv_idx = _split_indices(traj_data, dudrdvtype, durate)
    area_km2 = rn.mbr.get_h() * rn.mbr.get_w() / 1_000_000
    splits = {
        'du': du_idx,
        'dr': dr_idx,
        'dv': dv_idx,
    }
    split_stats = {
        name: _round_stats(_coverage_stats(rn, traj_data['edge_sets'], name, idx, area_km2, traj_data.get('node_sets')))
        for name, idx in splits.items()
    }
    split_stats['meta'] = {
        'dataset': dataset,
        'type': dudrdvtype,
        'durate': durate,
        'matching_version': int(MATCHING_VERSION),
        'match_backend': traj_data.get('match_backend', 'legacy' if MATCHING_VERSION == 0 else 'nearest'),
        'traj_num': int(len(traj_data['edge_sets'])),
        'train_num': int(len(train_idx)),
        'val_num': int(len(val_idx)),
        'test_num': int(len(test_idx)),
    }
    split_stats['road_network'] = _round_stats(_network_stats(rn))
    return split_stats
features=[    
    "node_coverage",
    "node_density_per_km2",
    "avg_total_degree",
    ]
def analyze_res_with_road(Datasets,key_task=['Du','Dr','Dv'],key_mia=['MIA']):
    print('analyze_res_with_road',features)
    all_features = {}
    for data in Datasets:
        all_features[data]= {}
        for urv in Urvs:
            all_features[data][urv] = {}
            for durate in DuRates:
                res = analyze_road_one_setting(data, urv, durate)
                all_features[data][urv][str(durate)] = res
    for mu in mu2:
        print('{:10s}'.format(mu),end='')
        for feat in features:
            for task_or_mia in [0,1]:
                key=[key_task,key_mia][task_or_mia]
                for task in Tasks:
                    task_res=[]
                    feat_res=[]
                    for model_group in [1]:
                        metric=metricses[model_group-1][0]
                        for data in Datasets:
                            for urv in Urvs:
                                for durate in DuRates:
                                    res_gap=1-json_item(data=data,mu=mu,task=task,urv=urv,key=key,durate=durate,model_group=model_group,metrics=metric,sims=True)
                                    task_res.append(res_gap)
                                    feature=all_features[data][urv][str(durate)]
                                    feat_res.append(abs(feature['du'][feat]-feature['dr'][feat]))
                    corr_res=pearsonr(np.array(task_res),np.array(feat_res))[0].item()
                    print('&{:.2f}'.format(corr_res),end='')
                print('   ',end='')
        print('\\\\')
if __name__ == '__main__':
    analyze_res_with_road(['Beijing'])
