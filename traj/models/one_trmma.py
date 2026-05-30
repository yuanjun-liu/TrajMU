exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import numpy as np
import sys
import torch,math,os
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from _nn.nData import random_seed,auto_device
from _nn.nBasic import save_weight,load_weight
from _tool.mFile import is_linux,check_dir
from _tool.mIO import loadZ_pk,saveZ_pk,loadZ_th,saveZ_th
import time
import numba
import csv
import geopandas as gpd
from haversine import haversine
from ast import literal_eval
import datetime as dt
import pandas as pd
import networkx as nx
from typing import List,Tuple
from datetime import timedelta
from rtree import Rtree
from queue import PriorityQueue
from multiprocessing import Pool
import random
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from collections import OrderedDict
from collections import Counter
from torch import optim
import pickle,json
device=auto_device()
random_seed(42)
if torch.cuda.is_available(): torch.backends.cudnn.deterministic = True
num_workers=0 if not is_linux else 2
sys.setrecursionlimit(200000)
def city_info(city:str):
    if city.lower() == 'porto':
        zone_range = [41.1395, -8.6911, 41.1864, -8.5521]
        ts = 15
        utc = 1
    elif city.lower() in ['beijing','t-drive','geolife']:
        zone_range = [39.7547, 116.1994, 40.0244, 116.5452]
        ts = 60
        utc = 0
    elif city.lower()=='chengdu':
        zone_range = [30.6443, 104.0288, 30.7416, 104.1375]
        ts = 12
        utc = 8
    elif city.lower()=='xian':
        zone_range = [34.2060, 108.9058, 34.2825, 109.0049]
        ts = 12
        utc = 8
    else: raise RuntimeError('bad city')
    return zone_range,ts,utc
def get_connection_strength_matrix_sparse(paths, seg_num):
    unigram = np.zeros(seg_num, dtype=np.int32)
    mat_all = [{} for i in range(seg_num)]
    start_time = time.time()
    for traj in paths:
        length = len(traj)
        for i in range(length):
            u = traj[i]
            unigram[u] += 1
            for j in range(i+1, length):
                if traj[j] in mat_all[u]:
                    mat_all[u][traj[j]] += 1
                else:
                    mat_all[u][traj[j]] = 1
    print("Attn Counting Time:" + '{:.3f}s'.format(time.time() - start_time))
    data = []
    for i, item in enumerate(mat_all):
        for k, v in item.items():
            data.append([i, k, v])
    print("Non-zero: {}".format(len(data)))
    return unigram, data
def calc_azimuth(v1):
    '''
    :param v1: [begin_lon, begin_lat, end_lon, end_lat]
    :return: azimuth
    '''
    v2 = [v1[0], v1[1], v1[0], 90]
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = (angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = (angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    if v1[2] < v1[0]:
        included_angle = 360 - included_angle
    return included_angle
rt_dict = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "unclassified": 5,
    "residential": 6,
    "motorway_link": 3,
    "trunk_link": 3,
    "primary_link": 3,
    "secondary_link": 4,
    "tertiary_link": 4,
    "living_street": 7,
}
def get_road_type(rt_str):
    avg = "secondary"
    if "[" in rt_str:
        rts = literal_eval(rt_str)
        codes = []
        for item in rts:
            if item not in rt_dict:
                codes.append(rt_dict[avg])
            else:
                codes.append(rt_dict[item])
        code = max(codes)
    else:
        if rt_str not in rt_dict:
            rt_str = avg
        code = rt_dict[rt_str]
    return code
def get_speed(rt):
    speed_info = {
        0: 33.0,
        1: 27.0,
        2: 22.0,
        3: 16.0,
        4: 11.0,
        5: 8.0,
        6: 6.0,
        7: 1.5,
    }
    return speed_info[rt]
def seginfo_constructor(edges, speeds, freqs):
    data = []
    for i in range(edges.shape[0]):
        tmp = edges.iloc[i]
        start = tmp['geometry'].coords[0]
        end = tmp['geometry'].coords[-1]
        azimuth = calc_azimuth([float(start[0]), float(start[1]), float(end[0]), float(end[1])])
        eid = tmp['fid']
        length = tmp['length']
        rt = get_road_type(tmp['highway'])
        speed = speeds[eid]
        if speed < 1e-2:
            speed = get_speed(rt)
        travel_time = length * 1.0 / speed
        data.append([eid,
                     tmp['u'],
                     tmp['v'],
                     round(length, 3),
                     rt,
                     "{},{}".format(start[0], start[1]),
                     "{},{}".format(end[0], end[1]),
                     azimuth,
                     freqs[eid],
                     round(travel_time, 3)])
    return data
def load_paths(trajfile, left=3, right=81, require_speed=False):
    trajs = pd.read_csv(trajfile, sep=",", header=None, names=['oid', 'tid', 'offsets', 'path', 'raw', 'low'])
    print("Trajectories Number: {}".format(trajs.shape[0]))
    paths = []
    offsets = []
    speeds = {}
    for i in range(trajs.shape[0]):
        tmp = trajs.iloc[i]
        traj = literal_eval(tmp['path'])
        if left <= len(traj) < right:
            path = []
            for seg in traj:
                segid = seg[0]
                path.append(segid)
                if require_speed:
                    seg_speed = seg[2]
                    if seg_speed >= 0.1:
                        if segid not in speeds:
                            speeds[segid] = []
                        speeds[segid].append(seg_speed)
            paths.append(path)
            offsets.append(literal_eval(tmp['offsets']))
    print("Paths Number: {}".format(len(paths)))
    return list(zip(paths, offsets)), speeds
def load_paths2(trajfile, left=3, right=81):
    trajs = pd.read_csv(trajfile, sep=",", header=None, names=['oid', 'tid', 'offsets', 'path', 'raw', 'low'])
    print("Trajectories Number: {}".format(trajs.shape[0]))
    res = []
    for i in range(trajs.shape[0]):
        tmp = trajs.iloc[i]
        traj = literal_eval(tmp['path'])
        if left <= len(traj) < right:
            res.append(traj)
    print("Paths Number: {}".format(len(res)))
    return res
def gen_vehicle_num(train_file,utc, seg_num):
    data = load_paths2(train_file, left=5, right=300)
    tz = dt.timezone(dt.timedelta(hours=utc))
    num_1d = 24
    vehicle_num = np.zeros((seg_num, num_1d*2), dtype=np.int32)
    start_time = time.time()
    for traj in data:
        for seg, ts, speed, _ in traj:
            if speed < 0.1 or speed >= 35:
                continue
            tm = dt.datetime.fromtimestamp(ts, tz)
            if tm.weekday() in [0, 1, 2, 3, 4]:
                idx = tm.hour
            else:
                idx = tm.hour + num_1d
            vehicle_num[seg, idx] += 1
    print("Traffic Popularity Time:" + '{:.3f}s'.format(time.time() - start_time))
    res = []
    for i in range(num_1d*2):
        for j in range(seg_num):
            res.append([i, j, vehicle_num[j, i]])
    return res
def load_edges(edges_shp):
    edges = gpd.read_file(edges_shp)
    eid = edges['fid'].tolist()
    u = edges['u'].tolist()
    v = edges['v'].tolist()
    length = edges['length'].tolist()
    data = []
    for i in range(len(eid)):
        data.append([eid[i], u[i], v[i], length[i]])
    df = pd.DataFrame(data, columns=['eid', 'source', 'target', 'length'])
    print("Number of Segments: {}, {}".format(df.shape[0], edges.shape[0]))
    return df
def get_edge_list(edges):
    eids = edges['eid'].tolist()
    targets = edges['target'].tolist()
    edge_list = {}
    for i in range(len(eids)):
        u = eids[i]
        tar = targets[i]
        v_set = edges.query('source == {}'.format(tar))['eid'].tolist()
        for v in v_set:
            key = str(u) + " " + str(v)
            edge_list[key] = 1
    return edge_list
def traj_freq(paths, edges_shp):
    segs = load_edges(edges_shp)
    data = get_edge_list(segs)
    for path in paths:
        for i in range(len(path)-1):
            key = str(path[i]) + " " + str(path[i+1])
            data[key] += 1
    data_rows = []
    for key, value in data.items():
        tmp = key.split(" ")
        data_rows.append([int(tmp[0]), int(tmp[1]), value])
    return data_rows
def gen_dam(root_map,train_file,utc,edges_shp):
    path_csm,path_seg,path_traffic,path_edge=os.path.join(root_map, "csm_all.txt"), os.path.join(root_map, "seg_info.csv"), os.path.join(root_map, "traffic_num.txt"), os.path.join(root_map, "weighted_edges.txt")
    if os.path.exists(path_csm) and os.path.exists(path_seg) and os.path.exists(path_traffic) and os.path.exists(path_edge):return
    print('build gen dam',root_map,train_file,utc,edges_shp)
    edges = gpd.read_file(edges_shp)
    seg_num = edges.shape[0]
    paths, speeds = load_paths(train_file, left=5, right=300, require_speed=True)
    paths = [item[0] for item in paths]
    speed_info = np.zeros(seg_num, dtype=float)
    for k, v in speeds.items():
        speed_info[k] = np.mean(v)
    print("==> speed size: {}".format(len(speeds)))
    unigram, mat_a = get_connection_strength_matrix_sparse(paths, seg_num)
    with open(path_csm, 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(mat_a)
    print("Saved csm_all.txt")
    data = seginfo_constructor(edges, speeds=speed_info, freqs=unigram)
    with open(path_seg, 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(data)
    print("Saved seg_info.csv")
    vehicle_num = gen_vehicle_num(train_file,utc, seg_num)
    with open(path_traffic, 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(vehicle_num)
    print("Saved traffic_num.txt")
    data_rows = traj_freq(paths, edges_shp)
    with open(path_edge, 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter=' ')
        fields_output_file.writerows(data_rows)
    print("Saved weighted_edges.txt")
def txt2npy(root_map):
    path_seg,path_popular=os.path.join(root_map, "segs_geo.npy"), os.path.join(root_map, "traffic_popularity.npy")
    if os.path.exists(path_seg) and os.path.exists(path_popular):return
    print('build txt2npy',root_map)
    edges = pd.read_csv(os.path.join(root_map, "seg_info.csv"), sep=" ", header=None, names=['eid', 'src', 'trg', 'len', 'rt', 'geo_src', 'geo_trg', 'azimuth', 'freq', 'travel_time'])
    seg_num = edges.shape[0]
    segs_geo = []
    for i in range(seg_num):
        tmp = edges.query("eid == {}".format(i)).iloc[0]
        u = [float(item) for item in tmp['geo_src'].split(",")]
        v = [float(item) for item in tmp['geo_trg'].split(",")]
        segs_geo.append(u + v)
    segs_geo = np.array(segs_geo, dtype=float)
    np.save(path_seg, segs_geo)
    traffic_data = pd.read_csv(os.path.join(root_map, "traffic_num.txt"), sep=" ", header=None, names=['row', 'col', 'value']).to_numpy(dtype=np.int32)
    time_delta = 3600
    num_1h = int(60 * 60 / time_delta)
    num_1d = 24 * num_1h
    vehicle_num = np.zeros((seg_num, num_1d*2), dtype=np.int32)
    for tmp in traffic_data:
        vehicle_num[tmp[1], tmp[0]] = tmp[2]
    np.save(os.path.join(root_map, "vehicle_num_{}-{}.npy".format(time_delta, vehicle_num.shape[1])), vehicle_num)
    traffic_popularity = np.array(vehicle_num, dtype=float)
    for t_idx in range(traffic_popularity.shape[1]):
        min_traffic = traffic_popularity[:, t_idx].min()
        max_traffic = traffic_popularity[:, t_idx].max()
        traffic_popularity[:, t_idx] = (traffic_popularity[:, t_idx] - min_traffic) * 2.0 / (max_traffic - min_traffic) - 1
    np.save(path_popular, traffic_popularity)
def get_road_graph(workspace):
    path=os.path.join(workspace, "road_graph_wtime")
    if os.path.exists(path):return
    print('build get_road_graph',workspace)
    edges = pd.read_csv(os.path.join(workspace, "seg_info.csv"), sep=" ", header=None, names=['eid', 'source', 'target', 'length', 'rt', 'geo_src', 'geo_trg', 'azimuth', 'freq', 'travel_time'])
    print("Number of Segments: {}".format(edges.shape[0]))
    G = nx.DiGraph(nodetype=int)
    eids = edges['eid'].tolist()
    lengths = edges['length'].tolist()
    times = edges['travel_time'].tolist()
    targets = edges['target'].tolist()
    for i in range(len(eids)):
        u = eids[i]
        tar = targets[i]
        out_edges = edges.query('source == ' + str(tar))
        v_set = out_edges['eid'].tolist()
        out_length = out_edges['length'].tolist()
        out_time = out_edges['travel_time'].tolist()
        for k in range(len(v_set)):
            G.add_edge(u, v_set[k], length=round(out_length[k], 3), time=round(out_time[k], 3))
        lens = round(lengths[i], 3)
        ts = round(times[i], 3)
        if G.has_node(u):
            G.nodes[u]['length'] = lens
            G.nodes[u]['time'] = ts
        else:
            G.add_node(u, length=lens, time=ts)
    pickle.dump(G, open(path, "wb"))
class SPoint:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng
    def __str__(self):
        return '({},{})'.format(self.lat, self.lng)
    def __repr__(self):
        return self.__str__()
    def __eq__(self, other):
        return self.lat == other.lat and self.lng == other.lng
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash(str(self.lat) + " " + str(self.lng))
DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS
EARTH_MEAN_RADIUS_METER = 6378137
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05
def distance(a, b):
    if a==b: return 0.0
    delta_lat = math.radians(b.lat - a.lat)
    delta_lng = math.radians(b.lng - a.lng)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(a.lat)) * math.cos(
        math.radians(b.lat)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d
class STPoint(SPoint):
    def __init__(self, lat, lng, time, data=None):
        super(STPoint, self).__init__(lat, lng)
        self.time = time
        self.data = data  
    def __str__(self):
        return str(self.__dict__)  
def cal_loc_along_line(a, b, rate):
    """
    convert rate to gps location
    """
    lat = a.lat + rate * (b.lat - a.lat)
    lng = a.lng + rate * (b.lng - a.lng)
    return SPoint(lat, lng)
class Trajectory:
    def __init__(self, pt_list):
        self.pt_list:List[STPoint] = pt_list
    def get_duration(self):
        return (self.pt_list[-1].time - self.pt_list[0].time).total_seconds()
    def get_distance(self):
        dist = 0.0
        pre_pt = self.pt_list[0]
        for pt in self.pt_list[1:]:
            tmp_dist = distance(pre_pt, pt)
            dist += tmp_dist
            pre_pt = pt
        return dist
    def get_avg_time_interval(self):
        point_time_interval = []
        for pre, cur in zip(self.pt_list[:-1], self.pt_list[1:]):
            point_time_interval.append((cur.time - pre.time).total_seconds())
        return sum(point_time_interval) / len(point_time_interval)
    def get_avg_distance_interval(self):
        point_dist_interval = []
        for pre, cur in zip(self.pt_list[:-1], self.pt_list[1:]):
            point_dist_interval.append(distance(pre, cur))
        return sum(point_dist_interval) / len(point_dist_interval)
    def get_mbr(self):
        return MBR.cal_mbr(self.pt_list)
    def get_start_time(self):
        return self.pt_list[0].time
    def get_end_time(self):
        return self.pt_list[-1].time
    def get_mid_time(self):
        return self.pt_list[0].time + (self.pt_list[-1].time - self.pt_list[0].time) / 2.0
    def get_centroid(self):
        mean_lat = 0.0
        mean_lng = 0.0
        for pt in self.pt_list:
            mean_lat += pt.lat
            mean_lng += pt.lng
        mean_lat /= len(self.pt_list)
        mean_lng /= len(self.pt_list)
        return SPoint(mean_lat, mean_lng)
    def query_trajectory_by_temporal_range(self, start_time, end_time):
        """
        Return the subtrajectory within start time and end time
        """
        traj_start_time = self.get_start_time()
        traj_end_time = self.get_end_time()
        if start_time > traj_end_time:
            return None
        if end_time <= traj_start_time:
            return None
        st = max(traj_start_time, start_time)
        et = min(traj_end_time + timedelta(seconds=1), end_time)
        start_idx = self.binary_search_idx(st)  
        if self.pt_list[start_idx].time < st:
            start_idx += 1
        end_idx = self.binary_search_idx(et)  
        if self.pt_list[end_idx].time < et:
            end_idx += 1
        sub_pt_list = self.pt_list[start_idx:end_idx]
        return Trajectory(sub_pt_list)
    def binary_search_idx(self, time):
        nb_pts = len(self.pt_list)
        if time < self.pt_list[0].time:
            return -1
        if time >= self.pt_list[-1].time:
            return nb_pts - 1
        left_idx = 0
        right_idx = nb_pts - 1
        while left_idx <= right_idx:
            mid_idx = int((left_idx + right_idx) / 2)
            if mid_idx < nb_pts - 1 and self.pt_list[mid_idx].time <= time < self.pt_list[mid_idx + 1].time:
                return mid_idx
            elif self.pt_list[mid_idx].time < time:
                left_idx = mid_idx + 1
            else:
                right_idx = mid_idx - 1
    def query_location_by_timestamp(self, time):
        """
        Return the GPS location given the time and trajectory (using linear interpolation).
        """
        idx = self.binary_search_idx(time)
        if idx == -1 or idx == len(self.pt_list) - 1:
            return None
        if self.pt_list[idx].time == time or (self.pt_list[idx + 1].time - self.pt_list[idx].time).total_seconds() == 0:
            return SPoint(self.pt_list[idx].lat, self.pt_list[idx].lng)
        else:
            dist_ab = distance(self.pt_list[idx], self.pt_list[idx + 1])
            if dist_ab == 0:
                return SPoint(self.pt_list[idx].lat, self.pt_list[idx].lng)
            dist_traveled = dist_ab * (time - self.pt_list[idx].time).total_seconds() / \
                            (self.pt_list[idx + 1].time - self.pt_list[idx].time).total_seconds()
            return cal_loc_along_line(self.pt_list[idx], self.pt_list[idx + 1], dist_traveled / dist_ab)
    def to_wkt(self):
        wkt = 'LINESTRING ('
        for pt in self.pt_list:
            wkt += '{} {}, '.format(pt.lng, pt.lat)
        wkt = wkt[:-2] + ')'
        return wkt
    def __hash__(self):
        return hash(self.pt_list[0].time.strftime('%Y%m%d%H%M%S') + '_' +
                    self.pt_list[-1].time.strftime('%Y%m%d%H%M%S'))
    def __eq__(self, other):
        return hash(self) == hash(other)
def gen_map(map_dir, out_dir):
    path_node=os.path.join(out_dir, "nodeOSM.txt") ; path_way=os.path.join(out_dir, "wayTypeOSM.txt")
    if os.path.exists(path_node) and os.path.exists(path_way):return
    print('build gen map')
    nodes = gpd.read_file(os.path.join(map_dir, "nodes.shp"))
    index = [i for i in range(nodes.shape[0])]
    nodes["fid"] = np.array(index, dtype=int)
    data = []
    nid_dict = {}
    for i in range(nodes.shape[0]):
        tmp = nodes.iloc[i]
        osmid = int(tmp['osmid'])
        fid = int(tmp['fid'])
        x = float(tmp['x'])
        y = float(tmp['y'])
        nid_dict[osmid] = fid
        data.append([fid, y, x])
    with open(path_node, 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)
    edges = gpd.read_file(os.path.join(map_dir, "edges.shp"))
    zone = [180, -180, 90, -90]
    rn_dict = {}
    data = []
    wayType = []
    for i in range(edges.shape[0]):
        tmp = edges.iloc[i]
        eid = int(tmp['fid'])
        u = int(tmp['u'])
        v = int(tmp['v'])
        points = tmp['geometry'].coords
        zone[0] = min(zone[0], np.min(points.xy[0]))
        zone[1] = max(zone[1], np.max(points.xy[0]))
        zone[2] = min(zone[2], np.min(points.xy[1]))
        zone[3] = max(zone[3], np.max(points.xy[1]))
        code = get_road_type(tmp['highway'])
        wayType.append([eid, code])
        row = [eid, nid_dict[u], nid_dict[v]]
        row.append(len(points))
        pts = []
        for lon, lat in points:
            row += [float(lat), float(lon)]
            pts.append([float(lat), float(lon)])
        data.append(row)
        tmp_dict = {"coords": pts, "length": float(tmp['length']), "level": code}
        rn_dict[eid] = tmp_dict
    print(zone)
    with open(os.path.join(out_dir, "rn_dict.json"), 'w') as fp:
        json.dump(rn_dict, fp)
    with open(os.path.join(out_dir, "edgeOSM.txt"), 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(data)
    with open(path_way, 'w') as fp:
        fields_output_file = csv.writer(fp, delimiter='\t')
        fields_output_file.writerows(wayType)
class CandidatePoint(SPoint):
    def __init__(self, lat, lng, eid, error, offset, rate):
        super(CandidatePoint, self).__init__(lat, lng)
        self.eid = eid
        self.error = error
        self.offset = offset
        self.rate = rate
    def __str__(self):
        return '{},{},{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error, self.offset, self.rate)
    def __repr__(self):
        return '{},{},{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error, self.offset, self.rate)
    def __hash__(self):
        return hash(self.__str__())
def get_graph(rn, trajs):
    freq = {}
    for i in range(len(rn.nodes)):
        nbrs = list(rn.neighbors(i))
        for item in nbrs:
            freq[(i, item)] = 1
    for traj in trajs:
        for pt1, pt2 in zip(traj.pt_list[:-1], traj.pt_list[1:]):
            key = (pt1.data['candi_pt'].eid, pt2.data['candi_pt'].eid)
            if key in freq:
                freq[key] += 1
            else:
                freq[key] = 1
    data = []
    for k, v in freq.items():
        data.append([k[0], k[1], v])
    return data
oid2uid={}
def parse_trajs(ori_dir, mode, low_cate, tz, scale,use_num=True): 
    if mode == 'train':
        ori_file = os.path.join(ori_dir, "traj_train.csv")
    elif mode == 'valid':
        ori_file = os.path.join(ori_dir, "traj_valid.csv")
    elif mode == 'test':
        ori_file = os.path.join(ori_dir, "traj_test.csv")
    else:
        raise NotImplementedError
    mm = pd.read_csv(ori_file, sep=",", header=None, names=['oid', 'tid', 'offsets', 'path', 'raw', 'low']).to_numpy()
    num = len(mm)
    if use_num:
        if mode == 'valid':
            num = min(10000, num)
        elif mode == 'test':
            num = min(50000, num)
    trajs = []
    my_uids,my_ts=[],[]
    for oid, tid, offsets, path, raw, low in mm[:num]:
        rec = literal_eval(raw)
        pt_list = []
        my_t=[]
        for attrs in rec:
            lng = float(attrs[0])
            lat = float(attrs[1])
            timestamp = int(attrs[2])
            lng_p = float(attrs[3])
            lat_p = float(attrs[4])
            rid = int(attrs[5])
            rate = float(attrs[6])
            offset = float(attrs[7])
            speed = float(attrs[8])
            idx = int(attrs[9])
            dist = haversine((lat, lng), (lat_p, lng_p), unit='m')
            candi_pt = CandidatePoint(lat_p, lng_p, rid, dist, offset, round(rate, 4))
            pt = STPoint(lat, lng, timestamp, {'candi_pt': candi_pt})
            my_t.append(np.array([lng_p,lat_p,timestamp]))
            pt.time_arr = dt.datetime.fromtimestamp(timestamp, tz)
            pt.speed = round(speed, 2)
            pt.cpath_idx = idx
            pt_list.append(pt)
        if len(pt_list) > 2:
            traj = Trajectory(pt_list)
            cpath = literal_eval(path)
            cpath, *_ = zip(*cpath)
            cpath = list(cpath)
            traj.cpath = cpath
            traj.low_idx = literal_eval(low)[low_cate]
            traj.tid = tid
            trajs.append(traj)
            if oid not in oid2uid:oid2uid[oid]=len(oid2uid)
            my_uids.append(oid2uid[oid])
            my_ts.append(np.array(my_t))
    return trajs,my_ts,np.array(my_uids)
def read_allts(ori_dir, low_cate, tz, scale):
    train_trajs,train_my_ts,train_my_uids=parse_trajs(ori_dir, "train", low_cate, tz, scale,False)
    valid_trajs,valid_my_ts,valid_my_uids=parse_trajs(ori_dir, "valid", low_cate, tz, scale,False)
    test_trajs,test_my_ts,test_my_uids=parse_trajs(ori_dir, "test", low_cate, tz, scale,False)
    trajs=[*train_trajs,*valid_trajs,*test_trajs]
    my_ts=np.array([*train_my_ts,*valid_my_ts,*test_my_ts],dtype=object)
    my_uid=np.concatenate([train_my_uids,valid_my_uids,test_my_uids])
    assert len(trajs)==len(my_ts)==len(my_uid)
    return trajs,my_uid,my_ts
def get_model_data(ori_dir, root_data, utc, low_cate, scale,call_tvt_urv,rt_if_exist=False):
    path_ts=os.path.join(root_data, "pre-ts.pk.zst")
    if os.path.exists(path_ts):
        if rt_if_exist:return True
        trajs,my_uid,my_ts,train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx=loadZ_pk(path_ts)
    else:
        print('build get_model_data',ori_dir, root_data, utc, low_cate, scale)
        tz = dt.timezone(dt.timedelta(hours=utc))
        check_dir(path_ts)
        trajs,my_uid,my_ts=read_allts(ori_dir,low_cate,tz,scale)
        train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx=call_tvt_urv(my_ts,my_uid)
        assert len(val_idx)>5 and len(test_idx)>5 and len(train_idx)>5 and len(du_idx)>5 and len(dr_idx)>5 and len(dv_idx)>5
        saveZ_pk(path_ts,[trajs,my_uid,my_ts,train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx])
    assert len(val_idx)>5 and len(test_idx)>5 and len(train_idx)>5 and len(du_idx)>5 and len(dr_idx)>5 and len(dv_idx)>5
    du_traj,dr_traj,dv_traj=[trajs[i] for i in du_idx],[trajs[i] for i in dr_idx],[trajs[i] for i in dv_idx]
    train_trajs,valid_trajs,test_out_trajs=[trajs[i] for i in train_idx],[trajs[i] for i in val_idx],[trajs[i] for i in test_idx]
    assert len(du_traj)>5 and len(dv_traj)>5 and len(dr_traj)>5 and len(train_trajs)>5 and len(valid_trajs)>5 and len(test_out_trajs)>5
    return train_trajs,valid_trajs,test_out_trajs,du_traj,dr_traj,dv_traj,my_ts[du_idx],my_ts[dr_idx],my_ts[dv_idx]
def api_preprocess(root_map,root_data,dataname,call_tvt_urv,rt_if_exist):
    """train,val,test,du,dr,dv,raw_du,raw_dr,raw_dv"""
    utc=city_info(dataname.lower())[-1]
    map_map=os.path.join(root_map, "map")
    gen_dam(root_map,os.path.join(root_map, "traj_train.csv"),utc,edges_shp=map_map)
    txt2npy(root_map)
    get_road_graph(root_map)
    gen_map(map_map,map_map)
    return get_model_data(root_map,root_data, utc,0, scale=100,call_tvt_urv=call_tvt_urv,rt_if_exist=rt_if_exist)
class MBR:
    def __init__(self, min_lat, min_lng, max_lat, max_lng):
        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng
    def contains(self, lat, lng):
        return self.min_lat <= lat < self.max_lat and self.min_lng <= lng < self.max_lng
    def center(self):
        return (self.min_lat + self.max_lat) / 2.0, (self.min_lng + self.max_lng) / 2.0
    def get_h(self):
        return distance(SPoint(self.min_lat, self.min_lng), SPoint(self.max_lat, self.min_lng))
    def get_w(self):
        return distance(SPoint(self.min_lat, self.min_lng), SPoint(self.min_lat, self.max_lng))
    def __str__(self):
        h = self.get_h()
        w = self.get_w()
        return '{}x{}m2'.format(h, w)
    def __eq__(self, other):
        return self.min_lat == other.min_lat and self.min_lng == other.min_lng \
               and self.max_lat == other.max_lat and self.max_lng == other.max_lng
    def to_wkt(self):
        return 'POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))'.format(self.min_lng, self.min_lat,self.min_lng, self.max_lat,self.max_lng, self.max_lat,self.max_lng, self.min_lat,self.min_lng, self.min_lat)
    @staticmethod
    def cal_mbr(coords):
        """
        Find MBR from coordinates
        Args:
        -----
        coords:
            list of Point()
        Returns:
        -------
        MBR()
        """
        min_lat = float('inf')
        min_lng = float('inf')
        max_lat = float('-inf')
        max_lng = float('-inf')
        for coord in coords:
            if coord.lat > max_lat:
                max_lat = coord.lat
            if coord.lat < min_lat:
                min_lat = coord.lat
            if coord.lng > max_lng:
                max_lng = coord.lng
            if coord.lng < min_lng:
                min_lng = coord.lng
        return MBR(min_lat, min_lng, max_lat, max_lng)
    @staticmethod
    def load_mbr(file_path):
        with open(file_path, 'r') as f:
            f.readline()
            attrs = f.readline()[:-1].split(';')
            mbr = MBR(float(attrs[1]), float(attrs[2]), float(attrs[3]), float(attrs[4]))
        return mbr
    @staticmethod
    def store_mbr(mbr, file_path):
        with open(file_path, 'w') as f:
            f.write('name;min_lat;min_lng;max_lat;max_lng;wkt\n')
            f.write('{};{};{};{};{};{}\n'.format(0, mbr.min_lat, mbr.min_lng, mbr.max_lat, mbr.max_lng, mbr.to_wkt()))
def bearing(a, b):
    """
    Calculate the bearing of ab
    """
    pt_a_lat_rad = math.radians(a.lat)
    pt_a_lng_rad = math.radians(a.lng)
    pt_b_lat_rad = math.radians(b.lat)
    pt_b_lng_rad = math.radians(b.lng)
    y = math.sin(pt_b_lng_rad - pt_a_lng_rad) * math.cos(pt_b_lat_rad)
    x = math.cos(pt_a_lat_rad) * math.sin(pt_b_lat_rad) - math.sin(pt_a_lat_rad) * math.cos(pt_b_lat_rad) * math.cos(
        pt_b_lng_rad - pt_a_lng_rad)
    bearing_rad = math.atan2(y, x)
    return math.fmod(math.degrees(bearing_rad) + 360.0, 360.0)
def project_pt_to_segment(a, b, t):
    """
    Args:
    -----
    a,b: start/end GPS location of a road segment
    t: raw point
    Returns:
    -------
    project: projected GPS point on road segment
    rate: rate of projected point location to road segment
    dist: haversine_distance of raw and projected point
    """
    ab_angle = bearing(a, b)
    at_angle = bearing(a, t)
    ab_length = distance(a, b)
    at_length = distance(a, t)
    delta_angle = at_angle - ab_angle
    meters_along = at_length * math.cos(math.radians(delta_angle))
    if ab_length == 0.0:
        rate = 0.0
    else:
        rate = meters_along / ab_length
    if rate >= 1:
        projection = SPoint(b.lat, b.lng)
        rate = 1.0
    elif rate <= 0:
        projection = SPoint(a.lat, a.lng)
        rate = 0.0
    else:
        projection = cal_loc_along_line(a, b, rate)
    dist = distance(t, projection)
    return projection, rate, dist
def project_pt_to_road(rn, t, rid):
    """
    Args:
    -----
    rn: road_network
    t: raw point
    rid: road edge id
    Returns:
    -------
    project: projected GPS point on road segment
    rate: rate of projected point location to road segment
    dist: haversine_distance of raw and projected point
    """
    edge_cords = rn.edgeCord[rid]
    dis = [distance(t, SPoint(edge_cords[2 * i], edge_cords[2 * i + 1])) for i in range(len(edge_cords) // 2)]
    idx = np.argmin(dis)
    candidate = []
    if idx != 0:
        candidate.append([*project_pt_to_segment(SPoint(edge_cords[2 * (idx - 1)], edge_cords[2 * (idx - 1) + 1]),
                                                 SPoint(edge_cords[2 * idx], edge_cords[2 * idx + 1]), t), idx])
    if idx != len(edge_cords) // 2 - 1:
        candidate.append([*project_pt_to_segment(SPoint(edge_cords[2 * idx], edge_cords[2 * idx + 1]),
                                                 SPoint(edge_cords[2 * (idx + 1)], edge_cords[2 * (idx + 1) + 1]), t),
                          idx + 1])
    best_candidate = candidate[0]
    if len(candidate) == 2 and candidate[0][2] > candidate[1][2]:
        best_candidate = candidate[1]
    projection, rate, dist, idx = best_candidate
    dist_to_end = (1 - rate) * distance(SPoint(edge_cords[2 * (idx - 1)], edge_cords[2 * (idx - 1) + 1]),
                                        SPoint(edge_cords[2 * idx], edge_cords[2 * idx + 1])) + rn.edgeOffset[rid][idx]
    if rn.edgeDis[rid] > 0:
        return projection, 1 - (dist_to_end / rn.edgeDis[rid]), dist
    else:
        return projection, 1, dist
def exp_prob(beta, x):
    """
    error distance weight.
    """
    return math.exp(-pow(x, 2) / pow(beta, 2))
class RoadNetworkMapFull:
    def __init__(self, dir, zone_range, unit_length):
        edgeFile = open(os.path.join(dir, 'edgeOSM.txt'))
        self.rtreeFile = os.path.join(dir, 'rtree')
        self.edgeDis = []
        self.edgeNode = []
        self.edgeCord = []
        self.edgeOffset = []
        self.nodeSet = set()
        self.nodeDict = {}
        self.edgeDict = {}
        self.edgeRevDict = {}
        self.nodeEdgeDict = {}
        self.nodeEdgeRevDict = {}
        self.zone_range = zone_range
        self.unit_length = unit_length
        self.minLat = 1e18
        self.maxLat = -1e18
        self.minLon = 1e18
        self.maxLon = -1e18
        self.edgeNum = 0
        self.nodeNum = 0
        self.valid_edge = {}
        self.valid_to_origin = {}
        self.valid_edge_cnt = 0
        self.edge_to_cluster = {}
        self.cluster_to_edge = {}
        self.cluster_neighbor = {}
        self.cluster_neighbor_edge = {}
        self.cluster_neighbor_cluster = {}
        for line in edgeFile.readlines():
            item_list = line.strip().split()
            a = int(item_list[1])
            b = int(item_list[2])
            self.edgeNode.append((a, b))
            self.nodeDict[a] = b
            if a not in self.nodeEdgeDict:
                self.nodeEdgeDict[a] = []
            if b not in self.nodeEdgeRevDict:
                self.nodeEdgeRevDict[b] = []
            self.nodeEdgeDict[a].append(self.edgeNum)
            self.nodeEdgeRevDict[b].append(self.edgeNum)
            self.nodeSet.add(a)
            self.nodeSet.add(b)
            num = int(item_list[3])
            dist = 0
            self.edgeCord.append(list(map(float, item_list[4:])))
            inzone_flag = True
            for i in range(num):
                tmplat = float(item_list[4 + i * 2])
                tmplon = float(item_list[5 + i * 2])
                self.minLat = min(self.minLat, tmplat)
                self.maxLat = max(self.maxLat, tmplat)
                self.minLon = min(self.minLon, tmplon)
                self.maxLon = max(self.maxLon, tmplon)
                inzone_flag = inzone_flag and self.inside_zone(tmplat, tmplon)
            if inzone_flag:
                self.valid_edge[self.edgeNum] = self.valid_edge_cnt
                self.valid_to_origin[self.valid_edge_cnt] = self.edgeNum
                self.valid_edge_cnt += 1
            offset = []
            for i in range(num - 1):
                dist += self.calSpatialDistance(float(item_list[4 + i * 2]), float(item_list[5 + i * 2]),
                                                float(item_list[6 + i * 2]), float(item_list[7 + i * 2]))
                offset.append(self.calSpatialDistance(float(item_list[4 + i * 2]), float(item_list[5 + i * 2]),
                                                float(item_list[6 + i * 2]), float(item_list[7 + i * 2])))
            self.edgeDis.append(dist)
            for i in range(len(offset) - 1, 0, -1):
                offset[i - 1] = offset[i - 1] + offset[i]
            offset.append(0)
            self.edgeOffset.append(offset)
            self.edgeNum += 1
        self.valid_edge_one = {}
        for (key, value) in self.valid_edge.items():
            self.valid_edge_one[key] = value + 1
        self.valid_to_origin_one = {}
        for (key, value) in self.valid_to_origin.items():
            self.valid_to_origin_one[key + 1] = value
        self.valid_edge_cnt_one = self.valid_edge_cnt + 1
        self.spatial_index = Rtree(self.rtreeFile)
        for rid in self.valid_edge.keys():
            edge_cords = self.edgeCord[rid]
            cords = []
            for i in range(len(edge_cords) // 2):
                cords.append(SPoint(edge_cords[2 * i], edge_cords[2 * i + 1]))
            mbr = MBR.cal_mbr(cords)
            self.spatial_index.insert(rid, (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        mid_point = SPoint((self.zone_range[0] + self.zone_range[2]) / 2,
                            (self.zone_range[1] + self.zone_range[3]) / 2)
        min_dist = 1e18
        best_cand = -1
        for rid in self.valid_edge.keys():
            gps, rate, dist = project_pt_to_road(self, mid_point, rid)
            if dist < min_dist:
                min_dist = dist
                best_cand = rid
        self.valid_to_origin_one[0] = best_cand  
        self.nodeNum = len(self.nodeSet)
        self.mbr = MBR(*zone_range)
        for eid in range(self.edgeNum):
            self.edgeRevDict[eid] = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            self.edgeDict[eid] = []
            if b in self.nodeEdgeDict:
                for nid in self.nodeEdgeDict[b]:
                    self.edgeDict[eid].append(nid)
                    self.edgeRevDict[nid].append(eid)
        edge_list = []
        edge_weight_list = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            if (a == b):
                continue
            edge_list.append((a, b))
            edge_weight_list.append(self.edgeDis[eid])
        print('edge Num: ', self.edgeNum)
        print('node Num: ', self.nodeNum)
        print('valid edge Num: ', self.valid_edge_cnt)
        self.wayType = {}
        wayFile = open(os.path.join(dir, 'wayTypeOSM.txt'))
        for line in wayFile.readlines():
            item_list = line.strip().split()
            roadId = int(item_list[0])
            wayId = int(item_list[-1])
            self.wayType[roadId] = wayId
        self.long_num = int((self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[0],
                                                self.zone_range[3]) + unit_length - 1) / unit_length)
        self.width_num = int((self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[2],
                                                       self.zone_range[1]) + unit_length - 1) / unit_length)
        self.cnn_graph = np.zeros((self.long_num, self.width_num)).astype(int)
        self.cnn_to_edge = np.ones((self.long_num, self.width_num)).astype(int) * self.valid_edge_cnt
        print('long length: ',
              self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[0], self.zone_range[3]),
              self.long_num)
        print('width length: ',
              self.calSpatialDistance(self.zone_range[0], self.zone_range[1], self.zone_range[2], self.zone_range[1]),
              self.width_num)
        print ('----- construct cnn graph ------')
        self.construct_cnn_graph()
        del self.spatial_index 
        self.spatial_index=Rtree(self.rtreeFile)
    def range_query(self, mbr: MBR) -> list:
        eids = self.spatial_index.intersection((mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        return list(eids)
    def nearest_query(self, gps: SPoint, return_type='spoint') -> SPoint:
        search_dist = 50
        while True:
            mbr = MBR(gps.lat - search_dist * LAT_PER_METER,
                      gps.lng - search_dist * LNG_PER_METER,
                      gps.lat + search_dist * LAT_PER_METER,
                      gps.lng + search_dist * LNG_PER_METER)
            candis = self.get_candidates(gps, mbr)
            if len(candis) > 0:
                best_cand = None
                min_err = 1e9
                for cand in candis:
                    if cand.error < min_err:
                        min_err = cand.error
                        if return_type == 'spoint':
                            best_cand = SPoint(cand.lat, cand.lng)
                        else:
                            best_cand = cand
                return best_cand
            else:
                search_dist = search_dist * 2
    def point_in_mbr(self, x: SPoint, mbr: MBR) -> bool:
        return mbr.min_lat <= x.lat <= mbr.max_lat and mbr.min_lng <= x.lng <= mbr.max_lng
    def get_candidates(self, x: SPoint, mbr: MBR) -> list:
        candi = self.range_query(mbr)
        refined_candi = []
        for eid in candi:
            projected, rate, dist = project_pt_to_road(self, x, eid)
            if self.point_in_mbr(projected, mbr):
                candidate = CandidatePoint(projected.lat, projected.lng, eid, dist, rate * self.edgeDis[eid], rate)
                refined_candi.append(candidate)
        return refined_candi
    def DDALine(self, x1, y1, x2, y2, valid_edge_id, way_id):
        if self.cnn_graph[x1,y1] < way_id:
            self.cnn_graph[x1,y1] = way_id
            self.cnn_to_edge[x1,y1] = valid_edge_id
        if (x1 == x2 and y1 == y2):
            return
        dx = x2 - x1
        dy = y2 - y1
        steps = 0
        if abs(dx) > abs(dy):
            steps = abs(dx)
        else:
            steps = abs(dy)
        delta_x = float(dx / steps)
        delta_y = float(dy / steps)
        x = x1 + 0.5
        y = y1 + 0.5
        for i in range(0, int(steps + 1)):
            if self.cnn_graph[int(x),int(y)] < way_id:
                self.cnn_graph[int(x),int(y)] = way_id
                self.cnn_to_edge[int(x),int(y)] = valid_edge_id
            x += delta_x
            y += delta_y
    def get_cnn_id(self, lat, lon):
        x = int ( (lon - self.zone_range[1]) / ((self.zone_range[3] - self.zone_range[1]) / self.long_num))
        y = int ( (lat - self.zone_range[0]) / ((self.zone_range[2] - self.zone_range[0]) / self.width_num))
        return x,y
    def draw_cnn_line(self, stlat, stlon, edlat, edlon, valid_edge_id, way_id):
        stx, sty = self.get_cnn_id(stlat, stlon)
        edx, edy = self.get_cnn_id(edlat, edlon)
        self.DDALine(stx, sty, edx, edy, valid_edge_id, way_id)
    def construct_cnn_graph(self):
        for i in range(self.edgeNum):
            if i in self.valid_edge:
                cord_len = len(self.edgeCord[i])
                for j in range(cord_len // 2 - 2):
                    stlat = self.edgeCord[i][j * 2]
                    stlon = self.edgeCord[i][j * 2 + 1]
                    edlat = self.edgeCord[i][j * 2 + 2]
                    edlon = self.edgeCord[i][j * 2 + 3]
                    self.draw_cnn_line(stlat, stlon, edlat, edlon, self.valid_edge[i], self.wayType[i])
    def inside_zone(self, lat, lon):
        return self.zone_range[0] <= lat and lat <= self.zone_range[2] and self.zone_range[1] <= lon and lon <= \
               self.zone_range[3]
    def calSpatialDistance(self, x1, y1, x2, y2):
        lat1 = (math.pi / 180.0) * x1
        lat2 = (math.pi / 180.0) * x2
        lon1 = (math.pi / 180.0) * y1
        lon2 = (math.pi / 180.0) * y2
        R = 6378.137
        t = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        if t > 1.0:
            t = 1.0
        d = math.acos(t) * R * 1000
        return d
    def edgeDistance(self, edgeId):
        return self.edgeDis[edgeId]
    def getEdgeNode(self, edgeId):
        return self.edgeNode[edgeId]
    def shortestPathAll(self, start, end=-1, with_route=False, max_len=1e18):
        pq = PriorityQueue()
        pq.put((0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            if dis > max_len:
                if end != -1:
                    return 1e18, []
                else:
                    return dis, pred
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    if dist[nid] > dist[id] + self.edgeDis[nid]:
                        dist[nid] = dist[id] + self.edgeDis[nid]
                        pred[nid] = id
                        pq.put((dist[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred
    def shortestAStarPath(self, start, end, with_route=False, max_len=1e18):
        pq = PriorityQueue()
        st = self.edgeCord[start][-2:]
        en = self.edgeCord[end][-2:]
        pq.put((self.calSpatialDistance(*st, *en), 0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while pq.qsize():
            h, dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            if h > max_len:
                return 1e18, []
            for nid in self.edgeDict[id]:
                if nid in self.valid_edge:
                    if dist[nid] > dist[id] + self.edgeDis[nid]:
                        dist[nid] = dist[id] + self.edgeDis[nid]
                        pred[nid] = id
                        st = self.edgeCord[nid][-2:]
                        pq.put((dist[nid] + self.calSpatialDistance(*st, *en), dist[nid], nid))
        if not with_route:
            pred = []
        return dist[end], pred
    def dotproduct(self, v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))
    def length(self, v):
        return math.sqrt(self.dotproduct(v, v))
    def cal_cosine(self, v1, v2):
        if (self.length(v1) < 1e-5 or self.length(v2) < 1e-5):
            cos_value = 1
        else:
            cos_value = self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2))
        return 0.5 + 0.5 * cos_value
    def cal_angle(self, eid, nid):
        ex, ey = self.edgeCord[eid][-2] - self.edgeCord[eid][0], self.edgeCord[eid][-1] - self.edgeCord[eid][1]
        nx, ny = self.edgeCord[nid][-2] - self.edgeCord[nid][0], self.edgeCord[nid][-1] - self.edgeCord[nid][1]
        v1 = (ex, ey)
        v2 = (nx, ny)
        if (self.length(v1) < 1e-5 or self.length(v2) < 1e-5):
            return 0
        else:
            return math.acos(round(self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2))))
    def shortestRankPathAll(self, start, end=-1, with_route=False):
        pq = PriorityQueue()
        pq.put((0, 0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        dist2 = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = 0
        dist2[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, dis2, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    en_rank = self.wayType[id] > self.wayType[nid]
                    if (dist[nid] > dist[id] + en_rank) or (
                            dist[nid] == dist[id] + en_rank and dist2[nid] > dist2[id] + self.edgeDis[nid]):
                        dist[nid] = dist[id] + en_rank
                        dist2[nid] = dist2[id] + self.edgeDis[nid]
                        pred[nid] = id
                        pq.put((dist[nid], dist2[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred
    def shortestAnglePathAll(self, start, end=-1, with_route=False):
        pq = PriorityQueue()
        pq.put((0, start))
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = 0
        pred[start] = -1
        nodeset = {}
        while (pq.qsize()):
            dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            for nid in self.edgeDict[id]:
                if (nid in self.valid_edge):
                    en_angle = self.cal_angle(id, nid)
                    if dist[nid] > dist[id] + en_angle:
                        dist[nid] = dist[id] + en_angle
                        pred[nid] = id
                        pq.put((dist[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred
    def shortestPath(self, start, end, stype='slen', with_route=True, max_len=1e18):
        if stype == 'slen':
            if end != -1:
                dis, pred = self.shortestAStarPath(start, end, with_route, max_len)
            else:
                dis, pred = self.shortestPathAll(start, end, with_route, max_len)
        elif stype == 'rlen':
            dis, pred = self.shortestRankPathAll(start, end, with_route)
        elif stype == 'alen':
            dis, pred = self.shortestAnglePathAll(start, end, with_route)
        if end == -1:
            return dis, pred
        if dis > max_len:
            return dis, []
        if with_route:
            id = end
            arr = [id]
            while pred[id] >= 0 and pred[id] < 1e18:
                id = pred[id]
                arr.append(id)
            arr = list(reversed(arr))
            return dis, arr
        else:
            return dis
    def output_dataset_part(self, data_file, start, end, w):
        f = open(data_file + '_%d_%d' % (start, end), 'w')
        for eid in range(start, end):
            if eid in self.valid_edge:
                connect = [eid]
                sdis, spred = self.shortestPath(start=eid, end=-1, stype='slen')
                rdis, rpred = self.shortestPath(start=eid, end=-1, stype='rlen')
                adis, apred = self.shortestPath(start=eid, end=-1, stype='alen')
                for nid in range(self.edgeNum):
                    if (nid in self.valid_edge) and (eid != nid) and sdis[nid] < 1e18 and rdis[nid] < 1e18 and adis[
                        nid] < 1e18:
                        connect.append(nid)
                if len(connect) > 1:
                    des_list = np.random.choice(connect[1:], size=min(len(connect[1:]), w), replace=False)
                    for des in des_list:
                        for ipred in [spred, rpred, apred]:
                            id = des
                            arr = [id]
                            while (ipred[id] >= 0 and ipred[id] < 1e18):
                                id = ipred[id]
                                arr.append(id)
                            arr = list(reversed(arr))
                            f.write(' '.join(list(map(str, arr))) + '\n')
    def output_dataset(self, data_size, data_num, data_file):
        w = data_size // self.valid_edge_cnt
        p = Pool(data_num)
        for i in range(data_num):
            start = min((self.edgeNum // data_num + 1) * i, self.edgeNum)
            end = min((self.edgeNum // data_num + 1) * (i + 1), self.edgeNum)
            if start < end:
                p.apply_async(self.output_dataset_part, args=(data_file, start, end, w))
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
    def output_train_dataset(self, output_data_file, data_num, data_file):
        g = open(output_data_file, 'w')
        for i in range(data_num):
            start = min((self.edgeNum // data_num + 1) * i, self.edgeNum)
            end = min((self.edgeNum // data_num + 1) * (i + 1), self.edgeNum)
            f = open(data_file + '_%d_%d' % (start, end), 'r')
            for line in f.readlines():
                g.write(line)
    def get_rid_rnfea_dict(self, dam, interval) -> np.array:
        freq = dam.vehicle_num.sum(axis=-1)
        freq_max = np.max(freq)
        freq_min = np.min(freq)
        dim = 18
        norm_feat = np.zeros([self.valid_edge_cnt_one, dim], dtype=np.float32)
        max_length = np.max(self.edgeDis)
        for rid in self.valid_edge.keys():
            norm_rid = [0. for _ in range(dim)]
            norm_rid[0] = np.log10(self.edgeDis[rid] + 1e-6) / np.log10(max_length)
            norm_rid[self.wayType[rid] + 1] = 1
            in_degree = 0
            for eid in self.edgeDict[rid]:
                if eid in self.valid_edge.keys():
                    in_degree += 1
            out_degree = 0
            for eid in self.edgeRevDict[rid]:
                if eid in self.valid_edge.keys():
                    out_degree += 1
            norm_rid[9] = in_degree
            norm_rid[10] = out_degree
            norm_rid[11] = (self.edgeCord[rid][0] - self.minLat) / (self.maxLat - self.minLat)
            norm_rid[12] = (self.edgeCord[rid][1] - self.minLon) / (self.maxLon - self.minLon)
            norm_rid[13] = (self.edgeCord[rid][-2] - self.minLat) / (self.maxLat - self.minLat)
            norm_rid[14] = (self.edgeCord[rid][-1] - self.minLon) / (self.maxLon - self.minLon)
            v1 = (self.edgeCord[rid][-2] - self.edgeCord[rid][0], self.edgeCord[rid][-1] - self.edgeCord[rid][1])
            v2 = (90 - self.edgeCord[rid][0], self.edgeCord[rid][1] - self.edgeCord[rid][1])
            norm_rid[15] = self.cal_cosine(v1, v2)
            norm_rid[16] = dam.seg_info.get_seg_travel_time(rid) / interval
            norm_rid[17] = (freq[rid] - freq_min) / (freq_max - freq_min)
            norm_feat[self.valid_edge_one[rid]] = np.array(norm_rid, dtype=np.float32)
        norm_feat[0] = np.array([0 for _ in range(dim)], dtype=np.float32)
        return norm_feat
    def get_nearest_seg_tmp(self, gps, trg_id):
        """
        Args:
        -----
        gps: [SPoint, tid]
        """
        step = 10
        search_dist = 10
        gps = SPoint(gps[0], gps[1])
        candis = []
        ids = []
        flag = True
        res = []
        while trg_id not in ids:
            mbr = MBR(gps.lat - search_dist * LAT_PER_METER,
                      gps.lng - search_dist * LNG_PER_METER,
                      gps.lat + search_dist * LAT_PER_METER,
                      gps.lng + search_dist * LNG_PER_METER)
            candis = self.get_candidates(gps, mbr)
            if flag and len(candis) > 0:
                res.append(search_dist)
                flag = False
            ids = [item.eid for item in candis]
            search_dist += step
        search_dist -= step
        res.append(search_dist)
        candis.sort(key=lambda elem: elem.error)
        ids = [item.eid for item in candis]
        idx = ids.index(trg_id)
        res.append(idx + 1)
        return res
    def get_nearest_seg(self, gps, vec_l, vec_p, candi_size, search_dist=100, beta=15):
        """
        Args:
        -----
        gps: [SPoint, tid]
        """
        step = 10
        gps = SPoint(gps[0], gps[1])
        candis = []
        while len(candis) == 0:
            mbr = MBR(gps.lat - search_dist * LAT_PER_METER,
                      gps.lng - search_dist * LNG_PER_METER,
                      gps.lat + search_dist * LAT_PER_METER,
                      gps.lng + search_dist * LNG_PER_METER)
            candis = self.get_candidates(gps, mbr)
            search_dist += step
        search_dist -= step
        candis.sort(key=lambda elem: elem.error)
        candis = candis[:candi_size]
        for candi in candis:
            e_lat, e_lng = self.edgeCord[candi.eid][-2] - self.edgeCord[candi.eid][0], self.edgeCord[candi.eid][-1] - self.edgeCord[candi.eid][1]
            v1 = (e_lat, e_lng)
            candi.cosv = self.cal_cosine(v1, vec_l)
            candi.cosv_pre = self.cal_cosine(v1, vec_p)
            f_lat, f_lng = gps.lat - self.edgeCord[candi.eid][0], gps.lng - self.edgeCord[candi.eid][1]
            l_lat, l_lng = self.edgeCord[candi.eid][-2] - gps.lat, self.edgeCord[candi.eid][-1] - gps.lng
            p_lat, p_lng = candi.lat - gps.lat, candi.lng - gps.lng
            candi.cosf = self.cal_cosine(v1, (f_lat, f_lng))
            candi.cosl = self.cal_cosine(v1, (l_lat, l_lng))
            candi.cos1 = self.cal_cosine((-f_lat, -f_lng), (l_lat, l_lng))
            candi.cos2 = self.cal_cosine((-f_lat, -f_lng), (p_lat, p_lng))
            candi.cos3 = self.cal_cosine((l_lat, l_lng), (p_lat, p_lng))
            candi.cosp = self.cal_cosine(v1, (p_lat, p_lng))
            candi.err_weight = exp_prob(beta, candi.error)
        return candis
    def get_gps_around(self, gps_seq, search_dist):
        ls_candis = []
        for gps in gps_seq:
            tmp = SPoint(gps[0], gps[1])
            mbr = MBR(tmp.lat - search_dist * LAT_PER_METER,
                      tmp.lng - search_dist * LNG_PER_METER,
                      tmp.lat + search_dist * LAT_PER_METER,
                      tmp.lng + search_dist * LNG_PER_METER)
            candis = self.get_candidates(tmp, mbr)
            ls_candis.append(candis)
        return ls_candis
    def get_trg_segs(self, gps_seq, candi_size, search_dist, beta):
        vecs = list(zip(gps_seq[:-1], gps_seq[1:]))
        vecs_later = vecs + [vecs[-1]]
        vecs_pre = [vecs[0]] + vecs
        ls_trg_segs = []
        for ds_pt, (p1, p2), (p3, p4) in zip(gps_seq, vecs_later, vecs_pre):
            segs = self.get_nearest_seg(ds_pt, (p2[0] - p1[0], p2[1] - p1[1]), (p4[0] - p3[0], p4[1] - p3[1]), candi_size, search_dist, beta)
            ls_trg_segs.append(segs)
        return ls_trg_segs
    def pt2seg(self, gps, eid):
        projected, rate, dist = project_pt_to_road(self, gps, eid)
        candidate = CandidatePoint(projected.lat, projected.lng, eid, dist, rate * self.edgeDis[eid], rate)
        return candidate
    def add_candi_attrs(self, candi, gps, vec_l, vec_p, mode, beta):
        e_lat, e_lng = self.edgeCord[candi.eid][-2] - self.edgeCord[candi.eid][0], self.edgeCord[candi.eid][-1] - \
                       self.edgeCord[candi.eid][1]
        v1 = (e_lat, e_lng)
        candi.cosv = self.cal_cosine(v1, vec_l)
        candi.cosv_pre = self.cal_cosine(v1, vec_p)
        if mode == 0:
            candi.cosv_pre = 1.
        if mode == 2:
            candi.cosv = 1.
        f_lat, f_lng = gps.lat - self.edgeCord[candi.eid][0], gps.lng - self.edgeCord[candi.eid][1]
        l_lat, l_lng = self.edgeCord[candi.eid][-2] - gps.lat, self.edgeCord[candi.eid][-1] - gps.lng
        p_lat, p_lng = candi.lat - gps.lat, candi.lng - gps.lng
        candi.cosf = self.cal_cosine(v1, (f_lat, f_lng))
        candi.cosl = self.cal_cosine(v1, (l_lat, l_lng))
        candi.cos1 = self.cal_cosine((-f_lat, -f_lng), (l_lat, l_lng))
        candi.cos2 = self.cal_cosine((-f_lat, -f_lng), (p_lat, p_lng))
        candi.cos3 = self.cal_cosine((l_lat, l_lng), (p_lat, p_lng))
        candi.cosp = self.cal_cosine(v1, (p_lat, p_lng))
        candi.err_weight = exp_prob(beta, candi.error)
        return candi
def gps2grid(pt, mbr, grid_size):
    """
    mbr:
        MBR class.
    grid size:
        int. in meter
    """
    LAT_PER_METER = 8.993203677616966e-06
    LNG_PER_METER = 1.1700193970443768e-05
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size
    lat = pt.lat
    lng = pt.lng
    locgrid_x = int((lat - mbr.min_lat) / lat_unit) + 1
    locgrid_y = int((lng - mbr.min_lng) / lng_unit) + 1
    return locgrid_x, locgrid_y
def get_normalized_t(first_pt, current_pt, time_interval):
    """
    calculate normalized t from first and current pt
    return time index (normalized time)
    """
    t = int(1 + ((current_pt.time_arr - first_pt.time_arr).seconds / time_interval))
    return t
class GPS2SegData(Dataset): 
    def __init__(self, rn, trajs, mbr, parameters, mode,is_train):
        self.parameters = parameters
        self.rn:RoadNetworkMapFull = rn
        self.mbr = mbr
        self.grid_size = parameters['grid_size']
        self.time_span = parameters['time_span']
        self.src_grid_seqs, self.src_gps_seqs, self.src_temporal_feas = [], [], []
        self.trg_rids = []
        self.is_train=is_train
        self.trajs = trajs
        self.keep_ratio = parameters['init_ratio']
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, index):
        traj = self.trajs[index]
        if self.is_train: 
            length = len(traj.pt_list)
            keep_index = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * self.keep_ratio))) + [length - 1]
        else: 
            keep_index = traj.low_idx
        src_list = np.array(traj.pt_list, dtype=object)
        src_list = src_list[keep_index].tolist()
        src_gps_seq, src_grid_seq, trg_rid = self.get_seqs(src_list)
        trg_candis = self.rn.get_trg_segs(src_gps_seq, self.parameters['candi_size'], self.parameters['search_dist'], self.parameters['beta'])
        candi_label, candi_id, candi_feat, candi_mask = self.get_candis_feats(trg_candis, trg_rid)
        src_grid_seq = torch.tensor(src_grid_seq)
        return src_grid_seq, trg_rid, candi_label, candi_id, candi_feat, candi_mask
    def get_candis_feats(self, ls_candi, trg_id):
        candi_id = []
        candi_feat = []
        candi_onehot = []
        candi_mask = []
        for candis, trg in zip(ls_candi, trg_id):
            candi_mask.append([1] * len(candis) + [0] * (self.parameters['candi_size'] - len(candis)))
            tmp_id = []
            tmp_feat = []
            tmp_onehot = [0] * self.parameters['candi_size']
            for candi in candis:
                tmp_id.append(candi.eid)
                tmp_feat.append([candi.err_weight, candi.cosv, candi.cosv_pre, candi.cosf, candi.cosl, candi.cos1, candi.cos2, candi.cos3, candi.cosp])
            tmp_id.extend([0] * (self.parameters['candi_size'] - len(candis)))
            tmp_feat.extend([[0] * len(tmp_feat[0])] * (self.parameters['candi_size'] - len(candis)))
            if trg in tmp_id:
                idx = tmp_id.index(trg)
                tmp_onehot[idx] = 1
            candi_id.append(tmp_id)
            candi_feat.append(tmp_feat)
            candi_onehot.append(tmp_onehot)
        candi_onehot = torch.tensor(candi_onehot)
        candi_id = torch.tensor(candi_id) + 1
        candi_feat = torch.tensor(candi_feat)
        candi_mask = torch.tensor(candi_mask, dtype=torch.float32)
        return candi_onehot, candi_id, candi_feat, candi_mask
    def get_seqs(self, ds_pt_list):
        ls_gps_seq = []
        ls_grid_seq = []
        mm_eids = []
        time_interval = self.time_span
        first_pt = ds_pt_list[0]
        for ds_pt in ds_pt_list:
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            if self.parameters['gps_flag']:
                locgrid_xid = (ds_pt.lat - self.rn.minLat) / (self.rn.maxLat - self.rn.minLat)
                locgrid_yid = (ds_pt.lng - self.rn.minLon) / (self.rn.maxLon - self.rn.minLon)
            else:
                locgrid_xid, locgrid_yid = gps2grid(ds_pt, self.mbr, self.grid_size)
            t = get_normalized_t(first_pt, ds_pt, time_interval)
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])
            mm_eids.append(ds_pt.data['candi_pt'].eid)
        return ls_gps_seq, ls_grid_seq, mm_eids
class CachedGPS2SegData(Dataset):
    def __init__(self, base_dataset: GPS2SegData, cache_path: str = ''):
        self.base = base_dataset
        self.parameters = base_dataset.parameters
        self.rn = base_dataset.rn
        self.mbr = base_dataset.mbr
        self.grid_size = base_dataset.grid_size
        self.time_span = base_dataset.time_span
        self.trajs = base_dataset.trajs
        self.is_train = base_dataset.is_train
        self.keep_ratio = base_dataset.keep_ratio
        self.cache_path = cache_path
        self._full_cache = None
        self._load_or_build()
    def _get_point_base_candidates(self, gps, candi_size, search_dist):
        step = 10
        gps_pt = SPoint(gps[0], gps[1])
        candis = []
        cur_dist = search_dist
        while len(candis) == 0:
            mbr = MBR(
                gps_pt.lat - cur_dist * LAT_PER_METER,
                gps_pt.lng - cur_dist * LNG_PER_METER,
                gps_pt.lat + cur_dist * LAT_PER_METER,
                gps_pt.lng + cur_dist * LNG_PER_METER,
            )
            candis = self.rn.get_candidates(gps_pt, mbr)
            cur_dist += step
        candis.sort(key=lambda elem: elem.error)
        candis = candis[:candi_size]
        return [
            (int(c.eid), float(c.lat), float(c.lng), float(c.error), float(c.offset), float(c.rate))
            for c in candis
        ]
    def _build_full_cache(self):
        cache = []
        for traj in self.trajs:
            full_pt_list = traj.pt_list
            point_cache = []
            for ds_pt in full_pt_list:
                point_cache.append(
                    self._get_point_base_candidates(
                        [ds_pt.lat, ds_pt.lng],
                        self.parameters['candi_size'],
                        self.parameters['search_dist'],
                    )
                )
            cache.append(point_cache)
        return cache
    def _load_or_build(self):
        if self.cache_path and os.path.exists(self.cache_path):
            self._full_cache = loadZ_pk(self.cache_path)
            if len(self._full_cache) == len(self.trajs):
                return
        self._full_cache = self._build_full_cache()
        if self.cache_path:
            check_dir(os.path.dirname(self.cache_path) + os.sep)
            saveZ_pk(self.cache_path, self._full_cache)
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, index):
        traj = self.trajs[index]
        if self.is_train:
            length = len(traj.pt_list)
            keep_index = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * self.keep_ratio))) + [length - 1]
        else:
            keep_index = traj.low_idx
        if not isinstance(keep_index, (list, tuple)):
            keep_index = list(keep_index)
        src_list = np.array(traj.pt_list, dtype=object)[keep_index].tolist()
        src_gps_seq, src_grid_seq, trg_rid = self.base.get_seqs(src_list)
        vecs = list(zip(src_gps_seq[:-1], src_gps_seq[1:]))
        vecs_later = vecs + [vecs[-1]]
        vecs_pre = [vecs[0]] + vecs
        all_base_candidates = self._full_cache[index]
        selected_base = [all_base_candidates[i] for i in keep_index]
        trg_candis = []
        for ds_pt, (p1, p2), (p3, p4), candis_info in zip(src_gps_seq, vecs_later, vecs_pre, selected_base):
            gps_pt = SPoint(ds_pt[0], ds_pt[1])
            vec_l = (p2[0] - p1[0], p2[1] - p1[1])
            vec_p = (p4[0] - p3[0], p4[1] - p3[1])
            candis = []
            for eid, lat, lng, error, offset, rate in candis_info:
                candi = CandidatePoint(lat, lng, eid, error, offset, rate)
                candi = self.rn.add_candi_attrs(candi, gps_pt, vec_l, vec_p, 1, self.parameters['beta'])
                candis.append(candi)
            trg_candis.append(candis)
        candi_label, candi_id, candi_feat, candi_mask = self.base.get_candis_feats(trg_candis, trg_rid)
        return (
            torch.tensor(src_grid_seq, dtype=torch.float32),
            trg_rid,
            candi_label,
            candi_id,
            candi_feat,
            candi_mask,
        )
def cache_gps2seg_dataset(dataset: GPS2SegData, cache_path: str = ''):
    return CachedGPS2SegData(dataset, cache_path=cache_path)
def api_pre_gsp2segdata(trajs,rn,mbr,paramters,mode,is_train,path=''):
    ...
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)
    def forward(self, query, key, value, attn_mask):
        bs, src_len = query.shape[0], query.shape[1]
        candi_num = key.shape[-2]
        query = query.unsqueeze(-2).repeat(1, 1, candi_num, 1)
        energy = torch.tanh(self.attn(torch.cat((query, key), dim=-1)))
        attention = self.v(energy).squeeze(-1)
        attention = attention.masked_fill(attn_mask == 0, -1e10)
        scores = F.softmax(attention, dim=-1)
        weighted = torch.bmm(scores.reshape(bs*src_len, candi_num).unsqueeze(-2), value.reshape(bs*src_len, candi_num, -1)).squeeze(-2)
        weighted = weighted.reshape(bs, src_len, -1)
        return scores, weighted
def sequence_mask(X, valid_len, value=0.):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
def sequence_mask3d(X, valid_len, valid_len2, value=0.):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    maxlen2 = X.size(2)
    mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None]
    mask2 = torch.arange((maxlen2), dtype=torch.float32,device=X.device)[None, :] < valid_len2[:, None]
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
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
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
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output
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
        x = self.norm(x)
        return x
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
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
        x = self.ff(x2)
        return x
class Transformer(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads) for _ in range(N)
        ])
        self.norm = Norm(d_model)
    def forward(self, src, mask3d=None):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
        return self.norm(x)
class Encoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters['hid_dim']
        input_dim = 3
        self.fc_in = nn.Linear(input_dim, parameters['hid_dim'])
        self.transformer = Transformer(parameters['hid_dim'], parameters['transformer_layers'], heads=4)
    def forward(self, src, src_len):
        max_src_len = src.size(1)
        bs = src.size(0)
        src_len = torch.tensor(src_len, device=src.device)
        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(bs, max_src_len, device=src.device)
        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        src = self.fc_in(src)
        outputs = self.transformer(src, mask3d)
        assert outputs.size(1) == max_src_len
        outputs = outputs * mask2d
        return outputs
class GPS2Seg(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.direction_flag = True 
        self.attn_flag = True 
        self.only_direction =False 
        self.emb_id = nn.Embedding(parameters['id_size'], 64)
        self.encoder = Encoder(parameters)
        parameters['hid_dim']=64
        fc_id_out_input_dim = parameters['hid_dim']
        if self.direction_flag:
            fc_id_out_input_dim += 9
        if self.only_direction:
            fc_id_out_input_dim = 9
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, parameters['hid_dim'])
        mlp_dim = parameters['hid_dim'] * 2
        if self.attn_flag:
            self.attn = Attention(parameters['hid_dim'])
            mlp_dim += parameters['hid_dim']
        self.prob_out = nn.Sequential(
            nn.Linear(mlp_dim, parameters['hid_dim'] * 2),
            nn.ReLU(),
            nn.Linear(parameters['hid_dim'] * 2, 1),
            nn.Sigmoid()
        )
        self.params = parameters
        self.hid_dim = parameters['hid_dim']
        self.init_weights()  
    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
    def forward(self, src, src_len, candi_ids, candi_feats, candi_masks):
        candi_num = candi_ids.shape[-1]
        candi_embedding = self.emb_id(candi_ids)
        if self.direction_flag:
            candi_embedding = torch.cat([candi_embedding, candi_feats], dim=-1)
        if self.only_direction:
            candi_embedding = candi_feats
        candi_vec = self.fc_id_out(candi_embedding)
        src = src.float()
        encoder_outputs = self.encoder(src, src_len)
        if self.attn_flag:
            _, context = self.attn(encoder_outputs, candi_vec, candi_vec, candi_masks)
            encoder_outputs = torch.cat((encoder_outputs, context), dim=-1)
        output_multi = encoder_outputs.unsqueeze(-2).repeat(1, 1, candi_num, 1)
        outputs_id = self.prob_out(torch.cat((output_multi, candi_vec), dim=-1)).squeeze(-1)
        outputs_id = outputs_id.masked_fill(candi_masks == 0, 0)
        return outputs_id
def _mma_train(model, iterator, optimizer, device):
    criterion_bce = nn.BCELoss(reduction='mean')
    epoch_train_id_loss = 0
    model.train()
    for i, batch in enumerate(iterator):
        src_seqs, src_lengths, _, candi_labels, candi_ids, candi_feats, candi_masks = batch
        src_seqs = src_seqs.to(device, non_blocking=True)
        candi_labels = candi_labels.float().to(device, non_blocking=True)
        candi_ids = candi_ids.to(device, non_blocking=True)
        candi_feats = candi_feats.to(device, non_blocking=True)
        candi_masks = candi_masks.to(device, non_blocking=True)
        output_ids = model(src_seqs, src_lengths, candi_ids, candi_feats, candi_masks)
        bce_loss = criterion_bce(output_ids, candi_labels) * candi_ids.shape[-1]
        optimizer.zero_grad(set_to_none=True)
        bce_loss.backward()
        optimizer.step()
        epoch_train_id_loss += bce_loss.item()
        if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
            print("==>{}: {}".format((i + 1) // (len(iterator) // 10), epoch_train_id_loss / (i + 1)))
    return epoch_train_id_loss / len(iterator)
def mma_evaluate(model, iterator, device):
    model.eval()
    epoch_train_id_loss = 0
    criterion_bce = nn.BCELoss(reduction='mean')
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_lengths, _, candi_labels, candi_ids, candi_feats, candi_masks = batch
            src_seqs = src_seqs.to(device, non_blocking=True)
            candi_labels = candi_labels.float().to(device, non_blocking=True)
            candi_ids = candi_ids.to(device, non_blocking=True)
            candi_feats = candi_feats.to(device, non_blocking=True)
            candi_masks = candi_masks.to(device, non_blocking=True)
            output_ids = model(src_seqs, src_lengths, candi_ids, candi_feats, candi_masks)
            bce_loss = criterion_bce(output_ids, candi_labels) * candi_ids.shape[-1]
            epoch_train_id_loss += bce_loss.item()
        print("==> Valid: {}".format(epoch_train_id_loss / (i + 1)))
        return epoch_train_id_loss / len(iterator)
def _mma_get_results(predict_id, target_id, lengths):
    predict_id = predict_id.detach().cpu().tolist()
    results = []
    for pred, trg, length in zip(predict_id, target_id, lengths):
        results.append([pred[:length], trg])
    return results
def mma_infer(model, iterator, device):
    data = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_lengths, trg_rids, _, candi_ids, candi_feats, candi_masks = batch
            src_seqs = src_seqs.to(device, non_blocking=True)
            candi_ids = candi_ids.to(device, non_blocking=True)
            candi_feats = candi_feats.to(device, non_blocking=True)
            candi_masks = candi_masks.to(device, non_blocking=True)
            output_ids = model(src_seqs, src_lengths, candi_ids, candi_feats, candi_masks)
            candi_size = candi_ids.shape[-1]
            output_tmp = (F.one_hot(output_ids.argmax(-1), candi_size) * candi_ids).sum(dim=-1) - 1
            results = _mma_get_results(output_tmp, trg_rids, src_lengths)
            data.extend(results)
    return data
def mma_collate_fn(data):
    src_seqs, trg_rids, candi_onehots, candi_ids, candi_feats, candi_masks = zip(*data)
    lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    candi_onehots = rnn_utils.pad_sequence(candi_onehots, batch_first=True, padding_value=0)
    candi_ids = rnn_utils.pad_sequence(candi_ids, batch_first=True, padding_value=0)
    candi_feats = rnn_utils.pad_sequence(candi_feats, batch_first=True, padding_value=0)
    candi_masks = rnn_utils.pad_sequence(candi_masks, batch_first=True, padding_value=0)
    return src_seqs, lengths, trg_rids, candi_onehots, candi_ids, candi_feats, candi_masks
def shrink_seq(seq):
    """remove repeated ids"""
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s == s0:
            continue
        else:
            new_seq.append(s)
        s0 = s
    return new_seq
def memoize(fn):
    '''
    Return a memoized version of the input function.
    The returned function caches the results of previous calls.
    Useful if a function call is expensive, and the function
    is called repeatedly with the same arguments.
    '''
    cache = dict()
    def wrapped(*v):
        key = tuple(v)  
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]
    return wrapped
def lcs_old(xs, ys):
    '''Return the longest subsequence common to xs and ys.
    Example
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    '''
    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i - 1], ys[j - 1]
            if xe == ye:
                return lcs_(i - 1, j - 1) + [xe]
            else:
                return max(lcs_(i, j - 1), lcs_(i - 1, j), key=len)
        else:
            return []
    return lcs_(len(xs), len(ys))
def lcs(xs, ys):
    '''Return the longest subsequence common to xs and ys.
    Example:
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    '''
    m, n = len(xs), len(ys)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if xs[i - 1] == ys[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_seq = []
    i, j = m, n
    while i > 0 and j > 0:
        if xs[i - 1] == ys[j - 1]:
            lcs_seq.append(xs[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs_seq.reverse()
    return lcs_seq
def cal_id_acc(predict, target):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len]
        target = [seq len]
        predict and target have been removed sos
    Returns:
    -------
        mean matched RID accuracy.
    """
    assert len(predict) == len(target)
    ttl = len(predict)
    cnt = np.sum(np.array(predict) == np.array(target))
    shr_trg_ids = shrink_seq(target)
    shr_pre_ids = shrink_seq(predict)
    correct_id_num = len(lcs(shr_trg_ids, shr_pre_ids))
    ttl_trg_id_num = len(shr_trg_ids)
    ttl_pre_id_num = len(shr_pre_ids)
    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    if rid_precision + rid_recall < 1e-6:
        rid_f1 = 0
    else:
        rid_f1 = 2 * rid_recall * rid_precision / (rid_precision + rid_recall)
    return rid_acc, rid_recall, rid_precision, rid_f1
def mma_args(city,rn):
    zone_range,ts,utc=city_info(city)
    args= {'device': device,'transformer_layers': 2,'candi_size': 10,'attn_flag': True,'direction_flag': True,'gps_flag': False,'search_dist': 50,'beta': 15,'gamma': 30,'min_lat': zone_range[0],'min_lng': zone_range[1],'max_lat': zone_range[2],'max_lng': zone_range[3],'city': city,'keep_ratio': 0.1,'grid_size': 50,'time_span': ts,'hid_dim':64,'id_emb_dim': 64,'dropout': 0.1,'id_size': rn.valid_edge_cnt_one,'n_epochs': 50,'batch_size': 256,'learning_rate': 1e-3,'decay_flag': True,'decay_ratio': 0.9,'clip': 1,'log_step': 1,'utc': utc,'small': False,'init_ratio': 0.5,'only_direction': False,'cate': "g2s",'threshold': 1}
    mbr = MBR(args['min_lat'], args['min_lng'], args['max_lat'], args['max_lng'])
    args['grid_num'] = gps2grid(SPoint(args['max_lat'], args['max_lng']), mbr, args['grid_size'])
    args['grid_num'] = (args['grid_num'][0] + 1, args['grid_num'][1] + 1)
    return args,mbr
def train_mma(root_data,root_model,dataname):
    path_mma=os.path.join(root_model,f'mma.th.zst') ; check_dir(path_mma)
    zone_range,ts,utc=city_info(dataname)
    map_root = os.path.join(root_data, dataname,'roadnet'); check_dir(map_root)
    rn = RoadNetworkMapFull(map_root, zone_range=zone_range, unit_length=50)
    args,mbr=mma_args(dataname,rn)
    traj_root = os.path.join(root_data, dataname)
    train_dataset = GPS2SegData(rn, traj_root, mbr, args, 'train',True)
    valid_dataset = GPS2SegData(rn, traj_root, mbr, args, 'valid',False)
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    print('Finish data preparing.')
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    train_iterator = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=mma_collate_fn, num_workers=num_workers, pin_memory=False)
    valid_iterator = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn= mma_collate_fn, num_workers=num_workers, pin_memory=False)
    model = GPS2Seg(args).to(device)
    print('model', str(model))
    best_valid_loss = float('inf')
    best_epoch = 0
    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'])
    stopping_count = 0
    train_times = []
    for epoch in range(args['n_epochs']):
        print("==> training {} ...".format(train_iterator.dataset.keep_ratio))
        start_time = time.time()
        train_id_loss = _mma_train(model, train_iterator, optimizer, device)
        end_time = time.time()
        epoch_secs = end_time - start_time
        train_times.append(end_time - start_time)
        print("==> validating...")
        valid_id_loss = mma_evaluate(model, valid_iterator, device)
        if valid_id_loss < best_valid_loss:
            best_valid_loss = valid_id_loss
            save_weight(model,path_mma)
            best_epoch = epoch
            stopping_count = 0
        else:
            stopping_count += 1
        if (epoch % args['log_step'] == 0) or (epoch == args['n_epochs'] - 1):
            print('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_secs) + 's')
            print('\tTrain RID Loss:' + str(train_id_loss))
            print('\tValid RID Loss:' + str(valid_id_loss))
        if args['decay_flag']:
            train_iterator.dataset.keep_ratio = max(args['keep_ratio'], train_iterator.dataset.keep_ratio * args['decay_ratio'])
        if stopping_count >= 5:
            print("==> [Info] Early Stop After Epoch {}.".format(epoch))
            break
    print('Best Epoch: {}, {}'.format(best_epoch, best_valid_loss))
    print('==> Best Epoch: {}, {}'.format(best_epoch, best_valid_loss))
    print('==> Training Time: {}, {}, {}'.format(np.mean(train_times), np.min(train_times), np.max(train_times)))
    print('==> Training Time: {}, {}, {}'.format(np.mean(train_times), np.min(train_times), np.max(train_times)))
def test_mma(root_data,root_model,city,name,save=False):
    path_res=os.path.join(root_model,f'mma-infer-{name}.pk.zst') ; check_dir(path_res)
    path_mma=os.path.join(root_model,f'mma.th.zst') ; check_dir(path_mma)
    zone_range,ts,utc=city_info(city)
    map_root = os.path.join(root_data, city, "roadnet"); check_dir(map_root)
    rn = RoadNetworkMapFull(map_root, zone_range=zone_range, unit_length=50)
    args,mbr=mma_args(city,rn)
    traj_root = os.path.join(root_data, city) 
    test_dataset = GPS2SegData(rn, traj_root, mbr, args, 'test')
    print('testing dataset shape: ' + str(len(test_dataset)))
    print('testing dataset shape: ' + str(len(test_dataset)))
    test_iterator = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=mma_collate_fn, num_workers=8, pin_memory=True)
    model = GPS2Seg(args)
    load_weight(model,path_mma)
    model=model.to(device)
    print('==> Model Loaded')
    print("==> Predicting...")
    if save and os.path.exists(path_res):
        pred_data=loadZ_pk(path_res)
    else:
        start_time = time.time()
        pred_data = mma_infer(model, test_iterator, device)
        end_time = time.time()
        epoch_secs = end_time - start_time
        print('Time: ' + str(epoch_secs) + 's')
        print('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / len(test_dataset) * 1000, len(test_dataset) / (end_time - start_time)))
        if save: saveZ_pk(path_res,pred_data)
    print("==> Starting Evaluation...")
    epoch_id1_loss = []
    epoch_recall_loss = []
    epoch_precision_loss = []
    epoch_f1_loss = []
    for tmp_predict, tmp_target in pred_data:
        rid_acc, rid_recall, rid_precision, rid_f1 = cal_id_acc(tmp_predict, tmp_target)
        epoch_id1_loss.append(rid_acc)
        epoch_recall_loss.append(rid_recall)
        epoch_precision_loss.append(rid_precision)
        epoch_f1_loss.append(rid_f1)
    test_id_acc, test_id_recall, test_id_precision, test_id_f1 = np.mean(epoch_id1_loss), np.mean(
        epoch_recall_loss), np.mean(epoch_precision_loss), np.mean(epoch_f1_loss)
    print(test_id_recall, test_id_precision, test_id_f1, test_id_acc)
    print('Time: ' + str(epoch_secs) + 's')
    print('\tTest RID Acc:' + str(test_id_acc) +
                    '\tTest RID Recall:' + str(test_id_recall) +
                    '\tTest RID Precision:' + str(test_id_precision) +
                    '\tTest RID F1 Score:' + str(test_id_f1))
def trmma_collate_fn(data0):
    data = []
    for item in data0:
        data.extend(item)
    da_routes, src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, trg_rids, trg_rates, \
    trg_rid_labels, d_rids, d_rates = zip(*data)
    src_lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    src_pro_feas = torch.vstack(src_pro_feas).squeeze(-1)
    src_seg_seqs = rnn_utils.pad_sequence(src_seg_seqs, batch_first=True, padding_value=0)
    src_seg_feats = rnn_utils.pad_sequence(src_seg_feats, batch_first=True, padding_value=0)
    trg_lengths = [len(seq) for seq in trg_rids]
    trg_rids = rnn_utils.pad_sequence(trg_rids, batch_first=True, padding_value=0)
    trg_rates = rnn_utils.pad_sequence(trg_rates, batch_first=True, padding_value=0)
    da_lengths = [len(seq) for seq in da_routes]
    da_routes = rnn_utils.pad_sequence(da_routes, batch_first=True, padding_value=0)
    da_pos = [torch.tensor(list(range(1, item + 1))) for item in da_lengths]
    da_pos = rnn_utils.pad_sequence(da_pos, batch_first=True, padding_value=0)
    d_rids = torch.vstack(d_rids).squeeze(-1)
    d_rates = torch.vstack(d_rates)
    max_da = max(da_lengths)
    trg_rid_labels = list(trg_rid_labels)
    for i in range(len(trg_rid_labels)):
        if trg_rid_labels[i].shape[1] < max_da:
            tmp = torch.zeros(trg_rid_labels[i].shape[0], max_da - trg_rid_labels[i].shape[1]) + 1e-6
            trg_rid_labels[i] = torch.cat([trg_rid_labels[i], tmp], dim=-1)
    trg_rid_labels = rnn_utils.pad_sequence(trg_rid_labels, batch_first=True, padding_value=1e-6)
    return src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates
def trmma_collate_fn_test(data):
    da_routes, src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, trg_gps_seqs, trg_rids, trg_rates, \
    trg_rid_labels, d_rids, d_rates = zip(*data)
    src_lengths = [len(seq) for seq in src_seqs]
    src_seqs = rnn_utils.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    src_pro_feas = torch.vstack(src_pro_feas).squeeze(-1)
    src_seg_seqs = rnn_utils.pad_sequence(src_seg_seqs, batch_first=True, padding_value=0)
    src_seg_feats = rnn_utils.pad_sequence(src_seg_feats, batch_first=True, padding_value=0)
    trg_lengths = [len(seq) for seq in trg_gps_seqs]
    trg_gps_seqs = rnn_utils.pad_sequence(trg_gps_seqs, batch_first=True, padding_value=0)
    trg_rids = rnn_utils.pad_sequence(trg_rids, batch_first=True, padding_value=0)
    trg_rates = rnn_utils.pad_sequence(trg_rates, batch_first=True, padding_value=0)
    da_lengths = [len(seq) for seq in da_routes]
    da_routes = rnn_utils.pad_sequence(da_routes, batch_first=True, padding_value=0)
    da_pos = [torch.tensor(list(range(1, item + 1))) for item in da_lengths]
    da_pos = rnn_utils.pad_sequence(da_pos, batch_first=True, padding_value=0)
    d_rids = torch.vstack(d_rids).squeeze(-1)
    d_rates = torch.vstack(d_rates)
    max_da = max(da_lengths)
    trg_rid_labels = list(trg_rid_labels)
    for i in range(len(trg_rid_labels)):
        if trg_rid_labels[i].shape[1] < max_da:
            tmp = torch.zeros(trg_rid_labels[i].shape[0], max_da - trg_rid_labels[i].shape[1])
            trg_rid_labels[i] = torch.cat([trg_rid_labels[i], tmp], dim=-1)
    trg_rid_labels = rnn_utils.pad_sequence(trg_rid_labels, batch_first=True, padding_value=0)
    return src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates
def trmma_train(model, iterator, optimizer, rid_features_dict, parameters, device):
    criterion_reg = nn.L1Loss(reduction='sum')
    criterion_bce = nn.BCELoss(reduction='sum')
    epoch_ttl_loss = 0
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    time_ttl = 0
    time_move = 0
    time_forward = 0
    time_loss = 0
    time_zero = 0
    time_gradient = 0
    time_update = 0
    time_ttl2 = 0
    t0 = time.time()
    model.train()
    for i, batch in enumerate(iterator):
        t1 = time.time()
        src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = batch
        src_pro_feas = src_pro_feas.to(device, non_blocking=True)
        trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
        src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
        src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
        src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
        trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
        trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
        da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
        da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
        d_rids = d_rids.to(device, non_blocking=True)
        d_rates = d_rates.to(device, non_blocking=True)
        time_move += time.time() - t1
        t2 = time.time()
        output_ids, output_rates = model(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths, src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio=parameters['tf_ratio'])
        time_forward += time.time() - t2
        t3 = time.time()
        trg_lengths_sub = [length - 2 for length in trg_lengths]
        loss_train_ids = criterion_bce(output_ids, trg_rid_labels) * parameters['lambda1'] / np.sum(np.array(trg_lengths_sub) * np.array(da_lengths))
        epoch_train_id_loss += loss_train_ids.item()
        ttl_loss = loss_train_ids
        if parameters['rate_flag']:
            loss_rates = criterion_reg(output_rates, trg_rates[1:-1]) * parameters['lambda2'] / sum(trg_lengths_sub)
            epoch_rate_loss += loss_rates.item()
            ttl_loss += loss_rates
        time_loss += time.time() - t3
        t4 = time.time()
        optimizer.zero_grad(set_to_none=True)
        time_zero += time.time() - t4
        t5 = time.time()
        ttl_loss.backward()
        time_gradient += time.time() - t5
        t6 = time.time()
        optimizer.step()
        time_update += time.time() - t6
        epoch_ttl_loss += ttl_loss.item()
        if len(iterator) >= 10 and (i + 1) % (len(iterator) // 10) == 0:
            print("==>{}: {}, {}, {}".format((i + 1) // (len(iterator) // 10), epoch_ttl_loss / (i + 1), epoch_train_id_loss / (i + 1), epoch_rate_loss / (i + 1)))
        time_ttl2 += time.time() - t1
    time_ttl += time.time() - t0
    return epoch_ttl_loss / len(iterator), epoch_train_id_loss / len(iterator), epoch_rate_loss / len(iterator)
def trmma_evaluate(model, iterator, rid_features_dict, parameters, device):
    criterion_reg = nn.L1Loss(reduction='sum')
    criterion_bce = nn.BCELoss(reduction='sum')
    epoch_train_id_loss = 0
    epoch_rate_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = batch
            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
            src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)
            output_ids, output_rates = model(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths,
                                                          src_pro_feas, rid_features_dict,
                                                          da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates,
                                                          teacher_forcing_ratio=0)
            trg_lengths_sub = [length - 2 for length in trg_lengths]
            loss_train_ids = criterion_bce(output_ids, trg_rid_labels) * parameters['lambda1'] / np.sum(np.array(trg_lengths_sub) * np.array(da_lengths))
            if parameters['rate_flag']:
                loss_rates = criterion_reg(output_rates, trg_rates[1:-1]) * parameters['lambda2'] / sum(trg_lengths_sub)
                epoch_rate_loss += loss_rates.item()
            epoch_train_id_loss += loss_train_ids.item()
        return (epoch_train_id_loss + epoch_rate_loss) / len(iterator), epoch_train_id_loss / len(iterator), epoch_rate_loss / len(iterator)
def trmma_get_results(predict_id, predict_rate, target_id, target_rate, target_gps, trg_len, routes, route_lengths, inverse_flag=True):
    if inverse_flag:
        predict_id = predict_id - 1
        target_id = target_id - 1
        routes = routes - 1
    predict_id = predict_id.permute(1, 0).detach().cpu().tolist()
    predict_rate = predict_rate.permute(1, 0).detach().cpu().tolist()
    target_gps = target_gps.permute(1, 0, 2).detach().cpu().tolist()
    target_id = target_id.permute(1, 0).detach().cpu().tolist()
    target_rate = target_rate.permute(1, 0).detach().cpu().tolist()
    routes = routes.permute(1, 0).detach().cpu().tolist()
    results = []
    for pred_seg, pred_rate, trg_id, trg_rate, trg_gps, length, route, route_len in zip(predict_id, predict_rate, target_id, target_rate, target_gps, trg_len, routes, route_lengths):
        results.append([pred_seg[:length], pred_rate[:length], trg_id[:length], trg_rate[:length], trg_gps[:length], route[:route_len]])
    return results
def trmma_infer(model, iterator, rid_features_dict, device):
    data = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = batch
            src_pro_feas = src_pro_feas.to(device, non_blocking=True)
            trg_rid_labels = trg_rid_labels.permute(1, 0, 2).to(device, non_blocking=True)
            src_seqs = src_seqs.permute(1, 0, 2).to(device, non_blocking=True)
            src_seg_seqs = src_seg_seqs.permute(1, 0).to(device, non_blocking=True)
            src_seg_feats = src_seg_feats.permute(1, 0, 2).to(device, non_blocking=True)
            trg_rids = trg_rids.permute(1, 0).long().to(device, non_blocking=True)
            trg_rates = trg_rates.permute(1, 0, 2).to(device, non_blocking=True)
            da_routes = da_routes.permute(1, 0).to(device, non_blocking=True)
            da_pos = da_pos.permute(1, 0).to(device, non_blocking=True)
            d_rids = d_rids.to(device, non_blocking=True)
            d_rates = d_rates.to(device, non_blocking=True)
            output_ids, output_rates = model(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths, src_pro_feas, rid_features_dict, da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio=-1)
            output_tmp = (F.one_hot(output_ids.argmax(-1), da_routes.shape[0]) * da_routes.permute(1, 0).unsqueeze(1).repeat(1, trg_rid_labels.shape[0], 1).permute(1, 0, 2)).sum(dim=-1)
            output_rates = output_rates.squeeze(2)
            trg_rates = trg_rates.squeeze(2)
            trg_gps_seqs = trg_gps_seqs.permute(1, 0, 2)
            trg_lengths_sub = [length - 2 for length in trg_lengths]
            results = trmma_get_results(output_tmp, output_rates, trg_rids[1:-1], trg_rates[1:-1], trg_gps_seqs[1:-1], trg_lengths_sub, da_routes, da_lengths)
            data.extend(results)
    return data
def cal_rn_dis_loss(predict_gps, predict_id, target_gps, target_id):
    """
    Calculate road network based MAE and RMSE between predicted and targeted GPS sequence.
    Args:
    -----
        sp_solver: shortest path solver
        predict_gps = [seq len, 2]
        predict_id = [seq len]
        target_gps = [seq len, 2]
        target_id = [seq len]
        predict and target have been removed sos
    Returns:
    -------
        MAE in meter.
        RMSE in meter.
    """
    ls_dis = []
    assert len(predict_id) == len(target_id) and len(predict_gps) == len(target_gps)
    trg_len = len(predict_gps)
    for i in range(trg_len):
        ls_dis.append(distance(SPoint(*predict_gps[i]), SPoint(*target_gps[i])))
    ls_dis = np.array(ls_dis)
    mae = ls_dis.mean()
    rmse = np.sqrt((ls_dis ** 2).mean())
    return mae, rmse
def calc_metrics(pred_seg, pred_gps, trg_id, trg_gps):
    rid_acc, rid_recall, rid_precision, rid_f1 = cal_id_acc(pred_seg, trg_id)
    mae, rmse = cal_rn_dis_loss(pred_gps, pred_seg, trg_gps, trg_id)
    return rid_recall, rid_precision, rid_f1, rid_acc, mae, rmse
class SparseDAM(object):
    def __init__(self, workspace, seg_num, mask_size=-1, lb_csmv=1):
        self.seg_num = seg_num
        self.mask_size = mask_size
        self.lb_csmv = lb_csmv
        data = pd.read_csv(os.path.join(workspace, "csm_all.txt"), sep=" ", header=None, names=['row', 'col', 'value']).to_numpy(dtype=np.int32)
        self.mat_col = np.zeros(seg_num, dtype=object)
        self.mat_row = np.zeros(seg_num, dtype=object)
        cnt = 0
        for i in range(seg_num):
            self.mat_col[i] = {}
            self.mat_row[i] = {}
        for tmp in data:
            self.mat_col[tmp[1]][tmp[0]] = tmp[2]
            self.mat_row[tmp[0]][tmp[1]] = tmp[2]
            cnt += 1
    def get_csm_value(self, o, d):
        value = self.mat_col[d][o]
        if value is None:
            value = 0
        return value
    def get_rank(self, o, k, d):
        o2k = self.get_csm_value(o, k)
        k2d = self.get_csm_value(k, d)
        res = k2d
        if o2k < res:
            res = o2k
        return res
    def get_col(self, d):
        return self.mat_col[d]
    def get_rank_list(self, o, d):
        src = self.mat_row[o]
        des = self.mat_col[d]
        candidates = []
        for k in des:
            k2d = des[k]
            if k2d >= self.lb_csmv:
                o2k = src[k]
                if o2k is not None:
                    rank = k2d
                    if rank > o2k:
                        rank = o2k
                    if rank >= self.lb_csmv:
                        candidates.append((k, rank))
        candidates.sort(key=lambda elem: elem[0])
        candidates.sort(key=lambda elem: elem[1], reverse=True)
        return candidates[: self.mask_size]
class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()
    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()
    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                OrderedDict.popitem(self, last=False)
    def get(self, key):
        res = OrderedDict.get(self, key)
        OrderedDict.move_to_end(self, key, last=True)
        return res
class SegInfo(object):
    def __init__(self, seginfo_file, cache_size=100000):
        segs = pd.read_csv(seginfo_file, sep=" ", header=None, names=['eid', 'src', 'trg', 'len', 'rt', 'geo_src', 'geo_trg', 'azimuth', 'freq', 'travel_time'])
        self.max_length = np.max(segs['len'])
        self.seg_num = segs.shape[0]
        self.__od_dist__ = LimitedSizeDict(size_limit=cache_size)
        self.__od_azimuth__ = LimitedSizeDict(size_limit=cache_size)
        self.__seg_info__ = np.zeros(self.seg_num, dtype=object)
        nodes = []
        gps = []
        for i in range(segs.shape[0]):
            seg = segs.iloc[i]
            bp = [float(item) for item in seg['geo_src'].split(",")]
            ep = [float(item) for item in seg['geo_trg'].split(",")]
            tmp = (
                float(seg['len']),
                float(seg['travel_time']),
                int(seg['rt']),
                float(seg['azimuth']),
                np.array(bp + ep, dtype=float),
                np.array(ep, dtype=float) - np.array(bp, dtype=float)
            )
            self.__seg_info__[seg['eid']] = tmp
            if seg['src'] not in nodes:
                nodes.append(seg['src'])
                gps.append(bp)
            if seg['trg'] not in nodes:
                nodes.append(seg['trg'])
                gps.append(ep)
        gps = pd.DataFrame(gps, columns=['lon', 'lat'])
        self.centric = np.array([gps.lon.mean(), gps.lat.mean()], dtype=float)
    def get_seg_length(self, seg):
        return self.__seg_info__[seg][0]
    def get_gps(self, seg, rate):
        geo = self.get_seg_geo(seg)
        gps = geo[:2] + rate * self.get_seg_vec(seg)
        gps = gps.tolist()
        gps.reverse()
        return gps
    def get_seg_travel_time(self, seg):
        return self.__seg_info__[seg][1]
    def get_seg_speed(self, seg):
        return self.__seg_info__[seg][0] / self.__seg_info__[seg][1]
    def get_seg_rt(self, seg):
        return self.__seg_info__[seg][2]
    def get_seg_azimuth(self, seg):
        return self.__seg_info__[seg][3]
    def get_seg_geo(self, seg):
        return self.__seg_info__[seg][4]
    def get_seg_vec(self, seg):
        return self.__seg_info__[seg][5]
    def get_rn_distance(self, path, o_r, d_r):
        if len(path) == 1:
            return self.__seg_info__[path[0]][0] * (d_r - o_r)
        dist = 0.0
        for i, seg in enumerate(path):
            if i == 0:
                tmp = self.__seg_info__[seg][0] * (1 - o_r)
            elif i == (len(path) - 1):
                tmp = self.__seg_info__[seg][0] * d_r
            else:
                tmp = self.__seg_info__[seg][0]
            dist += tmp
        return dist
    def get_path_distance(self, path):
        dist = 0.0
        for seg in path:
            dist += self.__seg_info__[seg][0]
        return dist
    def get_path_travel_time(self, path):
        tt = 0.0
        for seg in path:
            tt += self.__seg_info__[seg][1]
        return tt
    def get_od_dist(self, o, d):
        key = (o, d)
        if key in self.__od_dist__:
            res = self.__od_dist__.get(key)
        else:
            point1 = self.__seg_info__[o][4][2:]
            point2 = self.__seg_info__[d][4][:2]
            res = haversine((point1[1], point1[0]), (point2[1], point2[0]), unit="m")
            self.__od_dist__[key] = res
        return res
    def get_od_azimuth(self, o, d):
        key = (o, d)
        if key in self.__od_azimuth__:
            res = self.__od_azimuth__.get(key)
        else:
            o_ep = self.__seg_info__[o][4][2:].tolist()
            d_bp = self.__seg_info__[d][4][:2].tolist()
            res = calc_azimuth(o_ep + d_bp)
            self.__od_azimuth__[key] = res
        return res
def remove_circle(path_fixed):
    cur = 0
    while cur < len(path_fixed):
        eid = path_fixed[cur]
        idx = []
        for i in range(cur, len(path_fixed)):
            if path_fixed[i] == eid:
                idx.append(i)
        path_fixed = path_fixed[0: cur] + path_fixed[max(idx): ]
        cur += 1
    return path_fixed
def calc_cos_value(vec1, vec2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    a = vec1 * vec1
    b = vec2 * vec2
    c = vec1 * vec2
    denom = np.sqrt(a[0] + a[1]) * np.sqrt(b[0] + b[1])
    cos_value = (c[0] + c[1]) / denom if denom != 0 else 1.0
    return cos_value
def get_num_pts(time_span, time_interval):
    num_pts = 0  
    if time_span % time_interval > time_interval / 2:
        num_pts = time_span // time_interval  
    elif time_span > time_interval:
        num_pts = time_span // time_interval - 1
    return num_pts
def rate2gps(rn, rid, rate) -> SPoint:
    """
    Convert road rate to GPS on the road segment.
    Since one road contains several coordinates, iteratively computing length can be more accurate.
    Args:
    -----
    rn: road network
    rid, rate: single value from model prediction
    Returns:
    --------
    project_pt:
        projected GPS point on the road segment.
    """
    cords = np.array(rn.edgeCord[rid]).reshape(-1, 2).tolist()
    offset = rn.edgeDis[rid] * rate
    dist = 0  
    pre_dist = 0  
    if rate == 1.0:
        return SPoint(*cords[-1])
    if rate == 0.0:
        return SPoint(*cords[0])
    project_pt = SPoint(*cords[0])
    for i in range(len(cords) - 1):
        if i > 0:
            pre_dist += distance(SPoint(*cords[i - 1]), SPoint(*cords[i]))
        dist += distance(SPoint(*cords[i]), SPoint(*cords[i + 1]))
        if dist >= offset:
            if distance(SPoint(*cords[i]), SPoint(*cords[i + 1])) < 1e-6:  
                coor_rate = 0
            else:
                coor_rate = (offset - pre_dist) / distance(SPoint(*cords[i]), SPoint(*cords[i + 1]))
            project_pt = cal_loc_along_line(SPoint(*cords[i]), SPoint(*cords[i + 1]), coor_rate)
            break
    return project_pt
class DAPlanner(object):
    def __init__(self, root_map, id_size, utc):
        self.csm = SparseDAM(root_map, id_size)
        self.seg_info = SegInfo(os.path.join(root_map, "seg_info.csv"))
        self.G = pickle.load(open(os.path.join(root_map, "road_graph_wtime"), "rb"))
        print("Segment Nodes: {}, Edges: {}".format(len(self.G.nodes), len(self.G.edges)))
        self.vehicle_num = np.load(os.path.join(root_map, "vehicle_num_{}-48.npy".format(3600)))
        self.tz = dt.timezone(dt.timedelta(hours=utc))
        self.max_seq_len = 79
        self.freq_limit = 1
        self.dcsm_theta = 1
        self.no_path_cnt = 0
        self._route_cache = LimitedSizeDict(size_limit=200000)
    def planning_multi_batch(self, ods, ts):
        preds = []
        for i, od in enumerate(ods):
            route = self.planning_multi(od, ts[i])
            preds.append(route)
        return preds
    def planning_multi(self, od, t, mode='da', segs_flag=False):
        cache_key = None
        if (not segs_flag) and mode in ['time', 'length']:
            cache_key = (mode, tuple(od))
            if cache_key in self._route_cache:
                return list(self._route_cache.get(cache_key))
        pred = [od[0]]
        timestamp = t + self.seg_info.get_seg_travel_time(od[0])
        segs = []
        for i in range(len(od)-1):
            o = od[i]
            d = od[i+1]
            if pred[-1] != o:
                break
            if mode == 'da':
                col_d = self.csm.get_col(d)
                route = [o]
                seg_used = np.zeros(self.seg_info.seg_num, dtype=np.int32)
                seg_used[o] = 1
                seg_used[d] = 1
                while len(route) < self.max_seq_len and route[-1] != d:
                    out_segs = list(self.G.neighbors(route[-1]))
                    if len(out_segs) == 0:
                        break
                    nextseg = -1
                    next_max = -1
                    tie_cnt = 0
                    tie_nbrs = []
                    for seg in out_segs:
                        if seg == d:
                            nextseg = d
                            tie_cnt = 1
                            break
                        if seg_used[seg] >= self.freq_limit:
                            continue
                        curr_prob = col_d[seg] 
                        if curr_prob is None:
                            curr_prob = 0
                        if curr_prob > next_max:
                            nextseg = seg
                            tie_cnt = 1
                            next_max = curr_prob
                            tie_nbrs = [seg]
                        elif curr_prob == next_max:
                            tie_cnt += 1
                            tie_nbrs.append(seg)
                    if tie_cnt != 1:
                        if tie_cnt == 0:
                            tie_nbrs = out_segs
                        if next_max < self.dcsm_theta:
                            nextseg, _ = self.break_tie_angle(route[-1], tie_nbrs, d)
                        else:
                            nextseg, flag = self.break_tie_traffic_flow(tie_nbrs, timestamp)
                            if flag:
                                nextseg, _ = self.break_tie_angle(route[-1], tie_nbrs, d)
                    if nextseg == -1:
                        break
                    route.append(nextseg)
                    timestamp += self.seg_info.get_seg_travel_time(nextseg)
                    seg_used[nextseg] += 1
                if route[-1] != d:
                    try:
                        _, route = nx.bidirectional_dijkstra(self.G, o, d, weight="time")
                    except nx.exception.NetworkXNoPath as e:
                        self.no_path_cnt += 1
                        route = [o, d]
                route = remove_circle(route)
            elif mode == 'time':
                try:
                    _, route = nx.bidirectional_dijkstra(self.G, o, d, weight="time")
                except nx.exception.NetworkXNoPath as e:
                    self.no_path_cnt += 1
                    route = [o, d]
            elif mode == 'length':
                try:
                    _, route = nx.bidirectional_dijkstra(self.G, o, d, weight="length")
                except nx.exception.NetworkXNoPath as e:
                    self.no_path_cnt += 1
                    route = [o, d]
            else:
                raise NotImplementedError
            pred = pred + route[1:]
            segs.append(route)
        if segs_flag:
            return pred, segs
        else:
            if cache_key is not None:
                self._route_cache[cache_key] = tuple(pred)
            return pred
    def break_tie_angle(self, curr, tie_nbrs, d):
        curr_geo = self.seg_info.get_seg_geo(curr)
        curr_trg = curr_geo[2:]
        d_geo = self.seg_info.get_seg_geo(d)
        d_src = d_geo[:2]
        vec1 = d_src - curr_trg
        nextseg = -1
        next_max = -2
        tie_cnt = 0
        for seg in tie_nbrs:
            vec2 = self.seg_info.get_seg_vec(seg)
            cos_value = calc_cos_value(vec1, vec2)
            if cos_value > next_max:
                nextseg = seg
                tie_cnt = 1
                next_max = cos_value
            else:
                tie_cnt += 1
        flag = False
        return nextseg, flag
    def break_tie_traffic_flow(self, tie_nbrs, timestamp):
        idx, _ = self.get_time_idx2(timestamp)
        nextseg = -1
        next_max = -1
        tie_cnt = 0
        for seg in tie_nbrs:
            prob = self.vehicle_num[seg, idx]
            if prob > next_max:
                nextseg = seg
                tie_cnt = 1
                next_max = prob
            elif prob == next_max:
                tie_cnt += 1
        flag = False
        if tie_cnt > 1:
            flag = True
        return nextseg, flag
    def get_time_idx2(self, timestamp):
        time_arr = dt.datetime.fromtimestamp(timestamp, self.tz)
        if time_arr.weekday() in [0, 1, 2, 3, 4]:
            idx = time_arr.hour
        else:
            idx = time_arr.hour + 24
        t_r = (time_arr.minute * 60 + time_arr.second) * 1.0 / 3600
        return int(idx), t_r
    def get_interpolated_pts(self, src, trg, sub_seq, time_span, rn):
        num_pts = get_num_pts(trg.time - src.time, time_span)
        candi_d = trg.data['candi_pt']
        candi_o = src.data['candi_pt']
        pred_id = [candi_o.eid]
        pred_rate = [candi_o.rate]
        time_in = [src.time]
        forward_unit = self.seg_info.get_rn_distance(sub_seq, candi_o.rate, candi_d.rate) / (num_pts + 1)
        while num_pts > 0:
            forward_meter = forward_unit
            pointer = 0
            flag_find = False
            assert sub_seq[0] == pred_id[-1] and sub_seq[-1] == candi_d.eid
            while pointer < len(sub_seq) and flag_find == False:
                if pointer == 0:
                    try_meter = self.seg_info.get_seg_length(sub_seq[pointer]) * (1 - pred_rate[-1])
                elif pointer == (len(sub_seq) - 1):
                    try_meter = self.seg_info.get_seg_length(sub_seq[pointer]) * candi_d.rate
                else:
                    try_meter = self.seg_info.get_seg_length(sub_seq[pointer])
                if forward_meter > try_meter:
                    forward_meter -= try_meter
                    pointer += 1
                else:
                    flag_find = True
            if flag_find:
                id_tmp = sub_seq[pointer]
                rate_tmp = forward_meter / self.seg_info.get_seg_length(id_tmp)
                if pointer == 0:
                    rate_tmp += pred_rate[-1]
                sub_seq = sub_seq[pointer:]
            else:
                id_tmp = sub_seq[-1]
                start = 0
                if pred_id[-1] == id_tmp:
                    start = pred_rate[-1]
                unit_rate = (candi_d.rate - start) / (num_pts + 1)
                rate_tmp = unit_rate
                if pred_id[-1] == id_tmp:
                    rate_tmp += pred_rate[-1]
                sub_seq = sub_seq[-1:]
            pred_id.append(id_tmp)
            pred_rate.append(rate_tmp)
            time_in.append(time_in[-1] + time_span)
            num_pts -= 1
        res = [src]
        for eid, ratio, ts in zip(pred_id[1:], pred_rate[1:], time_in[1:]):
            projected = rate2gps(rn, eid, ratio)
            dist = 0.
            rate = ratio
            candi_pt = CandidatePoint(projected.lat, projected.lng, eid, dist, rate * self.seg_info.get_seg_length(eid), rate)
            pt = STPoint(projected.lat, projected.lng, ts, {'candi_pt': candi_pt})
            pt.time_arr = dt.datetime.fromtimestamp(ts, self.tz)
            res.append(pt)
        res.append(trg)
        return res
def get_label(cpath, trg_rid):
    label = []
    pre_rid = -1
    pre_prob = []
    for rid in trg_rid:
        if rid == pre_rid:
            tmp = pre_prob
        else:
            idx = 0
            if rid in cpath:
                idx = cpath.index(rid)
            tmp = [0] * len(cpath)
            tmp[idx] = 1
            pre_rid = rid
            pre_prob = tmp
        label.append(tmp)
    return label
def get_pro_features(ds_pt_list, hours):
    hour = np.bincount(hours).argmax()
    week = ds_pt_list[0].time_arr.weekday()
    if week in [5, 6]:
        hour += 24
    return hour
class TrajRecData(Dataset): 
    def __init__(self, rn, trajs, mbr, parameters, mode,is_train):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr  
        self.grid_size = parameters['grid_size']
        self.time_span = parameters['time_span']
        self.mode = mode
        self.keep_ratio = parameters['keep_ratio']
        self.is_train=is_train
        self.trajs = trajs
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, index):
        traj = self.trajs[index]
        if self.is_train : 
            length = len(traj.pt_list)
            keep_index = [0] + sorted(random.sample(range(1, length - 1), int((length - 2) * self.keep_ratio))) + [length - 1]
        else:
            keep_index = traj.low_idx
        src_list = np.array(traj.pt_list, dtype=object)
        src_list = src_list[keep_index].tolist()
        trg_list = traj.pt_list
        data = []
        for p1, p1_idx, p2, p2_idx in zip(src_list[:-1], keep_index[:-1], src_list[1:], keep_index[1:]):
            if (p1_idx + 1) < p2_idx:
                tmp_src_list = [p1, p2]
                ls_grid_seq, ls_gps_seq, hours, tmp_seg_seq = self.get_src_seq(tmp_src_list)
                features = get_pro_features(tmp_src_list, hours)
                mm_eids, mm_rates = self.get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]
                da_route = [self.rn.valid_edge_one[item] for item in path]
                src_seg_seq = [self.rn.valid_edge_one[item] for item in tmp_seg_seq]
                src_seg_feat = self.get_src_seg_feat(ls_gps_seq, tmp_seg_seq)
                label = get_label([self.rn.valid_edge_one[item] for item in path], mm_eids[1:-1])
                da_route = torch.tensor(da_route)
                src_grid_seq = torch.tensor(ls_grid_seq)
                src_pro_fea = torch.tensor(features)
                src_seg_seq = torch.tensor(src_seg_seq)
                src_seg_feat = torch.tensor(src_seg_feat)
                trg_rid = torch.tensor(mm_eids)
                trg_rate = torch.tensor(mm_rates)
                label = torch.tensor(label, dtype=torch.float32)
                d_rid = trg_rid[-1]
                d_rate = trg_rate[-1]
                data.append([da_route, src_grid_seq, src_pro_fea, src_seg_seq, src_seg_feat, trg_rid, trg_rate, label, d_rid, d_rate])
        return data
    def get_src_seg_feat(self, gps_seq, seg_seq):
        feats = []
        for ds_pt, seg in zip(gps_seq, seg_seq):
            gps = SPoint(ds_pt[0], ds_pt[1])
            candi = self.rn.pt2seg(gps, seg)
            feats.append([candi.rate])
        return feats
    def get_src_seq(self, ds_pt_list):
        hours = []
        ls_grid_seq = []
        ls_gps_seq = []
        first_pt = ds_pt_list[0]
        time_interval = self.time_span
        seg_seq = []
        for ds_pt in ds_pt_list:
            hours.append(ds_pt.time_arr.hour)
            t = get_normalized_t(first_pt, ds_pt, time_interval)
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            if self.parameters['gps_flag']:
                locgrid_xid = (ds_pt.lat - self.rn.minLat) / (self.rn.maxLat - self.rn.minLat)
                locgrid_yid = (ds_pt.lng - self.rn.minLon) / (self.rn.maxLon - self.rn.minLon)
            else:
                locgrid_xid, locgrid_yid = gps2grid(ds_pt, self.mbr, self.grid_size)
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])
            seg_seq.append(ds_pt.data['candi_pt'].eid)
        return ls_grid_seq, ls_gps_seq, hours, seg_seq
    def get_trg_seq(self, tmp_pt_list):
        mm_eids = []
        mm_rates = []
        for pt in tmp_pt_list:
            candi_pt = pt.data['candi_pt']
            mm_eids.append(self.rn.valid_edge_one[candi_pt.eid])
            mm_rates.append([candi_pt.rate])
        return mm_eids, mm_rates
def api_pre_trajrecdata(trajs,rn,mbr,paramters,mode,is_train,path=''):
    if path and os.path.exists(path):return NewTrajRecData(loadZ_pk(path))
    data=TrajRecData(rn,trajs,mbr,paramters,mode,is_train)
    res=[data[i] for i in range(len(data))]
    if path: saveZ_pk(path,res)
    return NewTrajRecData(res)
class NewTrajRecData(Dataset):
    def __init__(self,trajrecdata_ed):
        self.data=trajrecdata_ed
    def __len__(self):return len(self.data)
    def __getitem__(self, i):
        return self.data[i]
class GPSFormer(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([
            GPSLayer(d_model, heads) for _ in range(N)
        ])
    def forward(self, src, mask3d=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
        return x
class GPSEncoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.pro_features_flag = parameters['pro_features_flag']
        self.hid_dim = parameters['hid_dim']
        self.transformer = GPSFormer(parameters['hid_dim'], parameters['transformer_layers'], heads=parameters['heads'])
        if self.pro_features_flag:
            self.temporal = nn.Embedding(parameters['pro_input_dim'], parameters['pro_output_dim'])
            self.fc_hid = nn.Linear(parameters['hid_dim'] + parameters['pro_output_dim'], parameters['hid_dim'])
    def forward(self, src, src_len, pro_features):
        bs = src.size(1)
        max_src_len = src.size(0)
        mask3d = torch.ones(bs, max_src_len, max_src_len, device=src.device)
        mask2d = torch.ones(bs, max_src_len, device=src.device)
        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        mask2d = sequence_mask(mask2d, src_len).transpose(0, 1).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        src = src.transpose(0, 1)
        outputs = self.transformer(src, mask3d)
        outputs = outputs.transpose(0, 1)  
        assert outputs.size(0) == max_src_len
        outputs = outputs * mask2d
        hidden = torch.sum(outputs, dim=0) / src_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)
        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))
        return outputs, hidden
class GPSLayer(nn.Module):
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
        x = self.ff(x2)
        return x
class RouteLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.slf_attn = MultiHeadAttention(heads, d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_model * 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    def forward(self, route, gps, route_mask, inter_mask):
        route1 = self.dropout_1(self.slf_attn(route, route, route, route_mask))
        route_out = self.norm_1(route + route1)
        route2 = self.dropout_2(self.attn(route_out, gps, gps, inter_mask))
        route_out2 = self.norm_2(route_out + route2)
        x = self.ff(route_out2)
        return x
class GRLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.gps_enc = GPSLayer(d_model, heads, dropout)
        self.route_enc = RouteLayer(d_model, heads, dropout)
    def forward(self, route, route_mask, gps, gps_mask, inter_mask):
        gps_emb = self.gps_enc(gps, gps_mask)
        route_emb = self.route_enc(route, gps_emb, route_mask, inter_mask)
        return route_emb, gps_emb
class GRFormer(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([
            GRLayer(d_model, heads) for _ in range(N)
        ])
    def forward(self, src, route, mask3d, route_mask3d, inter_mask):
        x = route
        y = src
        for i in range(self.N):
            x, y = self.layers[i](x, route_mask3d, y, mask3d, inter_mask)
        return x
class GREncoder(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters['hid_dim']
        self.pro_features_flag = parameters['pro_features_flag']
        self.transformer = GRFormer(parameters['hid_dim'], parameters['transformer_layers'], heads=parameters['heads'])
        if self.pro_features_flag:
            self.temporal = nn.Embedding(parameters['pro_input_dim'], parameters['pro_output_dim'])
            self.fc_hid = nn.Linear(parameters['hid_dim'] + parameters['pro_output_dim'], parameters['hid_dim'])
    def forward(self, src, src_len, route, route_len, pro_features):
        bs = src.size(1)
        src_max_len = src.size(0)
        route_max_len = route.size(0)
        mask3d = torch.ones(bs, src_max_len, src_max_len, device=src.device)
        route_mask3d = torch.ones(bs, route_max_len, route_max_len, device=src.device)
        route_mask2d = torch.ones(bs, route_max_len, device=src.device)
        inter_mask = torch.ones(bs, route_max_len, src_max_len, device=src.device)
        mask3d = sequence_mask3d(mask3d, src_len, src_len)
        route_mask3d = sequence_mask3d(route_mask3d, route_len, route_len)
        route_mask2d = sequence_mask(route_mask2d, route_len).transpose(0,1).unsqueeze(-1).repeat(1, 1, self.hid_dim)
        inter_mask = sequence_mask3d(inter_mask, route_len, src_len)
        src = src.transpose(0, 1)
        route = route.transpose(0, 1)
        outputs = self.transformer(src, route, mask3d, route_mask3d, inter_mask)
        outputs = outputs.transpose(0, 1)  
        assert outputs.size(0) == route_max_len
        outputs = outputs * route_mask2d
        hidden = torch.sum(outputs, dim=0) / route_len.unsqueeze(-1).repeat(1, self.hid_dim)
        hidden = hidden.unsqueeze(0)
        if self.pro_features_flag:
            extra_emb = self.temporal(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=-1)))
        return outputs, hidden
class DecoderMulti(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.id_size = parameters['id_size']
        self.emb_id=nn.Parameter(torch.rand(parameters['id_size'], parameters['id_emb_dim']))
        self.dest_type = parameters['dest_type']
        self.rate_flag = parameters['rate_flag']
        self.prog_flag = parameters['prog_flag']
        self.rid_feats_flag = parameters['rid_feats_flag']
        rnn_input_dim = parameters['hid_dim']
        if self.rid_feats_flag:
            rnn_input_dim += parameters['rid_fea_dim']
        if self.rate_flag:
            rnn_input_dim += 1
        if self.dest_type in [1, 2]:
            rnn_input_dim += parameters['hid_dim']
            if self.rid_feats_flag:
                rnn_input_dim += parameters['rid_fea_dim']
            if self.rate_flag:
                rnn_input_dim += 1
        self.rnn = nn.GRU(rnn_input_dim, parameters['hid_dim'])
        self.attn_route = Attention(parameters['hid_dim'])
        if self.rate_flag:
            fc_rate_out_input_dim = parameters['hid_dim'] + parameters['hid_dim']
            self.fc_rate_out = nn.Sequential(
                nn.Linear(fc_rate_out_input_dim, parameters['hid_dim'] * 2),
                nn.ReLU(),
                nn.Linear(parameters['hid_dim'] * 2, 1),
                nn.Sigmoid()
            )
    def decoding_step(self, input_id, input_rate, hidden, route_outputs,
                      route_attn_mask, d_rids, d_rates, rid_features_dict, dt, observed_emb, observed_mask):
        rnn_input = self.emb_id[input_id]
        if self.rid_feats_flag:
            rnn_input = torch.cat([rnn_input, rid_features_dict[input_id]], dim=-1)
        if self.rate_flag:
            rnn_input = torch.cat((rnn_input, input_rate), dim=-1)
        if self.dest_type in [1, 2]:
            embed_drids = self.emb_id[d_rids]
            rnn_input = torch.cat((rnn_input, embed_drids), dim=-1)
            if self.rid_feats_flag:
                rnn_input = torch.cat([rnn_input, rid_features_dict[input_id]], dim=-1)
            if self.rate_flag:
                rnn_input = torch.cat((rnn_input, d_rates), dim=-1)
        rnn_input = rnn_input.unsqueeze(0)
        output, hidden = self.rnn(rnn_input, hidden)
        if False:
            observed_emb = observed_emb.unsqueeze(1)
            _, observed_weighted = self.observed_attn(hidden.permute(1, 0, 2), observed_emb, observed_emb, observed_mask.unsqueeze(1))
            query = hidden.permute(1, 0, 2) + observed_weighted
        else:
            query = hidden.permute(1, 0, 2)
        key = route_outputs.permute(1, 0, 2).unsqueeze(1)
        scores, weighted = self.attn_route(query, key, key, route_attn_mask.unsqueeze(1))  
        prediction_id = scores.squeeze(1).masked_fill(route_attn_mask == 0, 0)
        weighted = weighted.permute(1, 0, 2)
        if self.rate_flag:
            rate_input = torch.cat((hidden, weighted), dim=-1).squeeze(0)
            prediction_rate = self.fc_rate_out(rate_input)
        else:
            prediction_rate = torch.ones((prediction_id.shape[0], 1), dtype=torch.float32, device=hidden.device) / 2
        return prediction_id, prediction_rate, hidden
    def forward(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, hidden, rid_features_dict, routes, route_outputs, route_attn_mask, d_rids, d_rates, teacher_forcing_ratio):
        routes = routes.permute(1, 0)  
        outputs_id = torch.zeros([max_trg_len, batch_size, routes.shape[1]], device=hidden.device)
        rate_out_dim = 1
        outputs_rate = torch.zeros([max_trg_len, batch_size, rate_out_dim], device=hidden.device)
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        for t in range(1, max_trg_len):
            teacher_force = random.random() < teacher_forcing_ratio
            dt = None
            observed_emb = None
            observed_mask = None
            prediction_id, prediction_rate, hidden = self.decoding_step(input_id, input_rate, hidden, route_outputs, route_attn_mask, d_rids, d_rates, rid_features_dict, dt, observed_emb, observed_mask)
            if teacher_forcing_ratio == -1 and self.prog_flag:
                for i in range(batch_size):
                    if t < trg_len[i]:
                        prev_idx = (input_id[i] == routes[i]).nonzero(as_tuple=True)[0][0]
                        tmp_flag = True
                        while tmp_flag:
                            cur_idx = prediction_id[i].argmax()
                            if cur_idx < prev_idx:
                                prediction_id[i, cur_idx] = 1e-6
                            else:
                                tmp_flag = False
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate
            if teacher_force:
                input_id = trg_id[t]
                input_rate = trg_rate[t]
            else:
                input_id = (F.one_hot(prediction_id.argmax(dim=1), routes.shape[1]) * routes).sum(-1)
                input_rate = prediction_rate
        mask_trg = torch.ones([batch_size, max_trg_len], device=outputs_id.device)
        mask_trg = sequence_mask(mask_trg, torch.tensor(trg_len, device=outputs_id.device))
        outputs_rate = outputs_rate.permute(1, 0, 2)  
        outputs_rate = outputs_rate.masked_fill(mask_trg.unsqueeze(-1) == 0, 0)
        outputs_rate = outputs_rate.permute(1, 0, 2)
        return outputs_id, outputs_rate
class TrajRecovery(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.srcseg_flag = parameters['srcseg_flag']
        self.hid_dim = parameters['hid_dim']
        self.da_route_flag = parameters['da_route_flag']
        self.learn_pos = parameters['learn_pos']
        self.rid_feats_flag = parameters['rid_feats_flag']
        self.params = parameters
        self.emb_id = nn.Parameter(torch.rand(parameters['id_size'], parameters['id_emb_dim']))
        if self.learn_pos:
            max_input_length = 500
            self.pos_embedding_gps = nn.Embedding(max_input_length, parameters['hid_dim'])
            self.pos_embedding_route = nn.Embedding(max_input_length, parameters['hid_dim'])
        input_dim_gps = 3
        if self.learn_pos:
            input_dim_gps += parameters['hid_dim']
        if self.srcseg_flag:
            input_dim_gps += parameters['hid_dim'] + 1
        self.fc_in_gps = nn.Linear(input_dim_gps, parameters['hid_dim'])
        input_dim_route = parameters['hid_dim']
        if self.learn_pos:
            input_dim_route += parameters['hid_dim']
        if self.rid_feats_flag:
            input_dim_route += parameters['rid_fea_dim']
        self.fc_in_route = nn.Linear(input_dim_route, parameters['hid_dim'])
        if self.da_route_flag:
            self.encoder = GREncoder(parameters)
        else:
            self.encoder = GPSEncoder(parameters)
        self.decoder = DecoderMulti(parameters)
        self.init_weights()  
        self.timer1, self.timer2, self.timer3, self.timer4, self.timer5, self.timer6 = [], [], [], [], [], []
    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
    def forward(self, src, src_len, trg_id, trg_rate, trg_len, pro_features, rid_features_dict, da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio):
        t0 = time.time()
        max_trg_len = trg_id.size(0)
        batch_size = trg_id.size(1)
        self.decoder.emb_id = self.emb_id  
        gps_emb = src.float()
        if self.learn_pos:
            gps_pos = src[:, :, -1].long()
            gps_pos_emb = self.pos_embedding_gps(gps_pos)
            gps_emb = torch.cat([gps_emb, gps_pos_emb], dim=-1)
        if self.srcseg_flag:
            seg_emb = self.emb_id[src_seg_seqs]
            gps_emb = torch.cat((gps_emb, seg_emb, src_seg_feats), dim=-1)
        gps_in = self.fc_in_gps(gps_emb)
        gps_in_lens = torch.tensor(src_len, device=src.device)
        self.timer1.append(time.time() - t0)
        t1 = time.time()
        if self.da_route_flag:
            route_emb = self.emb_id[da_routes]
            if self.learn_pos:
                route_pos_emb = self.pos_embedding_route(da_pos)
                route_emb = torch.cat([route_emb, route_pos_emb], dim=-1)
            if self.rid_feats_flag:
                route_feats = rid_features_dict[da_routes]
                route_emb = torch.cat([route_emb, route_feats], dim=-1)
            route_in = self.fc_in_route(route_emb)
            route_in_lens = torch.tensor(da_lengths, device=src.device)
            self.timer2.append(time.time() - t1)
            t2 = time.time()
            route_outputs, hiddens = self.encoder(gps_in, gps_in_lens, route_in, route_in_lens, pro_features)
            self.timer3.append(time.time() - t2)
            t3 = time.time()
        else:
            _, hiddens = self.encoder(gps_in, gps_in_lens, pro_features)
            route_in_lens = torch.tensor(da_lengths, device=src.device)
            route_outputs = self.emb_id[da_routes]
        route_attn_mask = torch.ones(batch_size, max(da_lengths), device=src.device)  
        route_attn_mask = sequence_mask(route_attn_mask, route_in_lens)
        t4 = time.time()
        self.timer4.append(time.time() - t3)
        outputs_id, outputs_rate = self.decoder(max_trg_len, batch_size, trg_id, trg_rate, trg_len, hiddens, rid_features_dict, da_routes, route_outputs, route_attn_mask, d_rids, d_rates, teacher_forcing_ratio)
        final_outputs_id = outputs_id[1:-1]
        final_outputs_rate = outputs_rate[1:-1]
        t5 = time.time()
        self.timer5.append(time.time() - t4)
        self.timer6.append(t5 - t0)
        return final_outputs_id, final_outputs_rate
class TrajRecTestData(Dataset): 
    def __init__(self, rn, trajs, mbr, parameters,dam):
        self.parameters = parameters
        self.rn = rn
        self.mbr = mbr  
        self.grid_size = parameters['grid_size']
        self.time_span = parameters['time_span']
        self.dam = dam 
        if parameters['eid_cate'] == 'gps2seg':
            inferred_segs = loadZ_pk(parameters['inferred_seg_path'])
            predict_id, _ = zip(*inferred_segs)
        else:
            predict_id = []
        self.src_grid_seqs, self.src_gps_seqs, self.src_pro_feas = [], [], []
        self.trg_gps_seqs, self.trg_rids, self.trg_rates = [], [], []
        self.src_seg_seq = []
        self.src_seg_feats = []
        self.src_time_seq = []
        self.trg_time_seq = []
        self.routes = []
        self.labels = []
        self.groups = []
        self.src_mms = []
        route_time = 0
        for serial, traj in enumerate(trajs):
            trg_list = traj.pt_list.copy()
            src_list = np.array(traj.pt_list, dtype=object)
            src_list = src_list[traj.low_idx].tolist()
            _, src_gps_seq, _, seg_seq, time_seq = self.get_src_seq(src_list)
            if parameters['eid_cate'] in ['mm', 'nn', 'gps2seg']:
                seg_seq = predict_id[serial]
            src_mm = []
            for seg, (lat, lng) in zip(seg_seq, src_gps_seq):
                projected, rate, dist = project_pt_to_road(self.rn, SPoint(lat, lng), seg)
                src_mm.append([[projected.lat, projected.lng], seg, rate])
            self.src_mms.append(src_mm)
            for p1, p1_idx, p2, p2_idx, s1, s2, ts, mmf1, mmf2 in zip(src_list[:-1], traj.low_idx[:-1], src_list[1:], traj.low_idx[1:], seg_seq[:-1], seg_seq[1:], time_seq[:-1], src_mm[:-1], src_mm[1:]):
                if (p1_idx + 1) < p2_idx:
                    tmp_seg_seq = [s1, s2]
                    tmp_src_list = [p1, p2]
                    ls_grid_seq, ls_gps_seq, hours, _, _ = self.get_src_seq(tmp_src_list)
                    features = get_pro_features(tmp_src_list, hours)
                    mm_gps_seq, mm_eids, mm_rates, trg_time = self.get_trg_seq(trg_list[p1_idx: p2_idx + 1])
                    path = traj.cpath[p1.cpath_idx: p2.cpath_idx + 1]
                    if parameters['eid_cate'] in ['mm', 'nn', 'gps2seg']:
                        t0 = time.time()
                        path = self.dam.planning_multi([s1, s2], ts, mode=parameters['planner'])
                        route_time += time.time() - t0
                        mm_gps_seq[0] = mmf1[0]
                        mm_eids[0] = self.rn.valid_edge_one[mmf1[1]]
                        mm_rates[0] = [mmf1[2]]
                        mm_gps_seq[-1] = mmf2[0]
                        mm_eids[-1] = self.rn.valid_edge_one[mmf2[1]]
                        mm_rates[-1] = [mmf2[2]]
                    self.routes.append([self.rn.valid_edge_one[item] for item in path])
                    self.src_seg_seq.append([self.rn.valid_edge_one[item] for item in tmp_seg_seq])
                    self.src_seg_feats.append(self.get_src_seg_feat(ls_gps_seq, tmp_seg_seq))
                    self.labels.append(get_label([self.rn.valid_edge_one[item] for item in path], mm_eids[1:-1]))
                    self.trg_gps_seqs.append(mm_gps_seq)
                    self.trg_rids.append(mm_eids)
                    self.trg_rates.append(mm_rates)
                    self.src_grid_seqs.append(ls_grid_seq)
                    self.src_gps_seqs.append(ls_gps_seq)
                    self.src_pro_feas.append(features)
                    self.src_time_seq.append(time_seq)
                    self.trg_time_seq.append(trg_time)
                    self.groups.append(serial)
    def __len__(self):
        return len(self.src_grid_seqs)
    def __getitem__(self, index):
        src_grid_seq = self.src_grid_seqs[index]
        trg_gps_seq = self.trg_gps_seqs[index]
        trg_rid = self.trg_rids[index]
        trg_rate = self.trg_rates[index]
        da_route = self.routes[index]
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.float32)
        src_seg_seq = self.src_seg_seq[index]
        src_seg_feat = self.src_seg_feats[index]
        src_seg_seq = torch.tensor(src_seg_seq)
        src_seg_feat = torch.tensor(src_seg_feat)
        src_grid_seq = torch.tensor(src_grid_seq)
        trg_gps_seq = torch.tensor(trg_gps_seq)
        trg_rid = torch.tensor(trg_rid)
        trg_rate = torch.tensor(trg_rate)
        src_pro_fea = torch.tensor(self.src_pro_feas[index])
        da_route = torch.tensor(da_route)
        d_rid = trg_rid[-1]
        d_rate = trg_rate[-1]
        trg_gps_seq = trg_gps_seq
        trg_rid = trg_rid
        trg_rate = trg_rate
        return da_route, src_grid_seq, src_pro_fea, src_seg_seq, src_seg_feat, trg_gps_seq, trg_rid, trg_rate, label, d_rid, d_rate
    def get_src_seg_feat(self, gps_seq, seg_seq):
        feats = []
        for ds_pt, seg in zip(gps_seq, seg_seq):
            gps = SPoint(ds_pt[0], ds_pt[1])
            candi = self.rn.pt2seg(gps, seg)
            feats.append([candi.rate])
        return feats
    def get_src_seq(self, ds_pt_list):
        timestamps = []
        hours = []
        ls_grid_seq = []
        ls_gps_seq = []
        first_pt = ds_pt_list[0]
        time_interval = self.time_span
        seg_seq = []
        for ds_pt in ds_pt_list:
            timestamps.append(ds_pt.time)
            hours.append(ds_pt.time_arr.hour)
            t = get_normalized_t(first_pt, ds_pt, time_interval)
            ls_gps_seq.append([ds_pt.lat, ds_pt.lng])
            if self.parameters['gps_flag']:
                locgrid_xid = (ds_pt.lat - self.rn.minLat) / (self.rn.maxLat - self.rn.minLat)
                locgrid_yid = (ds_pt.lng - self.rn.minLon) / (self.rn.maxLon - self.rn.minLon)
            else:
                locgrid_xid, locgrid_yid = gps2grid(ds_pt, self.mbr, self.grid_size)
            ls_grid_seq.append([locgrid_xid, locgrid_yid, t])
            seg_seq.append(ds_pt.data['candi_pt'].eid)
        return ls_grid_seq, ls_gps_seq, hours, seg_seq, timestamps
    def get_trg_seq(self, tmp_pt_list):
        mm_gps_seq = []
        mm_eids = []
        mm_rates = []
        time_arrs = []
        for pt in tmp_pt_list:
            time_arr = pt.time_arr
            time_arrs.append([time_arr.month + 1, time_arr.day + 1, 1 if time_arr.weekday() in [0, 1, 2, 3, 4] else 2, time_arr.hour + 1, time_arr.minute + 1, time_arr.second + 1])
            candi_pt = pt.data['candi_pt']
            mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            mm_eids.append(self.rn.valid_edge_one[candi_pt.eid])
            mm_rates.append([candi_pt.rate])
        return mm_gps_seq, mm_eids, mm_rates, time_arrs
def api_pre_trajrectestdata(trajs,rn,mbr,paramters,dam,path=''):
    if path and os.path.exists(path):
        cache = loadZ_pk(path)
        if isinstance(cache, dict):
            return NewTrajRecTestData(cache['items'], cache.get('groups', []), cache.get('src_mms', []))
        return NewTrajRecTestData(cache)
    data=TrajRecTestData(rn,trajs,mbr,paramters,dam)
    res=[data[i] for i in range(len(data))]
    cache = {'items': res, 'groups': data.groups, 'src_mms': data.src_mms}
    if path: saveZ_pk(path,cache)
    return NewTrajRecTestData(cache['items'], cache['groups'], cache['src_mms'])
class NewTrajRecTestData(Dataset):
    def __init__(self,data_ed, groups=None, src_mms=None):
        self.data=data_ed
        self.groups=[] if groups is None else groups
        self.src_mms=[] if src_mms is None else src_mms
    def __len__(self):return len(self.data)
    def __getitem__(self, i):
        return self.data[i]
def toseq(rn, rids, rates, path, seg_info):
    """
    Convert batched rids and rates to gps sequence.
    Args:
    -----
    rn_dict:
        use for rate2gps()
    rids:
        [trg len, batch size, id one hot dim] in torch
    rates:
        [trg len, batch size] in torch
    Returns:
    --------
    seqs:
        [trg len, batch size, 2] in torch
    """
    seqs = []
    for seg, rate in zip(rids, rates):
        if seg != 0:
            r0 = rate
            pt = rate2gps(rn, seg, r0)
            seqs.append([pt.lat, pt.lng])
        else:
            seqs.append([(rn.zone_range[0] + rn.zone_range[2]) / 2, (rn.zone_range[1] + rn.zone_range[3]) / 2])
    return seqs
def trmma_arg(root_map,rn,dataname):
    zone_range,ts,utc=city_info(dataname)
    args= {'device': device,'transformer_layers': 4,'heads': 4,'tandem_fea_flag': True,'pro_features_flag': True,'srcseg_flag': False,'da_route_flag': True,'rate_flag': True,'prog_flag': False,'dest_type': 2,'gps_flag': True,'rid_feats_flag': True,'learn_pos': True,'search_dist': 50,'beta': 15,'gamma': 30,'rid_fea_dim': 18, 'pro_input_dim': 48,  'pro_output_dim': 8,'min_lat': zone_range[0],'min_lng': zone_range[1],'max_lat': zone_range[2],'max_lng': zone_range[3],'city': dataname,'keep_ratio': 0.1,'grid_size': 50,'time_span': ts,'hid_dim': 64,'id_emb_dim': 64,'dropout': 0.1,'id_size': rn.valid_edge_cnt_one,'lambda1': 10,'lambda2': 5,'n_epochs': 50,'batch_size': 256,'learning_rate': 1e-3,"lr_step": 2,"lr_decay": 0.8,'tf_ratio': 1,'decay_flag': True,'decay_ratio': 0.9,'clip': 1,'log_step': 1,'utc': utc,'small': False,'eid_cate': 'gps2seg','inferred_seg_path': '','planner': 'da','root_map': root_map,}
    args['planner']='length'
    mbr = MBR(args['min_lat'], args['min_lng'], args['max_lat'], args['max_lng'])
    args['grid_num'] = gps2grid(SPoint(args['max_lat'], args['max_lng']), mbr, args['grid_size'])
    args['grid_num'] = (args['grid_num'][0] + 1, args['grid_num'][1] + 1)
    return args,mbr
def train_trmma(root_data,root_model,dataname):
    path_trmma=os.path.join(root_model,f'trmma.th.zst') ; check_dir(path_trmma)
    map_root = os.path.join(root_data, dataname, "roadnet"); check_dir(map_root)
    traj_root = os.path.join(root_data, dataname)
    zone_range,ts,utc=city_info(dataname)
    rn = RoadNetworkMapFull(map_root, zone_range=zone_range, unit_length=50)
    args,mbr=trmma_arg(dataname,rn)
    dam = DAPlanner(args['root_map'], args['id_size'] - 1, args['utc'])
    rid_features_dict = torch.from_numpy(rn.get_rid_rnfea_dict(dam, ts)).to(device)
    train_dataset = TrajRecData(rn, traj_root, mbr, args, 'train',True)
    valid_dataset = TrajRecData(rn, traj_root, mbr, args, 'valid',False)
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    print('Finish data preparing.')
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    train_iterator = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=trmma_collate_fn, num_workers=num_workers, pin_memory=False)
    valid_iterator = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=trmma_collate_fn, num_workers=num_workers, pin_memory=False)
    model = TrajRecovery(args).to(device)
    print('model', str(model))
    ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
        ls_train_rate_loss, ls_train_id_loss, ls_train_mae, ls_train_rmse = [], [], [], [], [], [], [], []
    ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
        ls_valid_rate_loss, ls_valid_id_loss, ls_valid_mae, ls_valid_rmse = [], [], [], [], [], [], [], []
    best_valid_loss = float('inf')  
    best_epoch = 0
    lr = args['learning_rate']
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args['lr_step'], factor=args['lr_decay'], threshold=1e-3)
    stopping_count = 0
    train_times = []
    for epoch in range(args['n_epochs']):
        start_time = time.time()
        print("==> training {}, {}...".format(args['tf_ratio'], lr))
        t_train = time.time()
        train_loss, train_id_loss, train_rate_loss = trmma_train(model, train_iterator, optimizer, rid_features_dict, args, device)
        end_train = time.time()
        print("training: {}".format(end_train - t_train))
        ls_train_loss.append(train_loss)
        ls_train_id_loss.append(train_id_loss)
        ls_train_rate_loss.append(train_rate_loss)
        print("==> validating...")
        t_valid = time.time()
        valid_loss, valid_id_loss, valid_rate_loss = trmma_evaluate(model, valid_iterator, rid_features_dict, args, device)
        print("validating: {}".format(time.time() - t_valid))
        ls_valid_id_loss.append(valid_id_loss)
        ls_valid_rate_loss.append(valid_rate_loss)
        ls_valid_loss.append(valid_loss)
        end_time = time.time()
        epoch_secs = end_time - start_time
        train_times.append(end_train - t_train)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_weight(model,path_trmma)
            best_epoch = epoch
            stopping_count = 0
        else:
            stopping_count += 1
        if (epoch % args['log_step'] == 0) or (epoch == args['n_epochs'] - 1):
            print('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_secs) + 's')
            print('Epoch: ' + str(epoch + 1) + ' TF Ratio: ' + str(args['tf_ratio']))
            print('\tTrain Loss:' + str(train_loss) +
                            '\tTrain RID Loss:' + str(train_id_loss) +
                            '\tTrain Rate Loss:' + str(train_rate_loss))
            print('\tValid Loss:' + str(valid_loss) +
                            '\tValid RID Loss:' + str(valid_id_loss) +
                            '\tValid Rate Loss:' + str(valid_rate_loss))
        if args['decay_flag']:
            args['tf_ratio'] = args['tf_ratio'] * args['decay_ratio']
        scheduler.step(valid_id_loss)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr <= 0.9 * 1e-5:
            print("==> [Info] Early Stop since lr is too small After Epoch {}.".format(epoch))
            break
        if stopping_count >= 5:
            print("==> [Info] Early Stop After Epoch {}.".format(epoch))
            break
    print('Best Epoch: {}, {}'.format(best_epoch, best_valid_loss))
    print('==> Best Epoch: {}, {}'.format(best_epoch, best_valid_loss))
    print('==> Training Time: {}, {}, {}, {}'.format(np.sum(train_times) / 3600, np.mean(train_times), np.min(train_times), np.max(train_times)))
    print('==> Training Time: {}, {}, {}, {}'.format(np.sum(train_times) / 3600, np.mean(train_times), np.min(train_times), np.max(train_times)))        
def test_trmma(root_data,root_model,dataname,name,save=True):
    path_infer=os.path.join(root_model,f'trmma-infer-{name}.pk.zst') ; check_dir(path_infer)
    path_rec = os.path.join(root_model,f'trmma-rec-{name}.pk.zst') ; check_dir(path_rec)
    path_trmma=os.path.join(root_model,f'trmma.th.zst') ; check_dir(path_trmma)
    map_root = os.path.join(root_data, dataname, "roadnet"); check_dir(map_root)
    zone_range,ts,utc=city_info(dataname)
    rn = RoadNetworkMapFull(map_root, zone_range=zone_range, unit_length=50)
    args,mbr=trmma_arg(root_data,rn,dataname)
    dam = DAPlanner(args['root_map'], args['id_size'] - 1, args['utc'])
    rid_features_dict = torch.from_numpy(rn.get_rid_rnfea_dict(dam, ts)).to(device)
    traj_root = os.path.join(root_data, dataname)
    test_dataset = TrajRecTestData(rn, traj_root, mbr, args, 'test',False)
    print('testing dataset shape: ' + str(len(test_dataset)))
    test_iterator = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn= trmma_collate_fn_test, num_workers=8, pin_memory=True)
    model = TrajRecovery(args)
    load_weight(model,path_trmma)
    model=model.to(device)
    print('==> Model Loaded')
    print("==> Starting Prediction...")
    if save and os.path.exists(path_infer):
        data=loadZ_pk(path_infer)
    else:
        start_time = time.time()
        data = trmma_infer(model, test_iterator, rid_features_dict, device)
        end_time = time.time()
        epoch_secs = end_time - start_time
        print('Time: ' + str(epoch_secs) + 's')
        print('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / len(test_dataset) * 1000, len(test_dataset) / (end_time - start_time)))
        print('Inference Time: {}, {}, {}'.format(end_time - start_time, (end_time - start_time) / len(test_dataset) * 1000, len(test_dataset) / (end_time - start_time)))
        if save:saveZ_pk(path_infer,data)
    if save and os.path.exists(path_rec):
        results=loadZ_pk(path_rec)
    else:
        outputs = []
        for pred_seg, pred_rate, trg_id, trg_rate, trg_gps, route in data:
            pred_gps = toseq(rn, pred_seg, pred_rate, route, dam.seg_info)
            outputs.append([pred_gps, pred_seg, trg_gps, trg_id])
        test_trajs = pickle.load(open(os.path.join(traj_root, 'test_output.pkl'), "rb"))
        groups = Counter(test_dataset.groups)
        nums = []
        for i in range(len(test_trajs)):
            nums.append(groups[i])
        results = []
        for traj, num, src_mm in zip(test_trajs, nums, test_dataset.src_mms):
            tmp_all = outputs[:num]
            low_idx = traj.low_idx
            gps, segs, _ = zip(*src_mm)
            predict_ids = [segs[0]]
            predict_gps = [gps[0]]
            pointer = -1
            for p1_idx, p2_idx, seg, latlng in zip(low_idx[:-1], low_idx[1:], segs[1:], gps[1:]):
                if (p1_idx + 1) < p2_idx:
                    pointer += 1
                    tmp = tmp_all[pointer]
                    predict_gps.extend(tmp[0])
                    predict_ids.extend(tmp[1])
                predict_ids.append(seg)
                predict_gps.append(latlng)
            outputs = outputs[num:]
            mm_gps_seq = []
            mm_eids = []
            for i, pt in enumerate(traj.pt_list):
                candi_pt = pt.data['candi_pt']
                mm_eids.append(candi_pt.eid)
                mm_gps_seq.append([candi_pt.lat, candi_pt.lng])
            assert len(predict_gps) == len(mm_gps_seq) == len(predict_ids) == len(mm_eids)
            results.append([predict_gps, predict_ids, mm_gps_seq, mm_eids])
        if save:saveZ_pk(path_rec,results)
    print("==> Starting Evaluation...")
    epoch_id1_loss = []
    epoch_recall_loss = []
    epoch_precision_loss = []
    epoch_f1_loss = []
    epoch_mae_loss = []
    epoch_rmse_loss = []
    for pred_gps, pred_seg, trg_gps, trg_id in results:
        recall, precision, f1, loss_ids1, loss_mae, loss_rmse = calc_metrics(pred_seg, pred_gps, trg_id, trg_gps)
        epoch_id1_loss.append(loss_ids1)
        epoch_recall_loss.append(recall)
        epoch_precision_loss.append(precision)
        epoch_f1_loss.append(f1)
        epoch_mae_loss.append(loss_mae)
        epoch_rmse_loss.append(loss_rmse)
    test_id_recall, test_id_precision, test_id_f1, test_id_acc, test_mae, test_rmse = np.mean(epoch_recall_loss), np.mean(epoch_precision_loss), np.mean(epoch_f1_loss), np.mean(epoch_id1_loss), np.mean(epoch_mae_loss), np.mean(epoch_rmse_loss)
    print(test_id_recall, test_id_precision, test_id_f1, test_id_acc, test_mae, test_rmse)
    print('Time: ' + str(epoch_secs) + 's')
    print('\tTest RID Acc:' + str(test_id_acc) +
                    '\tTest RID Recall:' + str(test_id_recall) +
                    '\tTest RID Precision:' + str(test_id_precision) +
                    '\tTest RID F1 Score:' + str(test_id_f1) +
                    '\tTest MAE Loss:' + str(test_mae) +
                    '\tTest RMSE Loss:' + str(test_rmse))
