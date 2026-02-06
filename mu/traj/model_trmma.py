exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
from traj.models.one_trmma import *
from mu.MU import *
from mu.traj.loaddata import ts_split_area,ts_split_usr,train_rate,fix_trajs_num
from _nn.nFile import load_weight_mem,save_weight_mem
from rtree import Rtree
from mu.traj.mia_lstm import MIA as MIA3
from mu.metrics import _MIA2 as MIA4,basic_infer
def call_tvt_urv(dudrdv_type:str,du_rate):
    """idx of tr,val,te,du,dr,dv=fun(ts,uid,train_rate,du_rate)"""
    dudrdv_type=dudrdv_type.lower()
    if dudrdv_type =='usr':  return lambda ts,uid:ts_split_usr(ts,uid,du_rate,train_rate)
    if dudrdv_type =='area': return lambda ts,uid:ts_split_area(ts,du_rate,train_rate)
def slice_fn(du:GPS2SegData,num):
    if isinstance(du,list) or isinstance(du,np.ndarray):return du[:min(num,len(du))]
    du.trajs=du.trajs[:min(num,len(du.trajs))]
    return du
def _map_pre(data_name,root_map):
    path_map=os.path.join(root_map,data_name+'.map.pk.zst')
    if os.path.exists(path_map):
        rn:RoadNetworkMapFull= loadZ_pk(path_map)
        if os.path.exists(rn.rtreeFile+'.dat'):
            rn.spatial_index=Rtree(rn.rtreeFile)
            return rn
    zone_range,ts,utc=city_info(data_name)
    rn = RoadNetworkMapFull(os.path.join(root_map,'map'), zone_range=zone_range, unit_length=50)
    saveZ_pk(path_map,rn)
    return rn
class TrajMap(TaskModel):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.dtrain=self.dval=None
        self.args=None
        self._train_best_val_loss=float('inf');self._train_stop_count=0
        self._loss=nn.BCELoss(reduction='mean')
        self._train_weight=None
    def data_init(self,urv,rt_if_exist=False):
        """dr,du,dv,dtrain,dtest,dval,dr_raw,du_raw"""
        tvt_urv_root = self.root_data+f'-{urv}'
        x=api_preprocess(self.root_map,tvt_urv_root,self.data_name,call_tvt_urv(urv,self.du_rate),rt_if_exist) 
        if rt_if_exist:return x
        rn=_map_pre(self.data_name,self.root_map)
        assert len(rn.spatial_index)
        args,mbr=mma_args(self.data_name,rn)
        dtrain,dval,dtest,du,dr,dv,du_raw,dr_raw,dv_raw=x
        assert len(dr)>5 and len(dv)>5 and len(du)>5 and len(dtrain)>5 and len(dval)>5
        dtrain=GPS2SegData(rn,dtrain,mbr,args,'train',True)
        dr=GPS2SegData(rn,dr,mbr,args,'dr',True)
        du=GPS2SegData(rn,du,mbr,args,'du',True)
        dv=GPS2SegData(rn,dv,mbr,args,'dv',True)
        dval=GPS2SegData(rn,dval,mbr,args,'valid',False)
        assert len(dr)>5 and len(dv)>5 and len(du)>5 and len(dtrain)>5 and len(dval)>5
        dr,du,dv,dtrain,dval,dr_raw,du_raw=fix_trajs_num({k:v for k,v in zip('dr,du,dv,dtrain,dval,dr_raw,du_raw'.split(','),[dr,du,dv,dtrain,dval,dr_raw,du_raw])},self.du_rate,slice_fn).values()
        if DEBUG: dtrain=slice_fn(dtrain,1000);dval=slice_fn(dval,100);du=slice_fn(du,800);dr=slice_fn(dr,200);dv=slice_fn(dv,200)
        assert len(dr)>5 and len(dv)>5 and len(du)>5 and len(dtrain)>5 and len(dval)>5
        self.dtrain=dtrain;self.dr=dr;self.du=du
        self.dval=dval;self.args=args
        assert len(du)==len(du_raw) and len(dr)==len(dr_raw) 
        return {'dr':dr,'du':du,'dv':dv,'dtrain':dtrain,'dtest':None,'dval':dval,'dr_raw':dr_raw,'du_raw':du_raw}

    def get_collate_fn(self): 
        return mma_collate_fn
    def _call_new_model(self):
        self.model= GPS2Seg(self.args).to(device)
        return self.model
    def _call_new_opt(self,**kw):
        return optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
    def forward(self,x):
        src_seqs, src_lengths, _, candi_labels, candi_ids, candi_feats, candi_masks = x
        return self.model(src_seqs, src_lengths, candi_ids, candi_feats, candi_masks)
    def lossF(self,y,x): 
        src_seqs, src_lengths, _, candi_labels, candi_ids, candi_feats, candi_masks = x
        bce_loss = self._loss(y, candi_labels.float()) * candi_ids.shape[-1]
        return bce_loss
    def lossFdist(self,teacher,student): 
        return self._lossF_dist_cls(teacher,student)
    def train_after_epoch(self)->bool:
        """return: if exit train"""
        dval_loader=DataLoader(self.dval,self.args['batch_size'],num_workers=num_workers,collate_fn=self.get_collate_fn())
        self.model.eval()
        valid_id_loss = mma_evaluate(self.model,dval_loader , device)
        self.model.train()
        if valid_id_loss < self._train_best_val_loss:
            self._train_best_val_loss = valid_id_loss
            self._train_weight=save_weight_mem(self.model)
            self._train_stop_count = 0
        else:
            self._train_stop_count += 1
        if self.args['decay_flag']:
            self.dtrain.keep_ratio = max(self.args['keep_ratio'], self.dtrain.keep_ratio * self.args['decay_ratio'])
        if self._train_stop_count >= 5:
            print("==> [Info] Early Stop.")
            load_weight_mem(self.model, self._train_weight)
            return True
    def get_task_metrics(self): 
        self.du.is_train=self.dv.is_train=self.dr.is_train=False
        self.model.eval()
        def mia_call(y:torch.Tensor,x): 
            return torch.softmax(y[:,0,:],dim=-1)
        def mia_call4(y:torch.Tensor,x): 
            return torch.softmax(y.sort(dim=-1)[0].mean(dim=1),dim=-1) 
        def acc_f1(du,name): 
            path_res=os.path.join(self.root_model,f'mma-{self.urv}-{name}.pk.zst') ; check_dir(path_res)
            if 'Origin' in path_res:
                if os.path.exists(path_res):
                    pred_data=loadZ_pk(path_res)
                else:
                    pred_data = mma_infer(self.model, du, device)
                    print('save ',path_res) 
                    saveZ_pk(path_res,pred_data)
            else: pred_data = mma_infer(self.model, du, device)
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
            test_id_acc, test_id_recall, test_id_precision, test_id_f1 = np.mean(epoch_id1_loss), np.mean(epoch_recall_loss), np.mean(epoch_precision_loss), np.mean(epoch_f1_loss)
            return {'acc':test_id_acc.item(),'f1':test_id_f1.item()}
        dv_loader=DataLoader(self.dv,self.args['batch_size'],num_workers=num_workers,collate_fn=self.get_collate_fn())
        dr_loader=DataLoader(self.dr,self.args['batch_size'],num_workers=num_workers,collate_fn=self.get_collate_fn())
        du_loader=DataLoader(self.du,self.args['batch_size'],num_workers=num_workers,collate_fn=self.get_collate_fn())
        out_dr=basic_infer(self,dr_loader);out_du=basic_infer(self,du_loader);out_dv=basic_infer(self,dv_loader)
        mia5=MIA4([x.reshape(-1,10) for x in out_dr],[x.reshape(-1,10) for x in out_dv],[x.reshape(-1,10) for x in out_du])
        mia4=MIA4([mia_call4(y,0) for y in out_dr],[mia_call4(y,0) for y in out_dv],[mia_call4(y,0) for y in out_du])
        res_du=acc_f1(du_loader,'du') ; res_dr=acc_f1(dr_loader,'dr') ; res_dv=acc_f1(dv_loader,'dv')
        return {
            'MIA':'','mia_call':mia_call, 
                'MIA4':mia4, 
                'MIA5':mia5, 
                'F1_Du':res_du['f1'],'F1_Dr':res_dr['f1'],'F1_Dv':res_dv['f1'],
                'Acc_Du':res_du['acc'],'Acc_Dr':res_dr['acc'],'Acc_Dv':res_dv['acc'],
                }
class TrajRec(TaskModel):
    def __init__(self,**kw):
        super().__init__( **kw)
        self.dtrain=self.dval=None
        self._dr=self._du=self._dv=self._odr=self._odv=self._odu=None 
        self._rid_features_dict=None ; self._rn=None;self._dam=None 
        self._loss_reg= nn.L1Loss(reduction='sum');self._loss_bce=nn.BCELoss(reduction='sum')
        self._train_best_loss=float('inf') ; self._train_weight=None ; self._stop_count=0
        self._scheduler = None ; self._tf_ratio=1
        self._opt=None
    def data_init(self,urv,rt_if_exist=False):
        """dr,du,dv,dtrain,dtest,dval,dr_raw,du_raw & dr_eval,du_eval,dv_eval"""
        root_data=self.root_data.replace('TrajRec','TrajMap')
        tvt_urv_root = root_data+f'-{urv}'
        rn=_map_pre(self.data_name,self.root_map);self._rn=rn
        x=api_preprocess(self.root_map,tvt_urv_root,self.data_name,call_tvt_urv(urv,self.du_rate),rt_if_exist)
        zone_range,ts,utc=city_info(self.data_name)
        if rt_if_exist:return x
        dtrain,dval,dtest,_odu,_odr,_odv,du_raw,dr_raw,dv_raw=x
        args,mbr=trmma_arg(self.root_map,rn,self.data_name)
        dtrain,dval,_odu,_odr,_odv,du_raw,dr_raw,dv_raw=fix_trajs_num({k:v for k,v in zip('dtrain,dval,_odu,_odr,_odv,du_raw,dr_raw,dv_raw'.split(','),[dtrain,dval,_odu,_odr,_odv,du_raw,dr_raw,dv_raw])},self.du_rate,slice_fn).values()
        if DEBUG: dtrain=slice_fn(dtrain,1000);dval=slice_fn(dval,100);_odu=slice_fn(_odu,200);_odr=slice_fn(_odr,800);_odv=slice_fn(_odv,200)
        dtrain=TrajRecData(rn, dtrain, mbr, args, 'train',True)
        dr=TrajRecData(rn, deepcopy(_odr), mbr, args, 'dr',True) 
        du=TrajRecData(rn, deepcopy(_odu), mbr, args, 'du',True)
        dv=TrajRecData(rn, deepcopy(_odv), mbr, args, 'dv',True)
        dval=TrajRecData(rn, dval, mbr, args, 'valid',False) 
        root_mma=self.root_model[:self.root_model.index('TrajRec')]+f'TrajMap{self.du_rate}/Origin/'
        dam = DAPlanner(self.root_map, args['id_size'] - 1, utc)
        args['inferred_seg_path']=os.path.join(root_mma,f'mma-{self.urv}-dr.pk.zst');_dr=TrajRecTestData(rn,deepcopy(_odr), mbr, args,dam) 
        args['inferred_seg_path']=os.path.join(root_mma,f'mma-{self.urv}-du.pk.zst');_du=TrajRecTestData(rn, deepcopy(_odu), mbr, args,dam) 
        args['inferred_seg_path']=os.path.join(root_mma,f'mma-{self.urv}-dv.pk.zst');_dv=TrajRecTestData(rn, deepcopy(_odv), mbr, args,dam)
        self._rid_features_dict = torch.from_numpy(rn.get_rid_rnfea_dict(dam, ts)).to(device)
        self.dtrain=dtrain;self.dr=dr;self.du=du ;self.args=args
        self.dval=dval;self._du=_du;self._dr=_dr;self._dv=_dv
        self._odr=_odr;self._odv=_odv;self._odu=_odu;self._dam=dam
        return {'dr':dr,'du':du,'dv':dv,'dtrain':dtrain,'dtest':None,'dval':dval,'dr_eval':_dr,'dv_eval':_dv,'du_eval':_du,'dr_raw':dr_raw,'du_raw':du_raw}


    def get_collate_fn(self): 
        return  trmma_collate_fn
    def get_collate_fn_test(self): 
        return trmma_collate_fn_test
    def _call_new_model(self):
        self.model= TrajRecovery(self.args).to(device)
        return self.model
    def _call_new_opt(self,**kw):
        if 'lr' not in kw:kw['lr']=1e-3
        self._opt= optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), **kw)
        return self._opt
    def _call_new_sch(self,opt):
        self._scheduler= optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=2, factor=0.8, threshold=1e-3)
        return None
    def forward(self,x):
        if len(x)==15: 
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_gps_seqs, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = x
        else: 
            src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = x
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
        output_ids, output_rates = self.model(src_seqs, src_lengths, trg_rids, trg_rates, trg_lengths, src_pro_feas, self._rid_features_dict, da_routes, da_lengths, da_pos, src_seg_seqs, src_seg_feats, d_rids, d_rates, teacher_forcing_ratio=self._tf_ratio)
        return output_ids,output_rates
    def lossF(self,y,x): 
        src_seqs, src_pro_feas, src_seg_seqs, src_seg_feats, src_lengths, trg_rids, trg_rates, trg_lengths, trg_rid_labels, da_routes, da_lengths, da_pos, d_rids, d_rates = x
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
        output_ids,output_rates=y
        trg_lengths_sub = [length - 2 for length in trg_lengths]
        loss_train_ids = self._loss_bce(output_ids, trg_rid_labels) * 10 / np.sum(np.array(trg_lengths_sub) * np.array(da_lengths))
        ttl_loss = loss_train_ids
        loss_rates = self._loss_reg(output_rates, trg_rates[1:-1]) * 5 / sum(trg_lengths_sub)
        ttl_loss += loss_rates
        return ttl_loss
    def lossFdist(self,teacher,student):
        return self._lossF_dist_cls(teacher[1],student[1])
    def train_after_epoch(self)->bool:
        """return: if exit train"""
        dv_loader=DataLoader(self.dval,self.bs,False,collate_fn=self.get_collate_fn(),pin_memory=False,num_workers=num_workers)
        self.model.eval()
        valid_loss, valid_id_loss, valid_rate_loss = trmma_evaluate(self.model, dv_loader, self._rid_features_dict, self.args, device)
        self.model.train()
        if valid_loss < self._train_best_loss:
            self._train_best_loss = valid_loss
            self._train_weight=save_weight_mem(self.model)
            self._stop_count = 0
        else:
            self._stop_count += 1
        self._tf_ratio = self._tf_ratio * 0.9
        self._scheduler.step(valid_id_loss)
        lr = self._opt.param_groups[0]['lr']
        if lr <= 0.9 * 1e-5:
            print("==> [Info] Early Stop since lr is too small")
            load_weight_mem(self.model,self._train_weight) ; return True
        if self._stop_count >= 20: 
            print("==> [Info] Early Stop")
            load_weight_mem(self.model,self._train_weight) ; return True
        return False
    def get_task_metrics(self): 
        self.model.eval()
        def infer_traj(du:TrajRecTestData,odu):
            loader=DataLoader(du,self.bs,False,collate_fn=trmma_collate_fn_test,num_workers=num_workers)
            data = trmma_infer(self.model, loader, self._rid_features_dict, device)
            outputs = []
            for pred_seg, pred_rate, trg_id, trg_rate, trg_gps, route in data:
                pred_gps = toseq(self._rn, pred_seg, pred_rate, route, self._dam.seg_info)
                outputs.append([pred_gps, pred_seg, trg_gps, trg_id])
            groups = Counter(du.groups)
            nums = []
            for i in range(len(odu)):
                nums.append(groups[i])
            results = []
            for traj, num, src_mm in zip(odu, nums, du.src_mms):
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
            return results
        rec_du=infer_traj(self._du,self._odu)
        rec_dr=infer_traj(self._dr,self._odr)
        rec_dv=infer_traj(self._dv,self._odv)
        def acc_f1_mae_rmse(results):
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
            return {'acc':test_id_acc.item(),'f1':test_id_f1.item(),'mae':test_mae.item(),'rmse':test_rmse.item()}
        def mia_call(y,x):
            output_ids, output_rates=y 
            prob=torch.sort(output_ids,dim=-1,descending=True)[0][:,:,:10]
            prob=torch.softmax(prob,dim=-1)
            return prob[0]
        res_du=acc_f1_mae_rmse(rec_du) ; res_dr=acc_f1_mae_rmse(rec_dr) ; res_dv=acc_f1_mae_rmse(rec_dv)
        return {'MIA':'', 'mia_call':mia_call,
                'F1_Du':res_du['f1'],'F1_Dr':res_dr['f1'],'F1_Dv':res_dv['f1'],
                'Acc_Du':res_du['acc'],'Acc_Dr':res_dr['acc'],'Acc_Dv':res_dv['acc'],
                'MAE_Du':res_du['mae'],'MAE_Dr':res_dr['mae'],'MAE_Dv':res_dv['mae'],
                'RMSE_Du':res_du['rmse'],'RMSE_Dr':res_dr['rmse'],'RMSE_Dv':res_dv['rmse'],
                'MIA3':MIA3([x[0] for x in rec_du],[x[0] for x in rec_dr],[x[0] for x in rec_dv]),
                 }
