exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import numpy as np
from mu.MU import *
from traj.models.one_mlsimp import *
from mu.traj.loaddata import load_traj_raw,ts_split
from traj.data.load_trajs import traj_bbox,traj_tbox
from mu.traj.mia_lstm import MIA as MIA3
from traj.sim.measures import ITS

fix_traj_len={'Porto':500,'Beijing':110,'Xian':400}
cr={'Porto':0.01,'Beijing':0.05,'Xian':0.01}
def _data_pre(data,root,fix_traj_len):
    path=os.path.join(root,f'ts-n{fix_traj_len}.pk.zst')
    if os.path.exists(path):
        ts,UIDS=loadZ_pk(path)
        return ts,UIDS
    ts,uids,*other=load_traj_raw(data)
    TS=[];UIDS=[]
    for i,(t,uid) in enumerate(zip(ts,uids)):
        if len(t)<fix_traj_len: ts[i]=None
        else:
            TS.append(np.array(t)[:fix_traj_len])
            UIDS.append(int(uid))
    del ts,uids
    if len(TS)<1500:
        print('len TS',len(TS))
        raise RuntimeError('TS not enough')
    ts=np.array(TS,dtype=object)
    UIDS=np.array(UIDS,dtype=int)
    saveZ_pk(path,[ts,UIDS])
    return ts,UIDS
def _data_init(root_data,data_name,bbox,urv,durate,bs,rt_if_exist):
    root_data=root_data+f'-{urv}' ; check_dir(root_data+os.path.sep)
    path=os.path.join(root_data,f'ts-n{fix_traj_len[data_name]}.pk.zst')
    if os.path.exists(path):
        if rt_if_exist:return True
        dr,du,dv,dtrain,dtest,dval,_dtrain,_dval,_dtest,_du,_dr,_dv,du_idx = loadZ_pk(path)
        return dr,du,dv,dtrain,dtest,dval,_dtrain,_dr,_du,_dv,du_idx
    ts,UIDS=_data_pre(data_name,root_data.replace(str(durate),'').replace(f'init-{urv}',''),fix_traj_len[data_name])
    rate=1 
    train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx=ts_split(ts,UIDS,urv,durate,train_num=1000*rate) 
    train_idx=train_idx[:1000*rate] 
    val_idx,test_idx,du_idx,dr_idx,dv_idx=val_idx[:200*rate],test_idx[:200*rate],du_idx[:200*rate],dr_idx[:800*rate],dv_idx[:400*rate]
    _dtrain,_dval,_dtest,_du,_dr,_dv=ts[train_idx],ts[val_idx],ts[test_idx],ts[du_idx],ts[dr_idx],ts[dv_idx]
    _dtrain,_dval,_dtest,_du,_dr,_dv=np.array(_dtrain,dtype=float), np.array(_dval,dtype=float), np.array(_dtest,dtype=float), np.array(_du,dtype=float), np.array(_dr,dtype=float), np.array(_dv,dtype=float)
    api_tbert_pretrain(root_data=root_data,ts_train=_dtrain,bbox=bbox,bs=bs)
    dtrain=api_pre_gnndata(root_data,'dtrain',_dtrain,bbox,bs=bs)
    dval=api_pre_gnndata(root_data,'dval',_dval,bbox,bs=bs)
    dtest=api_pre_gnndata(root_data,'dtest',_dtest,bbox,bs=bs)
    du=api_pre_gnndata(root_data,'du',deepcopy(_du),bbox,bs=bs)
    dr=api_pre_gnndata(root_data,'dr',deepcopy(_dr),bbox,bs=bs)
    dv=api_pre_gnndata(root_data,'dv',deepcopy(_dv),bbox,bs=bs)
    assert len(du)>5 and len(_du)>5
    assert len(du)==len(_du)
    saveZ_pk(path,[dr,du,dv,dtrain,dtest,dval,_dtrain,_dval,_dtest,_du,_dr,_dv,du_idx])
    api_pre_test(root_data,'du',bbox,_du,data_name)
    api_pre_test(root_data,'dr',bbox,_dr,data_name)
    api_pre_test(root_data,'dv',bbox,_dv,data_name)
    print('build data over, please re-run');raise DataInitFinishExp()
class TrajSimp(TaskModel):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.model:GAT=None 
        self.bbox=traj_bbox[self.data_name];self.bbox.append(traj_tbox[self.data_name])
        self.dtrain:GraphSimpDataset=None ; self._dtrain=self._dr=self._du=self._dv=None
        self.id_du,self.id_dtrain,self.id_dr,self.id_dv=None,None,None,None
        self._diff_trajs_idx=None ; self.du_idx=None
    def data_init(self,uvr,rt_if_exist=False):
        """dr,du,dv,dtrain,dtest,dval,dr_raw,du_raw"""
        x=_data_init(self.root_data,self.data_name,self.bbox,uvr,self.du_rate,bs=self.bs,rt_if_exist=rt_if_exist)
        if rt_if_exist:return True
        dr,du,dv,dtrain,dtest,dval,_dtrain,_dr,_du,_dv,du_idx=x
        self._dtrain=_dtrain;self._dr=_dr;self._du=_du;self._dv=_dv;self.du_idx=du_idx
        dr_raw=self._dr;du_raw=self._du
        return {'dr':dr,'du':du,'dv':dv,'dtrain':dtrain,'dtest':dtest,'dval':dval,'dr_raw':dr_raw,'du_raw':du_raw}

    def get_collate_fn(self): 
        return GraphSimpcollate
    def _call_new_model(self):
        self.model= GAT().to(device)
        return self.model
    def _call_new_opt(self,**kw):
        if 'lr' not in kw:kw['lr']=1e-4
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), weight_decay=0,**kw)
    def _call_new_sch(self,opt):
        return None
    def model_pretrain(self,path_train):
        """time"""
        root_model=path_train[:-6]
        gnn_path= root_model+'pre-gnn.th.zst'
        simp_path=root_model+'pre-simp.th.zst'
        diff_path=root_model+'pre-diff.th.zst' ; diff_path='' 
        tim_path= root_model+'pre-tim.pk'
        traj_idx_path=root_model+'pre-diff-idx.pk.zst'
        if os.path.exists(gnn_path) and os.path.exists(tim_path):
            self.model=self.call_new_model();self.model.load_state_dict(loadZ_th(gnn_path))
            if os.path.exists(traj_idx_path): 
                self._diff_trajs_idx=loadZ_pk(traj_idx_path)
            return load_pk(tim_path)
        if 'origin' in path_train or 'Origin' in path_train:
            name='dtrain'
            ts=self._dtrain
        elif 'retrain' in path_train or 'Retrain' in path_train:
            name='dr'
            ts=self._dr
        else:raise RuntimeError('only origin and retrain can be pretrained')
        t1=time.time()
        graph_train_dataset = api_pre_gnndata(self.root_data+f'-{self.urv}',name)
        simp_trajs_idx=None
        diff_trajs_idx=None
        load_model=False
        for i in range(1 if DEBUG else 9): 
            simp_trajs_idx = train_graphsimp(graph_train_dataset, gnn_path, diff_trajs_idx,load_model,DEBUG=DEBUG)
            simp,diff=get_model_diffusimp(simp_path,diff_path,load_model)
            diff_trajs_idx = train_diffusimp(ts,self.bbox, simp_trajs_idx,simp,diff,simp_path,diff_path,DEBUG=DEBUG)
            load_model=True
        t2=time.time()
        saveZ_pk(tim_path,t2-t1)
        self.model=self.call_new_model();self.model.load_state_dict(loadZ_th(gnn_path))
        self._diff_trajs_idx=diff_trajs_idx ; saveZ_pk(traj_idx_path,diff_trajs_idx)
        return t2-t1
    def train_before_train(self):
        self.id_dr=id(self.dr);self.id_du=id(self.du);self.id_dv=id(self.dv);self.id_dtrain=id(self.dtrain)
        self.dtrain.update_simp(self._diff_trajs_idx)
        diff_du=[self._diff_trajs_idx[i] for i in self.du_idx]
        diff_dr=[x for i,x in enumerate(self._diff_trajs_idx) if i not in self.du_idx]
        self.du.update_simp(diff_du)
        self.dr.update_simp(diff_dr)
    def forward(self,x):
        trajs_feature, trajs_edge_index, trajs_point_node_index, trajs_seg_node_index, trajs_emb, trajs_neighbor, amply_labels = x
        traj_point_embS=[]
        for i in range(len(trajs_feature)):
            traj_feature = trajs_feature[i]
            traj_edge_index = trajs_edge_index[i]
            traj_point_emb =self.model(traj_feature,traj_edge_index)
            traj_point_embS.append(traj_point_emb)
        return torch.stack(traj_point_embS)
    def lossF(self,y,x): 
        trajs_feature, trajs_edge_index, trajs_point_node_index, trajs_seg_node_index, trajs_emb, trajs_neighbor, amply_labels = x
        align_losses, uniform_losses,mutual_losses=0,0,0 ; simp_trajs=[]
        traj_point_embS=y
        for i in range(len(trajs_feature)):
            traj_point_node_index = trajs_point_node_index[i]
            traj_neighbor = trajs_neighbor[i]
            if amply_labels[i]<0:
                amply_label=None
            else:
                amply_label=amply_labels[i]
            align_loss, uniform_loss, important_simp,mutual_loss = self.model.loss(traj_point_embS[i][traj_point_node_index],traj_neighbor,amply_label)
            align_losses+=align_loss;uniform_losses+=uniform_loss;mutual_losses+=mutual_loss ; simp_trajs.append(important_simp)
        simp_trajs = torch.stack(simp_trajs)
        simp_trajs_norm = torch.norm(simp_trajs, p=2, dim=1).unsqueeze(-1)
        simp_trajs = simp_trajs/simp_trajs_norm
        if isinstance(trajs_emb,list) or isinstance(trajs_emb,tuple): trajs_emb = torch.stack(trajs_emb) 
        trajs_emb_norm = torch.norm(trajs_emb,p=2,dim=1).unsqueeze(-1)
        trajs_emb = trajs_emb/trajs_emb_norm
        batch_losses = F.mse_loss(simp_trajs@simp_trajs.T,trajs_emb@trajs_emb.T)
        align_losses = (align_losses/len(trajs_feature)).mean()
        uniform_losses = (uniform_losses/len(trajs_feature)).mean()
        if mutual_losses!=0:
            mutual_losses = mutual_losses/len(trajs_feature)
        losses = align_losses + 0.3 * uniform_losses + 0.5 * batch_losses + 1 * mutual_losses
        return losses
    def get_task_metrics(self): 
        def infer_simp(gnn:GAT,root_data,name,ts_test,cr=0.0025):
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
                simp_trajs.append(np.array(simp_traj))
            return simp_trajs
        def infer_wq_simp(gnn:GAT,root_data,name,ts_test,cr=0.0025):
            trajs=ts_test
            adjust_tensor,query1,query2=api_pre_test(root_data,name,bbox=self.bbox,ts_test=ts_test,city=self.data_name)
            graph_test_dataset=api_pre_gnndata(root_data,name)
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
                simp_trajs.append(np.array(simp_traj))
            return simp_trajs
        root_data=self.root_data+f'-{self.urv}'
        simp_ts_du=infer_simp(self.model,root_data=root_data, name='du', ts_test=self._du,cr=cr[self.data_name])
        simp_ts_dr=infer_simp(self.model,root_data=root_data, name='dr', ts_test=self._dr,cr=cr[self.data_name])
        simp_ts_dv=infer_simp(self.model,root_data=root_data, name='dv', ts_test=self._dv,cr=cr[self.data_name])
        simpQ_ts_du=infer_wq_simp(self.model,root_data=root_data, name='du', ts_test=self._du,cr=cr[self.data_name])
        simpQ_ts_dr=infer_wq_simp(self.model,root_data=root_data, name='dr', ts_test=self._dr,cr=cr[self.data_name])
        simpQ_ts_dv=infer_wq_simp(self.model,root_data=root_data, name='dv', ts_test=self._dv,cr=cr[self.data_name])
        def val_sed(raw_ts,simp_ts):
            errs=[]
            for t1,t2 in zip(raw_ts,simp_ts):
                err=SED_fast(np.array(t1),np.array(t2))
                errs.append(err)
            res=np.array(errs)
            res=res[~np.isinf(res)]
            assert len(res)
            return res.mean().item()
        def mia_imp(y,x):
            trajs_feature, trajs_edge_index, trajs_point_node_index, trajs_seg_node_index, trajs_emb, trajs_neighbor, amply_labels =x
            traj_point_embS=y
            res=[]
            for i in range(len(trajs_feature)):
                z,neighbor=traj_point_embS[i][trajs_point_node_index[i]],trajs_neighbor[i]
                align_loss = 0
                for i in range(neighbor.size(0)):
                    pos = z[neighbor[i]]
                    align_loss += (z-pos).norm(dim=1).pow(2)
                align_loss = align_loss/neighbor.size(0)
                sim = torch.cdist(z,z).pow(2)
                uniform_loss = sim.mul(-2).exp().mean(dim=1).log()
                important = torch.sigmoid(align_loss* uniform_loss)
                res.append(important.detach().cpu())
            return torch.stack(res)
        def range_f1(du,name):
            return validation(self.model,root_data=root_data, nameQ=name, nameD='dr', tsD=self._dr,tsQ=du, city=self.data_name, bbox=self.bbox,cr=cr[self.data_name])
        def SEDwQ(du,name): 
            root_data=self.root_data+f'-{self.urv}'
            return val_sed_wQ(self.model,root_data=root_data, nameQ=name, nameD='dr',tsD=self._dr,tsQ=du,city=self.data_name,bbox=self.bbox,cr=cr[self.data_name])
        def ITSE(ts_ori,ts_simp):
            res=[]
            for t1,t2 in zip(ts_ori,ts_simp):
                res.append(ITS(t1,t2,len(t1)))
            return np.array(res).mean().item()
        res= {
            'MIA':'','mia_call':mia_imp, 
            'MIA3':MIA3(simp_ts_du,simp_ts_dr,simp_ts_dv),'MIA3Q':MIA3(simpQ_ts_du,simpQ_ts_dr,simpQ_ts_dv),
                'RangeF1Du':range_f1(self._du,'du'),'RangeF1Dr':range_f1(self._dr,'dr'),'RangeF1Dv':range_f1(self._dv,'dv'),
                'SEDwQ_Du':SEDwQ(self._du,'du'),'SEDwQ_Dr':SEDwQ(self._dr,'dr'),'SEDwQ_Dv':SEDwQ(self._dv,'dv'),
                'SED_Du':val_sed(self._du,simp_ts_du),'SED_Dr':val_sed(self._dr,simp_ts_dr),'SED_Dv':val_sed(self._dv,simp_ts_dv),
                'ITS_Du':ITSE(self._du,simp_ts_du),'ITS_Dr':ITSE(self._dr,simp_ts_dr),'ITS_Dv':ITSE(self._dv,simp_ts_dv),
                'ITS_wQ_Du':ITSE(self._du,simpQ_ts_du),'ITS_wQ_Dr':ITSE(self._dr,simpQ_ts_dr),'ITS_wQ_Dv':ITSE(self._dv,simpQ_ts_dv),
                }
        return res
    