exec("import sys \nif __name__=='__main__':  sys.path.extend(['./'])")
import os
from traj.models.one_blue import *
from mu.traj.loaddata import datasets,t_add_noise,load_traj_raw,ts_split,fix_trajs_num
from mu.MU import TaskModel,DataInitFinishExp
from traj.data.load_trajs import traj_bbox
from traj.data.process_ts import t_len
from _tool.mIO import loadZ_pk,saveZ_pk,load_pk
from _nn.nBasic import auto_device,to_device
from _tool.mData import ids_shrink
from _tool.mFile import is_mac
device=auto_device()
min_traj_per_usr=200
max_len_traj=1000
def _data_pre(data,root):
    path=os.path.join(root,f'ts-u{min_traj_per_usr}.pk.zst')
    if os.path.exists(path):
        TS,UIDS=loadZ_pk(path)
        return TS,UIDS
    ts,uids,*other=load_traj_raw(data)
    TS=[];UIDS=[]
    for i,(t,uid) in enumerate(zip(ts,uids)):
        if t_len(t)<1e-2: ts[i]=None
        else:
            TS.append(np.array(t))
            UIDS.append(int(uid))
    print('num traj, |T|>1k:',len(ts),len(TS))
    TS,UIDS=np.array(TS,dtype=object),np.array(UIDS,dtype=int)
    usr_count=np.zeros(max(UIDS)+1,dtype=int)
    for u in UIDS:
        usr_count[u]+=1
    ts,uid=[],[]
    for t,u in zip(TS,UIDS):
        if usr_count[u]<min_traj_per_usr:continue
        ts.append(t)
        uid.append(u)
    if len(ts)>10000:
        del TS,UIDS
        TS,UIDS=np.array(ts,dtype=object),np.array(uid,dtype=int)
    UIDS=ids_shrink(UIDS)
    print(f'num usr: {len(set(UIDS))}, num traj: {len(TS)}')
    saveZ_pk(path,[TS,UIDS])
    return TS,UIDS
def _data_init(dataname,dataroot,urv,durate,rt_if_exist): 
    """dr,du,dv,dtrain,dtest,dval,du_noise,dr_noise,dv_noise,num_cls"""
    path=os.path.join(dataroot,f'ts-u{min_traj_per_usr}-{urv}.pk.zst')
    if os.path.exists(path):
        if rt_if_exist:return True
        dr,du,dv,dtrain,dtest,dval,du_noise,dr_noise,dv_noise,num_cls=loadZ_pk(path)
        return dr,du,dv,dtrain,dtest,dval,du_noise,dr_noise,dv_noise,num_cls+1
    TS,UIDS=_data_pre(dataname,dataroot.replace(str(durate),'').replace('init',''))
    ts=TS ; lb=UIDS
    train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx=ts_split(ts,UIDS,urv,durate)
    train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx=fix_trajs_num({k:v for k,v in zip('train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx'.split(','),[train_idx,val_idx,test_idx,du_idx,dr_idx,dv_idx])},durate).values()
    num_cls=max(lb[train_idx])
    dr=preprocess(ts[dr_idx],lb[dr_idx])
    dv=preprocess(ts[dv_idx],lb[dv_idx])
    du=preprocess(ts[du_idx],lb[du_idx])
    dtrain=preprocess(ts[train_idx],lb[train_idx])
    dtest=preprocess(ts[test_idx],lb[test_idx])
    dval=preprocess(ts[val_idx],lb[val_idx])
    du_noise=preprocess( np.array([t_add_noise(t) for t in  ts[du_idx]],dtype=object),lb[du_idx]) 
    dr_noise=preprocess( np.array([t_add_noise(t) for t in  ts[dr_idx]],dtype=object),lb[dr_idx])
    dv_noise=preprocess( np.array([t_add_noise(t) for t in  ts[dv_idx]],dtype=object),lb[dv_idx])
    saveZ_pk(path,[dr,du,dv,dtrain,dtest,dval,du_noise,dr_noise,dv_noise,num_cls])
    print('build data over, please re-run');raise DataInitFinishExp()
class TrajSim(TaskModel): 
    def __init__(self, bs=256,**kw):
        super().__init__(**kw)
        bbox=traj_bbox[self.data_name]
        self.mbr={'min_lon':bbox[0][0],'max_lon':bbox[0][1],'min_lat':bbox[1][0],'max_lat':bbox[1][1]}
        self.dv_noise=self.du_noise=self.dr_noise=None 
    def get_collate_fn(self): 
        return lambda x: collate_fn_pretrain(x,self.mbr)
    def get_collate_fn_test(self): 
        return lambda x: collate_fn_inference(x,self.mbr)
    def _call_new_model(self):
        self.model=Net().to(self.device)
        return self.model
    def _call_new_opt(self,**kw):
        return get_optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))
    def _call_new_sch(self,opt):
        return get_scheduler(opt)
    def forward(self,data): 
        """x_s5, x_s3, x_s2"""
        x = data['x']
        time_x = data['time_x']
        traj_len_s5 = data['traj_len_s5']
        patch_len_s2 = data['patch_len_s2']
        traj_len_s2 = data['traj_len_s2']
        patch_len_s3 = data['patch_len_s3']
        traj_len_s3 = data['traj_len_s3']
        x = self.model.spa_emb(x)
        tx = self.model.time_emb(time_x)
        x = x + tx
        x_s5, x_s3, x_s2 = self.model.encoder(x, traj_len_s5, traj_len_s3, patch_len_s3, traj_len_s2, patch_len_s2)
        return x_s5, x_s3, x_s2
    def lossF(self,y,data): 
        mask_y = data['mask_y']
        traj_len_s5 = data['traj_len_s5']
        traj_len_s2 = data['traj_len_s2']
        traj_len_s3 = data['traj_len_s3']
        x_s5, x_s3, x_s2=y
        x_s5_out = self.model.decoder(x_s5, x_s3, x_s2, traj_len_s5, traj_len_s3, traj_len_s2)
        x_s5_out = x_s5_out[:, 1:]
        mask_y = mask_y.reshape(-1).bool()
        x_s5_out = x_s5_out.reshape(-1, x_s5_out.shape[-1])
        x_s5_out = x_s5_out[mask_y]
        spa_y = data['x']
        spa_y = spa_y.reshape(-1, spa_y.shape[-1])
        spa_y = spa_y[mask_y]
        spa_y_hat = self.model.predictor_spa(x_s5_out)
        spa_loss = F.mse_loss(spa_y_hat, spa_y)
        time_y = data['time_x']
        time_y = time_y.reshape(-1, time_y.shape[-1])
        time_y = time_y[mask_y]
        time_y_hat = self.model.predictor_time(x_s5_out)
        time_loss = F.mse_loss(time_y_hat, time_y)
        return spa_loss+time_loss
    def train_after_batch(self)->bool:
        """return: if exit train"""
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if torch.mps.is_available(): torch.mps.empty_cache()
    def get_task_metrics(self):
        self.model.eval()
        @torch.no_grad()
        def inference_fn(model, data_loader):
            embeddings = []
            for batch_data in data_loader:
                batch_data=to_device(batch_data)
                x_s5, x_s3, x_s2 = model(batch_data)
                emb=x_s2[:, 0]
                embeddings.append(emb.detach().cpu().numpy())
            return np.vstack(embeddings)
        def hit_k(truth:np.ndarray,pred:np.ndarray):
            return np.sum(truth==pred,axis=1).mean().item()
        emb_du=inference_fn(model=self, data_loader=DataLoader(self.du,self.bs,False,pin_memory=True,collate_fn=self.get_collate_fn_test()))
        emb_du2=inference_fn(model=self, data_loader=DataLoader(self.du_noise,self.bs,False,pin_memory=True,collate_fn=self.get_collate_fn_test()))
        emb_dr=inference_fn(model=self, data_loader=DataLoader(self.dr,self.bs,False,pin_memory=True,collate_fn=self.get_collate_fn_test()))
        emb_dr2=inference_fn(model=self, data_loader=DataLoader(self.dr_noise,self.bs,False,pin_memory=True,collate_fn=self.get_collate_fn_test()))
        emb_dv=inference_fn(model=self, data_loader=DataLoader(self.dv,self.bs,False,pin_memory=True,collate_fn=self.get_collate_fn_test()))
        emb_dv2=inference_fn(model=self, data_loader=DataLoader(self.dv_noise,self.bs,False,pin_memory=True,collate_fn=self.get_collate_fn_test()))
        def mr_hr(query_emb,full_databse_emb):
            dists = query_emb @ full_databse_emb.T
            mr_res = mr(dists).item()
            scores = np.argsort(dists, axis=-1)[:, ::-1][:, :10]
            truth=np.arange(len(query_emb)).reshape(-1,1)
            hr_res={k:hit_k(truth,scores[:,:k]) for k in [1,5,10]}
            return {'MR':mr_res,'HR':hr_res}
        def mia_call(y,x):
            x_s5, x_s3, x_s2=y
            return x_s2[:, 0] 
        res_du=mr_hr(emb_du2,np.concat([emb_du,emb_dr]))
        res_dv=mr_hr(emb_dv2,np.concat([emb_dv,emb_dr]))
        res_dr=mr_hr(emb_dr2,np.concat([emb_dr,emb_dv]))
        return {'MIA':'','mia_call':mia_call,
            'MR_Du':res_du['MR'],'MR_Dr':res_dr['MR'],'MR_Dv':res_dv['MR'],
            'HR1_Du':res_du['HR'][1],'HR1_Dr':res_dr['HR'][1],'HR1_Dv':res_dv['HR'][1],
            'HR5_Du':res_du['HR'][5],'HR5_Dr':res_dr['HR'][5],'HR5_Dv':res_dv['HR'][5],
            'HR10_Du':res_du['HR'][10],'HR10_Dr':res_dr['HR'][10],'HR10_Dv':res_dv['HR'][10],
        }
    def data_init(self,uvt,rt_if_exist=False):
        """dr,du,dv,dtrain,dtest,dval,dr_raw,du_raw"""
        x=_data_init(self.data_name,self.root_data,uvt,self.du_rate,rt_if_exist)
        if rt_if_exist:return x
        dr,du,dv,dtrain,dtest,dval,self.du_noise,self.dr_noise,self.dv_noise,_=x
        dr_raw=[t['gps_seq'] for t in dr]
        du_raw=[t['gps_seq'] for t in du]
        return {'dr':dr,"du":du,'dv':dv,'dtrain':dtrain,'dtest':dtest,'dval':dval,'dr_raw':dr_raw,'du_raw':du_raw}
