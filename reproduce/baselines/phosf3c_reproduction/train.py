from __future__ import annotations
import json,os,platform,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score,balanced_accuracy_score,brier_score_loss,confusion_matrix,matthews_corrcoef,precision_recall_fscore_support,roc_auc_score
from torch.utils.data import DataLoader
from .dataset import ShardedDataset
from .model import PhosF3C

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def _optimize_runtime():
 torch.set_float32_matmul_precision("high")
 torch.backends.cuda.matmul.allow_tf32=True
 torch.backends.cudnn.allow_tf32=True
 torch.backends.cudnn.benchmark=True
 try:torch.backends.cuda.enable_flash_sdp(True)
 except Exception:pass
 try:torch.backends.cuda.enable_mem_efficient_sdp(True)
 except Exception:pass
 try:torch.set_num_threads(min(4,os.cpu_count() or 4))
 except Exception:pass
def predict(model,loader,device,amp):
 model.eval();ys=[];ps=[];ids=[]
 with torch.inference_mode():
  for x,y,sample in loader:
   with torch.autocast("cuda",torch.float16,enabled=amp):logits=model(x.to(device,non_blocking=True))
   ys.extend(y.numpy());ps.extend(torch.softmax(logits.float(),1)[:,1].cpu().numpy());ids.extend(sample)
 return np.asarray(ys),np.asarray(ps),ids
def metrics(y,p):
 z=p>=.5;tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel();precision,recall,f1,_=precision_recall_fscore_support(y,z,average="binary",zero_division=0);return {"acc":float((z==y).mean()),"bacc":balanced_accuracy_score(y,z),"precision":precision,"recall":recall,"f1":f1,"sn":tp/max(1,tp+fn),"sp":tn/max(1,tn+fp),"mcc":matthews_corrcoef(y,z),"roc_auc":roc_auc_score(y,p),"pr_auc":average_precision_score(y,p),"brier":brier_score_loss(y,p),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
def train_one(task,split,embeddings,seed,config,output,gpu):
 seed_all(seed);_optimize_runtime();device=torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu");amp=bool(config.get("amp",True)) and device.type=="cuda";sets=[ShardedDataset(split,embeddings,n) for n in ("train","val","test")];workers=max(1,int(config.get("num_workers",4)))
 kwargs={"batch_size":int(config["batch_size"]),"num_workers":workers,"pin_memory":device.type=="cuda","persistent_workers":workers>0}
 if workers>0:kwargs["prefetch_factor"]=max(2,int(config.get("prefetch_factor",4)))
 loaders=[DataLoader(x,shuffle=i==0,**kwargs) for i,x in enumerate(sets)];model=PhosF3C(alpha=float(config["alpha"]),**config.get("model_kwargs",{})).to(device)
 compile_mode=config.get("compile_mode","reduce-overhead")
 if compile_mode!="none" and hasattr(torch,"compile") and device.type=="cuda":
  try:model=torch.compile(model,mode=compile_mode)
  except Exception:pass
 criterion=torch.nn.CrossEntropyLoss();optimizer=torch.optim.Adam(model.parameters(),lr=float(config["learning_rate"]),betas=tuple(config.get("betas",[.9,.999])),weight_decay=float(config.get("weight_decay",1e-4)));scaler=torch.amp.GradScaler("cuda",enabled=amp);best=-np.inf;state=None;stale=0;history=[];started=time.perf_counter();tag=f"{task}_seed{seed}"
 for folder in ("checkpoints","predictions","run_results","history","logs"):(output/folder).mkdir(parents=True,exist_ok=True)
 for epoch in range(1,int(config["max_epochs"])+1):
  model.train();loss_sum=0.
  accumulation_steps=max(1,int(config.get("gradient_accumulation_steps",1)))
  optimizer.zero_grad(set_to_none=True)
  for batch_index,(x,y,_) in enumerate(loaders[0]):
   x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
   with torch.autocast("cuda",torch.float16,enabled=amp):loss=criterion(model(x),y)/accumulation_steps
   scaler.scale(loss).backward()
   if (batch_index+1)%accumulation_steps==0 or batch_index+1==len(loaders[0]):
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
   loss_sum+=float(loss)*accumulation_steps*len(y)
  vy,vp,_=predict(model,loaders[1],device,amp);score=roc_auc_score(vy,vp)
  if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch;stale=0
  else:stale+=1
  vm=metrics(vy,vp);row={"epoch":epoch,"train_loss":loss_sum/len(sets[0]),"val_roc_auc":score,"val_pr_auc":vm["pr_auc"],"best_val_roc_auc":best,"stale":stale,"elapsed_sec":time.perf_counter()-started};history.append(row)
  if stale>=int(config["patience"]):break
 train_time=time.perf_counter()-started;model.load_state_dict(state);started=time.perf_counter();y,p,ids=predict(model,loaders[2],device,amp);infer=time.perf_counter()-started;prediction=pd.read_csv(Path(split)/"test.csv");prediction["y_true"]=y;prediction["y_prob"]=p;prediction["y_pred"]=(p>=.5).astype(int);prediction.to_csv(output/"predictions"/f"{tag}.csv",index=False);pd.DataFrame(history).to_csv(output/"history"/f"{tag}.csv",index=False);(output/"logs"/f"{tag}.txt").write_text(f"task={task}\nmodel=phosf3c\nseed={seed}\ntrain_time_sec={train_time}\ninference_time_sec={infer}\n",encoding="utf-8");checkpoint=output/"checkpoints"/f"{tag}.pt";torch.save({"model_state_dict":model.state_dict(),"task":task,"model":"phosf3c","seed":seed},checkpoint);result={"task":task,"task_type":"PTM","model":"PhosF3C","seed":seed,"split_seed":config["split_seed"],"best_epoch":best_epoch,"epochs_trained":len(history),"best_validation_roc_auc":best,"n_train":len(sets[0]),"n_val":len(sets[1]),"n_test":len(sets[2]),"train_time_sec":train_time,"inference_time_sec":infer,"throughput_seq_per_sec":len(y)/max(infer,1e-12),"total_params":sum(x.numel() for x in model.parameters()),"trainable_params":sum(x.numel() for x in model.parameters() if x.requires_grad),"gpu_id":gpu,"gpu_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU","torch_version":torch.__version__,"python_version":platform.python_version(),"checkpoint_path":str(checkpoint),**metrics(y,p)};(output/"run_results"/f"{tag}.json").write_text(json.dumps(result,indent=2));return result
