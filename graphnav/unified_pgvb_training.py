"""Visual-only training utilities for the actual integrated UnifiedPGVB-v2 model."""
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .full_experiment import new_model
from .unified_pgvb import UnifiedPGVB


def load_unified(payload,device='cuda'):
 if payload.get('format')!='unified-pgvb-v2':raise ValueError('Integrated V2 checkpoint required')
 geometry=new_model(SimpleNamespace(**payload['config']['geometry_args']),torch.device(device))
 del geometry.vision_proj;del geometry.visual_score
 model=UnifiedPGVB(geometry=geometry).to(device);model.load_state_dict(payload['model'],strict=True)
 model.joint_ready=False;model.direction_ready=False;model.target_ready=False
 return model

def configure_visual_training(model):
 model.train().requires_grad_(False);model.pgvb.train().requires_grad_(True)
 if model.geometry is not None:model.geometry.eval()
 model.joint_ready=False;model.direction_ready=False;model.target_ready=False
 return [name for name,p in model.named_parameters() if p.requires_grad]

def frame_order(ids,seed,epoch):return np.random.default_rng(seed+epoch).permutation(sorted(ids)).tolist()

def weak_landmark_loss(scores,labels,frame_ids,type_ids):
 scores=scores.float();pos=labels==1;neg=labels==0;known=pos|neg;count=pos.sum(-1);nc=neg.sum(-1)
 if not (count>0).all():raise ValueError('Reliable positive region required; do not fabricate visibility')
 plse=torch.logsumexp(scores.masked_fill(~pos,float('-inf')),-1);kl=plse-(scores*pos).sum(-1)/count-count.float().log();region=torch.where(nc>0,torch.logsumexp(scores.masked_fill(~known,float('-inf')),-1)-plse,kl).mean()
 bce=((F.softplus(-scores)*pos).sum(-1)/count+(F.softplus(scores)*neg).sum(-1)/nc.clamp_min(1)).mean()
 compatible=(pos.float()@(~neg).float().T)==0;eligible=compatible&compatible.T&(frame_ids[:,None]==frame_ids[None,:])&(type_ids[:,None]==type_ids[None,:]);eligible.fill_diagonal_(False)
 cross=scores@pos.float().T/count.float()[None];own=(scores*pos).sum(-1)/count;penalty=F.softplus(1+cross-own[:,None]);pair=(penalty*eligible).sum()/eligible.sum().clamp_min(1)
 return dict(total=region+.2*bce+.3*pair,region=region,bce=bce,pair=pair,pairs=eligible.sum())

class CachedLandmarkFrames(Dataset):
 def __init__(self,data,subset):
  self.data=data;self.by_frame={}
  for i,r in enumerate(data['records']):
   if r['subset']==subset:self.by_frame.setdefault(r['frame_id'],[]).append(i)
  self.frame_ids=sorted(self.by_frame)
 def __len__(self):return len(self.frame_ids)
 def __getitem__(self,fid):
  rows=[self.data['records'][i] for i in self.by_frame[fid]];queries=[self.data['queries'][r['query_key']] for r in rows]
  return self.data['frames'][fid],rows,queries

def collate_cached_frames(items):
 frames=[x[0] for x in items];rows=[r for x in items for r in x[1]];queries=[q for x in items for q in x[2]];take=[j for j,x in enumerate(items) for _ in x[1]]
 return dict(unique_images=torch.stack([f['image_features'] for f in frames]),unique_valid=torch.stack([f['valid'] for f in frames]),take=torch.tensor(take),name_text=torch.stack([q['name_text'] for q in queries]),type_text=torch.stack([q['type_text'] for q in queries]),polygon_tokens=torch.stack([r['polygon_tokens'] for r in rows]),labels=torch.stack([r['labels'] for r in rows]),frame_ids=torch.tensor([r['frame_id'] for r in rows]),type_ids=torch.tensor([0 if r['type']=='Building' else 1 for r in rows]),row_indices=torch.tensor(take))

def move_batch(batch,device):
 b={k:v.to(device,non_blocking=True) for k,v in batch.items()};take=b['take'];inputs=dict(image_features=b['unique_images'][take],valid=b['unique_valid'][take],name_text=b['name_text'],type_text=b['type_text'],polygon_tokens=b['polygon_tokens'])
 return inputs,b['labels'],b['frame_ids'],b['type_ids']
