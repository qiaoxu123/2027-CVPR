"""Frozen input controls. All transforms use public inputs, never true labels."""
import numpy as np
import torch


class ControlledPublicBatch:
 def __init__(self,base,ids,mode):
  self.base=base;self.data=base.data;self.mode=mode;self.vec=base.vec;self.device=base.device
  eligible=[i for i in ids if self.data['queries'][i]['anchor_valid'].any()]
  order=np.random.default_rng(42).permutation(eligible).tolist()
  self.donor={i:order[(j+1)%len(order)] for j,i in enumerate(order)}
  self.unspecified=self.data['phrases'].index('Unspecified spatial relation.')
 def build(self,ids):
  x,old=self.base.build(ids);mode=self.mode
  if mode=='shape_topology_off':return {**x,'node_shape':torch.zeros_like(x['node_shape']),'edges':torch.zeros_like(x['edges'])},old
  if mode=='angle_projection_off':return {**x,'angle_kernel':torch.full_like(x['angle_kernel'],1/8)},old
  if mode=='target_only':
   e=x['entity_steps'].clone();rel=x['relation_steps'].clone();valid=torch.zeros_like(x['step_valid'])
   for k,i in enumerate(ids):
    q=self.data['queries'][i]
    for j in torch.nonzero(q['anchor_valid']).flatten().tolist():
     count=int(q['step_valid'][j].sum());e[k,j,0]=x['entity_steps'][k,j,max(count-1,0)];rel[k,j,0]=self.vec[self.unspecified].to(self.device);valid[k,j,0]=True
   return {**x,'entity_steps':e,'relation_steps':rel,'step_valid':valid},old
  if mode not in ['language_roles_shuffle','path_sentences_shuffle']:raise ValueError('Unknown public control')
  ent=x['entity_steps'].clone();rel=x['relation_steps'].clone();donor_texts=[]
  for k,i in enumerate(ids):
   original=self.data['queries'][i];donor=self.data['queries'][self.donor.get(i,i)];donor_texts.append(donor['text']);dr=max(int(donor['anchor_valid'].sum()),1)
   for j in torch.nonzero(original['anchor_valid']).flatten().tolist():
    dj=j%dr;count=max(int(donor['step_valid'][dj].sum()),1)
    for t in range(4):
     rel[k,j,t]=self.vec[int(donor['relation_ids'][dj,t%count])].to(self.device)
     if mode=='language_roles_shuffle':ent[k,j,t]=self.vec[int(donor['entity_ids'][dj,t%count])].to(self.device)
  if mode=='language_roles_shuffle':
   pieces=[self.base.tokens[t] for t in donor_texts];length=max(len(v) for v,m in pieces);tokens=torch.zeros(len(ids),length,768);mask=torch.zeros(len(ids),length,dtype=torch.bool)
   for k,(v,m) in enumerate(pieces):tokens[k,:len(v)]=v;mask[k,:len(v)]=m
   x={**x,'tokens':tokens.to(self.device),'token_mask':mask.to(self.device)}
  return {**x,'entity_steps':ent,'relation_steps':rel},old


def public_type_prior(cache,ids,prepared=None):
 """Untrained reference: language target noun type + public angular histogram.

 No north/left->world rule and no target ID selection. Useful to test whether
 learned improvement is more than a class-conditioned map-density prior.
 """
 x,old=prepared if prepared is not None else cache.build(ids);rows=[]
 for k,i in enumerate(ids):
  q=cache.data['queries'][i];j=next((j for j in range(16) if q['anchor_valid'][j]),0);n=int(q['step_valid'][j].sum());phrase=cache.data['phrases'][int(q['entity_ids'][j,max(n-1,0)])].casefold()
  import re
  if re.search(r'\b(car|van|truck|vehicle|bus|suv|sedan)\b',phrase):kind=[7,9]
  elif re.search(r'\b(building|house|home|warehouse|church|roof|hall|complex)\b',phrase):kind=[1]
  elif re.search(r'\b(parking|lot|ground|grass|field|area|garden)\b',phrase):kind=[5,6]
  else:kind=list(range(10))
  mask=x['node_valid'][k].clone()&torch.isin(x['node_types'][k],torch.tensor(kind,device=cache.device))
  if not mask.any():mask=x['node_valid'][k]
  prob=mask.float()/mask.sum();rows.append(torch.einsum('n,rnk->rk',prob,x['angle_kernel'][k].float()))
 p=torch.stack(rows);return p/p.sum(-1,keepdim=True).clamp_min(1e-9)
