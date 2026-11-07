"""Fixed local probability pooling for the first public Geometry candidate."""
from dataclasses import replace
import math
import torch

def decode_first_candidate(field, grid_xy, *, radius=2, temperature=.5):
 q=field.belief_logits.shape[-1];side=math.isqrt(q)
 if side*side!=q or temperature<=0:raise ValueError('Square grid and positive temperature required')
 xy=field.candidate_xy.clone();context=field.candidate_context.clone()
 for b in range(len(xy)):
  top=int(field.candidate_ids[b,0]);row,col=divmod(top,side)
  ids=torch.tensor([r*side+c for r in range(max(0,row-radius),min(side,row+radius+1)) for c in range(max(0,col-radius),min(side,col+radius+1))],device=xy.device)
  w=(field.belief_logits[b,ids].float()/temperature).softmax(-1)
  xy[b,0]=(grid_xy[b,ids].float()*w[:,None]).sum(0)
  if field.dense_context is None:raise ValueError('Dense context required for aligned pooling')
  context[b,0]=(field.dense_context[b,ids].float()*w[:,None]).sum(0).to(context.dtype)
 return replace(field,candidate_xy=xy,candidate_context=context)
