"""Public full-token matching for spatially distinct Geometry hypotheses."""
import torch
from torch import nn

class DistinctCandidateLanguageRanker(nn.Module):
 def __init__(self,context_dim=128,text_dim=768,hidden=64):
  super().__init__();self.query=nn.Linear(context_dim,hidden);self.key=nn.Linear(text_dim,hidden);self.value=nn.Linear(text_dim,hidden)
  self.score=nn.Sequential(nn.Linear(context_dim+hidden+3,hidden),nn.GELU(),nn.Linear(hidden,1));nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)
 def forward(self,context,candidate_xy,geometry_logits,tokens,token_mask):
  if not token_mask.bool().any(-1).all():raise ValueError('At least one valid public language token required')
  q=self.query(context);k=self.key(tokens);attention=(q@k.transpose(-1,-2))/q.shape[-1]**.5;attention=attention.masked_fill(~token_mask[:,None].bool(),-1e4).softmax(-1);language=attention@self.value(tokens)
  centered=geometry_logits-geometry_logits[:,:1];x=torch.cat([context,language,candidate_xy,centered[:,:,None]],-1);delta=self.score(x).squeeze(-1)
  return geometry_logits+delta
