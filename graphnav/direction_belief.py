"""Minimal learned word-conditioned Gaussian displacement kernel, no target inputs.

A3 world-grid rows increase NORTH and columns EAST. Relative terms require an
explicit reference heading in radians from world +X toward +Y; UAV yaw alone
is not an instruction reference frame. The 32 learned parameters are offsets
and uncertainties for eight discrete words, not a full language encoder.
"""
import torch
from torch import nn
from torch.nn import functional as F
WORDS=('north','south','east','west','front','behind','left','right')


def rotate_means(means,ids,heading):
 c,s=heading.cos(),heading.sin();rot=torch.stack((c*means[:,0]-s*means[:,1],s*means[:,0]+c*means[:,1]),-1)
 return torch.where((ids>=4)[:,None],rot,means)


def apply_offset_kernel(anchor,kernel):
 if anchor.ndim!=3 or kernel.ndim!=3 or anchor.shape[0]!=kernel.shape[0]:raise ValueError('Batched Anchor and offset kernel required')
 n=anchor.shape[-1]
 if anchor.shape[-2]!=n or kernel.shape[-2:]!=(2*n-1,2*n-1):raise ValueError('Matching square world grid required')
 # conv2d is cross-correlation: flip target-minus-anchor offsets exactly once.
 p=F.conv2d(anchor[None].float(),kernel[:,None].flip(-1,-2).float(),padding=n-1,groups=len(anchor))[0].clamp_min(0)
 return p/p.sum((-1,-2),keepdim=True).clamp_min(1e-20)


class DirectionBelief(nn.Module):
 def __init__(self,ignore_direction=False):
  super().__init__();self.parameters_by_word=nn.Embedding(8,4);nn.init.zeros_(self.parameters_by_word.weight);self.ignore_direction=bool(ignore_direction)

 def forward(self,anchor,direction,span,reference_heading=None):
  if anchor.shape[1:]!=(28,28) or span.shape!=(len(anchor),2) or direction.shape!=(len(anchor),):raise ValueError('28-grid, public meter span and one direction ID required')
  if not torch.isfinite(anchor).all() or not torch.isfinite(span).all() or (span<=0).any() or (anchor<0).any() or (anchor.sum((-1,-2))<=0).any():raise ValueError('Finite positive Anchor probabilities and map span required')
  if (direction<0).any() or (direction>=8).any():raise ValueError('Known discrete word ID required')
  if not self.ignore_direction and (direction>=4).any():
   if reference_heading is None or not torch.isfinite(reference_heading[direction>=4]).all():raise ValueError('Relative direction has unknown reference heading; abstain')
  ids=torch.zeros_like(direction) if self.ignore_direction else direction
  params=self.parameters_by_word(ids);mu=100*params[:,:2];sigma=(30*params[:,2:].exp()).clamp(3,400)
  axis=torch.arange(-27,28,device=anchor.device,dtype=torch.float32);gy,gx=torch.meshgrid(axis,axis,indexing='ij');dx=gx[None]*span[:,0,None,None]/28;dy=gy[None]*span[:,1,None,None]/28
  if (ids>=4).any():
   angle=torch.nan_to_num(reference_heading);c=angle.cos()[:,None,None];s=angle.sin()[:,None,None];x=c*dx+s*dy;y=-s*dx+c*dy;dx=torch.where((ids>=4)[:,None,None],x,dx);dy=torch.where((ids>=4)[:,None,None],y,dy)
  energy=-.5*((dx-mu[:,0,None,None])/sigma[:,0,None,None]).square()-.5*((dy-mu[:,1,None,None])/sigma[:,1,None,None]).square()
  kernel=energy.flatten(1).softmax(-1).reshape(-1,55,55)+1e-10
  return apply_offset_kernel(anchor/anchor.sum((-1,-2),keepdim=True),kernel)
