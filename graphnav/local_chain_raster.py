"""Differentiable evidence projection onto the unchanged original world lattice.

Local patch centers are evidence locations, not camera-following Stop targets.
No private target coordinate or persistent previous-Goal state is accepted.
"""
import math
import torch
from .full_localization import GoalField
from .stage1 import preserved_proposal_order


def rasterized_goal_logprob(joint_logits,local_uv,local_valid,*,global_cells):
    grid=math.isqrt(global_cells)
    if grid*grid!=global_cells or joint_logits.shape!=(len(local_uv),global_cells+local_uv.shape[1]):
        raise ValueError('square global field and matching local points required')
    if local_valid.shape!=local_uv.shape[:2] or local_valid.dtype!=torch.bool:
        raise ValueError('explicit local validity required')
    if not torch.isfinite(local_uv).all():raise ValueError('finite public world coordinates required')
    logits=torch.cat((joint_logits[:,:global_cells],joint_logits[:,global_cells:].masked_fill(~local_valid,float('-inf'))),-1)
    probability=logits.float().softmax(-1)
    result=probability[:,:global_cells].clone()
    xy=(local_uv.float()*grid-.5).clamp(0,grid-1)
    x0=xy[...,0].floor().long();y0=xy[...,1].floor().long()
    x1=(x0+1).clamp(max=grid-1);y1=(y0+1).clamp(max=grid-1)
    wx=xy[...,0]-x0;wy=xy[...,1]-y0
    mass=probability[:,global_cells:]
    for x,y,weight in ((x0,y0,(1-wx)*(1-wy)),(x1,y0,wx*(1-wy)),(x0,y1,(1-wx)*wy),(x1,y1,wx*wy)):
        result=result.scatter_add(1,y*grid+x,mass*weight)
    return result.clamp_min(1e-30).log()


def raster_goal_field(base,global_xy,local_uv,joint_logits,local_valid):
    if not local_valid.any():return base
    logits=rasterized_goal_logprob(joint_logits,local_uv,local_valid,global_cells=base.belief_logits.shape[-1])
    ids=preserved_proposal_order(logits,base.candidate_ids)
    dense=base.dense_context
    return GoalField(logits,ids,global_xy.gather(1,ids[...,None].expand(-1,-1,2)),
                     dense.gather(1,ids[...,None].expand(-1,-1,dense.shape[-1])),dense_context=dense)
