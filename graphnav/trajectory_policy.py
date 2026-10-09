"""Candidate-aligned multimodal waypoint and calibrated Stop head candidates.

Model inputs are belief/context, current pose and public map size. Human
remaining paths and destination labels are accepted only by the loss function.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def remaining_waypoints(trajectory, prefix_index, *, count=8, spacing_m=50.):
    """Interpolate actual remaining human path at bounded macro-action distances.

    Source z is interpolated with XY arclength; no destination-derived path is
    synthesized. A short remaining path repeats its actual observed endpoint.
    """
    path=np.asarray(trajectory,dtype=np.float64)
    if path.ndim!=2 or path.shape[1]<3 or not np.isfinite(path).all():
        raise ValueError('finite human xyz trajectory required')
    if not 0<=prefix_index<len(path) or count<1 or spacing_m<=0:
        raise ValueError('invalid waypoint supervision settings')
    tail=path[prefix_index:,:3]
    lengths=np.linalg.norm(np.diff(tail[:,:2],axis=0),axis=-1)
    cumulative=np.r_[0.,np.cumsum(lengths)]
    # Keep the final source pose at repeated arclength values (e.g. vertical motion).
    keep=np.r_[np.diff(cumulative)>1e-8,True]
    cumulative,tail=cumulative[keep],tail[keep]
    distances=np.minimum(np.arange(1,count+1)*spacing_m,cumulative[-1])
    return np.stack([np.interp(distances,cumulative,tail[:,axis]) for axis in range(3)],-1).astype(np.float32)


@dataclass
class TrajectoryOutput:
    paths_world: torch.Tensor
    mode_logits: torch.Tensor
    candidate_logits: torch.Tensor
    stop_logits: torch.Tensor

    def select(self):
        # Mode scores are conditional on a candidate, not an arbitrary rank offset.
        scores=self.candidate_logits[...,None]+self.mode_logits.log_softmax(-1)
        ids=scores.flatten(1).argmax(1)
        candidate=ids//scores.shape[-1];mode=ids%scores.shape[-1]
        batch=torch.arange(len(ids),device=ids.device)
        return candidate,mode,self.paths_world[batch,candidate,mode,0],self.stop_logits[batch,candidate]


class CandidateTrajectoryStop(nn.Module):
    def __init__(self,hidden=128,modes=3,waypoints=8,max_step_m=50.,max_vertical_m=10.):
        super().__init__()
        self.modes,self.waypoints=modes,waypoints
        self.max_step_m,self.max_vertical_m=max_step_m,max_vertical_m
        self.state=nn.Sequential(nn.Linear(10,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.fusion=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.path=nn.Linear(hidden,modes*waypoints*3)
        self.mode=nn.Linear(hidden,modes)
        self.rank=nn.Linear(hidden,1)
        self.stop=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.path.weight);nn.init.zeros_(self.path.bias)
        nn.init.zeros_(self.rank.weight);nn.init.zeros_(self.rank.bias)
        nn.init.zeros_(self.mode.weight);nn.init.zeros_(self.mode.bias)

    def forward(self,*,candidate_xy,candidate_context,candidate_logits,pose_world,
                bounds,ground_level,belief_entropy,observed_fraction):
        """All inputs observable: metric xyz+yaw pose; XY bounds and known ground."""
        b,k,_=candidate_xy.shape
        span=bounds[:,2:]-bounds[:,:2]
        goals=bounds[:,:2,None].transpose(1,2)+candidate_xy*span[:,None]
        delta=goals-pose_world[:,None,:2]
        dist=delta.norm(dim=-1,keepdim=True)
        features=torch.cat((delta/100.,dist/100.,
            torch.sin(pose_world[:,3])[:,None,None].expand(-1,k,1),
            torch.cos(pose_world[:,3])[:,None,None].expand(-1,k,1),
            ((pose_world[:,2]-ground_level)/100.)[:,None,None].expand(-1,k,1),
            candidate_logits.softmax(-1)[...,None],
            belief_entropy[:,None,None].expand(-1,k,1),
            observed_fraction[:,None,None].expand(-1,k,1),
            (dist<=20.).to(delta.dtype)),dim=-1)
        hidden=self.fusion(torch.cat((candidate_context,self.state(features)),dim=-1))
        residual=torch.tanh(self.path(hidden).view(b,k,self.modes,self.waypoints,3))
        step=torch.arange(1,self.waypoints+1,device=delta.device,dtype=delta.dtype)
        direction=delta/dist.clamp_min(1e-6)
        distance=torch.minimum(dist[...,None],step[None,None,:,None]*self.max_step_m)
        prior_xy=pose_world[:,None,None,:2]+direction[...,None,:]*distance
        # Straight / left / right initial hypotheses break winner-take-all symmetry.
        # Lateral offsets vanish at the candidate, so no label-derived path is used.
        offsets=torch.tensor([0.]+[(-1. if i%2 else 1.) for i in range(1,self.modes)],device=delta.device,dtype=delta.dtype)
        perpendicular=torch.stack((-direction[...,1],direction[...,0]),-1)
        progress=(distance.squeeze(-1)/dist.clamp_min(1e-6)).clamp(0,1)
        lateral=perpendicular[:,:,None,None,:]*torch.sin(progress[:,:,None,:,None]*torch.pi)*offsets[None,None,:,None,None]*20.
        xy=prior_xy[:,:,None]+lateral+residual[...,:2]*20.
        xy=torch.maximum(torch.minimum(xy,bounds[:,None,None,None,2:]),bounds[:,None,None,None,:2])
        # Ensure each predicted segment is executable under the shared movement limits.
        previous=pose_world[:,None,None,:2].expand(-1,k,self.modes,-1)
        bounded=[]
        for j in range(self.waypoints):
            move=xy[:,:,:,j]-previous
            move=move*(self.max_step_m/move.norm(dim=-1,keepdim=True).clamp_min(self.max_step_m))
            previous=previous+move;bounded.append(previous)
        xy=torch.stack(bounded,dim=3)
        z=pose_world[:,None,None,None,2]+residual[...,2]*self.max_vertical_m
        z=torch.maximum(torch.minimum(z,ground_level[:,None,None,None]+100),ground_level[:,None,None,None]+10)
        previous_z=pose_world[:,None,None,2].expand(-1,k,self.modes)
        bounded_z=[]
        for j in range(self.waypoints):
            previous_z=previous_z+(z[:,:,:,j]-previous_z).clamp(-self.max_vertical_m,self.max_vertical_m)
            bounded_z.append(previous_z)
        z=torch.stack(bounded_z,dim=3)
        paths=torch.cat((xy,z[...,None]),-1)
        return TrajectoryOutput(paths,self.mode(hidden),candidate_logits+self.rank(hidden).squeeze(-1),
                                self.stop(hidden).squeeze(-1))


def trajectory_losses(out,*,candidate_world_xy,pose_world,target_world,teacher_world,radius=20.,valid_teacher_z=None,stop_pos_weight=None):
    """Destination and teacher future are label-only arguments."""
    distances=(candidate_world_xy-target_world[:,None,:2]).norm(dim=-1)
    weights=torch.softmax(-distances.square()/(2*20.**2),dim=-1)
    ranking=-(weights*F.log_softmax(out.candidate_logits,-1)).sum(-1).mean()
    nearest=distances.argmin(-1);batch=torch.arange(len(nearest),device=nearest.device)
    paths=out.paths_world[batch,nearest]
    errors=F.smooth_l1_loss(paths/50.,teacher_world[:,None].expand_as(paths)/50.,reduction='none')
    weights_xyz=torch.ones_like(teacher_world)
    if valid_teacher_z is not None:
        weights_xyz[...,2]=valid_teacher_z.to(errors.dtype)
    per_mode=(errors*weights_xyz[:,None]).sum((-1,-2))/weights_xyz.sum((-1,-2)).clamp_min(1)[:,None]
    best=per_mode.detach().argmin(-1)
    path=per_mode[batch,best].mean()
    mode=F.cross_entropy(out.mode_logits[batch,nearest],best)
    arrived=(pose_world[:,:2]-target_world[:,:2]).norm(dim=-1)<=radius
    # Close to a wrong candidate is a hard Stop negative even at an actual arrival pose.
    stop_labels=(arrived[:,None] & (distances<=radius)).to(out.stop_logits.dtype)
    stop=F.binary_cross_entropy_with_logits(out.stop_logits,stop_labels,pos_weight=stop_pos_weight)
    return dict(total=ranking+path+.1*mode+.2*stop,ranking=ranking,path=path,mode=mode,stop=stop)
