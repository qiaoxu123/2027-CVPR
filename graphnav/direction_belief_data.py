"""Controlled synthetic geometry and label-only target distributions."""
import numpy as np
import torch
UNIT=np.array([[0,1],[0,-1],[1,0],[-1,0],[1,0],[-1,0],[0,1],[0,-1]],np.float32)


def grid_world(span):
 v=(torch.arange(28,device=span.device,dtype=torch.float32)+.5)/28;y,x=torch.meshgrid(v,v,indexing='ij');return torch.stack((x,y),-1)[None]*span[:,None,None]


def point_distribution(points,span,sigma=10.):
 world=grid_world(span);energy=-.5*((world-points[:,None,None])/sigma).square().sum(-1);return energy.flatten(1).softmax(-1).reshape(-1,28,28)


def synthetic(n,seed):
 rng=np.random.default_rng(seed);ids=np.arange(n)%8;rng.shuffle(ids);span=rng.uniform(280,600,(n,2)).astype(np.float32);centers=rng.uniform(.25,.75,(n,2)).astype(np.float32)*span;heading=rng.uniform(-np.pi,np.pi,n).astype(np.float32);units=UNIT[ids].copy()
 relative=ids>=4;c=np.cos(heading[relative]);s=np.sin(heading[relative]);u=units[relative].copy();units[relative,0]=c*u[:,0]-s*u[:,1];units[relative,1]=s*u[:,0]+c*u[:,1]
 perp=np.stack((-units[:,1],units[:,0]),-1);points=centers+units*rng.uniform(30,90,(n,1))+perp*rng.normal(0,8,(n,1));points=np.clip(points,span/56,span-span/56).astype(np.float32)
 span=torch.from_numpy(span);centers=torch.from_numpy(centers);target=torch.from_numpy(points);anchor=point_distribution(centers,span,sigma=6.)
 return dict(anchor=anchor,direction=torch.from_numpy(ids).long(),span=span,heading=torch.from_numpy(heading),target=point_distribution(target,span),target_world=target,anchor_center=centers)
