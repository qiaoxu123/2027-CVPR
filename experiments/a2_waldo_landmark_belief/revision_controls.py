"""Frozen TRAIN-only mechanism checks, never validation-set tuning."""
import numpy as np
import torch

CONTROLS=('real','no_text','no_geometry','permuted_links','ambiguous','image_relative','failed_detection')


def controlled_inputs(network,meta,control):
    if control not in CONTROLS:raise ValueError('Unknown diagnostic control')
    result=dict(network)
    if control=='no_text':
        for key in ('instruction_tokens','landmark_name_tokens'):result[key]=torch.zeros_like(network[key])
    if control=='failed_detection':
        for key in ('box_valid','pair_valid'):result[key]=torch.zeros_like(network[key])
    if control=='image_relative':
        footprint=np.asarray(meta['footprint_world'],dtype=np.float64)
        right=footprint[1]-footprint[0];down=footprint[3]-footprint[0]
        if min(np.linalg.norm(right),np.linalg.norm(down))<=1e-8:raise ValueError('Degenerate public camera footprint')
        basis=np.stack((right/np.linalg.norm(right),down/np.linalg.norm(down)))
        if not np.allclose(basis@basis.T,np.eye(2),atol=1e-3):raise ValueError('Nonorthogonal public camera footprint')
        geometry=network['pair_geometry'].clone();rotation=torch.as_tensor(basis.T,dtype=geometry.dtype,device=geometry.device)
        geometry[...,:2]=network['pair_geometry'][...,:2]@rotation
        direction=torch.stack((network['pair_geometry'][...,4],network['pair_geometry'][...,3]),-1)@rotation
        geometry[...,4]=direction[...,0];geometry[...,3]=direction[...,1]
        result['pair_geometry']=geometry
    result['control']=control if control in ('no_text','no_geometry','permuted_links','ambiguous') else 'real'
    return result
