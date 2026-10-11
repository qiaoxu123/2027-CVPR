"""Strict public inference and optional GAT; original grid/controller interface."""
import numpy as np
import torch
from .language import parse,vector
from .reasoning import infer,mix


def forward(graph,instruction,public,method='A2',weight=.25,visual_evidence=None,reference_visual=None):
    required={'static_logits','grid_xy','bounds','pose_world'}
    if set(public)!=required:raise ValueError('P5 public contract forbids labels/target/description keys')
    bounds=np.asarray(public['bounds']);xy=bounds[:2]+np.asarray(public['grid_xy'])*(bounds[2:]-bounds[:2])
    language=parse(instruction,graph)
    result=infer(graph,language,xy,public['static_logits'],relations=method in ('A2','A3','A4'),
                 yaw=float(public['pose_world'][3]),visual_evidence=visual_evidence,reference_visual=reference_visual)
    result.update(language=language,world_grid=xy)
    result['logits']=mix(public['static_logits'],result['heatmap'],weight)
    return result


def model_inputs(graph,result,device='cpu'):
    top=np.argsort(result['static'],kind='stable')[::-1][:8]
    selected=graph.local(result['world_grid'][top],mandatory=result['mandatory'],limit=192)
    if not len(selected):raise ValueError('empty public local object graph')
    reverse=np.full(len(graph.ids),-1,int);reverse[selected]=np.arange(len(selected))
    edge=graph.edge_index;use=(reverse[edge[0]]>=0)&(reverse[edge[1]]>=0)
    p=dict(nodes=torch.tensor(graph.features[selected],device=device),
           edge_index=torch.tensor(reverse[edge[:,use]],device=device,dtype=torch.long),
           edge_attr=torch.tensor(graph.edge_attr[use],device=device),
           language=torch.tensor(vector(result['language']),device=device),
           node_prior=torch.tensor(result['node_unary'][selected],device=device,dtype=torch.float32),
           kernels=torch.tensor(result['kernels'][selected],device=device),
           static=torch.tensor(result['static']*result['relation_field'],device=device,dtype=torch.float32))
    p['static']/=p['static'].sum().clamp_min(1e-12)
    return p,selected


def reconstruct(result,selected,delta):
    """Keep unselected objects and continuous regions; learned delta only local."""
    device=delta.device;score=torch.tensor(result['node_unary'],device=device,dtype=torch.float32)
    score=score.index_add(0,torch.tensor(selected,device=device),delta)
    belief=score.softmax(0);kernel=torch.tensor(result['kernels'],device=device)
    continuous=torch.tensor(result['static']*result['relation_field'],device=device,dtype=torch.float32)
    continuous/=continuous.sum().clamp_min(1e-12)
    heat=.5*(belief@kernel)+.5*continuous;heat/=heat.sum().clamp_min(1e-12)
    return dict(object_belief=belief,heatmap=heat)


def learned(graph,result,model,device='cpu'):
    p,selected=model_inputs(graph,result,device);local=model(p)
    full=reconstruct(result,selected,local['delta'])
    return full,selected


def controller_field(original_field,new_logits,grid_xy):
    """Rebuild aligned candidates/contexts without modifying frozen modules."""
    import copy
    field=copy.copy(original_field)
    logits=torch.as_tensor(new_logits,dtype=original_field.belief_logits.dtype,
                           device=original_field.belief_logits.device)
    if logits.shape!=original_field.belief_logits.shape:raise ValueError('unchanged original grid required')
    if original_field.dense_context is None:raise ValueError('dense context required for new candidates')
    xy=torch.as_tensor(grid_xy,device=logits.device,dtype=original_field.candidate_xy.dtype)
    if xy.ndim==2:xy=xy[None].expand(len(logits),-1,-1)
    if xy.shape[:2]!=logits.shape:raise ValueError('aligned original candidate grid required')
    ids=torch.topk(logits,original_field.candidate_ids.shape[1],dim=1).indices
    field.belief_logits=logits;field.candidate_ids=ids
    field.candidate_xy=xy.gather(1,ids[...,None].expand(-1,-1,2))
    context=original_field.dense_context
    field.candidate_context=context.gather(1,ids[...,None].expand(-1,-1,context.shape[-1]))
    return field


def observed_forward(graph,instruction,public,memory,*,weight,cutoff_step,control='real'):
    """Attribute updates apply to target and reference hypotheses, observed only.

    A4 uses identical A2 graph/relations and an observed RGB memory. No GT keys.
    Black/Shuffled controls must already have been applied during observe().
    Deployment requires a Dev-supported acceptance calibration; no safe coverage
    is claimed from pixel rules alone.
    """
    if control not in ('real','black','shuffled'):raise ValueError('RGB control')
    language=parse(instruction,graph)
    for value in memory.nodes.values():
        if value['step']>cutoff_step:
            raise ValueError('future memory')
        if value['control']!=control:raise ValueError('control/memory mismatch')
    evidence={i:memory.evidence(entity,len(graph.ids)) for i,entity in enumerate(language['entities'])}
    return forward(graph,instruction,public,method='A4',weight=weight,
                   visual_evidence=evidence[0],reference_visual={i:v for i,v in evidence.items() if i})

def trajectory_step(policy,state,original_field,new_logits,grid_xy):
    """Use unchanged existing frozen policy on a fully aligned new GoalField.

    The experiment orchestrator must enforce localization gates before calling
    this integration hook. This function does not change Stop, action budget,
    success radius, or the controller implementation.
    """
    return policy.predict_field(state,controller_field(original_field,new_logits,grid_xy))
