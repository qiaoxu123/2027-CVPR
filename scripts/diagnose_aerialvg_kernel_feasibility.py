"""Posterior-mixture Oracle feasibility on cached Train/internalDev states only.

Necessary LP constraints ignore lower-prior competing observed wrong cells.
Feasible solutions are therefore accepted only after actual frozen update verifies
success. Numerical infeasibility is evidence, not a formal exact-arithmetic proof.
"""
import argparse,collections,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.train_object_query_belief import load

EPS=1e-9


def necessary_constraints(base,prior,support,observed,legal,u,gain,bound=2.):
    base=np.asarray(base,float);prior=np.maximum(np.asarray(prior,float),EPS);support=np.asarray(support,float);observed=np.asarray(observed,bool);legal=np.asarray(legal,bool);k=support.shape[1];A=[];b=[]
    def lower(vector,threshold):
        # Actual density is floored at EPS before logarithms.
        if threshold>EPS:A.append(-vector);b.append(-threshold)
    def upper(vector,threshold):
        if threshold<EPS*(1-1e-8):return False
        A.append(vector);b.append(threshold);return True
    for v in np.flatnonzero(~legal):
        gap=(base[v]-base[u])/gain
        if not observed[u] and not observed[v]:
            if gap>1e-8:return None,'unobserved_prior_floor'
        elif not observed[u]:
            limit=-gap
            if limit < -bound-1e-8:return None,'signed_cap'
            if limit<bound and not upper(support[v],prior[v]*np.exp(limit)):return None,'density_floor'
        elif not observed[v]:
            if gap>bound+1e-8:return None,'signed_cap'
            if gap>-bound:lower(support[u],prior[u]*np.exp(gap))
        elif gap>=0:
            if gap>2*bound+1e-8:return None,'signed_cap'
            lower(support[u],prior[u]*np.exp(gap-bound))
            if not upper(support[v],prior[v]*np.exp(bound-gap)):return None,'density_floor'
            ratio=np.exp(gap)*prior[u]/prior[v]
            if ratio<1-1e-6:return None,'unsupported_floor_case'
            A.append(ratio*support[v]-support[u]);b.append(0.)
            if ratio>1+1e-8:lower(support[u],ratio*EPS)
        # Observed wrong cells with base[v]<base[u] impose a disjunction; omit
        # them in this necessary relaxation, then check actual updated logits.
    A=np.asarray(A,float).reshape(-1,k);b=np.asarray(b,float)
    if len(A):
        scale=np.maximum(np.maximum(np.abs(A).max(1),np.abs(b)),1e-14);A=A/scale[:,None];b=b/scale
    return (A,b),None


def solve_goal(base,prior,support,observed,legal,u,gain,bound=2.):
    constraints,reason=necessary_constraints(base,prior,support,observed,legal,u,gain,bound)
    if constraints is None:return [],dict(status='unsupported' if reason=='unsupported_floor_case' else 'structural_infeasible',reason=reason)
    A,b=constraints;k=support.shape[1];solutions=[];statuses=[]
    objectives=[-support[u],support[~legal].sum(0)-support[legal].sum(0)]
    for objective in objectives:
        result=linprog(objective,A_ub=A if len(A) else None,b_ub=b if len(b) else None,A_eq=np.ones((1,k)),b_eq=[1.],bounds=[(0,1)]*k,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        statuses.append(int(result.status))
        if result.success:
            weights=np.maximum(result.x,0);weights/=weights.sum()
            violation=float(np.max(A@weights-b)) if len(A) else 0.
            if violation<=1e-7 and not any(np.allclose(weights,w,atol=1e-8) for w in solutions):solutions.append(weights)
    return solutions,dict(status='feasible_relaxation' if solutions else 'numerically_infeasible' if all(s==2 for s in statuses) else 'numerical_unknown',solver_status=statuses)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',required=True);p.add_argument('--oracles',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.experiment_root);out=Path(a.output)
    if out.exists():raise ValueError('Refusing overwrite')
    checkpoint=root/'aerialvg_native_plain_sigma8_s0_20261101/best.pt';sha=file_sha256(checkpoint);previous=json.loads(Path(a.oracles).read_text())
    if previous['checkpoint_sha']!=sha:raise ValueError('Frozen Oracle/checkpoint mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);model=ObjectQueryRanker().cuda();model.load_state_dict(state['model']);model.eval().requires_grad_(False);config=state['config'];gain=float(model.fusion_gain.clamp_min(0).tanh());results={};group_sets={}
    with torch.inference_mode():
        for part,directory in [('train','aerialvg_native_training_train256_20261101'),('dev','aerialvg_native_training_dev128_20261101')]:
            meta,rows=load(root/directory)
            if meta['partition']!=part or meta['base_sha']!=config['base_sha']:raise ValueError('Invalid diagnostic split')
            group_sets[part]={r['group'] for r in rows};lookup={(r['episode'],r['step']):r for r in rows}
            selected=[r for r in previous['details'][part] if not r['any_positive_single_query_full_mass_repairs'] and not r['GT_anchor_full_mass_repairs'] and r['relaxed_full_mass_margin'] is not None and r['relaxed_full_mass_margin']>=-1e-6]
            records=[]
            for item in selected:
                row=lookup[(item['episode'],item['step'])];base=row['base_logits'].cuda().float()[None];grid=row['grid_world'].cuda().float()[None];world=row['query_world'].cuda().float()[None];valid=row['valid'].cuda()[None];observed=row['observed'].cuda()[None];ids=valid[0].nonzero().flatten()
                kernel=torch.exp(-torch.cdist(grid,world).square()/(2*config['kernel_sigma']**2))*observed[...,None]*valid[:,None]
                support=kernel[0]/kernel[0].sum(0,keepdim=True).clamp_min(EPS);prior=base[0].softmax(-1)*observed[0];prior=prior/prior.sum().clamp_min(EPS)
                # GT only defines the Oracle objective region, never model inference.
                target=row['labels']['target_world'][:2].cuda();distance=(grid[0]-target).norm(dim=-1);legal=distance<=20;B=base[0].cpu().numpy();P=prior.cpu().numpy();K=support[:,ids].cpu().numpy();O=observed[0].cpu().numpy();L=legal.cpu().numpy();verified=None;attempts=[]
                for u in np.flatnonzero(L):
                    weights_list,status=solve_goal(B,P,K,O,L,int(u),gain,config['bound']);attempts.append(dict(goal=int(u),**status))
                    for weights in weights_list:
                        q=torch.full((1,len(valid[0])),-float('inf'),device='cuda');w=torch.tensor(weights,device='cuda',dtype=torch.float32);q[0,ids]=w.log();null=torch.tensor([-float('inf')],device='cuda')
                        updated,delta=model.update_belief(base,q,null,kernel,observed,valid,bound=config['bound']);winner=int(updated[0].argmax());margin=float(updated[0,legal].max()-updated[0,~legal].max())
                        if bool(legal[winner]):verified=dict(oracle_weights=weights.tolist(),valid_query_indices=ids.cpu().tolist(),updated_NE=float(distance[winner]),positive_minus_negative_margin=margin,delta_max=float(delta.abs().max()));break
                    if verified is not None:break
                all_infeasible=bool(attempts) and all(r['status'] in ['structural_infeasible','numerically_infeasible'] for r in attempts)
                outcome='verified_mixture_witness' if verified is not None else 'necessary_LP_numerically_infeasible' if all_infeasible else 'unknown_relaxation_not_verified'
                records.append(dict(episode=row['episode'],step=row['step'],outcome=outcome,witness=verified,attempts=attempts))
            results[part]=dict(states=len(records),counts=dict(collections.Counter(r['outcome'] for r in records)),rows=records);print(json.dumps(dict(partition=part,states=len(records),counts=results[part]['counts'])),flush=True)
    if group_sets['train']&group_sets['dev'] or file_sha256(checkpoint)!=sha:raise ValueError('Split overlap or changed checkpoint')
    atomic_json(dict(base_commit='fcfde0bcd9e7f6cc66d0a1f23302ecd4ec9289f6',checkpoint_sha=sha,checkpoint_unchanged=True,source_oracles_sha=file_sha256(a.oracles),summary=results,protocol='Read-only cached Train/internalDev. Conditional residual subset only. Arbitrary full-mass posterior mixtures over existing valid queries with frozen kernels/gain/bound/support; no model training. Necessary LP ignores lower-prior observed competitors; only actual frozen update witnesses count as successes. LP numerical infeasibility is not a formal exact-arithmetic certificate; unverified relaxations remain unknown. Oracle solutions are label-dependent, never deployed or used for protocol tuning.'),out)


if __name__=='__main__':main()
