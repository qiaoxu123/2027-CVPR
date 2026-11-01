"""Read-only Train/internalDev posterior/kernel Oracle decomposition, never deployment."""
import argparse,collections,json,math
from pathlib import Path
import torch
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.train_object_query_belief import load


def one_hot_log_scores(scores,null,valid,index,mass):
    """Represent exact hypothetical visual mass, retaining frozen gain and bound."""
    if not 0<=mass<=1 or not bool(valid[0,index]):raise ValueError('Invalid oracle mass/index')
    q=torch.full_like(scores.float(),-float('inf'));q[0,index]=math.log(mass) if mass>0 else -float('inf')
    n=torch.tensor([math.log1p(-mass) if mass<1 else -float('inf')],device=scores.device)
    return q,n


def relaxed_margin(base,legal,observed,cap):
    """Optimistic independent signed residual box, not a realizable posterior claim."""
    if not bool(legal.any()):return None
    positive=(base[legal]+cap*observed[legal]).max()
    negative=(base[~legal]-cap*observed[~legal]).max() if bool((~legal).any()) else positive-1
    return float(positive-negative)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.experiment_root);output=Path(a.output)
    if output.exists():raise ValueError('Refusing overwrite')
    checkpoint=root/'aerialvg_native_plain_sigma8_s0_20261101/best.pt';sha=file_sha256(checkpoint);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['config']['relation']:raise ValueError('Unchanged plain ranker required')
    device='cuda';model=ObjectQueryRanker().to(device);model.load_state_dict(state['model']);model.eval().requires_grad_(False);config=state['config'];results={};details={};groups={}
    with torch.inference_mode():
        for partition,dirname in [('train','aerialvg_native_training_train256_20261101'),('dev','aerialvg_native_training_dev128_20261101')]:
            meta,rows=load(root/dirname)
            if meta['partition']!=partition or meta['base_sha']!=config['base_sha']:raise ValueError('Train/internalDev metadata mismatch')
            groups[partition]={r['group'] for r in rows};records=[];counts=collections.Counter();probabilities=collections.defaultdict(list)
            for row in rows:
                x={k:(v.to(device).float() if v.is_floating_point() else v.to(device))[None] for k,v in row.items() if torch.is_tensor(v)}
                base=x['base_logits'];grid=x['grid_world'];world=x['query_world'];valid=x['valid'];observed=x['observed']
                kernel=torch.exp(-torch.cdist(grid,world).square()/(2*config['kernel_sigma']**2))*observed[...,None]*valid[:,None]
                with torch.autocast('cuda',dtype=torch.bfloat16):q,null=model(x['features'],x['text'],x['text_mask'],x['geometry'],valid)
                actual,delta=model.update_belief(base,q,null,kernel,observed,valid,bound=config['bound'])
                # Labels only score the already-computed actual model and define explicit Oracles.
                target=row['labels']['target_world'][:2].to(device);distance=(grid[0]-target).norm(dim=-1);legal=distance<=20;a0=bool(legal[base[0].argmax()]);updated=bool(legal[actual[0].argmax()]);counts['states']+=1;counts['a0_correct']+=int(a0);counts['updated_correct']+=int(updated);counts['corrected']+=int(not a0 and updated);counts['damaged']+=int(a0 and not updated)
                if a0 or not bool(valid.any()):continue
                winner=int(q[0].argmax());query_distance=(world[0]-target).norm(dim=-1);rank_correct=bool(query_distance[winner]<=20);counts['a0_wrong_rank_correct']+=int(rank_correct)
                if not rank_correct:continue
                scores=torch.cat((q.float(),null.float()[:,None]),-1).softmax(-1)[0];mass=float(scores[:-1].sum());positive=(query_distance<=20)&valid[0];key='failed' if not updated else 'succeeded';probabilities[key+'_visual_mass'].append(mass);probabilities[key+'_positive_query_mass'].append(float(scores[:-1][positive].sum()));probabilities[key+'_null_probability'].append(float(scores[-1]))
                if updated:continue
                counts['failed_correct_rank']+=1;has_peak=bool(kernel[0,:,winner].sum()>0);peak=int(kernel[0,:,winner].argmax()) if has_peak else None;peak_wrong=has_peak and not bool(legal[peak]);counts['ranked_candidate_kernel_peak_outside20']+=int(peak_wrong);counts['ranked_candidate_has_no_kernel_support']+=int(not has_peak)
                def correct(q2,n2,k2):
                    logits,_=model.update_belief(base,q2,n2,k2,observed,valid,bound=config['bound']);return bool(legal[logits[0].argmax()])
                same_mass=[];full_mass=[]
                for index in positive.nonzero().flatten().tolist():
                    q2,n2=one_hot_log_scores(q,null,valid,index,mass);same_mass.append(correct(q2,n2,kernel))
                    q3,n3=one_hot_log_scores(q,null,valid,index,1.);full_mass.append(correct(q3,n3,kernel))
                winner_q,winner_n=one_hot_log_scores(q,null,valid,winner,mass);top_only=correct(winner_q,winner_n,kernel)
                gt_kernel=kernel.clone();gt_kernel[0,:,winner]=torch.exp(-distance.square()/(2*config['kernel_sigma']**2))*observed[0]
                gt_same=correct(winner_q,winner_n,gt_kernel);gt_q,gt_n=one_hot_log_scores(q,null,valid,winner,1.);gt_full=correct(gt_q,gt_n,gt_kernel)
                oracle_same=any(same_mass);oracle_full=any(full_mass)
                full_cap=float(model.fusion_gain.clamp_min(0).tanh()*config['bound'])
                fixed_margin=relaxed_margin(base[0],legal,observed[0],full_cap*mass);full_margin=relaxed_margin(base[0],legal,observed[0],full_cap)
                full_impossible=full_margin is None or full_margin < -1e-6
                residual_unresolved=not oracle_full and not gt_full
                for name,value in dict(top1_only_same_mass_repairs=top_only,any_positive_single_query_same_mass_repairs=oracle_same,any_positive_single_query_full_mass_repairs=oracle_full,GT_anchor_same_mass_repairs=gt_same,GT_anchor_full_mass_repairs=gt_full,confidence_intervention_adds_query_repair=oracle_full and not oracle_same,GT_anchor_adds_same_mass_repair=gt_same and not oracle_same,neither_query_nor_GT_full_mass_repairs=residual_unresolved,relaxed_full_mass_impossible=full_impossible,unresolved_and_relaxed_full_impossible=residual_unresolved and full_impossible,unresolved_but_relaxed_full_not_excluded=residual_unresolved and not full_impossible).items():counts[name]+=int(value)
                records.append(dict(episode=row['episode'],step=row['step'],visual_mass=mass,positive_query_mass=float(scores[:-1][positive].sum()),null_probability=float(scores[-1]),ranked_candidate_NE=float(query_distance[winner]),ranked_candidate_kernel_peak_NE=float(distance[peak]) if peak is not None else None,top1_only_same_mass_repairs=top_only,any_positive_single_query_same_mass_repairs=oracle_same,any_positive_single_query_full_mass_repairs=oracle_full,GT_anchor_same_mass_repairs=gt_same,GT_anchor_full_mass_repairs=gt_full,relaxed_same_mass_margin=fixed_margin,relaxed_full_mass_margin=full_margin))
            results[partition]=dict(counts=dict(counts),means={k:sum(v)/len(v) for k,v in probabilities.items()});details[partition]=records
            print(json.dumps(dict(partition=partition,**results[partition])),flush=True)
    if groups['train']&groups['dev']:raise ValueError('Split group overlap')
    if file_sha256(checkpoint)!=sha:raise ValueError('Checkpoint changed')
    atomic_json(dict(checkpoint_sha=sha,checkpoint_unchanged=True,base_commit='fcfde0bcd9e7f6cc66d0a1f23302ecd4ec9289f6',summary=results,details=details,protocol='Frozen actual outputs first. Oracle interventions are post-label mathematical diagnostics on same states/weights/sigma/bound/observed mask: one-hot positive-query posterior at same visual mass, then mass1; exact GT-anchored kernel separately. GT anchoring introduces unavailable perfect localization and is never a model result. One-hot maxima are not the exhaustive mixture upper bound. Interventions overlap, not exclusive root causes. No training/rollout/Test/new validation labels.'),output)


if __name__=='__main__':main()
