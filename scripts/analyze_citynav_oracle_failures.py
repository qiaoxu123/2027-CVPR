"""Read-only TRAIN Oracle attribution; automatic-box hypotheses stay unverified."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np,torch
from graphnav.checkpointing import atomic_json
from graphnav.object_query_belief import ObjectQueryRanker
from scripts.train_object_query_belief import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['records','cache','native-checkpoint','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('No overwrite')
    meta,rows=load(a.cache)
    if meta['partition']!='train':raise ValueError('TRAIN only')
    lookup={(r['episode'],r['step']):r for r in rows};records=[json.loads(x) for x in Path(a.records).read_text().splitlines()]
    native=ObjectQueryRanker().cuda();state=torch.load(a.native_checkpoint,map_location='cpu',weights_only=True);native.load_state_dict(state['model']);native.eval().requires_grad_(False)
    eligible=[r for r in records if r['oracle_authorized']];wrong=[r for r in eligible if not r['methods']['A0']['correct']]
    failures=[];counts=Counter();geometry=Counter()
    with torch.inference_mode():
        for r in eligible:
            row=lookup[(r['episode'],r['step'])];base=row['base_logits'].float().cuda();grid=row['grid_world'].float().cuda();observed=row['observed'].cuda();target=row['labels']['target_world'][:2].float().cuda()
            dist=(grid-target).norm(dim=-1);good=dist<=20
            if r['methods']['C_target_bounded']['correct']:continue
            kernel=torch.exp(-dist.square()/(2*8.**2))[None,:,None]*observed[None,:,None]
            q=torch.tensor([[20.]],device='cuda');null=torch.tensor([-20.],device='cuda')
            updated,delta=native.update_belief(base[None],q,null,kernel,observed[None],torch.ones(1,1,dtype=torch.bool,device='cuda'),bound=2.)
            error=float(dist[updated[0].argmax()])
            if abs(error-r['methods']['C_target_bounded']['error_m'])>1e-4:raise ValueError('Frozen C reproduction mismatch')
            p0=base.softmax(-1)*observed;p0=p0/p0.sum().clamp_min(1e-9)
            density=kernel[0,:,0]/kernel[0,:,0].sum().clamp_min(1e-9)
            ratio=(density.clamp_min(1e-9).log()-p0.clamp_min(1e-9).log()).clamp(-2,2)
            old=int(base.argmax());legal=(updated[0].masked_fill(~good,-1e4)).argmax()
            if not good.any():reason='no_20m_grid_cell'
            elif not (good&observed).any():reason='no_observed_legal_grid'
            elif r['fusion_limit']['bounded_impossible']:reason='analytic_amplitude_bound'
            elif observed[old] and observed[legal] and abs(float(ratio[old]-ratio[legal]))<1e-5 and abs(float(ratio[old]))>1.999:
                reason='both_peaks_same_clipped_shift'
            elif not observed[old]:reason='unobserved_A0_peak_preserved_plus_density_interaction'
            else:reason='prior_kernel_clipping_interaction_unknown'
            counts[reason]+=1
            failures.append(dict(sample_id=r['sample_id'],category=r['category'],A0_error=r['methods']['A0']['error_m'],C_error=error,
                reason=reason,old_peak_observed=bool(observed[old]),best_good_observed=bool(observed[legal]),
                old_ratio=float(ratio[old]),good_ratio=float(ratio[legal]),old_delta=float(delta[0,old]),good_delta=float(delta[0,legal])))
    def rates(subset):
        result=dict(states=len(subset))
        for k in ['top1_match','top5_match','selected_injected_box']:
            result['A_'+k]=sum(r['oracle_details'].get('A_box_exists',{}).get(k,False) for r in subset)
        result['A_matching_and_Belief_correct']=sum(r['oracle_details'].get('A_box_exists',{}).get('top1_match',False) and r['methods']['A_box_exists']['correct'] for r in subset)
        for name in ['B_existing15_selected','B_existing900_selected','B_added_box_selected']:
            ds=[r['oracle_details'][name] for r in subset if r['oracle_details'].get(name,{}).get('attempted')]
            mass=[d['evidence_mass'] for d in ds];mapping=[d['query_world_error_m'] for d in ds]
            result[name]=dict(attempted=len(ds),mass_min=min(mass) if mass else None,mass_median=float(np.median(mass)) if mass else None,
                             mass_lt_half=sum(x<.5 for x in mass),mapped_center_gt20=sum(x>20 for x in mapping))
        return result
    # Counterfactual differences isolate learned evidence mass vs unit-mass, and mapping vs known target position.
    differences=dict(B_added_wrong_C_box_bounded_correct=sum(not r['methods']['B_added_box_selected']['correct'] and r['methods']['C_box_bounded']['correct'] for r in eligible),
        C_box_bounded_wrong_C_target_bounded_correct=sum(not r['methods']['C_box_bounded']['correct'] and r['methods']['C_target_bounded']['correct'] for r in eligible),
        C_target_bounded_wrong_C_box_bounded_correct=sum(not r['methods']['C_target_bounded']['correct'] and r['methods']['C_box_bounded']['correct'] for r in eligible),
        C_target_bounded_wrong_C_target_unbounded_correct=sum(not r['methods']['C_target_bounded']['correct'] and r['methods']['C_target_unbounded']['correct'] for r in eligible))
    atomic_json(dict(only_TRAIN=True,human_verified_states=sum(r['actual_visibility_verified'] for r in records),
                     automatic_hypotheses_only=True,eligible=rates(eligible),eligible_A0_wrong=rates(wrong),
                     C_target_bounded_failure_counts=dict(counts),C_target_bounded_failures=failures,
                     paired_counterfactual_differences=differences,warning='Detection/matching attribution remains conditional on unverified pixel registration; fusion algebra is independently checked'),Path(a.output))


if __name__=='__main__':main()
