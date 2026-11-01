"""TRAIN-only C counterfactuals: isolate clipping and observed-prior renormalization."""
import argparse,json
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json
from scripts.train_object_query_belief import load


def observed_density_or_base(density,observed,base):
    """An empty observed grid supplies no evidence; never argmax an all-mask field."""
    if not observed.any():return base
    return density.clamp_min(1e-30).log().masked_fill(~observed,-1e4)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['records','cache','native-checkpoint','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('No overwrite')
    meta,rows=load(a.cache)
    if meta['partition']!='train':raise ValueError('TRAIN-only offline analysis')
    lookup={(r['episode'],r['step']):r for r in rows};records=[json.loads(x) for x in Path(a.records).read_text().splitlines()]
    checkpoint=torch.load(a.native_checkpoint,map_location='cpu',weights_only=True);gain=checkpoint['model']['fusion_gain'].clamp_min(0).tanh().cuda()
    output=[]
    with torch.inference_mode():
        for r in records:
            if not r['oracle_authorized']:continue
            row=lookup[(r['episode'],r['step'])];base=row['base_logits'].float().cuda();grid=row['grid_world'].float().cuda();obs=row['observed'].cuda()
            distance=(grid-row['labels']['target_world'][:2].float().cuda()).norm(dim=-1)
            density=torch.exp(-distance.square()/(2*8.**2))*obs
            density=density/density.sum().clamp_min(1e-9)
            p0=base.softmax(-1);conditional=p0*obs;conditional=conditional/conditional.sum().clamp_min(1e-9)
            ratio_cond=density.clamp_min(1e-9).log()-conditional.clamp_min(1e-9).log()
            ratio_global=density.clamp_min(1e-9).log()-p0.clamp_min(1e-9).log()
            variants={'original_conditional_clipped':base+gain*ratio_cond.clamp(-2,2)*obs,
                      'conditional_unclipped':base+gain*ratio_cond*obs,
                      'global_prior_clipped':base+gain*ratio_global.clamp(-2,2)*obs,
                      'global_prior_unclipped':base+gain*ratio_global*obs,
                      'ideal_density_observed_only':observed_density_or_base(density,obs,base)}
            errors={name:float(distance[value.argmax()]) for name,value in variants.items()}
            if abs(errors['original_conditional_clipped']-r['methods']['C_target_bounded']['error_m'])>1e-4:raise ValueError('Original algebra did not reproduce frozen C')
            output.append(dict(sample_id=r['sample_id'],category=r['category'],instance_group=r['instance_group'],
                               A0_error=r['methods']['A0']['error_m'],Native_error=r['methods']['Native15']['error_m'],
                               observed_A0_probability_mass=float(p0[obs].sum()),errors=errors,
                               mode='UNVERIFIED_POSITION_HYPOTHESIS_OFFLINE_NOT_DEPLOYABLE'))
    summary={}
    for label,items in [('all',output),('A0_wrong',[r for r in output if r['A0_error']>20]),('A0_correct',[r for r in output if r['A0_error']<=20])]:
        summary[label]={}
        for name in output[0]['errors'] if output else []:
            summary[label][name]=dict(states=len(items),R1=sum(r['errors'][name]<=20 for r in items)/len(items) if items else None,
                NE=sum(r['errors'][name] for r in items)/len(items) if items else None,
                corrected_A0=sum(r['A0_error']>20 and r['errors'][name]<=20 for r in items),
                damaged_A0=sum(r['A0_error']<=20 and r['errors'][name]>20 for r in items))
    atomic_json(dict(only_TRAIN=True,no_training=True,human_verified_states=0,scope='Explicit C position-hypothesis mechanism audit, not a deployment improvement',
                     summary=summary,records=output),Path(a.output))


if __name__=='__main__':main()
