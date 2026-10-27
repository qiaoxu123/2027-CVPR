"""Training-internal calibration only; diagnostic or insufficient data stays off."""
from common import *
import argparse
import numpy as np
import torch
from train import load_groups,evaluate_rows
from schema import to_device
from fusion import local_mix


def calibration_eligible(state,positives,negatives,ranked_views,smoke=False):
    metrics=state['metrics']
    return (not smoke and not state['diagnostic_only'] and metrics['eligible'] and metrics['new_errors']==0
            and positives>=8 and negatives>=8 and ranked_views>=8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);model,state=load_branch(args.checkpoint,args.device);groups=load_groups(args.smoke)
    validate_cache_signature(state,json.loads((ROOT/'data/manifest.json').read_text())['signature'])
    rows=groups['calibration'];summary,records=evaluate_rows(model,rows,args.device)
    known=[r for r in records if r['match_valid']];temp=1.
    if known:
        def nll(t):
            z=torch.tensor([r['match_logit']/t for r in known]);y=torch.tensor([r['match'] for r in known])
            return float(torch.nn.functional.binary_cross_entropy_with_logits(z,y))
        temp=min([.5,1.,2.,4.],key=nll)
    probes=[]
    with torch.no_grad():
        predictions=[model(**to_device(r['public'],args.device)) for r in rows]
        for threshold in [.5,.7,.9]:
            for alpha in [0.,.1,.25,.5]:
                gains=losses=0;errors=[];used=0
                for r,o in zip(rows,predictions):
                    s={k:v.to(args.device) for k,v in r['public']['spatial'].items()};y=r['labels'];gate=float(torch.sigmoid(o['match_logit']/temp))>=threshold
                    out=local_mix(s['base_logits'],s['grid_world'],s['candidate_xy']+o['offset'],o['scores'],s['candidate_support'],alpha if gate else 0.)
                    before=int(s['base_logits'].argmax());after=int(out['logits'].argmax());target=y['target_xy'].to(args.device)
                    bd=float((s['grid_world'][before]-target).norm());ad=float((s['grid_world'][after]-target).norm())
                    gains+=bd>20 and ad<=20;losses+=bd<=20 and ad>20;errors.append(ad);used+=out['changed']
                probes.append(dict(threshold=threshold,alpha=alpha,gains=gains,losses=losses,xy_error=float(np.mean(errors)) if errors else 1e6,updates=used))
    valid=[v for v in probes if v['losses']==0];best=min(valid,key=lambda v:(-v['gains'],v['xy_error'],v['alpha'],v['threshold']))
    positives=sum(r['match']>0 for r in known);negatives=len(known)-positives
    # Small/partial/diagnostic caches cannot certify gating. Keep safe fallback.
    eligible=calibration_eligible(state,positives,negatives,summary['ranked_views'],args.smoke)
    enabled=eligible and best['alpha']>0 and best['gains']>0
    result=dict(enabled=bool(enabled),alpha=best['alpha'] if enabled else 0.,threshold=best['threshold'],temperature=temp,
        checkpoint_sha256=sha(args.checkpoint),config_sha256=sha(ROOT/'configs/frozen.json'),calibration_positive_views=positives,calibration_negative_views=negatives,
        candidate_summary=summary,probes=probes,reason='enabled after internal calibration' if enabled else 'insufficient evidence or no safe measured gain: exact main-model fallback',
        scope='train-internal calibration only; XY improvements here are calibration, not heldout results')
    if args.output.exists():raise ValueError('Refuse overwrite of calibration')
    save(result,args.output);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
