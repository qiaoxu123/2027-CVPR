"""Read fixed predictions: weak tri-class calibration and object clustering."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.bidirectional_core.stats import interval
from .dataset import city_rows
from .data import write
from .association import weights as public_weights


def run(a):
    out=Path(a.output)
    if out.exists():raise ValueError('no overwrite')
    out.mkdir(parents=True);rows={r['metadata']['sample_id']:r for r in city_rows(a.features)};m=json.loads(Path(a.metrics).read_text());selected=m['selected'];result={}
    for control in ['real','zero','shuffled']:
        recs=[json.loads(x) for x in (Path(a.metrics).parent/(selected+'_'+control+'_predictions.jsonl')).read_text().splitlines()];confusion=np.zeros((3,3),int);g=[];identity=[];truncated=[]
        for p in recs:
            r=rows[p['sample_id']];lab=r['labels'];gt=np.full(len(lab['positive_roi_mask']),2);gt[lab['semantic_valid_mask'].numpy()&~lab['positive_roi_mask'].numpy()]=1;gt[lab['positive_roi_mask'].numpy()]=0;pred=np.asarray(p['judgment_prob']).reshape(-1,3).argmax(-1)
            for x,y in zip(gt,pred):confusion[x,y]+=1
            if p['metrics']['identity_r1'] is not None:
                g.append(p['map_name']+':'+p['target_object_id']);identity.append(p['metrics']['identity_r1'])
        f1=[]
        for c in range(3):
            tp=confusion[c,c];den=confusion[c].sum()+confusion[:,c].sum();f1.append(float(2*tp/den) if den else None)
        result[control]=dict(confusion_true_rows_pred_columns=confusion.tolist(),classes=['Match','Mismatch','Unknown'],f1=f1,macro_f1=float(np.mean([x for x in f1 if x is not None])),identity_object_cluster_95ci=interval(identity,groups=g,seed=20261011),independent_objects=len(set(g)),annotation_status='weak original identity and conservative relation/attribute judgments; not human calibrated semantic truth')
    chain=[];limited=[];real=[json.loads(x) for x in (Path(a.metrics).parent/(selected+'_real_predictions.jsonl')).read_text().splitlines()]
    for p in real:
        r=rows[p['sample_id']];s=r['state'];v=r['metadata']['roi_objects'];lab=r['labels'];bounds=np.array(s['public']['bounds']);xy=bounds[:2]+np.array(s['public']['grid_xy'])*(bounds[2:]-bounds[:2]);logits=np.array(s['public']['static_logits']);dist=np.linalg.norm(xy-np.array(r['sealed']['target_world']),axis=1);before=np.argsort(logits,kind='stable')[::-1];wrong=dist[before[0]]>20;updated=logits.copy();gap=None
        if lab['identity_positive'].any():
            w=public_weights(s,v);support=(w[lab['identity_positive'].numpy()]>0).any(0)
            if support.any():gap=float(logits.max()-logits[support].max());updated[support]+=.025
        after=np.argsort(updated,kind='stable')[::-1];correct=dist[after[0]]<=20;limited.append(dict(sample_id=p['sample_id'],family_id=p['family_id'],map_name=p['map_name'],metrics=dict(goal_r1=float(correct),recovery=float(wrong and correct),damage=float(not wrong and not correct))))
        if wrong and p['target_rank'] is not None:chain.append(dict(sample_id=p['sample_id'],family_id=p['family_id'],target_rank=p['target_rank'],identity_top1_correct=p['target_rank']==1,accepted=p['accepted_candidate'] is not None,required_centroid_logit_gap=gap,bounded_residual_could_beat_static=gap is not None and gap<.025))
    result['chain']=dict(visible_static_error_states=len(chain),visible_static_error_families=len({r['family_id'] for r in chain}),identity_correct_states=sum(r['identity_top1_correct'] for r in chain),identity_correct_families=len({r['family_id'] for r in chain if r['identity_top1_correct']}),identity_correct_but_abstained_states=sum(r['identity_top1_correct'] and not r['accepted'] for r in chain),margin_below_fixed_bound_states=sum(r['bounded_residual_could_beat_static'] for r in chain),note='Post-hoc diagnostic. Same .025 bound identity oracle uses GT only to select an actually seen instance; not normal inference, not threshold retuning.')
    from experiments.bidirectional_core.stats import summarize_rows
    result['bounded_identity_oracle']={k:summarize_rows(limited,k,seed=20261011) for k in ['goal_r1','recovery','damage']}
    write(out/'chain_cases.json',chain);write(out/'bounded_identity_oracle_predictions.json',limited)
    write(out/'metrics.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--metrics',required=True);p.add_argument('--output',required=True);run(p.parse_args())
if __name__=='__main__':main()
