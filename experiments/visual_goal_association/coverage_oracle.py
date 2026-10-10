"""Post-hoc identity-only centroid upper bound for actually observed P1 IDs."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.bidirectional_core.stats import paired_difference
from .evaluate import load_prepared,write_rows,write_json,SEED
from .coverage import stats


def run(prepared,coverage,output,objects):
    root=Path(output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);data=load_prepared(prepared);states={r['sample_id']:r for r in data['public']};raw=json.loads(Path(objects).read_text());result={}
    for path in sorted(Path(coverage).glob('*_predictions.jsonl')):
        rows=[]
        for line in path.read_text().splitlines():
            item=json.loads(line);state=states[item['sample_id']];lab=data['labels'][item['sample_id']];xy=state['bounds'][:2]+state['grid_xy']*(state['bounds'][2:]-state['bounds'][:2]);logits=state['logits'].copy()
            # Identity selects the public mapped object, never a goal-nearest cell.
            oid=lab['target_object_id'];valid=oid in item['observed_object_ids'];credible=bool(item['metrics']['target_credible'])
            target=raw[state['map_name']].get(oid);grid=None
            if valid and target:
                center=np.asarray(target['contour'],float).mean(0);grid=int(np.square(xy-center).sum(1).argmin())
                logits[grid]+=float(logits.max()-logits.min()+1)
            distance=np.linalg.norm(xy-lab['target_world'],axis=1);static=int(state['logits'].argmax());rank=np.argsort(-logits);a=bool(distance[static]<=20);b=bool(distance[rank[0]]<=20)
            item['metrics']=dict(r1=float(b),r4=float((distance[rank[:4]]<=20).any()),r8=float((distance[rank[:8]]<=20).any()),ne=float(distance[rank[0]]),recovery=float(not a and b),damage=float(a and not b),net_recovery=float(b)-float(a),error_credible=float(not a and credible),credible_recoverable=float(not a and credible and b))
            item.update(oracle_diagnostic_only=True,oracle_associated_cell=grid);rows.append(item)
        name=path.name.removesuffix('_predictions.jsonl');result[name]=rows;write_rows(root/(name+'_predictions.jsonl'),rows)
    summary={s:stats(r) for s,r in result.items()}
    paired={s:paired_difference(result['current'],r,'r1',seed=SEED,repetitions=2000) for s,r in result.items()}
    write_json(root/'metrics.json',dict(models=summary,paired_vs_current=paired,oracle_only=True,note='all actual observed IDs; public centroid mapping; no RGB semantic claim; credible subset separately counted',seed=SEED))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--coverage',required=True);p.add_argument('--output',required=True);p.add_argument('--objects',required=True);a=p.parse_args();run(a.prepared,a.coverage,a.output,a.objects)
