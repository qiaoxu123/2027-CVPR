"""Read existing evaluation artifacts only; no datasets, inference, or tuning."""
import argparse,json,csv
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--saved-run',type=Path,required=True);p.add_argument('--label-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(p.read_text()) for p in sorted((a.saved_run/'rows').glob('*.json'))]
    assert len(rows)==2470 and all(r['error'] is None for r in rows)
    groups={k:[] for k in ('no_support','bound_insufficient','sufficient_failure','corrected','damaged')}
    for r in rows:
        c=r['correction_capacity'];g=c['geometry_correct'];f=c['fused_correct']
        if not g:
            key='no_support' if not c['correct_region_observed'] else 'bound_insufficient' if c['bound_insufficient'] else 'corrected' if f else 'sufficient_failure'
            groups[key].append(r)
        elif not f:groups['damaged'].append(r)
    result=dict(scope='Saved prior Val artifacts only. No new Val inference, labels or parameter selection.',groups={})
    for key,rr in groups.items():
        result['groups'][key]=dict(n=len(rr),unique_physical_targets=len({tuple(r['episode_id'][:2]) for r in rr}),
            geometry_gap_quantiles=np.quantile([r['correction_capacity']['wrong_minus_correct_geometry_peak'] for r in rr],[0,.1,.5,.9,1]).tolist(),
            actual_delta_quantiles=np.quantile([r['correction_capacity']['actual_delta_max'] for r in rr],[0,.5,1]).tolist(),
            geometry_NE=float(np.mean([r['geometry']['NE'] for r in rr])),fused_NE=float(np.mean([r['fused']['NE'] for r in rr])),
            anchor_zero=sum(r['public_anchor_count']==0 for r in rr),
            visual_only_region_hit=sum(bool(r['correction_capacity']['visual_only_top1_correct_region_proxy']) for r in rr),
            examples=[dict(episode_id=r['episode_id'],instruction=r['instruction'],geometry_NE=r['geometry']['NE'],fused_NE=r['fused']['NE'],capacity=r['correction_capacity']) for r in rr[:3]])
    audit=json.loads(a.label_audit.read_text());exceptions=audit['exceptions'];result['TRAIN_label_conflicts']=dict(summary=audit['summary'],n=len(exceptions),unique_physical_targets=len({':'.join(r['episode_id'].split(':')[:2]) for r in exceptions}),rows=exceptions)
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'saved_failure_groups.json').write_text(json.dumps(result,indent=2))
    with (a.output/'saved_failure_groups.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['group','episode_id','instruction','geometry_NE','fused_NE','geometry_gap','visual_only_region_hit'])
        for key,rr in groups.items():
            for r in rr:writer.writerow([key,json.dumps(r['episode_id']),r['instruction'],r['geometry']['NE'],r['fused']['NE'],r['correction_capacity']['wrong_minus_correct_geometry_peak'],r['correction_capacity']['visual_only_top1_correct_region_proxy']])
    print(json.dumps({k:v['n'] for k,v in result['groups'].items()}))

if __name__=='__main__':main()
