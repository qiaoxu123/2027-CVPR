"""Summarize frozen final runs without selecting models from Unseen outcomes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.summarize_closedloop import read_run,paired_stats,failures
from graphnav.checkpointing import file_sha256,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default='reports/sbf_protocol/final_frozen_evaluation_plan.json')
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing final analysis overwrite')
    plan=json.loads(Path(a.plan).read_text());primary=plan['primary_candidate']
    seen_paths={'a3_basic':'runs/closedloop_baseline/a3_seen_full','a3_trajectory':'runs/closedloop_seen_altitude_guard/a3','gat':'runs/closedloop_seen_altitude_guard/gat','visual':'runs/closedloop_seen_altitude_guard/visual','joint_current':'runs/closedloop_seen_altitude_guard/joint_current','joint_history':'runs/closedloop_seen_altitude_guard/joint_history','full_seed0':'runs/closedloop_seen_altitude_guard/full_seed0','full_seed1':'runs/full_seed1_seen','full_seed2':'runs/full_seed2_seen'}
    for name in ('g0_trajectory','g1_p1_v2_trajectory','g1_p1_v3_trajectory'):seen_paths[name]='runs/p1_matched_trajectory_seen/'+name
    baselines={s:read_run('runs/closedloop_baseline/sbf_val_'+s+'_full') for s in ('seen','unseen')}
    result=dict(plan_sha256=file_sha256(a.plan),primary_candidate=primary,selection=plan['primary_selection'],provenance=provenance(),models={},paired_vs_sbf={},seed_summary={},acceptance={},note='Frozen candidates only. No selection or tuning from Unseen. Fractions in CSV/JSON, meters for NE; display percentages separately. SBF is the one available local paper-based checkpoint, not author-official or three-seed training.')
    rows=[];maps=[];episodes=[]
    keys=('SR','SPL','OSR','NE','nDTW','sDTW','steps','inference_ms','False_Stop_Rate')
    def collect(name,split,path,run):
        report,trace,manifest=run
        if manifest!=baselines[split][2]:raise ValueError('final denominator/episode manifest differs from SBF: '+name)
        rows.append(dict(model=name,split=split,**{k:report['metrics'].get(k) for k in keys},episodes=len(trace),seconds=report['seconds'],peak_allocated_gib=report['gpu_peak_allocated_gib']))
        result['models'].setdefault(name,{})[split]=dict(path=path,metrics=report['metrics'],seconds=report['seconds'],peak_allocated_gib=report['gpu_peak_allocated_gib'],failures=failures(trace),observed_visual_cache=report.get('observed_visual_cache'),checkpoint_sha256=report['checkpoint_sha256'],head_checkpoint_sha256=report.get('head_checkpoint_sha256'),dynamic_checkpoint_sha256=report.get('dynamic_checkpoint_sha256'),graphnav_commit=report['graphnav_commit'])
        maps.extend(dict(model=name,split=split,map_name=m,episodes=v['episodes'],**{k:v.get(k) for k in keys}) for m,v in report['by_map'].items())
        episodes.extend(dict(model=name,split=split,index=i,map_name=r['map_name'],termination=r['termination'],**{k:r['metrics'].get(k) for k in keys if k!='False_Stop_Rate'}) for i,r in trace.items())
        if name!='sbfnav':result['paired_vs_sbf'].setdefault(name,{})[split]=paired_stats(baselines[split][1],trace)
    for split in ('seen','unseen'):collect('sbfnav',split,'runs/closedloop_baseline/sbf_val_'+split+'_full',baselines[split])
    for name,description in plan['candidates'].items():
        path='runs/final_closedloop_unseen/'+name;run=read_run(path)
        for flag,metadata in description['checkpoints'].items():
            report_key={'--checkpoint':'checkpoint_sha256','--head-checkpoint':'head_checkpoint_sha256','--dynamic-checkpoint':'dynamic_checkpoint_sha256','--visual-checkpoint':'visual_checkpoint_sha256'}[flag]
            if run[0].get(report_key)!=metadata['sha256']:raise ValueError('frozen candidate weights differ: '+name+' '+flag)
        collect(name,'unseen',path,run)
        if name in seen_paths:collect(name,'seen',seen_paths[name],read_run(seen_paths[name]))
    for split in ('seen','unseen'):
        values={k:np.asarray([result['models'][f'full_seed{i}'][split]['metrics'][k] for i in range(3)]) for k in keys if k!='False_Stop_Rate'}
        result['seed_summary'][split]={k:dict(values=v.tolist(),mean=float(v.mean()),std_sample=float(v.std(ddof=1)),minimum=float(v.min()),maximum=float(v.max())) for k,v in values.items()}
    result['seed_uncertainty']=plan['seed_uncertainty']+' Mean/std/range over3 seeds; no narrow normal confidence claim with only3 training seeds. Paired episode/map bootstrap is separate from training-seed variability.'
    sbf=result['models']['sbfnav']['unseen']['metrics'];candidate=result['models'][primary]['unseen']['metrics'];seen=result['models'][primary]['seen']['metrics'];sbf_seen=result['models']['sbfnav']['seen']['metrics']
    result['acceptance']=dict(numeric_level2=candidate['SR']>=.95*sbf['SR'] and candidate['SPL']>=.90*sbf['SPL'],numeric_level3=candidate['SR']>=sbf['SR'] and candidate['SPL']>=sbf['SPL'],level2_min_SR=.95*sbf['SR'],level2_min_SPL=.90*sbf['SPL'],unseen_SR_ratio=candidate['SR']/sbf['SR'],unseen_SPL_ratio=candidate['SPL']/sbf['SPL'],seen_SR_delta=seen['SR']-sbf_seen['SR'],seen_SPL_delta=seen['SPL']-sbf_seen['SPL'],all_three_seeds_numeric_level2=all(result['models'][f'full_seed{i}']['unseen']['metrics']['SR']>=.95*sbf['SR'] and result['models'][f'full_seed{i}']['unseen']['metrics']['SPL']>=.90*sbf['SPL'] for i in range(3)),note='Numeric thresholds only. Level1 pipeline/no-future checks and Seen/Stop/constraint audits require separate assessment. SBF has no policy Stop, so its False-Stop Rate is null and cannot be treated as zero error. No physical flight/swept collision claim.')
    out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(result,indent=2))
    for name,table in [('comparison',rows),('maps',maps),('episodes',episodes)]:
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps(dict(output=str(out),primary=primary,acceptance=result['acceptance'])))


if __name__=='__main__':main()
