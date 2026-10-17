"""Aggregate only complete GeomBelief experiments; paired IDs and map clusters."""
import csv,json,hashlib,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import aggregate
from graphnav.geombelief_inputs import ARMS
from scripts.geombelief_map_experiments import OUT,OLD,HERE
from scripts.polygon_ablation_statistics import paired_ci,paired_rows,table

LABELS={'correct':'Language + Correct Geometry','name_type':'Language + Name/Type Only','shuffled':'Language + Shuffled Geometry','distance':'Language + Distance Only','direction':'Language + Direction Only','distance_direction':'Language + Distance + Direction'}

def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]
def csvwrite(name,data):
    keys=list(dict.fromkeys(k for r in data for k in r)) or ['status']
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)

def map_paths(arm,seed,split,endpoint):
    if arm=='correct':
        job=f'trained__descriptor_only__full_polygon__seed{seed}'
        cp=OLD/'training'/f'shared_full_polygon__full_polygon__seed{seed}'/'best.pt'
        folder=OLD/endpoint/job
    else:
        job=f'map_geometry__{arm}__seed{seed}';cp=OUT/'training'/job/'best.pt';folder=OUT/endpoint/job
    if endpoint=='localization':return folder/f'{split}.json',folder/f'{split}.localization.jsonl',cp
    return folder/split/'metrics.json',folder/split/'predictions.jsonl',cp

def summarize():
    OUT.mkdir(parents=True,exist_ok=True);data={};result=[];means=[];stats={};manifest=[]
    for endpoint in ('localization','navigation'):
        metrics=('R@1','R@4','R@8','NE') if endpoint=='localization' else ('SR','SPL','OSR','NE','False_Stop_Rate')
        for split,n in [('val_seen',2426 if endpoint=='localization' else 2470),('val_unseen',2679 if endpoint=='localization' else 2697)]:
            for arm in ('correct',*ARMS):
                members=[]
                for seed in (0,1,2):
                    metric,trace,cp=map_paths(arm,seed,split,endpoint)
                    complete=metric.exists() and trace.exists()
                    r=dict(model='GeomBelief',arm=LABELS[arm],arm_key=arm,endpoint=endpoint,split=split,seed=seed,status='Done' if complete else 'Pending',expected_samples=n,result_path=str(metric),predictions=str(trace),checkpoint=str(cp),checkpoint_sha256=file_sha256(cp) if cp.exists() else '',reused=arm=='correct')
                    if complete:
                        actual=rows(trace);assert len(actual)==n,(trace,len(actual),n)
                        assert len({x['query_id'] if endpoint=='localization' else tuple(x['episode_id']) for x in actual})==n
                        summary=json.loads(metric.read_text())['metrics'];r.update(samples=n,**{k:summary[k] for k in metrics});members.append(r)
                        data[endpoint,split,arm,seed]=actual
                    result.append(r);manifest.append(r.copy())
                if len(members)==3:
                    m=dict(endpoint=endpoint,split=split,arm=LABELS[arm],arm_key=arm,seeds=3,samples_per_seed=n)
                    for k in metrics:
                        values=[r[k] for r in members]
                        m[k+'_mean']=float(np.mean(values)) if all(v is not None for v in values) else None
                        m[k+'_std']=float(np.std(values,ddof=1)) if all(v is not None for v in values) else None
                    means.append(m)
            tests=[('correct','name_type'),('correct','shuffled'),('distance_direction','name_type'),('distance','name_type'),('direction','name_type'),('distance_direction','distance'),('distance_direction','direction')]
            paired_metrics=list(metrics[:3] if endpoint=='localization' else ('SR','SPL'))
            for left,right in tests:
                if not all((endpoint,split,a,s) in data for a in (left,right) for s in (0,1,2)):continue
                deltas=[]
                for seed in (0,1,2):
                    d,m,ids=paired_rows(data[endpoint,split,left,seed],data[endpoint,split,right,seed],endpoint,paired_metrics)
                    if deltas:assert m==maps and ids==identities
                    maps,identities=m,ids;deltas.append(d)
                    stats[f'{endpoint}/{split}/{left}-minus-{right}/seed{seed}']=paired_ci(d,m,paired_metrics)
                stats[f'{endpoint}/{split}/{left}-minus-{right}/seed_mean']=paired_ci(np.mean(deltas,0),maps,paired_metrics)
    csvwrite('map_geometry_results.csv',result);csvwrite('map_geometry_seed_summary.csv',means)
    control=[];control_data={};formal=json.loads((OLD/'frozen_plan.json').read_text())['base_checkpoint']
    for split,n in [('val_seen',2470),('val_unseen',2697)]:
        for arm in ('predicted_guarded_blend','predicted_direct','predicted_waypoint','oracle_direct','oracle_guarded_blend','rule_arrival'):
            if arm=='predicted_guarded_blend':
                folder=OUT/'controller'/'predicted_guarded_blend'/split
            else:folder=OUT/'controller'/arm/split
            ok=bool(folder and (folder/'metrics.json').exists() and (folder/'predictions.jsonl').exists())
            r=dict(arm=arm,split=split,status='Done' if ok else 'Pending',expected_episodes=n,checkpoint_sha256=file_sha256(formal),result_path=str(folder),oracle_diagnostic=arm.startswith('oracle'))
            if ok:
                rr=rows(folder/'predictions.jsonl');assert len(rr)==n
                mm=aggregate([x['metrics'] for x in rr]);r.update(episodes=n,**{k:mm[k] for k in ('SR','SPL','OSR','NE','False_Stop_Rate')})
                control_data[split,arm]=rr
                failures=[x for x in rr if not x['metrics']['SR']]
                r['failed_episodes']=len(failures)
                r['failed_with_visited_success_region']=sum(x['metrics']['OSR']>0 for x in failures)
                r['failed_without_visited_success_region']=sum(x['metrics']['OSR']==0 for x in failures)
                r['policy_stop_episodes']=sum(x['termination']=='policy_stop' for x in rr)
                r['stagnation_episodes']=sum(x['termination']=='stagnation' for x in rr)
                r['horizon_episodes']=sum(x['termination']=='horizon' for x in rr)
            control.append(r);manifest.append(r.copy())
        for left,right in [('predicted_waypoint','predicted_direct'),('predicted_guarded_blend','predicted_waypoint'),('predicted_guarded_blend','predicted_direct'),('oracle_direct','predicted_direct'),('oracle_guarded_blend','predicted_guarded_blend'),('predicted_guarded_blend','rule_arrival')]:
            if all((split,a) in control_data for a in (left,right)):
                d,m,ids=paired_rows(control_data[split,left],control_data[split,right],'navigation',['SR','SPL'])
                stats[f'controller/{split}/{left}-minus-{right}']=paired_ci(d,m,['SR','SPL'])
    csvwrite('controller_ablation_results.csv',control)
    atomic_json(stats,OUT/'paired_statistics.json')
    atomic_json(dict(model='GeomBelief',parent_commit='34d2974',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol= json.loads((OUT/'frozen_plan.json').read_text()),experiments=manifest,completed_rows=sum(r['status']=='Done' for r in manifest),
        source_sha256={str(p.relative_to(HERE)):file_sha256(p) for p in [*HERE.glob('scripts/geombelief*.py'),*HERE.glob('graphnav/geombelief*.py')]},
        bootstrap='10000 paired draws seed20261017; seed-mean deltas, not pooled seeds as independent maps'),OUT/'experiment_manifest.json')
    report='# GeomBelief: Language-Grounded Map Geometry\n\n'+(OUT/'MAP_GEOMETRY_INPUT_AUDIT.md').read_text().split('\n',1)[1]
    report+='\n## Per-seed results\n\n'+table(result,['endpoint','split','arm','seed','status','R@1','R@4','R@8','SR','SPL','OSR','NE','False_Stop_Rate'])
    report+='\n## Three-seed means and sample standard deviations\n\n'+table(means,['endpoint','split','arm','R@1_mean','R@1_std','SR_mean','SR_std','SPL_mean','SPL_std','NE_mean'])
    cis=[]
    for key,v in stats.items():
        if '/seed_mean' not in key:continue
        for name,s in v['metrics'].items():cis.append(dict(comparison=key,metric=name,delta_pp=100*s['delta'],map_CI95_pp=[100*x for x in s['map_cluster_episode_weighted_ci95']],maps=v['independent_maps']))
    report+='\n## Paired map-cluster intervals\n\n'+table(cis,['comparison','metric','delta_pp','map_CI95_pp','maps'])
    report+='\nThe three predeclared primary contrasts are Correct−Name/Type, Correct−Shuffle and Distance+Direction−Name/Type. Other simple-geometry contrasts and R@4/R@8 are secondary. Rates are fractions in per-seed tables; effects in the CI table are percentage points.\n'
    report+='\nConfidence intervals are pointwise, not multiplicity-adjusted. Episode bootstrap, equal-map estimates, every map and leave-one-map-out are in paired_statistics.json. Four Unseen maps are map blocks, not four independent cities. No missing or partial result is treated as zero.\n'
    report+='\n## Conclusions\n\n'
    for left,right in [('correct','name_type'),('correct','shuffled'),('distance_direction','name_type')]:
        for endpoint,metric in [('localization','R@1'),('navigation','SR')]:
            key=f'{endpoint}/val_unseen/{left}-minus-{right}/seed_mean'
            if key in stats:
                s=stats[key]['metrics'][metric];lo,hi=s['map_cluster_episode_weighted_ci95'];support='positive map-cluster support' if lo>0 else 'negative map-cluster support' if hi<0 else 'interval crosses zero; no clear map-level support'
                report+=f'- {left} − {right}, Unseen {metric}: {s["delta"]*100:.3f} pp, 95% CI [{lo*100:.3f}, {hi*100:.3f}]; {support}.\n'
            else:report+=f'- {left} − {right}, Unseen {metric}: Pending complete three-seed paired results.\n'
    report+='\nCorrect-minus-Name tests all map geometry together, not uniquely language–geometry interaction. A language-shuffle factorial would be needed to isolate that interaction; none is claimed here. Fixed-head transfer and possibly memorized Seen map/name priors limit causal interpretation of navigation.\n'
    (OUT/'MAP_GEOMETRY_ABLATION_REPORT.md').write_text(report)
    control_report=(OUT/'CONTROL_ATTRIBUTION_REPORT.md').read_text().split('\n## Results')[0]
    control_report+='\n## Results\n\n'+table(control,['arm','split','status','episodes','SR','SPL','OSR','NE','False_Stop_Rate','failed_without_visited_success_region','failed_with_visited_success_region'])
    control_report+='\nFull-split paired controller intervals are in paired_statistics.json. Oracle gains measure recoverable error under the specified intervention; they are not an additive percentage of failure uniquely caused by localization. OSR−SR identifies visited-but-not-retained success, not automatically a Stop error. Learned Stop versus rule uses the same 0.001 m gate.\n'
    (OUT/'CONTROL_ATTRIBUTION_REPORT.md').write_text(control_report)
    for dest in (HERE/'docs/PAPER_EXPERIMENT_TODO.md',HERE/'reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md'):
        text=dest.read_text().split('\n<!-- GEOMBELIEF_REMAINING -->')[0]
        text+='\n<!-- GEOMBELIEF_REMAINING -->\n\n## GeomBelief independent contribution experiments\n\n'
        text+=f'Archived Polygon baseline 34d2974 retained. New complete metric rows: {sum(r["status"]=="Done" for r in manifest)}/{len(manifest)} (includes reused controls). '
        target='../reports/paper_experiments/geombelief_remaining/MAP_GEOMETRY_ABLATION_REPORT.md' if dest.parent.name=='docs' else 'geombelief_remaining/MAP_GEOMETRY_ABLATION_REPORT.md'
        text+=f'See [map geometry report]({target}) and reports/paper_experiments/geombelief_remaining/experiment_manifest.json for exact paths, SHA, missing experiments and paired statistics. P1 fair visual-input retraining remains Pending until explicitly completed; existing Real/Zero/Shuffle/No-belief results are reused.\n'
        dest.write_text(text)

if __name__=='__main__':summarize()
