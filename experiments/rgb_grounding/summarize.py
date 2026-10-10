"""Archive small, reproducible evidence without copying weights or datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from .evaluate_features import load
from experiments.bidirectional_core.stats import paired_difference
from experiments.bidirectional_core.cache import jsonl
from graphnav.checkpointing import file_sha256


def read(path):return json.loads(Path(path).read_text())


def population(root):
    rows=load(root);forward=[r for r in rows if r['metadata']['direction']=='forward'];counts={}
    for name,group in [('all',rows),('forward',forward),('reverse',[r for r in rows if r['metadata']['direction']=='reverse'])]:
        positives={(r['metadata']['map_name'],o['object_id']) for r in group for o,p in zip(r['metadata']['roi_objects'],r['labels']['positive_roi_mask']) if p}
        counts[name]=dict(states=len(group),families=len({r['metadata']['family_id'] for r in group}),maps=len({r['metadata']['map_name'] for r in group}),unique_positive_objects=len(positives),candidate_rois=sum(len(r['metadata']['roi_objects']) for r in group),human_verified_attribute_labels=0)
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True);args=p.parse_args();runs=Path(args.runs);assets=Path(args.assets);output=Path(args.output)
    if output.exists():raise ValueError('refusing aggregate overwrite')
    output.mkdir(parents=True)
    sources={}
    names=['semantic_seen_frozen','external_aerial_comparison','external_refseg_comparison','runtime_verified','stage1/representations_dev','stage1/representations_seen_frozen','stage1/georsclip_seen_frozen','stage1/skyclip_seen_frozen','stage1/fusion_with_causal_history','goal_dev_highgain_aerial_city_adapt']
    for name in names:sources[name]=read(runs/name/'metrics.json')
    for path in (runs/'goal_seen_frozen').glob('*/metrics.json'):sources['goal_seen_frozen/'+path.parent.name]=read(path)
    for path in (runs/'foundation_goal_seen').glob('*/metrics.json'):sources['foundation_goal_seen/'+path.parent.name]=read(path)
    audit=dict(city_train=population(runs/'features_city_train'),city_dev=population(runs/'features_city_dev'),city_seen=population(runs/'features_city_seen_judged'),aerial_split=read(runs/'aerial_pilot/split_audit.json'),dior=read(runs/'external_dior_audit.json'),leakage=read(runs/'external_leakage_audit/audit.json'),city_internal_dev_split=read(runs/'data_internal_dev/selection.json'))
    comparison=[];sem=sources['semantic_seen_frozen'];native=jsonl(runs/'semantic_seen_frozen/native_real_predictions.jsonl');paired={}
    selection=read(runs/'foundation_selection_frozen.json')
    for name,metrics in sem['models'].items():
        comparison.append(dict(model=name,metric='weak_region_r1',n=metrics['forward_r1']['n'],mean=metrics['forward_r1']['mean'],map_ci_low=metrics['forward_r1']['map_cluster_95ci'][0],map_ci_high=metrics['forward_r1']['map_cluster_95ci'][1]))
    for name in ['georsclip','skyclip']:
        method=selection['models'][name]['method'];d=sources['stage1/'+name+'_seen_frozen'];m=d['models'][method]['forward_r1'];preds=jsonl(runs/'stage1'/(name+'_seen_frozen')/(method+'_predictions.jsonl'));paired[name]=paired_difference(native,preds,'forward_r1',repetitions=2000);comparison.append(dict(model=name+'_'+method,metric='weak_region_r1',n=m['n'],mean=m['mean'],map_ci_low=m['map_cluster_95ci'][0],map_ci_high=m['map_cluster_95ci'][1]))
    for name,d in sources.items():
        if name.startswith(('goal_seen_frozen/','foundation_goal_seen/')):
            for control,m in d['models'].items():
                s=m['forward_r1'];comparison.append(dict(model=name+'/'+control,metric='goal_r1_xy20_forward',n=s['n'],mean=s['mean'],map_ci_low=s['map_cluster_95ci'][0],map_ci_high=s['map_cluster_95ci'][1]))
    with (output/'comparison.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=['model','metric','n','mean','map_ci_low','map_ci_high']);writer.writeheader();writer.writerows(comparison)
    trained={path.parent.name:read(path) for path in (runs/'heads').glob('*/training_complete.json')}
    downloads=[]
    for path in assets.rglob('*.provenance.json'):
        d=read(path)
        if d.get('success'):downloads.append({k:d[k] for k in ['name','url','revision','bytes','sha256','seconds','actual_proxy'] if k in d})
    manifest=dict(training=trained,downloads=downloads,source_audit_sha256={p.name:file_sha256(p) for p in (runs/'source_audit').glob('*.json')},results_sha256={name:file_sha256(runs/name/'metrics.json') for name in sources},frozen_goal_config=read(runs/'calibration_dev_confidence_complete/frozen_config.json'),foundation_selection=selection,source_main_sha='0c2584421f812821791091f734371380b759c24d',branch='research/rgb-language-grounding-20261010',backbones_frozen=True,unseen_evaluations=0,new_closedloop_runs=0,stop_reason='no fixed Seen model passes Goal R1 map-bootstrap lower>0, Net Recovery>0, real>shuffled gate')
    # Image provenance for 480 files stays local; report compact aggregate.
    image_downloads=[d for d in downloads if d['name'].startswith('aerialvg/images/')]
    manifest['aerial_downloads_summary']=dict(files=len(image_downloads),bytes=sum(d['bytes'] for d in image_downloads),individual_provenance_local=str(assets/'aerialvg/images'))
    manifest['downloads']=[d for d in downloads if not d['name'].startswith('aerialvg/images/')]
    (output/'data_audit.json').write_text(json.dumps(audit,indent=2));(output/'model_manifest.json').write_text(json.dumps(manifest,indent=2));(output/'metrics.json').write_text(json.dumps(dict(sources=sources,paired_foundations_vs_native=paired),indent=2))
    # Small complete matching predictions, CityNav only; no gated captions.
    prediction_dir=output/'city_predictions';prediction_dir.mkdir()
    for path in (runs/'semantic_seen_frozen').glob('*_predictions.jsonl'):(prediction_dir/path.name).write_bytes(path.read_bytes())
    print(json.dumps(dict(results=len(sources),training_configs=len(trained),comparison_rows=len(comparison),audit=audit['city_seen'],stop_reason=manifest['stop_reason']),indent=2))


if __name__=='__main__':main()
