"""Offline episode subgroups and honest timing-boundary comparison."""
from common import *
import sys,re
import numpy as np

def bucket(value,edges):
    for low,high,label in edges:
        if low<=value<high:return label
    raise ValueError(value)

def main():
    sys.path.insert(0,str(ARCHIVE))
    from graphnav.workspace_paths import current_paths
    args=current_paths(read(MAIN/'runs/graphnav_v1_stage1/full_seen/metrics.json')['args']);sys.path.insert(0,args['sbf_source'])
    from sbfnav.config import load_config
    from sbfnav.dataset import CityNavDataset
    cfg=load_config(args['config'])
    distances=[(0,20,'[0,20)'),(20,50,'[20,50)'),(50,100,'[50,100)'),(100,200,'[100,200)'),(200,float('inf'),'[200,inf)')]
    errors=[(0,20,'[0,20)'),(20,50,'[20,50)'),(50,100,'[50,100)'),(100,float('inf'),'[100,inf)')]
    patterns={'distance':r'\b(near|nearby|close|beside|besides|next to|adjacent|far|nearest|closest)\b',
              'direction':r'\b(left|right|north|south|east|west|front|behind|back|opposite|across)\b',
              'boundary':r'\b(edge|corner|end|intersection|boundary|along)\b',
              'between':r'\bbetween\b','count_or_order':r'\b(one|two|three|four|five|first|second|third|last|row|group)\b'}
    rows=[];checks=[]
    for split,stem in [('val_seen','full_seen'),('val_unseen','full_unseen'),('test_unseen','full_test_unseen')]:
        # Explicit user-authorized offline audit of already-frozen Test outputs;
        # no policy execution, parameter tuning, or new checkpoint selection.
        assert sha(MAIN/'runs/graphnav_v1_stage1/best.pt')=='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a'
        dataset=CityNavDataset(args['data_root'],split,version=cfg['data']['version'],allow_test_unseen=split=='test_unseen');path=MAIN/'runs/graphnav_v1_stage1'/stem/'predictions.jsonl';traces=list(jsonl(path));assert len(traces)==len(dataset)
        recomputed=[]
        for trace in traces:
            r=dataset[trace['index']];assert list(r.episode_id)==trace['episode_id'];gt=np.asarray(r.target_position)[:2];poses=np.asarray(trace['poses'])[:,:2];diag=trace['step_diagnostics'][0]
            start=float(np.linalg.norm(poses[0]-gt));ne=float(np.linalg.norm(poses[-1]-gt));pathlength=float(np.linalg.norm(np.diff(poses,axis=0),axis=1).sum());sr=int(ne<=20);osr=int(np.linalg.norm(poses-gt,axis=1).min()<=20);spl=sr*start/max(pathlength,start,1e-8)
            values=dict(NE=ne,SR=sr,OSR=osr,SPL=spl)
            for k,v in values.items():assert abs(v-trace['metrics'][k])<1e-5,(split,trace['index'],k,v,trace['metrics'][k])
            recomputed.append(values)
            static=np.asarray(diag['static_top16_world_xy'][0]);belief=np.asarray(diag['belief_top16_world_xy'][0]);e0=float(np.linalg.norm(static-gt));eb=float(np.linalg.norm(belief-gt))
            matches=[name for name,pattern in patterns.items() if re.search(pattern,r.instruction,re.I)]
            rows.append(dict(split=split,index=trace['index'],episode_id=':'.join(map(str,r.episode_id)),map_name=r.map_name,start_distance_m=start,start_distance_bin=bucket(start,distances),initial_static_error_m=e0,initial_static_error_bin=bucket(e0,errors),initial_static_R1=int(e0<=20),initial_observed_R1=int(eb<=20),relation_tags=';'.join(matches) or 'no_regex_match',**values))
        formal=read(path.parent/'metrics.json')['metrics'];result={k:float(np.mean([r[k] for r in recomputed])) for k in ('NE','SR','OSR','SPL')}
        for k,v in result.items():assert abs(v-formal[k])<1e-6,(split,k)
        checks.append(dict(split=split,episodes=len(traces),source=str(path),source_sha256=sha(path),aggregate_matches_formal=True,**result))
    # Exact saved weak-projection visibility labels are available for full Seen only.
    import torch
    cache=PROJECT/'Ablations/GraphNav/phrase-mask-20261018/phrase_mask_results/seen_cache';manifest=read(cache/'manifest.json');visibility={}
    for shard in manifest['shards']:
        data=torch.load(cache/shard['file'],map_location='cpu',weights_only=True,mmap=True)
        for meta,presence in zip(data['metadata'],data['labels']['presence'].tolist()):
            visibility[meta['episode_index']]={1:'strict_visible',0:'outside',-1:'uncertain'}[int(presence)]
        del data
    assert len(visibility)==2470
    for r in rows:r['visibility']=visibility[r['index']] if r['split']=='val_seen' else 'not_evaluated'
    csvwrite(OUT/'episode_statistics.csv',rows);groups=[]
    distance_summary=[]
    for split in ('val_seen','val_unseen','test_unseen'):
        values=np.asarray([r['start_distance_m'] for r in rows if r['split']==split])
        distance_summary.append(dict(split=split,episodes=len(values),mean_m=float(values.mean()),median_m=float(np.median(values)),min_m=float(values.min()),max_m=float(values.max()),q25_m=float(np.quantile(values,.25)),q75_m=float(np.quantile(values,.75))))
    csvwrite(OUT/'start_distance_summary.csv',distance_summary)
    for split in ('val_seen','val_unseen','test_unseen'):
        subset=[r for r in rows if r['split']==split];total=len(subset)
        definitions=[('start_distance_bin',r[2]) for r in distances]+[('initial_static_error_bin',r[2]) for r in errors]
        definitions += [('relation_tag',key) for key in [*patterns,'no_regex_match']]
        if split=='val_seen':definitions += [('visibility',k) for k in ('strict_visible','outside','uncertain')]
        for group,label in definitions:
            selected=[r for r in subset if label in r['relation_tags'].split(';')] if group=='relation_tag' else [r for r in subset if r[group]==label]
            n=len(selected)
            groups.append(dict(split=split,group=group,label=label,episodes=n,fraction_of_all_split_percent=100*n/total,SR_percent=100*np.mean([r['SR'] for r in selected]) if n else '',initial_static_R1_percent=100*np.mean([r['initial_static_R1'] for r in selected]) if n else '',initial_observed_R1_percent=100*np.mean([r['initial_observed_R1'] for r in selected]) if n else '',NE_m=np.mean([r['NE'] for r in selected]) if n else '',scope='episode-weighted (not deduplicated query recall); regex relation groups overlap; visibility is saved weak-projection supervision, not human identity truth'))
    csvwrite(OUT/'subgroup_statistics.csv',groups);dump(OUT/'offline_metric_verification.json',checks)
    dump(OUT/'visibility_provenance.json',dict(manifest=str(cache/'manifest.json'),manifest_sha256=sha(cache/'manifest.json'),counts={k:sum(v==k for v in visibility.values()) for k in ('strict_visible','outside','uncertain')},definition='Reused fixed strict weak-projection/support labels from phrase-mask audit; only full Seen has these saved labels. No inference from task distance or heatmap color.'))
    # Preserve units and sample granularity: HETT samples are whole episodes.
    gp=PROJECT/'Ablations/GraphNav/profiling-val-unseen-20261016/profiling_val_unseen/full/profile.json';g=read(gp)
    hp=PROJECT/'Ablations/HETT/hett-main-system-opt/artifacts/profiling/profile_val_unseen_full.json';h=read(hp)
    ep=gp.parents[1]/'profile_val_unseen_full.json';ge=read(ep)
    launch_path=gp.parents[1]/'launch_record.json';launch=read(launch_path)
    assert h['device'] in launch['device']
    timings=[]
    for population,values in [('all_decisions',g['all']),('excluding_first_decision',g['excluding_first'])]:
        for component,v in values.items():timings.append(dict(method='GeomBelief frozen C',source=str(gp),source_sha256=sha(gp),population=population,granularity='per decision',component=component,mean_ms=v['mean'],min_ms=v['min'],variance_ms2=v['population_variance'],hardware=h['device']))
    components={'rollout':h['rollout_seconds'],'environment':h['environment_observation_seconds'],'other':h['other_rollout_seconds'],**h['model_gpu_seconds_by_component']}
    for component,v in components.items():timings.append(dict(method='HETT historical local teacherfix epoch9',source=str(hp),source_sha256=sha(hp),population='all_episodes',granularity='per episode',component=component,mean_ms=v['mean']*1000,min_ms=v['min']*1000,variance_ms2=v['variance']*1e6,hardware=h['device']))
    for method,source,data in [('GeomBelief frozen C',ep,ge)]:
        parts={'rollout':data['rollout_seconds'],'environment':data['environment_observation_seconds'],'other':data['other_rollout_seconds'],**data['model_gpu_seconds_by_component']}
        for component,v in parts.items():timings.append(dict(method=method,source=str(source),source_sha256=sha(source),population='all_episodes',granularity='per episode (HETT-style module spans)',component=component,mean_ms=v['mean']*1000,min_ms=v['min']*1000,variance_ms2=v['variance']*1e6,hardware=h['device']))
    for method,source,data in [('GeomBelief frozen C',ep,ge),('HETT historical local teacherfix epoch9',hp,h)]:
        first=data['samples'][0]
        for key in ('rollout_seconds','model_inference_gpu_seconds','environment_observation_seconds'):
            timings.append(dict(method=method,source=str(source),source_sha256=sha(source),population='first_episode_warmup',granularity='per episode; excludes model loading',component=key,mean_ms=first[key]*1000,min_ms=first[key]*1000,variance_ms2='',hardware=h['device']))
    common_ids=set(r['sample_id'] for r in ge['samples'])&set(r['sample_id'] for r in h['samples'])
    csvwrite(OUT/'profiling_comparison.csv',timings)
    dump(OUT/'profiling_scope.json',dict(GeomBelief_source=str(gp),GeomBelief_whole_episode_source=str(ep),GeomBelief_whole_episode_sha256=sha(ep),GeomBelief_launch_record=str(launch_path),GeomBelief_launch_sha256=sha(launch_path),GeomBelief_first_episode=ge['samples'][0],common_episode_ids=len(common_ids),GeomBelief_environment=g.get('environment'),GeomBelief_timing=g.get('timing'),HETT_source=str(hp),HETT_checkpoint=h['checkpoint'],HETT_checkpoint_epoch=h['checkpoint_epoch'],HETT_samples=h['sample_count'],HETT_first_sample=h['samples'][0],HETT_excluding_first=h['rollout_seconds_excluding_first_sample'],strictly_matched=False,
         reason='Same GPU model and compatible whole-episode module-span reports exist; per-decision GeomBelief remains a separate scope. Model/checkpoint identities, native controls/step counts and run conditions differ. Historical HETT is a locally trained teacherfix model, not the official published weight. Geometric subcomponent is not isolated in HETT. No official controlled speedup ratio is justified.',
         misleading_directory_probe=dict(path=str(PROJECT/'Ablations/GraphNav/diagnostics_20261013/profile_hett/metrics.json'),actual_model=read(PROJECT/'Ablations/GraphNav/diagnostics_20261013/profile_hett/metrics.json')['model'],decision='Excluded: directory name is not evidence of HETT model identity.')))
    lines=['# Offline subgroups and profiling','',f'All {len(rows)} archived formal episodes were rescored in XY; per-episode and aggregate NE/SR/OSR/SPL match original records. No navigation replay, checkpoint selection or training was run.','',
        'Subgroup SR is actual final navigation success. R@1 columns are initial, episode-weighted localization; they are not deduplicated-query R@1 or closed-loop success. Distance bins are half-open and cover the entire split. Relation tags are lexical, overlap, and do not establish correct language attachment or a geographic reference frame.','',
        'Saved strict projection visibility is reused for Seen2470: 332 visible / 1860 outside / 278 uncertain. Unseen/Test visibility remains not evaluated; no guessed visibility labels were created. Weak projected polygon identity is not human verification.','',
        f'Same GPU model whole-episode evidence exists for both methods, with {len(common_ids)} common IDs. GeomBelief mean rollout/model time is {ge["rollout_seconds"]["mean"]*1000:.2f}/{ge["model_gpu_seconds_by_component"]["model_inference_gpu_seconds"]["mean"]*1000:.2f}ms per episode; historical HETT local teacherfix epoch9 is {h["rollout_seconds"]["mean"]*1000:.2f}/{h["model_gpu_seconds_by_component"]["model_inference_gpu_seconds"]["mean"]*1000:.2f}ms. Native policy/step counts and training identity differ; these are not official controlled speedup estimates. The separate 22,144 GeomBelief decision timings are not mixed into per-episode values. Model/vision/environment and first-episode warmup are separated; model-load startup is not included. A folder named profile_hett actually identifies its model as GraphNav-V1 and is explicitly excluded from HETT evidence.']
    (OUT/'SUPPLEMENTARY_ANALYSIS.md').write_text('\n'.join(lines)+'\n');print('rescored episodes',len(rows),'subgroups',len(groups),'timing rows',len(timings),flush=True)

if __name__=='__main__':main()
