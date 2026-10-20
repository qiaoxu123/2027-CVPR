"""Select illustrative real validation cases; no checkpoint/parameter selection."""
from common import *
import sys
import numpy as np

def main():
    sys.path.insert(0,str(ARCHIVE))
    from graphnav.workspace_paths import current_paths
    args=current_paths(read(MAIN/'runs/graphnav_v1_stage1/full_seen/metrics.json')['args'])
    sys.path.insert(0,args['sbf_source'])
    from sbfnav.config import load_config
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from graphnav.closedloop import PublicEpisode
    from graphnav.a3_runtime import public_static_record
    from sbfnav.planning_state import PlanningStateBuilder
    from graphnav.closedloop import PublicCatalog
    cfg=load_config(args['config']);catalog=CityReferCatalog(Path(args['data_root'])/'cityrefer')
    builder=PlanningStateBuilder(args['data_root'],PublicCatalog(catalog),canvas_size=cfg['model']['canvas_size'],field_size=28,max_height_above_ground_m=cfg['state']['max_height_above_ground_m'],trajectory_radius_m=cfg['state']['trajectory_radius_m'])
    chosen=[];counts={}
    for split,stem in [('val_seen','full_seen'),('val_unseen','full_unseen')]:
        dataset=CityNavDataset(args['data_root'],split,version=cfg['data']['version'])
        source=MAIN/'runs/graphnav_v1_stage1'/stem/'predictions.jsonl'
        rows=list(jsonl(source));counts[split]={}
        selectors={
            'success':lambda r:r['metrics']['SR']==1,
            'wrong_goal_false_stop':lambda r:r['metrics']['false_stop']==1 and r['metrics']['OSR']==0,
            'visited_but_failed':lambda r:r['metrics']['OSR']==1 and r['metrics']['SR']==0,
        }
        for label,predicate in selectors.items():
            eligible=[r for r in rows if predicate(r)];counts[split][label]=len(eligible)
            if eligible: chosen.append((f'formal_{split}_{label}',split,source,eligible[0],'formal',dataset))
    # Actual belief winners, not merely success differences on diverged paths.
    split='val_unseen';dataset=CityNavDataset(args['data_root'],split,version=cfg['data']['version'])
    source=ARCHIVE/'visual_evaluations/calibrated_residual_full_unseen_real/predictions.jsonl'
    rows=list(jsonl(source));groups={k:[] for k in ('corrected','reinforced','degraded','insufficient')}
    for row in rows:
        target=np.asarray(dataset[row['index']].target_position)[:2]
        ds=row['step_diagnostics']
        e0=np.linalg.norm(np.asarray(ds[0]['static_top16_world_xy'][0])-target)
        errors=[np.linalg.norm(np.asarray(d['belief_top16_world_xy'][0])-target) for d in ds]
        if e0>20 and errors[-1]<=20:groups['corrected'].append(row)
        if e0<=20 and errors[-1]<=20 and row['metrics']['SR']==1:groups['reinforced'].append(row)
        if e0<=20 and errors[-1]>20:groups['degraded'].append(row)
        if e0>20 and errors[-1]>20:groups['insufficient'].append(row)
    counts['calibrated_visual_val_unseen']={k:len(v) for k,v in groups.items()}
    for label in ('corrected','reinforced','degraded'):
        if groups[label]:chosen.append((f'visual_{label}',split,source,groups[label][0],'calibrated_residual',dataset))
    output=[]
    for name,split,path,row,model,dataset in chosen:
        record=dataset[row['index']]
        assert list(record.episode_id)==row['episode_id']
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        static=public_static_record(public,builder.transform(public.map_name),catalog,max_landmarks=16)
        metrics=read(path.parent/'metrics.json')
        cp=MAIN/'runs/graphnav_v1_stage1/best.pt'
        residual=ARCHIVE/'calibrated_residual/best.pt' if model=='calibrated_residual' else None
        data_path=WORKSPACE/'DATA/hett/release/refined_citynav/processed_citynav'/f'citynav_{split}.json'
        item=dict(case_id=name,split=split,index=row['index'],episode_id=row['episode_id'],map_name=row['map_name'],instruction=record.instruction,
                  target_xyz=np.asarray(record.target_position).tolist(),public_map=static,trace=row,source_predictions=str(path),source_predictions_sha256=sha(path),
                  source_metrics=str(path.parent/'metrics.json'),source_metrics_sha256=sha(path.parent/'metrics.json'),
                  checkpoint=str(cp),checkpoint_sha256=sha(cp),residual_checkpoint=str(residual) if residual else None,residual_checkpoint_sha256=sha(residual) if residual else None,
                  dataset_version=cfg['data']['version'],dataset_path=str(data_path),dataset_sha256=sha(data_path),model_identity=model,
                  selection='First qualifying episode in immutable source order, per declared outcome. Illustrative outcome-selected case; not an unbiased estimate of frequency or a checkpoint selection.',
                  image_use='Real CityNav orthophoto crops at executed poses. Local-only RGB previews until underlying SensatUrban image redistribution terms are confirmed; published derived figures omit RGB pixels.')
        output.append(item)
    with (OUT/'cases.jsonl').open('w') as f:
        for r in output:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    dump(OUT/'case_selection_counts.json',counts)
    print(json.dumps(dict(cases=[(r['case_id'],r['episode_id']) for r in output],counts=counts)))

if __name__=='__main__':main()
