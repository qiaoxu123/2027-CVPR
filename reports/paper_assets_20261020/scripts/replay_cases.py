"""Frozen inference at archived observed poses only; never rerun navigation/train.

GT is not supplied to the policy. Matched-path controls are explicitly distinct
from the controls' independent closed-loop trajectories.
"""
from common import *
import sys,os
import numpy as np

def prob(x):
    x=np.asarray(x,dtype=np.float64);v=np.exp(x-x.max());return v/v.sum()

def change(p,q,xy):
    return dict(belief_L1=float(np.abs(p-q).sum()),KL_p_to_q_nats=float(np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(q,1e-300)))),
                top1_shift_m=float(np.linalg.norm(xy[p.argmax()]-xy[q.argmax()])),top1_cell_changed=int(p.argmax()!=q.argmax()))

def main():
    import torch
    sys.path.insert(0,str(ARCHIVE))
    from graphnav.workspace_paths import current_paths
    from graphnav.stage1_runtime import GraphNavStage1Policy
    from graphnav.calibrated_residual_runtime import install_calibrated_residual
    from graphnav.closedloop import PublicEpisode,PublicCatalog
    from graphnav.dynamic_runtime import NativeObservationBuilder
    args=current_paths(read(MAIN/'runs/graphnav_v1_stage1/full_seen/metrics.json')['args'])
    sys.path.insert(0,args['sbf_source'])
    from sbfnav.config import load_config
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    cfg=load_config(args['config']);catalog=CityReferCatalog(Path(args['data_root'])/'cityrefer')
    datasets={s:CityNavDataset(args['data_root'],s,version=cfg['data']['version']) for s in ('val_seen','val_unseen')}
    base=PlanningStateBuilder(args['data_root'],PublicCatalog(catalog),canvas_size=cfg['model']['canvas_size'],field_size=cfg['model']['field_size'],max_height_above_ground_m=cfg['state']['max_height_above_ground_m'],trajectory_radius_m=cfg['state']['trajectory_radius_m'])
    builder=NativeObservationBuilder(base,Path(args['data_root'])/'rgbd',OUT/'local_only/render_scratch')
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    destination=OUT/'fields';destination.mkdir(exist_ok=True)
    local=OUT/'local_only/observations';local.mkdir(parents=True,exist_ok=True)
    policy=None;kind=None;manifest=[];stats=[]
    cases=list(jsonl(OUT/'cases.jsonl'))
    for case in cases:
        name=case['case_id'];cache=destination/f'{name}.npz';meta=destination/f'{name}.json'
        if cache.exists() and meta.exists():
            item=read(meta);assert item['source_predictions_sha256']==case['source_predictions_sha256']
            manifest.append(item);stats.extend(item['statistics']);print('reuse',name,flush=True);continue
        if kind!=case['model_identity']:
            if policy is not None:del policy;torch.cuda.empty_cache()
            kind=case['model_identity']
            policy=GraphNavStage1Policy(case['checkpoint'],OUT/'local_only/public_inputs')
            if case['residual_checkpoint']:install_calibrated_residual(policy,case['residual_checkpoint'],case['checkpoint'])
        record=datasets[case['split']][case['index']]
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        captured=[]
        def hook(module,inputs,result):
            captured.append({k:v[0].detach().float().cpu().numpy().copy() for k,v in [('static',result.static.belief_logits),('belief',result.belief.belief_logits),('residual',result.delta),('observed',result.observed)]})
        handle=policy.model.register_forward_hook(hook)
        poses=np.asarray(case['trace']['poses']);frames=[];allfields={};checks=[]
        controls=('real',) if kind=='formal' else ('real','zero','shuffle','no-belief')
        for control in controls:
            policy.visual_control=control;policy.set_episode(public,base.transform(public.map_name),catalog)
            captured.clear()
            for i,diagnostic in enumerate(case['trace']['step_diagnostics']):
                step=diagnostic['observation_step'];state=builder.build_from_poses(public,poses[:step+1])
                prediction=policy.predict(state,public.instruction)
                if control=='real':
                    errors={k:float(np.max(np.abs(np.asarray(prediction.diagnostics[k])-np.asarray(diagnostic[k])))) for k in ('static_top16_world_xy','belief_top16_world_xy','selected_goal_xy')}
                    errors['stop_probability']=abs(prediction.diagnostics['stop_probability']-diagnostic['stop_probability'])
                    checks.append(dict(step=step,**errors))
                    if any(v> (1e-6 if k=='stop_probability' else 1e-4) for k,v in errors.items()):
                        raise AssertionError(f'Replay mismatch {name} step {step}: {errors}; original sources remain untouched')
                    obs=state.current_observation
                    frames.append(dict(step=step,pose=poses[step].tolist(),footprint_world=np.asarray(obs['footprint_world']).tolist()))
                    np.savez_compressed(local/f'{name}_step{step}.npz',bgr=obs['bgr'],footprint_world=obs['footprint_world'],depth_valid=obs['depth_valid'])
            allfields[control]={k:np.stack([s[k] for s in captured]) for k in captured[0]}
        handle.remove()
        xy=policy.world[0].float().cpu().numpy();target=np.asarray(case['target_xyz'])[:2];statistics=[]
        for control,v in allfields.items():
            previous=None
            for i,(s,b) in enumerate(zip(v['static'],v['belief'])):
                ps,pb=prob(s),prob(b)
                for comparison,ref in [('static_to_observed',ps)]+([('previous_to_current',previous)] if previous is not None else []):
                    entry=dict(case_id=name,split=case['split'],episode_id=':'.join(map(str,case['episode_id'])),model_identity=kind,control=control,step=frames[i]['step'],comparison=comparison,
                               **change(ref,pb,xy),reference_top1_error_m=float(np.linalg.norm(xy[ref.argmax()]-target)),observed_top1_error_m=float(np.linalg.norm(xy[pb.argmax()]-target)),
                               residual_max_abs=float(np.abs(v['residual'][i]).max()),source_predictions_sha256=case['source_predictions_sha256'],
                               scope='archived real executed pose prefix; controls are matched-path offline interventions, not independent rollouts')
                    statistics.append(entry)
                previous=pb
        np.savez_compressed(cache,world_xy=xy,poses=poses,target_xyz=case['target_xyz'],**{f'{control}_{key}':v for control,fields in allfields.items() for key,v in fields.items()})
        item=dict(case_id=name,checkpoint_sha256=case['checkpoint_sha256'],residual_checkpoint_sha256=case['residual_checkpoint_sha256'],source_predictions_sha256=case['source_predictions_sha256'],
                  dataset_sha256=case['dataset_sha256'],source_code={str(p.relative_to(ARCHIVE)):sha(p) for p in [ARCHIVE/'graphnav/stage1_runtime.py',ARCHIVE/'graphnav/stage1.py',ARCHIVE/'graphnav/calibrated_residual_runtime.py',ARCHIVE/'graphnav/renderer.py']},
                  frames=frames,replay_checks=checks,replay_matches_saved_diagnostics=True,arrays=str(cache.relative_to(OUT)),arrays_sha256=sha(cache),statistics=statistics)
        dump(meta,item);manifest.append(item);stats.extend(statistics)
        print(json.dumps(dict(case=name,steps=len(frames),controls=controls,replay_parity=True)),flush=True)
    csvwrite(OUT/'belief_change_statistics.csv',stats)
    dump(OUT/'replay_manifest.json',manifest)

if __name__=='__main__':main()
