"""Evaluate SBFNav with the shared, GT-free rollout and immutable episode manifests."""
import argparse
from collections import defaultdict, Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from graphnav.closedloop import (ClosedLoopProtocol, PublicEpisode, PublicCatalog,
                                aggregate, rollout, score_trace)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sbf-source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', choices=['val_seen', 'val_unseen'], required=True)
    parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['sbfnav','a3','a3-trajectory','dynamic'], default='sbfnav')
    parser.add_argument('--dynamic-checkpoint')
    parser.add_argument('--visual-checkpoint')
    parser.add_argument('--visual-history',action='store_true')
    parser.add_argument('--disable-visual',action='store_true')
    parser.add_argument('--observed-cache')
    parser.add_argument('--head-checkpoint')
    parser.add_argument('--stop-threshold',type=float,default=.5)
    parser.add_argument('--head-execution',choices=['joint','fixed-goal','fixed-goal-no-stop','direct-goal','guarded-path','guarded-blend','guarded-direct'],default='joint')
    args = parser.parse_args()
    started_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    started_diff=subprocess.check_output(['git','diff','HEAD','--binary'])
    evaluator_sha256=sha(__file__)
    output = Path(args.output)
    if output.exists(): raise ValueError('refusing to overwrite evaluation')
    output.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.sbf_source).resolve()))
    from sbfnav.config import load_config, apply_overrides, validate_config
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from sbfnav.navigation import BoundedWaypointController
    from sbfnav.model_factory import build_backbone, build_model
    from sbfnav.checkpointing import load_checkpoint
    from sbfnav.runtime import SBFNavRuntime
    from sbfnav.mapdata import GROUND_LEVEL
    from sbfnav.validation import evenly_spaced_indices
    config = apply_overrides(load_config(args.config), ['data.root='+args.data_root])
    validate_config(config)
    if config['selector']['selection_mode'] != 'selector':
        raise ValueError('only trained non-oracle selector accepted')
    seed = config['experiment']['seed']
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    nav, altitude = config['navigation'], config['altitude']
    protocol = ClosedLoopProtocol(horizon=nav['horizon'],horizontal_step_m=nav['horizontal_step_m'],
        vertical_step_m=nav['vertical_step_m'],min_altitude_m=altitude['min_above_ground_m'],
        max_altitude_m=altitude['max_above_ground_m'],stagnation_steps=nav['stagnation_steps'],
        success_radius_m=nav['success_distance_m'])
    dataset = CityNavDataset(args.data_root,args.split,version=config['data']['version'])
    catalog = CityReferCatalog(Path(args.data_root)/'cityrefer')
    indices = evenly_spaced_indices(len(dataset), args.max_episodes)
    episodes = [dataset[i] for i in indices]
    manifest = [{'index':int(i),'episode_id':list(record.episode_id),
                 'start_pose':record.start_pose.tolist()} for i,record in zip(indices,episodes)]
    (output/'episodes.json').write_text(json.dumps(manifest,indent=2))
    state = config['state']
    builder = PlanningStateBuilder(args.data_root, PublicCatalog(catalog),
        canvas_size=config['model']['canvas_size'],field_size=config['model']['field_size'],
        max_height_above_ground_m=state['max_height_above_ground_m'],
        trajectory_radius_m=state['trajectory_radius_m'])
    if args.model=='sbfnav':
        backbone = build_backbone(config).cuda(); model = build_model(config).cuda()
        checkpoint = load_checkpoint(args.checkpoint,model=model,backbone=backbone,map_location='cuda')
        runtime = SBFNavRuntime(model,backbone,config,torch.device('cuda'))
    else:
        from graphnav.a3_runtime import A3WaypointPolicy,A3TrajectoryPolicy
        if args.model=='dynamic':
            if not args.dynamic_checkpoint or not args.visual_checkpoint:raise ValueError('dynamic and visual checkpoints required')
            from graphnav.dynamic_runtime import DynamicWaypointPolicy,NativeObservationBuilder
            runtime=DynamicWaypointPolicy(args.checkpoint,args.dynamic_checkpoint,args.visual_checkpoint,str(output/'public_inputs'),history=args.visual_history,visual=not args.disable_visual,observed_cache=args.observed_cache,head_checkpoint=args.head_checkpoint,stop_threshold=args.stop_threshold,execution=args.head_execution)
            builder=NativeObservationBuilder(builder,Path(args.data_root)/'rgbd',output/'observations')
        elif args.model=='a3-trajectory':
            if not args.head_checkpoint:raise ValueError('trained trajectory head checkpoint required')
            runtime=A3TrajectoryPolicy(args.checkpoint,args.head_checkpoint,output/'public_inputs',stop_threshold=args.stop_threshold,execution=args.head_execution)
        else:runtime=A3WaypointPolicy(args.checkpoint,output/'public_inputs')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    controller = BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,
        vertical_step_m=protocol.vertical_step_m,minimum_above_ground_m=protocol.min_altitude_m,
        maximum_above_ground_m=protocol.max_altitude_m)
    rows, maps, terminations = [], defaultdict(list), Counter()
    started = time.perf_counter()
    with (output/'predictions.jsonl').open('w') as stream:
        for position,record in enumerate(episodes):
            # IDs are used only to prepare supplied CityRefer language bindings and output labels.
            # Neither record, target, IDs, nor expert future enters the policy or rollout.
            public = PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),
                                   tuple(catalog.referenced_landmarks(record)),
                                   tuple(catalog.referenced_landmark_names(record)))
            if args.model!='sbfnav': runtime.set_episode(public,builder.transform(public.map_name),catalog)
            poses, termination, timings, diagnostics = rollout(public,builder,controller,runtime.predict,
                GROUND_LEVEL[record.map_name],protocol,synchronize=torch.cuda.synchronize)
            metrics = score_trace(poses,record.target_position,record.trajectory,termination,
                radius=protocol.success_radius_m,decision_seconds=timings)
            rows.append(metrics); maps[record.map_name].append(metrics); terminations[termination]+=1
            stream.write(json.dumps(dict(index=record.index,episode_id=record.episode_id,
                map_name=record.map_name,poses=poses.tolist(),termination=termination,
                metrics=metrics,decision_seconds=timings,step_diagnostics=diagnostics))+'\n'); stream.flush()
            if (position+1)%16==0:
                print(json.dumps(dict(completed=position+1,total=len(episodes),elapsed=time.perf_counter()-started)),flush=True)
    report = dict(model=('SBFNav local paper-based reproduction' if args.model=='sbfnav'
                        else ('A3 + causal dynamic visual/graph residual belief' if args.model=='dynamic' else ('A3 + trained trajectory/rank/Stop; geometry/pose only' if args.model=='a3-trajectory'
                              else 'A3 + bounded Top1 waypoint; no trained trajectory or stop'))),split=args.split,
        metrics=aggregate(rows),by_map={k:aggregate(v) for k,v in maps.items()},
        terminations=dict(terminations),protocol=asdict(protocol),config=config,
        checkpoint_sha256=sha(args.checkpoint),checkpoint_epoch=checkpoint.get('completed_epoch',checkpoint.get('epoch')),
        head_checkpoint_sha256=sha(args.head_checkpoint) if args.head_checkpoint else None,
        dynamic_checkpoint_sha256=sha(args.dynamic_checkpoint) if args.dynamic_checkpoint else None,
        visual_checkpoint_sha256=sha(args.visual_checkpoint) if args.visual_checkpoint else None,
        episode_manifest_sha256=sha(output/'episodes.json'),args=vars(args),seconds=time.perf_counter()-started,
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        input_sha256={str(p):sha(p) for p in [dataset.annotation_path,Path(args.data_root)/'cityrefer/objects.json',
            Path(args.data_root)/'cityrefer/processed_descriptions.json']},
        sbf_source_sha256={str(p.relative_to(args.sbf_source)):sha(p)
            for p in sorted(Path(args.sbf_source).glob('sbfnav/**/*.py'))},
        graphnav_commit=started_commit,evaluator_sha256=evaluator_sha256,
        graphnav_working_diff_sha256=hashlib.sha256(started_diff).hexdigest(),
        inference_target_access=False,
        warning=('Legacy SBF checkpoint selected by Val Unseen proxy; no new selection by Unseen.'
                 if args.model=='sbfnav' else 'Geometric waypoint execution baseline, not full GraphNav.'))
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['metrics']),flush=True)


if __name__ == '__main__': main()
