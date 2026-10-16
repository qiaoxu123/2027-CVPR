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


def validate_stage1_frozen_evaluation(args):
    """Require completed Seen selection and unchanged controls for held-out splits."""
    if args.split not in ('val_unseen', 'test_unseen'):
        return
    if not args.frozen_final_candidate:
        raise ValueError('Stage1 held-out evaluation requires a Seen-selected frozen manifest')
    frozen = json.loads(Path(args.frozen_final_candidate).read_text())
    if frozen.get('checkpoint_sha256') != sha(args.checkpoint) or not frozen.get('all_stages_complete'):
        raise ValueError('Held-out checkpoint differs from completed, frozen Stage1 selection')
    epochs = frozen.get('completed_epochs', {})
    if any(epochs.get(stage, 0) < minimum for stage, minimum in [('A', 3), ('B', 3), ('C', 5)]):
        raise ValueError('Full Stage1 A/B/C training required before held-out evaluation')
    if (frozen.get('visual_control') != args.stage1_visual_control
            or frozen.get('history_control') != args.stage1_history_control
            or frozen.get('stop_threshold') != args.stop_threshold
            or frozen.get('active_observation') is not False):
        raise ValueError('Held-out inference parameters differ from frozen Seen candidate')
    if getattr(args,'stage1_resolution_eval',False) and frozen.get('grid_size')!=args.stage1_grid_size:
        raise ValueError('Resolution differs from its separately frozen manifest')
    if args.split == 'test_unseen' and args.max_episodes is not None:
        raise ValueError('Test Unseen requires full evaluation, without episode subsampling')


def configure_stage1_controls(args):
    """Keep formal defaults; allow explicit Val-Seen-only interventions."""
    if getattr(args,'stage1_resolution_eval',False):
        if args.stage1_grid_size!=384:raise ValueError('Full resolution intervention must be the fixed 384 grid')
        if args.stage1_visual_control!='real' or args.stage1_history_control!='checkpoint':raise ValueError('Resolution evaluation requires unchanged visual/history controls')
        args.stop_threshold=.3
    elif args.stage1_diagnostic:
        if args.split!='val_seen': raise ValueError('Stage1 interventions are Val-Seen-only')
        if not np.isfinite(args.stop_threshold) or not 0<args.stop_threshold<1: raise ValueError('Stop threshold must be in (0,1)')
        if args.stage1_grid_size is not None and not 2<=args.stage1_grid_size<=384: raise ValueError('grid size must be in [2,384]')
    else:
        if args.stage1_grid_size is not None: raise ValueError('grid override requires --stage1-diagnostic')
        args.stop_threshold=.3
    args.head_execution='guarded-blend';args.explore_candidates=0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sbf-source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', choices=['val_seen', 'val_unseen','test_unseen'], required=True)
    parser.add_argument('--frozen-final-candidate',help='Completed Seen-selected frozen candidate, required for Test Unseen')
    parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume',action='store_true',help='Resume exactly the same evaluation from flushed episode traces')
    parser.add_argument('--model', choices=['sbfnav','a3','a3-trajectory','dynamic','target-evidence','joint-active','static-transfer','graphnav-v1'], default='sbfnav')
    parser.add_argument('--stage1-residual-checkpoint',help='Opt-in separately trained residual; Seen validation only')
    parser.add_argument('--stage1-resolution-eval',action='store_true',help='Fixed 384-resolution intervention; original weights and controls, separate frozen manifest')
    parser.add_argument('--stage1-save-fields',action='store_true',help='Save initial public fields for inspection')
    parser.add_argument('--profile', action='store_true', help='Synchronized per-decision component timings')
    parser.add_argument('--stage1-diagnostic', action='store_true', help='Val-Seen-only grid/Stop interventions')
    parser.add_argument('--stage1-grid-size', type=int, help='Actual candidate field resolution for diagnostics')
    parser.add_argument('--stage1-visual-control',choices=['real','zero','shuffle','no-belief'],default='real')
    parser.add_argument('--stage1-history-control',choices=['checkpoint','current','history'],default='checkpoint')
    parser.add_argument('--disable-joint-active',action='store_true')
    parser.add_argument('--joint-execution',choices=['checkpoint','guarded-blend'],default='checkpoint')
    parser.add_argument('--reference-a3-checkpoint',help='Original A3 provenance for legacy head migration')
    parser.add_argument('--dynamic-checkpoint')
    parser.add_argument('--evidence-checkpoint',help='Required for opt-in target evidence evaluation')
    parser.add_argument('--active-observation',action='store_true',help='Opt-in bounded XY/Z sensing actions')
    parser.add_argument('--active-max-actions',type=int,default=2)
    parser.add_argument('--joint-active-observation',action='store_true',
        help='Explicitly enable active sensing for joint-active; OFF by default')
    parser.add_argument('--joint-stop-guard-m',type=float,default=20.,
        help='Maximum XY distance to selected predicted goal for joint Stop; 0 disables only this guard')
    parser.add_argument('--joint-stop-threshold',type=float,default=None,
        help='Override checkpoint joint Stop probability threshold for paired diagnostics')
    parser.add_argument('--joint-visual-control',choices=('full','no-belief','zero-vectors','shuffle-vectors'),
        default='full',help='Causal image-appearance ablation under matched observed support')
    parser.add_argument('--visual-checkpoint')
    parser.add_argument('--visual-history',action='store_true')
    parser.add_argument('--disable-visual',action='store_true')
    parser.add_argument('--graph-control',choices=['full','self','zero-edge-features'],default='full')
    parser.add_argument('--visual-control',choices=['full','zero-vectors','same-type-shuffle'],default='full')
    parser.add_argument('--observed-cache')
    parser.add_argument('--evaluation-concurrency',type=int,choices=(1,2),default=1,
        help='Execution metadata only; maximum simultaneous evaluators, not an isolated latency claim')
    parser.add_argument('--head-checkpoint')
    parser.add_argument('--stop-threshold',type=float,default=.5)
    parser.add_argument('--explore-candidates',type=int,default=0)
    parser.add_argument('--head-execution',choices=['joint','fixed-goal','fixed-goal-no-stop','direct-goal','guarded-path','guarded-blend','guarded-direct'],default='joint')
    args = parser.parse_args()
    if args.stage1_resolution_eval:
        torch.set_num_threads(1)
        import cv2
        cv2.setNumThreads(1)
    if args.model=='graphnav-v1':
        configure_stage1_controls(args)
        validate_stage1_frozen_evaluation(args)
    if (args.profile or args.stage1_diagnostic or args.stage1_resolution_eval or args.stage1_grid_size is not None) and args.model!='graphnav-v1':
        raise ValueError('Stage1 diagnostic controls require --model graphnav-v1')
    if args.stage1_residual_checkpoint and (args.model!='graphnav-v1' or args.split!='val_seen'):
        raise ValueError('Calibrated residual selection requires Seen validation')
    if args.profile and args.split=='test_unseen': raise ValueError('Profiling must use validation data')
    if args.split=='test_unseen' and args.model!='graphnav-v1':
        if args.model!='joint-active' or args.max_episodes is not None or not args.frozen_final_candidate:
            raise ValueError('Test Unseen requires full evaluation of explicitly frozen joint Seen Best after20 epochs')
        frozen=json.loads(Path(args.frozen_final_candidate).read_text())
        if frozen.get('completed_training_epochs')!=20 or not 1<=frozen.get('epoch',0)<=20 or frozen.get('checkpoint_sha256')!=sha(args.checkpoint):
            raise ValueError('Test Unseen checkpoint must equal frozen Seen Best after20 completed epochs')
    if args.active_observation and args.model != 'target-evidence':
        raise ValueError('active observation requires the target-evidence policy')
    if args.joint_active_observation and args.model!='joint-active':
        raise ValueError('--joint-active-observation requires --model joint-active')
    if (args.joint_visual_control!='full' or args.joint_stop_guard_m!=20.
            or args.joint_stop_threshold is not None) and args.model!='joint-active':
        raise ValueError('joint controls require --model joint-active')
    if args.joint_stop_guard_m<0 or not np.isfinite(args.joint_stop_guard_m):
        raise ValueError('joint Stop guard must be finite and nonnegative')
    if args.joint_stop_threshold is not None and not 0<=args.joint_stop_threshold<=1:
        raise ValueError('joint Stop threshold must be in [0,1]')
    if args.active_max_actions < 0:
        raise ValueError('active-max-actions must be nonnegative')
    started_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    started_diff=subprocess.check_output(['git','diff','HEAD','--binary'])
    evaluator_sha256=sha(__file__)
    source_root=Path(__file__).resolve().parents[1]
    implementation_sha256={str(p.relative_to(source_root)):sha(p) for p in sorted((source_root/'graphnav').glob('*.py'))}
    output = Path(args.output)
    if output.exists() and not args.resume: raise ValueError('refusing to overwrite evaluation')
    output.mkdir(parents=True,exist_ok=args.resume)
    signature=dict(args={k:v for k,v in vars(args).items() if k!='resume' and v is not None},checkpoint_sha256=sha(args.checkpoint))
    if args.resume:
        previous_signature=json.loads((output/'eval_started.json').read_text())
        previous_signature['args']={k:v for k,v in previous_signature['args'].items() if v is not None}
        if previous_signature!=signature:raise ValueError('Resume checkpoint or evaluation controls differ')
    else:(output/'eval_started.json').write_text(json.dumps(signature,indent=2))
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
    dataset = CityNavDataset(args.data_root,args.split,version=config['data']['version'],
                             allow_test_unseen=args.split=='test_unseen')
    catalog = CityReferCatalog(Path(args.data_root)/'cityrefer')
    indices = evenly_spaced_indices(len(dataset), args.max_episodes)
    episodes = [dataset[i] for i in indices]
    manifest = [{'index':int(i),'episode_id':list(record.episode_id),
                 'start_pose':record.start_pose.tolist()} for i,record in zip(indices,episodes)]
    if args.resume and json.loads((output/'episodes.json').read_text())!=manifest:raise ValueError('Resume episode manifest differs')
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
        if args.model=='graphnav-v1':
            from graphnav.stage1_runtime import GraphNavStage1Policy
            from graphnav.dynamic_runtime import NativeObservationBuilder
            runtime=GraphNavStage1Policy(args.checkpoint,output/'public_inputs',
                visual_control=args.stage1_visual_control,history_control=args.stage1_history_control)
            builder=NativeObservationBuilder(builder,Path(args.data_root)/'rgbd',output/'observations')
        elif args.model=='joint-active':
            from graphnav.joint_active_runtime import JointActivePolicy
            from graphnav.dynamic_runtime import NativeObservationBuilder
            runtime=JointActivePolicy(
                args.checkpoint, output/'public_inputs',
                active_observation=args.joint_active_observation,
                stop_guard_m=args.joint_stop_guard_m,
                stop_threshold=args.joint_stop_threshold,
                visual_control=args.joint_visual_control,
            )
            builder=NativeObservationBuilder(builder,Path(args.data_root)/'rgbd',output/'observations')
        elif args.model in ('dynamic','target-evidence'):
            if not args.dynamic_checkpoint or not args.visual_checkpoint:raise ValueError('dynamic and visual checkpoints required')
            from graphnav.dynamic_runtime import DynamicWaypointPolicy,NativeObservationBuilder
            policy_class=DynamicWaypointPolicy
            options={}
            if args.model=='target-evidence':
                if not args.evidence_checkpoint:
                    raise ValueError('target-evidence requires a trained evidence checkpoint')
                from graphnav.target_runtime import TargetEvidenceWaypointPolicy
                policy_class=TargetEvidenceWaypointPolicy
                options=dict(evidence_checkpoint=args.evidence_checkpoint,
                             active_observation=args.active_observation,
                             active_max_actions=args.active_max_actions)
            elif args.active_observation or args.evidence_checkpoint:
                raise ValueError('active observation and target evidence require --model target-evidence')
            runtime=policy_class(args.checkpoint,args.dynamic_checkpoint,args.visual_checkpoint,str(output/'public_inputs'),
                history=args.visual_history,visual=not args.disable_visual,observed_cache=args.observed_cache,
                head_checkpoint=args.head_checkpoint,stop_threshold=args.stop_threshold,execution=args.head_execution,
                explore_candidates=args.explore_candidates,graph_control=args.graph_control,visual_control=args.visual_control,
                **options)
            if args.model=='target-evidence' or runtime.use_visual:
                builder=NativeObservationBuilder(builder,Path(args.data_root)/'rgbd',output/'observations')
        elif args.model=='static-transfer':
            if not args.head_checkpoint or not args.reference_a3_checkpoint:raise ValueError('original head and reference A3 required for explicit transfer')
            from graphnav.static_transfer_runtime import StaticTransferTrajectoryPolicy
            runtime=StaticTransferTrajectoryPolicy(args.checkpoint,args.head_checkpoint,args.reference_a3_checkpoint,output/'public_inputs',stop_threshold=args.stop_threshold,execution=args.head_execution)
        elif args.model=='a3-trajectory':
            if not args.head_checkpoint:raise ValueError('trained trajectory head checkpoint required')
            runtime=A3TrajectoryPolicy(args.checkpoint,args.head_checkpoint,output/'public_inputs',stop_threshold=args.stop_threshold,execution=args.head_execution,explore_candidates=args.explore_candidates)
        else:runtime=A3WaypointPolicy(args.checkpoint,output/'public_inputs')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    controller = BoundedWaypointController(horizontal_step_m=protocol.horizontal_step_m,
        vertical_step_m=protocol.vertical_step_m,minimum_above_ground_m=protocol.min_altitude_m,
        maximum_above_ground_m=protocol.max_altitude_m)
    if args.stage1_residual_checkpoint:
        from graphnav.calibrated_residual_runtime import install_calibrated_residual
        install_calibrated_residual(runtime,args.stage1_residual_checkpoint,args.checkpoint)
    profiler=None
    if args.model=='graphnav-v1':
        runtime.stop_threshold=args.stop_threshold
        if args.stage1_grid_size is not None: runtime.args.grid_size=args.stage1_grid_size
        if args.profile:
            from graphnav.profiling import DecisionProfiler
            profiler=DecisionProfiler(torch.cuda.synchronize)
            profiler.instrument(runtime,builder)
    if args.stage1_save_fields:
        from graphnav.diagnostic_fields import install_field_capture
        install_field_capture(runtime,output/'fields')
    rows, maps, terminations = [], defaultdict(list), Counter()
    retrieval_queries=set();retrieval_rows=[];active_counts=Counter();stop_gate_counts=Counter()
    started = time.perf_counter()
    completed=[]
    if args.resume:
        completed=[json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
        if [r['index'] for r in completed]!=[r.index for r in episodes[:len(completed)]]:raise ValueError('Resume trace order differs')
    with (output/'predictions.jsonl').open('a' if args.resume else 'w') as stream:
        for position,record in enumerate(episodes):
            # IDs are used only to prepare supplied CityRefer language bindings and output labels.
            # Neither record, target, IDs, nor expert future enters the policy or rollout.
            public = PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),
                                   tuple(catalog.referenced_landmarks(record)),
                                   tuple(catalog.referenced_landmark_names(record)))
            if position<len(completed):
                saved=completed[position]
                poses=np.asarray(saved['poses']);termination=saved['termination'];timings=saved['decision_seconds'];diagnostics=saved['step_diagnostics']
            else:
                if profiler is not None: profiler.start_episode(record.episode_id)
                if args.model!='sbfnav': runtime.set_episode(public,builder.transform(public.map_name),catalog)
                if profiler is not None: profiler.episode_id=list(record.episode_id)
                poses, termination, timings, diagnostics = rollout(public,builder,controller,runtime.predict,
                    GROUND_LEVEL[record.map_name],protocol,synchronize=torch.cuda.synchronize,profiler=profiler)
                if profiler is not None: profiler.finish_episode()
            metrics = score_trace(poses,record.target_position,record.trajectory,termination,
                radius=protocol.success_radius_m,decision_seconds=timings)
            if args.model=='graphnav-v1':
                query=(record.map_name,record.instruction)
                if query not in retrieval_queries:
                    retrieval_queries.add(query)
                    row={}
                    for label,points in [('initial_static',diagnostics[0]['static_top16_world_xy']),
                                         ('dynamic_first',diagnostics[0]['belief_top16_world_xy']),
                                         ('dynamic_final',diagnostics[-1]['belief_top16_world_xy'])]:
                        distance=np.linalg.norm(np.asarray(points)-record.target_position[:2],axis=-1)
                        row[label+'_NE_m']=float(distance[0])
                        for k in (1,4,8,16):row[label+f'_R@{k}/20m']=float(distance[:k].min()<=20)
                    row['dynamic_first_step']=diagnostics[0]['observation_step']
                    row['dynamic_final_step']=diagnostics[-1]['observation_step']
                    row['history_unique_views']=diagnostics[-1]['history_unique_views']
                    row['history_duplicates']=diagnostics[-1]['history_duplicates']
                    visual=diagnostics[-1]['appearance_top1_world_xy']
                    row['appearance_spatial_r1_at20m']=float(np.linalg.norm(np.asarray(visual)-record.target_position[:2])<=20) if visual is not None else 0.
                    retrieval_rows.append(row)
            if args.model=='joint-active':
                active_counts.update(d['active_action'] for d in diagnostics if 'active_action' in d)
                stop_gate_counts.update(d['stop_reject_reason'] for d in diagnostics
                                        if 'stop_reject_reason' in d)
                query=(record.map_name,record.instruction)
                if query not in retrieval_queries:
                    retrieval_queries.add(query)
                    # Private destination is accessed only after the entire rollout.
                    distances=np.linalg.norm(np.asarray(diagnostics[-1]['belief_top8_world_xy'])-record.target_position[:2],axis=-1)
                    row=dict(NE_m=float(distances[0]),**{f'R@{k}/20m':float(distances[:k].min()<=20) for k in (1,4,8)})
                    visual=diagnostics[-1]['appearance_top1_world_xy']
                    row['appearance_spatial_r1_at20m']=float(np.linalg.norm(np.asarray(visual)-record.target_position[:2])<=20) if visual is not None else 0.
                    retrieval_rows.append(row)
            rows.append(metrics); maps[record.map_name].append(metrics); terminations[termination]+=1
            if position>=len(completed):
                stream.write(json.dumps(dict(index=record.index,episode_id=record.episode_id,
                    map_name=record.map_name,poses=poses.tolist(),termination=termination,
                    metrics=metrics,decision_seconds=timings,step_diagnostics=diagnostics))+'\n'); stream.flush()
            if (position+1)%16==0:
                print(json.dumps(dict(completed=position+1,total=len(episodes),elapsed=time.perf_counter()-started)),flush=True)
    if profiler is not None: profiler.save(output)
    report = dict(model=('Joint A3 + observed appearance/history + trained trajectory/Stop; active sensing opt-in' if args.model=='joint-active' else ('SBFNav local paper-based reproduction' if args.model=='sbfnav'
                        else ('A3 + causal dynamic visual/graph residual belief' if args.model=='dynamic' else ('A3 + target-conditioned observed RGB evidence + optional active sensing' if args.model=='target-evidence' else ('A3 + trained trajectory/rank/Stop; geometry/pose only' if args.model=='a3-trajectory'
                              else 'A3 + bounded Top1 waypoint; no trained trajectory or stop'))))),split=args.split,
        metrics=aggregate(rows),by_map={k:aggregate(v) for k,v in maps.items()},
        terminations=dict(terminations),protocol=asdict(protocol),config=config,
        checkpoint_sha256=sha(args.checkpoint),checkpoint_epoch=checkpoint.get('completed_epoch',checkpoint.get('epoch')),
        head_checkpoint_sha256=sha(args.head_checkpoint) if args.head_checkpoint else None,
        dynamic_checkpoint_sha256=sha(args.dynamic_checkpoint) if args.dynamic_checkpoint else None,
        visual_checkpoint_sha256=sha(args.visual_checkpoint) if args.visual_checkpoint else None,
        evidence_checkpoint_sha256=sha(args.evidence_checkpoint) if args.evidence_checkpoint else None,
        episode_manifest_sha256=sha(output/'episodes.json'),args=vars(args),seconds=time.perf_counter()-started,
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        input_sha256={str(p):sha(p) for p in [dataset.annotation_path,Path(args.data_root)/'cityrefer/objects.json',
            Path(args.data_root)/'cityrefer/processed_descriptions.json']},
        sbf_source_sha256={str(p.relative_to(args.sbf_source)):sha(p)
            for p in sorted(Path(args.sbf_source).glob('sbfnav/**/*.py'))},
        graphnav_commit=started_commit,evaluator_sha256=evaluator_sha256,
        graphnav_working_diff_sha256=hashlib.sha256(started_diff).hexdigest(),
        implementation_sha256=implementation_sha256,
        inference_target_access=False,
        warning=('Legacy SBF checkpoint selected by Val Unseen proxy; no new selection by Unseen.'
                 if args.model=='sbfnav' else
                 'Shared macro-action benchmark with orthophoto/vertical-depth observations; no swept 3D collision or perspective-camera claim.'))
    if args.model=='static-transfer':
        report['model']='Compatible static A3/GAT + transferred unchanged original Trajectory/Stop; guarded-blend'
        report['reference_a3_checkpoint_sha256']=sha(args.reference_a3_checkpoint)
        report['static_variant']=runtime.static_variant;report['static_graph_control']=runtime.static_control
        report['controller_transfer_note']='Original head provenance verified against original A3; compatible new static features are an explicitly evaluated distribution shift, not controller retraining.'
    if args.model=='graphnav-v1':
        report['model']='GraphNav-V1 Stage1; initialized from verified A3 Epoch14 and pretrained controller; bounded visual residual + causal quality history; original Guarded-Blend, no active sensing'
        report['goal_retrieval']=dict(unique_queries=len(retrieval_rows),
            **{key:float(np.mean([r[key]for r in retrieval_rows]))for key in retrieval_rows[0]},
            note='Private targets accessed only after rollout. Initial static, first observed belief, and final observed belief are separate. R@1/4/8 preserve legacy FP32 topk(8) tie ordering; R@16 extends that ordering. Final observation time differs per episode.')
        if args.stage1_residual_checkpoint:
            report['model']='Frozen GraphNav-V1 + calibrated visual residual research intervention'
            report['calibrated_residual_sha256']=sha(args.stage1_residual_checkpoint)
        report['stage1_protocol']=runtime.protocol
        report['stage1_inference_controls']=dict(diagnostic=args.stage1_diagnostic,stop_threshold=runtime.stop_threshold,grid_size=runtime.args.grid_size,profile=args.profile)
        report['visual_backbone_parameters']=sum(p.numel()for p in runtime.native.model.parameters())
        report['bert_backbone_parameters']=sum(p.numel()for p in runtime.bert.model.parameters())
    if getattr(runtime,'encoder_visual',None) is not None:
        report['observed_visual_cache']=dict(getattr(runtime.encoder_visual,'cache_metrics',{}),
            backbone_encoded_images=runtime.encoder_visual.backbone.encoded_images,
            cache_protocol='Exact currently observed ROI pixels + actual frozen model/head/precision SHA; no future gallery',
            timing_note='Cross-model warm cache affects latency. New vector requests and actual backbone encodes may differ through patch-memory reuse; not target visibility or accuracy.')
    if args.model=='joint-active':
        report['goal_retrieval']=dict(unique_queries=len(retrieval_rows),
            **{key:float(np.mean([r[key] for r in retrieval_rows])) for key in retrieval_rows[0]},
            note='Final observed/history belief. Deduplicate map+instruction, first source target. Different from initial static recall and closed-loop SR. Appearance spatial retrieval is not roof-color/shape classification accuracy.')
        report['active_action_counts']=dict(active_counts)
        report['stop_guard_diagnostics']=dict(stop_gate_counts)
        report['joint_inference_controls']=dict(
            active_observation=args.joint_active_observation,
            stop_guard_m=args.joint_stop_guard_m,
            stop_threshold=(runtime.stop_threshold),
            visual_control=args.joint_visual_control,
            note='GT-free candidate proximity; intervention metrics require new paired rollouts',
        )
        report['frozen_final_candidate_sha256']=sha(args.frozen_final_candidate) if args.frozen_final_candidate else None
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['metrics']),flush=True)


if __name__ == '__main__': main()
