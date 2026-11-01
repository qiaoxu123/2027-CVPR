"""Verify completed frozen native rollouts against source episodes and recomputed scores."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import score_trace,aggregate
from graphnav.workspace_paths import PROJECT


def verify(directory,frozen_path,expected_episodes,require_full=False):
    root=Path(directory);frozen=json.loads(Path(frozen_path).read_text())
    report=json.loads((root/'metrics.json').read_text());native=json.loads((root/'native_experiment.json').read_text())
    manifest=json.loads((root/'episodes.json').read_text());traces=[json.loads(s) for s in (root/'predictions.jsonl').read_text().splitlines()]
    if not native['metrics_written'] or report['inference_target_access'] is not False:raise ValueError('Incomplete or invalid inference report')
    if len(manifest)!=expected_episodes or len(traces)!=expected_episodes or report['metrics']['episodes']!=expected_episodes:raise ValueError('Incomplete rollout set')
    if report['episode_manifest_sha256']!=file_sha256(root/'episodes.json'):raise ValueError('Episode manifest changed')
    if native['native_checkpoint_sha']!=frozen['native_checkpoint_sha'] or file_sha256(native['native_checkpoint'])!=frozen['native_checkpoint_sha']:raise ValueError('Frozen native checkpoint mismatch')
    if report['checkpoint_sha256']!=frozen['base_checkpoint_sha'] or file_sha256(report['args']['checkpoint'])!=frozen['base_checkpoint_sha']:raise ValueError('Frozen base checkpoint mismatch')
    if native.get('history_window',1)!=frozen['native_history_window'] or native['control']!=frozen['native_control']:raise ValueError('Frozen native controls mismatch')
    for key in ['stop_threshold','head_execution']:
        if report['args'][key]!=frozen[key]:raise ValueError('Frozen controller mismatch: '+key)
    if report['args']['explore_candidates']!=0 or frozen['active_observation'] is not False:raise ValueError('Unexpected active control')
    for path,digest in report['input_sha256'].items():
        if file_sha256(path)!=digest:raise ValueError('Dataset annotation changed')
    sbf=PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source';sys.path.insert(0,str(sbf))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.validation import evenly_spaced_indices
    source=CityNavDataset(report['args']['data_root'],report['args']['split'],version='processed_citynav',allow_test_unseen=report['args']['split']=='test_unseen')
    if require_full and (report['args']['max_episodes'] is not None or expected_episodes!=len(source)):raise ValueError('Full evaluation required without subsampling')
    expected=evenly_spaced_indices(len(source),expected_episodes)
    if [r['index'] for r in manifest]!=list(expected):raise ValueError('Unexpected episode selection')
    scores=[];decisions=0
    for episode,trace,index in zip(manifest,traces,expected):
        record=source[index]
        if episode['episode_id']!=list(record.episode_id) or trace['episode_id']!=episode['episode_id'] or trace['index']!=record.index:raise ValueError('Episode identity mismatch')
        if not np.allclose(trace['poses'][0],record.start_pose,atol=1e-5):raise ValueError('Initial pose changed')
        diagnostics=trace['step_diagnostics']
        if len(diagnostics)!=len(trace['decision_seconds']):raise ValueError('Decision trace count mismatch')
        for step,d in enumerate(diagnostics):
            if d['observation_step']!=step or d['native_object_checkpoint_sha']!=frozen['native_checkpoint_sha'] or d['native_object_history_window']!=frozen['native_history_window'] or d['native_object_control']!=frozen['native_control'] or d['active_observation'] is not False:raise ValueError('Per-step frozen inference mismatch')
        if trace['termination']=='policy_stop':
            final=diagnostics[-1]
            if final['stop_probability']<frozen['stop_threshold'] or final['predicted_goal_distance']>1e-3 or not final['stop_altitude_guard']:raise ValueError('Original Stop guard violated')
        decisions+=len(diagnostics)
        score=score_trace(trace['poses'],record.target_position,record.trajectory,trace['termination'],radius=report['protocol']['success_radius_m'],decision_seconds=trace['decision_seconds'])
        for key,value in score.items():
            if not math.isclose(value,trace['metrics'][key],rel_tol=1e-7,abs_tol=1e-6):raise ValueError('Per-episode score mismatch: '+key)
        scores.append(score)
    metrics=aggregate(scores)
    for key,value in metrics.items():
        if isinstance(value,(float,int)) and not math.isclose(value,report['metrics'][key],rel_tol=1e-7,abs_tol=1e-6):raise ValueError('Aggregate mismatch: '+key)
    result=dict(passed=True,full_split=require_full,split=report['args']['split'],episodes=expected_episodes,decisions=decisions,metrics=metrics,freeze_sha=file_sha256(frozen_path),trace_sha=file_sha256(root/'predictions.jsonl'),report_sha=file_sha256(root/'metrics.json'),note='Verifies source episode selection, frozen controls/weights, per-step provenance and recomputed scoring. Does not alone prove efficacy or absence of every possible implementation leak.')
    atomic_json(result,root/'verification.json');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--frozen',required=True);p.add_argument('--expected-episodes',required=True,type=int);p.add_argument('--require-full',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.directory,a.frozen,a.expected_episodes,a.require_full),indent=2))


if __name__=='__main__':main()
