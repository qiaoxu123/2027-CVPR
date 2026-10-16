"""Run unchanged frozen V1 on all Val-Unseen episodes with HETT timing."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.workspace_paths import PROJECT, current_paths


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'profiling_val_unseen'
HETT=PROJECT/'Ablations/HETT/hett-main-system-opt'


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True).strip()


def summary(directory):
    profile=json.loads((directory/'profile_episodes.json').read_text())
    metrics=json.loads((directory/'metrics.json').read_text())
    assert profile['samples']==metrics['metrics']['episodes']==2697
    official=PROJECT/'GraphNav/runs/graphnav_v1_stage1/full_unseen'
    ref=json.loads((official/'metrics.json').read_text())
    assert metrics['checkpoint_sha256']==ref['checkpoint_sha256']
    assert metrics['episode_manifest_sha256']==ref['episode_manifest_sha256']
    source=[json.loads(l) for l in (directory/'predictions.jsonl').open()]
    assert len(source)==2697 and [r['index'] for r in source]==list(range(2697))
    assert len({tuple(r['episode_id']) for r in source})==2697
    samples=[]
    for i,(row,pred) in enumerate(zip(profile['samples_detail'],source)):
        assert row['episode_id']==pred['episode_id']
        policy=row['static_graph_gpu_ms']+row['visual_residual_history_gpu_ms']+row['trajectory_stop_gpu_ms']
        samples.append(dict(index=i,sample_id=str(tuple(row['episode_id'])),map=row['map_name'],
                            rollout_seconds=row['rollout_ms']/1000,
                            language_gpu_seconds=row['language_gpu_ms']/1000,
                            vision_gpu_seconds=row['vision_gpu_ms']/1000,
                            policy_gpu_seconds=policy/1000,
                            model_inference_gpu_seconds=row['model_inference_gpu_ms']/1000,
                            environment_observation_seconds=row['environment_ms']/1000,
                            other_rollout_seconds=row['other_rollout_ms']/1000,
                            steps=pred['metrics']['steps'],decisions=row['decisions'],navigation_error=pred['metrics']['NE']))
    def stats(values):
        a=np.asarray(values);return dict(count=len(a),min=float(a.min()),mean=float(a.mean()),variance=float(a.var()),stddev=float(a.std()),max=float(a.max()))
    model_keys=('language_gpu_seconds','vision_gpu_seconds','policy_gpu_seconds','model_inference_gpu_seconds')
    report=dict(checkpoint=metrics['args']['checkpoint'],checkpoint_sha256=metrics['checkpoint_sha256'],
                split='val_unseen',sample_count=2697,unique_sample_count=2697,variance_method='population variance (ddof=0)',
                rollout_seconds=stats([r['rollout_seconds'] for r in samples]),
                rollout_seconds_excluding_first_sample=stats([r['rollout_seconds'] for r in samples[1:]]),
                model_gpu_seconds_by_component={k:stats([r[k] for r in samples]) for k in model_keys},
                environment_observation_seconds=stats([r['environment_observation_seconds'] for r in samples]),
                other_rollout_seconds=stats([r['other_rollout_seconds'] for r in samples]),
                navigation_metrics=metrics['metrics'],samples=samples,
                first_sample=samples[0],top5_slowest_samples=sorted(samples,key=lambda r:r['rollout_seconds'],reverse=True)[:5],
                top5_hardest_by_navigation_error=sorted(samples,key=lambda r:r['navigation_error'],reverse=True)[:5],
                timing=profile['timing'],reference_profiler=str(HETT/'multiagent/profile_inference.py'))
    atomic_json(report,OUT/'profile_val_unseen_full.json')
    keys=[k for k in samples[0] if k.endswith('_seconds')]
    rows=[dict(component=k,min_ms=stats([r[k] for r in samples])['min']*1000,
               mean_ms=stats([r[k] for r in samples])['mean']*1000,
               population_variance_ms2=stats([r[k] for r in samples])['variance']*1e6) for k in keys]
    for k,v in profile['all'].items():
        if k in ('static_graph_gpu_ms','visual_residual_history_gpu_ms','trajectory_stop_gpu_ms'):
            rows.append(dict(component=k,min_ms=v['min'],mean_ms=v['mean'],population_variance_ms2=v['population_variance']))
    with (OUT/'profiling_summary_ms.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    old={r['index']:r for r in map(json.loads,(official/'predictions.jsonl').open())}
    outcome_changes=[r['index'] for r in source if r['metrics']['SR']!=old[r['index']]['metrics']['SR'] or r['termination']!=old[r['index']]['termination']]
    atomic_json(dict(checkpoint_matches=True,manifest_matches=True,episodes=2697,outcome_changes=outcome_changes,
                     metric_deltas={k:metrics['metrics'][k]-ref['metrics'][k] for k in ('SR','SPL','NE')}),OUT/'navigation_parity.json')


def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=current_paths(json.loads((PROJECT/'GraphNav/runs/graphnav_v1_stage1/full_unseen/metrics.json').read_text())['args'])
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    command=[sys.executable,'-u','-m','scripts.evaluate_closedloop','--model','graphnav-v1',
             '--checkpoint',base['checkpoint'],'--sbf-source',base['sbf_source'],'--config',base['config'],
             '--data-root',base['data_root'],'--split','val_unseen','--profile-hett',
             '--frozen-final-candidate',str(PROJECT/'GraphNav/runs/graphnav_v1_stage1/frozen_final_candidate.json')]
    sha=file_sha256(base['checkpoint'])
    assert sha=='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a'
    record=dict(started=time.time(),checkpoint_sha256=sha,grid_size=28,stop_threshold=.3,
                context='original reference-only; no global landmark intervention',training=False,
                protocol='HETT whole-episode CUDA Event module spans and synchronized observation wall time',
                hett_profiler_sha256=file_sha256(HETT/'multiagent/profile_inference.py'),
                device=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True),
                gpu_before=gpu_snapshot(),command=command,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(record,OUT/'launch_record.json')
    for name,count in [('smoke4',4),('full',None)]:
        destination=OUT/name
        if (destination/'metrics.json').exists():continue
        if destination.exists():raise RuntimeError(f'Preserve existing incomplete timing attempt: {destination}')
        cmd=command+['--output',str(destination)]
        if count is not None:cmd+=['--max-episodes',str(count)]
        with (OUT/f'{name}.log').open('a') as log,(OUT/'gpu_activity.jsonl').open('a') as activity:
            child=subprocess.Popen(cmd,env=environment,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                trace=destination/'predictions.jsonl'
                n=sum(1 for _ in trace.open()) if trace.exists() else 0
                atomic_json(dict(job=name,pid=child.pid,completed=n,total=count or 2697,updated=time.time()),OUT/'progress.json')
                activity.write(json.dumps(dict(time=time.time(),expected_pid=child.pid,processes=gpu_snapshot()))+'\n');activity.flush()
                time.sleep(10)
            if child.returncode:raise RuntimeError(f'{name} failed with returncode {child.returncode}; see log')
        smoke=json.loads((destination/'profile_episodes.json').read_text())
        assert smoke['samples']==(count or 2697)
        assert file_sha256(base['checkpoint'])==sha
        for row in smoke['samples_detail']:
            total=sum(row[k] for k in ('language_gpu_ms','vision_gpu_ms','static_graph_gpu_ms','visual_residual_history_gpu_ms','trajectory_stop_gpu_ms'))
            assert abs(total-row['model_inference_gpu_ms'])<1e-5
        print(name,'complete',flush=True)
    summary(OUT/'full')
    atomic_json(dict(complete=True,finished=time.time(),episodes=2697),OUT/'complete.json')
    atomic_json(dict(complete=True,completed=2697,total=2697,updated=time.time()),OUT/'progress.json')


if __name__=='__main__':
    try:main()
    except Exception as error:
        OUT.mkdir(exist_ok=True);atomic_json(dict(error=str(error),time=time.time()),OUT/'failure.json');raise
