"""Run sequential Val-Seen diagnostics without changing frozen official results."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from graphnav.workspace_paths import current_paths, PROJECT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(PROJECT/'Ablations/GraphNav/diagnostics_20261013'))
    parser.add_argument('--episodes',type=int,default=128)
    parser.add_argument('--heatmap-episodes',type=int,default=4)
    args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    base=current_paths(json.loads(Path('runs/graphnav_v1_stage1/full_seen/metrics.json').read_text())['args'])
    commands=[('profile',args.episodes,['--profile'])]
    commands += [(f'heatmap_{g}',args.heatmap_episodes,['--stage1-grid-size',str(g),'--profile']) for g in (28,384)]
    commands += [(f'stop_{t}',args.episodes,['--stop-threshold',str(t)]) for t in (.1,.3,.5,.7,.9)]
    commands += [('profile_hett',args.episodes,['--profile'])]
    commands += [('no_belief',args.episodes,['--stage1-visual-control','no-belief'])]
    status=[]
    for name,count,extra in commands:
        destination=root/name
        cmd=[sys.executable,'-m','scripts.evaluate_closedloop','--model','graphnav-v1',
            '--stage1-diagnostic','--sbf-source',base['sbf_source'],'--config',base['config'],
            '--data-root',base['data_root'],'--checkpoint',base['checkpoint'],'--split','val_seen',
            '--max-episodes',str(count),'--stop-threshold','.3','--output',str(destination),*extra]
        if name=='profile_hett':
            cmd[cmd.index('--split')+1]='val_unseen'
            cmd.remove('--stage1-diagnostic')
            cmd += ['--frozen-final-candidate',str(Path('runs/graphnav_v1_stage1/frozen_final_candidate.json').resolve())]
        entry=dict(name=name,command=cmd,started=time.time())
        print(json.dumps(entry),flush=True)
        if (destination/'metrics.json').exists():
            entry['status']='already_complete'
        elif destination.exists():
            entry['status']='incomplete_existing_output'
        else:
            with (root/(name+'.log')).open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            entry['returncode']=result.returncode
            entry['status']='complete' if result.returncode==0 else 'failed'
        entry['finished']=time.time();status.append(entry)
        (root/'queue.json').write_text(json.dumps(status,indent=2))
        print(json.dumps(entry),flush=True)
    rows=[]
    for entry in status:
        metrics=root/entry['name']/'metrics.json'
        if metrics.exists():
            report=json.loads(metrics.read_text());m=report['metrics']
            rows.append(dict(name=entry['name'],**report['stage1_inference_controls'],
                **{k:m[k] for k in ('episodes','SR','SPL','OSR','NE','steps','False_Stop_Rate','false_stop_episode_fraction')},
                seconds=report['seconds'],peak_gib=report['gpu_peak_allocated_gib'],
                terminations=json.dumps(report['terminations']),manifest_sha=report['episode_manifest_sha256'],checkpoint_sha=report['checkpoint_sha256']))
    if rows:
        with (root/'comparison.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    if any(s['status'] not in ('complete','already_complete') for s in status):sys.exit(1)


if __name__=='__main__':main()
