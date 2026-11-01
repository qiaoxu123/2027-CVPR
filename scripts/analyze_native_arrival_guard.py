"""Exact earlier-stop prefix counterfactuals using public decision diagnostics only.

Target/reference are accessed solely by score_trace after the stop decision.
This does not extrapolate trajectories after an earlier original stop.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from graphnav.workspace_paths import PROJECT,WORKSPACE
from graphnav.closedloop import score_trace,aggregate
from graphnav.checkpointing import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--native',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
 from sbfnav.dataset import CityNavDataset
 directories={'baseline':Path(a.baseline),'native':Path(a.native)};reports={k:json.loads((v/'metrics.json').read_text()) for k,v in directories.items()}
 assert reports['baseline']['episode_manifest_sha256']==reports['native']['episode_manifest_sha256']
 source=CityNavDataset(WORKSPACE/'DATA/sbfnav-data','val_seen',version='processed_citynav');results={}
 for tag,directory in directories.items():
  episodes=[json.loads(s) for s in (directory/'predictions.jsonl').read_text().splitlines()]
  assert len(episodes)==reports[tag]['metrics']['episodes']
  results[tag]={}
  for radius in [.001,5.,10.,15.,20.]:
   rows=[]
   for episode in episodes:
    first=None
    # This decision has no target/reference input.
    for step,diagnostic in enumerate(episode['step_diagnostics']):
     distance=diagnostic['predicted_goal_distance']
     if distance is not None and distance<=radius and diagnostic['stop_probability']>=.3 and diagnostic['stop_altitude_guard']:
      first=step;break
    poses=np.array(episode['poses']) if first is None else np.array(episode['poses'][:first+1])
    termination=episode['termination'] if first is None else 'policy_stop'
    seconds=episode['decision_seconds'] if first is None else episode['decision_seconds'][:first+1]
    record=source[episode['index']]
    metrics=score_trace(poses,record.target_position,record.trajectory,termination,radius=20.,decision_seconds=seconds)
    if radius==.001:
     for key in ['SR','OSR','SPL','NE','steps']:assert abs(metrics[key]-episode['metrics'][key])<1e-5,(key,metrics,episode['metrics'])
    rows.append(dict(index=episode['index'],first_stop_step=first,metrics=metrics))
   results[tag][str(radius)]=dict(metrics=aggregate([r['metrics'] for r in rows]),rows=rows)
 summary={tag:{r:v['metrics'] for r,v in configurations.items()} for tag,configurations in results.items()}
 atomic_json(dict(summary=summary,results=results,protocol='Public pStop>=0.3, predicted-goal distance<=radius, original altitude condition. radius>=original1e-3; same rule both models. Exact original reconstruction asserted. Posthoc Seen diagnostic, not prospective full-evaluation evidence.'),Path(a.output));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
