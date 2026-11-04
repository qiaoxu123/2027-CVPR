"""Post-hoc public-landmark diagnostics on sealed old rollouts; never a policy."""
import argparse,csv,json
from pathlib import Path
import numpy as np,shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.landmark_navigation_diagnostic import public_region


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 tasks={t['index']:t for t in json.loads(a.tasks.read_text())};rows=[json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()];events=[];details=[]
 for r in rows:
  if r['method']!='E4_PGVB':continue
  region=public_region(tasks[r['index']]['public_polygons'],tasks[r['index']]['world_bounds']);ds=r['step_diagnostics'];previous=None
  for k,d in enumerate(ds):
   z=np.load(d['audit_artifact']);b=z['bounds'];resolution=(b[2:]-b[:2])/28;tol=min(20.,float(np.linalg.norm(resolution)/2));goal=np.asarray(d['selected_goal_xy']);ne=d['public_landmark_goal_distance_m'];actual=float(shapely.distance(shapely.Point(r['poses'][k][:2]),region));dist=d['predicted_goal_distance'];pstop=d['stop_probability'];jump=float(np.linalg.norm(goal-previous)) if previous is not None else 0.
   item=dict(index=r['index'],step=k,goal_xy=goal.tolist(),goal_NE=ne,actual_NE=actual,jump_m=jump,pgvb_active=d['pgvb_active'],stop_probability=pstop,goal_distance=dist,arrival_guard=dist<=1e-3,quantization_tolerance_m=tol,within_quantization=dist<=tol,goal_correct=ne<=20,actually_arrived=actual<=20,stop_probability_ready=pstop>=.3,altitude_guard=d['stop_altitude_guard'],pgvb_reason=d.get('pgvb_reason'),candidate_id=int(z['final_candidate_ids'][0]))
   if 'pgvb_probability' in z:item['pgvb_peak_probability']=float(z['pgvb_probability'].max())
   if jump>20:
    item['previous_goal_NE']=ds[k-1]['public_landmark_goal_distance_m'];item['transition_class']=('correct' if item['previous_goal_NE']<=20 else 'wrong')+'->'+('correct' if ne<=20 else 'wrong');events.append(item.copy())
   details.append(item);previous=goal
 assert len(events)==50
 stop_block=[x for x in details if x['stop_probability_ready'] and x['altitude_guard'] and not x['arrival_guard']]
 summary=dict(jumps=50,jump_classes={key:sum(x['transition_class']==key for x in events) for key in ['correct->correct','wrong->correct','correct->wrong','wrong->wrong']},jump_episodes=len({x['index'] for x in events}),
   blocked_ready_stop=len(stop_block),blocked_actual_and_goal_correct=sum(x['actually_arrived'] and x['goal_correct'] for x in stop_block),blocked_actual_correct_goal_wrong=sum(x['actually_arrived'] and not x['goal_correct'] for x in stop_block),
   quantization_would_release=sum(x['within_quantization'] for x in stop_block),quantization_release_but_not_arrived=sum(x['within_quantization'] and not x['actually_arrived'] for x in stop_block),warning='post-hoc fixed-trajectory mechanism checks, not closed-loop SR or threshold selection')
 for n,v in [('summary.json',summary),('jumps50.json',events),('steps.json',details)]: (a.output/n).write_text(json.dumps(v,indent=2))
 with (a.output/'steps.csv').open('w') as f:
  keys=sorted({k for x in details for k in x});w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(details)
 for i in [1,13,34,39]:
  selected=[x for x in details if x['index']==i];fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True);xx=[x['step'] for x in selected]
  for key,label in [('goal_NE','Goal NE'),('actual_NE','UAV NE')]:axes[0].plot(xx,[x[key] for x in selected],'-o',label=label)
  axes[0].axhline(20,color='gray',ls='--');axes[0].legend();axes[0].set_ylabel('Polygon distance m')
  axes[1].plot(xx,[x['goal_distance'] for x in selected],'-o',label='Distance to current Goal');axes[1].plot(xx,[x['quantization_tolerance_m'] for x in selected],ls='--',label='Half-cell diagonal (not tuned)');axes[1].legend();axes[1].set_ylabel('m')
  axes[2].plot(xx,[x['stop_probability'] for x in selected],'-o',label='Learned Stop p');axes[2].axhline(.3,color='gray',ls='--');axes[2].scatter([x['step'] for x in selected if x['jump_m']>20],[.95 for x in selected if x['jump_m']>20],c='red',label='Goal jump >20m');axes[2].legend();axes[2].set_xlabel('Causal decision step');fig.suptitle(f'{i}: {tasks[i]["landmark_name"]} | sealed old E4, no new inference');fig.tight_layout();fig.savefig(a.output/f'task_{i:02d}.png',dpi=140);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
