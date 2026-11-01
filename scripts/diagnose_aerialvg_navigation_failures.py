"""Existing128 archived traces only. No dataset labels, new rollouts or tuning."""
import argparse,collections,json
from pathlib import Path
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256


def distance_bounds(point,endpoints,radii):
    """Triangle inequalities from archived endpoint NE; never reconstruct a target."""
    d=np.linalg.norm(np.asarray(endpoints)-np.asarray(point),axis=-1);r=np.asarray(radii)
    lower=float(np.max(np.abs(d-r)));upper=float(np.min(d+r))
    if lower>upper+1e-4:raise ValueError('Inconsistent archived endpoint distances')
    return lower,upper


def cause_category(goal_bounds,goal_distance,termination,missed_near):
    if goal_bounds is not None and goal_bounds[0]>20+1e-4:return '定位偏差证据'
    if goal_bounds is not None and goal_bounds[1]<=20-1e-4 and goal_distance>1e-3 and termination in ['horizon','stagnation']:return '控制或收敛未完成证据'
    if missed_near:return '已到达但未停：Stop与目标跟踪混合'
    return '未知'


def summarize(rows):
    result={}
    for name,subset in [('all',rows),('failed',[r for r in rows if not r['SR']]),('OSR_success_SR_failure',[r for r in rows if r['OSR'] and not r['SR']])]:
        causes=collections.Counter(r['cause'] for r in subset if not r['SR']);n=len(subset)
        result[name]=dict(episodes=n,SR_success=sum(r['SR'] for r in subset),OSR_success=sum(r['OSR'] for r in subset),causes={k:dict(count=v,fraction=v/n if n else None) for k,v in causes.items()},false_policy_stop=sum(r['false_policy_stop'] for r in subset),certified_near_not_stopped=sum(r['certified_near_not_stopped'] for r in subset),goal_jumps_gt20=sum(r['goal_jumps_gt20'] for r in subset),goal_transitions=sum(r['goal_transitions'] for r in subset),jumps_when_agent_moves_le5m=sum(r['jumps_when_agent_moves_le5m'] for r in subset),small_agent_move_transitions=sum(r['small_agent_move_transitions'] for r in subset),persistent_wrong_goal_ge3=sum(r['persistent_wrong_goal_ge3'] for r in subset),correct_to_wrong_goal_switches=sum(r['correct_to_wrong_goal_switches'] for r in subset),moves_away_from_selected_goal=sum(r['moves_away_from_selected_goal'] for r in subset),unknown_accumulation_mechanism=sum(not r['SR'] for r in subset))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.experiment_root)
    names=dict(baseline='aerialvg_original_navigation_seen128_20261101',current='aerialvg_native_navigation_seen128_20261101',history8='aerialvg_native_temporal_navigation_seen128_20261101')
    reports={};traces={};provenance={}
    for tag,name in names.items():
        directory=root/name;report=json.loads((directory/'metrics.json').read_text());rows=[json.loads(s) for s in (directory/'predictions.jsonl').read_text().splitlines()]
        if report['args']['split']!='val_seen' or len(rows)!=128 or report['metrics']['episodes']!=128:raise ValueError('Only complete existing128 archived trajectories allowed')
        reports[tag]=report;traces[tag]=rows;provenance[tag]=dict(metrics_sha=file_sha256(directory/'metrics.json'),trace_sha=file_sha256(directory/'predictions.jsonl'))
    first=reports['baseline']
    for tag in names:
        if reports[tag]['episode_manifest_sha256']!=first['episode_manifest_sha256'] or reports[tag]['checkpoint_sha256']!=first['checkpoint_sha256'] or reports[tag]['input_sha256']!=first['input_sha256'] or reports[tag]['protocol']!=first['protocol']:raise ValueError('Archived paired provenance mismatch')
        if [r['episode_id'] for r in traces[tag]]!=[r['episode_id'] for r in traces['baseline']]:raise ValueError('Archive episode mismatch')
    output={tag:[] for tag in names}
    for i in range(128):
        endpoints=[traces[tag][i]['poses'][-1][:2] for tag in names];radii=[traces[tag][i]['metrics']['NE'] for tag in names]
        for tag in names:
            row=traces[tag][i];d=row['step_diagnostics'];poses=np.asarray(row['poses'])[:,:2];goals=[np.asarray(x['selected_goal_xy']) if x.get('selected_goal_xy') is not None else None for x in d];bounds=[distance_bounds(g,endpoints,radii) if g is not None else None for g in goals];last=goals[-1];last_bound=bounds[-1];final_goal_distance=float(np.linalg.norm(last-poses[-1])) if last is not None else None
            missed=[]
            for t,diagnostic in enumerate(d):
                upper=distance_bounds(poses[t],endpoints,radii)[1]
                if upper<=20-1e-4 and not (row['termination']=='policy_stop' and t==len(d)-1):
                    reasons=[]
                    if diagnostic['stop_probability']<.3:reasons.append('stop_probability_lt_0.3')
                    if diagnostic.get('predicted_goal_distance',float('inf'))>1e-3:reasons.append('exact_predicted_goal_arrival_guard')
                    if not diagnostic['stop_altitude_guard']:reasons.append('altitude_guard')
                    missed.append(dict(step=t,reasons=reasons))
            jumps=smalljumps=transitions=smalltransitions=switches=away=0;run=0;persistent=False
            for t,g in enumerate(goals):
                wrong=bounds[t] is not None and bounds[t][0]>20+1e-4
                if t and g is not None and goals[t-1] is not None:
                    motion=float(np.linalg.norm(g-goals[t-1]));agent_motion=float(np.linalg.norm(poses[t]-poses[t-1]));transitions+=1;jumps+=int(motion>20);smalltransitions+=int(agent_motion<=5);smalljumps+=int(motion>20 and agent_motion<=5)
                    switches+=int(bounds[t-1][1]<=20-1e-4 and wrong)
                    run=run+1 if wrong and bounds[t-1][0]>20+1e-4 and motion<=1 else int(wrong)
                else:run=int(wrong)
                persistent|=run>=3
                if g is not None and t+1<len(poses):away+=int(np.linalg.norm(poses[t+1]-g)>np.linalg.norm(poses[t]-g)+1e-3)
            success=bool(row['metrics']['SR']);cause='成功' if success else cause_category(last_bound,final_goal_distance,row['termination'],bool(missed))
            output[tag].append(dict(index=row['index'],episode_id=row['episode_id'],SR=success,OSR=bool(row['metrics']['OSR']),termination=row['termination'],NE=row['metrics']['NE'],cause=cause,terminal_goal_target_distance_bounds=last_bound,terminal_agent_goal_distance=final_goal_distance,false_policy_stop=not success and row['termination']=='policy_stop',certified_near_not_stopped=bool(missed),certified_near_rejections=missed,goal_jumps_gt20=jumps,goal_transitions=transitions,jumps_when_agent_moves_le5m=smalljumps,small_agent_move_transitions=smalltransitions,persistent_wrong_goal_ge3=persistent,correct_to_wrong_goal_switches=switches,moves_away_from_selected_goal=away))
    helped=[r['index'] for old,r in zip(output['current'],output['history8']) if r['SR'] and not old['SR']];harmed=[r['index'] for old,r in zip(output['current'],output['history8']) if not r['SR'] and old['SR']]
    result=dict(base_commit='fcfde0bcd9e7f6cc66d0a1f23302ecd4ec9289f6',provenance=provenance,summary={tag:summarize(rows) for tag,rows in output.items()},history8_vs_current=dict(helped=len(helped),harmed=len(harmed),helped_indices=helped,harmed_indices=harmed,mechanism='Changing history is the paired intervention, but trajectories diverge. Missing per-frame query posteriors/current-only delta prevents identifying evidence accumulation as a cause.'),rows=output,protocol='Only existing128 archived Val-Seen traces read as explicitly requested; no source Val-Seen/Test labels or new rollouts. Target-distance bounds use triangle inequalities from paired endpoint NE, not guessed target coordinates. Causes are mutually exclusive evidence categories, not uniquely established root causes. Unknown retained.20m success is position-only.')
    atomic_json(result,Path(a.output));print(json.dumps(dict(summary=result['summary'],history8_vs_current=result['history8_vs_current']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
