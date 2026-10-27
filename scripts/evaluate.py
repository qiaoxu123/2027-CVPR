"""Paired original-paper navigation; private scoring occurs after policy prediction."""
from common import *
import argparse
import time
import numpy as np
import torch
from policy import RefinementPolicy
from prepare import DetailBuilder
from graphnav.dynamic_runtime import NativeObservationBuilder
from graphnav.closedloop import PublicEpisode,PublicCatalog,ClosedLoopProtocol,rollout,score_trace,aggregate
from sbfnav.dataset import CityNavDataset,CityReferCatalog
from sbfnav.planning_state import PlanningStateBuilder
from sbfnav.navigation import BoundedWaypointController
from sbfnav.mapdata import GROUND_LEVEL


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['val_seen','val_unseen'],required=True);p.add_argument('--arm',choices=['baseline','refinement'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--calibration',type=Path);p.add_argument('--limit',type=int);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control',default='real',choices=['real','zero','no_language','no_depth','no_height','no_geometry','no_offset','shuffle_depth']);args=p.parse_args()
    torch.set_num_threads(2);c=configuration();verify_sources(c)
    if args.output.exists():raise ValueError('Refuse to overwrite evaluation')
    if args.arm=='refinement' and (not args.checkpoint or not args.calibration):raise ValueError('Refinement requires checkpoint and calibrated gate')
    args.output.mkdir(parents=True);entries=json.loads(evaluation_manifest(c,args.split).read_text())['episodes']
    if args.limit:entries=entries[:args.limit]
    signature=dict(original_sha256=sha(CHECKPOINT),implementation=implementation_hashes(),entries=entries,arm=args.arm,control=args.control,
        checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None,calibration_sha256=sha(args.calibration) if args.calibration else None)
    ds=CityNavDataset(DATA,args.split,version=c['data_version']);cat=CityReferCatalog(DATA/'cityrefer');signature['annotation_sha256']=sha(ds.annotation_path)
    save(signature,args.output/'signature.json')
    policy=RefinementPolicy(args.output/'query',checkpoint=args.checkpoint if args.arm=='refinement' else None,calibration=args.calibration if args.arm=='refinement' else None,control=args.control)
    active=policy.calibration['enabled'] and args.control!='zero'
    builder=(DetailBuilder if active else NativeObservationBuilder)(PlanningStateBuilder(DATA,PublicCatalog(cat),canvas_size=384,field_size=28),RGBD,args.output/'render')
    render_seconds=[];original_build=builder.build_from_poses
    def timed(*a,**kw):
        tick=time.perf_counter();s=original_build(*a,**kw);render_seconds.append(time.perf_counter()-tick);return s
    builder.build_from_poses=timed
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    rows=[];started=time.perf_counter()
    for entry in entries:
        rec=ds[int(entry['index'])]
        if rec.map_name!=entry['map_name']:raise ValueError('Evaluation episode map differs from frozen manifest')
        ep=PublicEpisode(rec.map_name,rec.instruction,rec.start_pose.copy(),tuple(cat.referenced_landmarks(rec)),tuple(cat.referenced_landmark_names(rec)))
        policy.set_episode(ep,builder.transform(rec.map_name),cat);tick=time.perf_counter()
        poses,term,times,diags=rollout(ep,builder,controller,policy.predict,GROUND_LEVEL[rec.map_name],ClosedLoopProtocol(),synchronize=torch.cuda.synchronize)
        metrics=score_trace(poses,rec.target_position,rec.trajectory,term,decision_seconds=times)
        row=dict(index=int(entry['index']),map_name=rec.map_name,metrics=metrics,poses=poses.tolist(),termination=term,diagnostics=diags,seconds=time.perf_counter()-tick)
        with (args.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row);save(dict(status='evaluating',completed=len(rows),total=len(entries)),args.output/'status.json');print(f'{len(rows)}/{len(entries)}',flush=True)
    diags=[d for r in rows for d in r['diagnostics']];seconds=time.perf_counter()-started
    result=aggregate([r['metrics'] for r in rows]);result.update(episodes=len(rows),gate_enabled=bool(active),local_updates=sum(d['local_updated'] for d in diags),
        mean_builder_ms=1000*float(np.mean(render_seconds)),rollout_ms_per_decision=1000*seconds/max(len(diags),1),
        scope='paired subset only; original 28x28 and XY metrics; disabled gate equals original policy')
    save(result,args.output/'summary.json');save(dict(status='complete',completed=len(rows)),args.output/'status.json');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
