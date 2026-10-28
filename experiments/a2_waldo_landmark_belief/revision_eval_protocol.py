"""Shared post-rollout metrics and immutable evaluation protocol helpers."""
import base64,zlib
from pathlib import Path
import numpy as np
import cv2
from .paths import ROOT,sha,bootstrap
bootstrap()
from graphnav.landmark_projection import world_to_frame

FULL_COUNTS=dict(val_seen=2470,val_unseen=2697)


def controller_consistency(row,bounds,ground,protocol):
    """Replay saved public commands with the original frozen controller."""
    from sbfnav.navigation import BoundedWaypointController
    controller=BoundedWaypointController(horizontal_step_m=50.,vertical_step_m=10.,
        minimum_above_ground_m=10.,maximum_above_ground_m=100.)
    actual=np.asarray(row['poses'],dtype=np.float64)
    if actual.ndim!=2 or actual.shape[1]!=4 or not np.isfinite(actual).all():
        raise ValueError('Invalid saved poses')
    commands=row['step_diagnostics']
    if not 0<len(commands)<=protocol.horizon:raise ValueError('Invalid decision count')
    replay=[actual[0]];stagnant=0;termination='horizon'
    for i,command in enumerate(commands):
        waypoint=np.asarray(command['commanded_xyz'],dtype=np.float64)
        if waypoint.shape!=(3,) or not np.isfinite(waypoint).all():raise ValueError('Invalid saved command')
        if command['commanded_stop']:
            termination='policy_stop'
        else:
            next_pose=controller.step(replay[-1],waypoint,ground_level=ground,xy_bounds=bounds)
            displacement=np.linalg.norm(next_pose[:3]-replay[-1][:3])
            replay.append(next_pose)
            stagnant=stagnant+1 if displacement<=protocol.stagnation_tolerance_m else 0
            if stagnant>=protocol.stagnation_steps:termination='stagnation'
        if termination!='horizon':
            if i!=len(commands)-1:raise ValueError('Commands recorded after termination')
            break
    if termination=='horizon' and len(commands)!=protocol.horizon:raise ValueError('Truncated horizon trace')
    expected=np.asarray(replay)
    if (termination!=row['termination'] or expected.shape!=actual.shape
        or not np.allclose(expected,actual,rtol=0,atol=1e-8)):
        raise ValueError('Saved trajectory does not follow frozen controller commands')
    return True


def evaluation_code_hashes():
    names=('revision_evaluation.py','revision_eval_protocol.py','revision_freeze.py','revision_policy.py','revision_runtime.py',
           'revision_fusion.py','revision_model.py','revision_artifacts.py','revision_confirmation.py','revision_calibrate.py',
           'revision_preflight.py','calibration_protocol.py','policy.py','perception.py','paths.py',
           'revision_results.py','revision_paired_stats.py')
    return {n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names}


def sensor_payload(observation,candidates):
    bgr=observation['bgr']
    if bgr.shape!=(384,384,3):raise ValueError('Native384 sensor required')
    valid=(bgr.max(-1)>0).reshape(-1)
    packed=base64.b64encode(zlib.compress(np.packbits(valid).tobytes())).decode()
    return dict(shape=[384,384],valid_bitmap=packed,footprint_world=np.asarray(observation['footprint_world']).tolist(),
                candidates=[dict(kind=int(c['kind']),poly=c['poly'],confidence=float(c.get('confidence',0.))) for c in candidates])


def target_observation_coverage(sensor,target_xy,target_kind):
    shape=sensor['shape'];bits=np.frombuffer(zlib.decompress(base64.b64decode(sensor['valid_bitmap'])),dtype=np.uint8)
    valid=np.unpackbits(bits)[:int(np.prod(shape))].reshape(shape).astype(bool)
    pixel=world_to_frame(np.asarray(target_xy,dtype=np.float32)[None,:2],sensor['footprint_world'],(*shape,3))[0]
    visible=bool(np.isfinite(pixel).all() and (pixel>=0).all() and (pixel<=383).all())
    if visible:visible=bool(valid[int(pixel[1]),int(pixel[0])])
    point=near=False
    for candidate in sensor['candidates']:
        if candidate['kind']!=target_kind:continue
        signed=cv2.pointPolygonTest(np.asarray(candidate['poly'],np.float32),tuple(map(float,target_xy[:2])),True)
        point=point or signed>=0;near=near or signed>=-20
    return dict(target_visible=visible,candidate_point_covered=bool(visible and point),candidate_within20m=bool(visible and near))


def recall_row(diagnostics,target_xy):
    if not diagnostics:raise ValueError('No observed belief for Recall')
    result={}
    for label,points in (('initial_static',diagnostics[0]['static_top16_world_xy']),('dynamic_first',diagnostics[0]['belief_top16_world_xy']),('dynamic_final',diagnostics[-1]['belief_top16_world_xy'])):
        distance=np.linalg.norm(np.asarray(points)-np.asarray(target_xy)[:2],axis=-1)
        result[label+'_NE_m']=float(distance[0])
        for k in (1,4,8,16):result[label+f'_R@{k}/20m']=float(distance[:k].min()<=20)
    return result


def pilot_decision(a0,a2):
    if not a0 or len(a0)!=len(a2):raise ValueError('Paired pilot population mismatch')
    gains=losses=changed=updates=0;ne=[]
    for before,after in zip(a0,a2):
        if 'index' in before and before['index']!=after['index']:raise ValueError('Pilot episode identities differ')
        b=before['metrics'];a=after['metrics']
        gains+=int(a['SR']>b['SR']);losses+=int(a['SR']<b['SR']);ne.append(a['NE']-b['NE'])
        changed+=int(not np.array_equal(np.asarray(before['poses']),np.asarray(after['poses'])))
        updates+=sum(bool(d.get('a2_updated',False)) for d in after['step_diagnostics'])
    checks=dict(sr_losses_not_above_gains=losses<=gains,no_mean_ne_increase=float(np.mean(ne))<=0,
                nonzero_updates=updates>0,changed_real_trajectories=changed>0)
    return dict(passed=all(checks.values()),checks=checks,episodes=len(a0),corrected_episodes=gains,broken_episodes=losses,
                mean_ne_difference_m=float(np.mean(ne)),updated_steps=updates,changed_trajectories=changed,
                scope='TRAIN-internal real closed-loop pilot; final Seen/Unseen benefit not established')


def require_population(scope,entries):
    expected=64 if scope=='pilot' else FULL_COUNTS[scope]
    if len(entries)!=expected or len({e['index'] for e in entries})!=expected:raise ValueError('Incomplete/duplicate evaluation population')
    if scope!='pilot' and {e['index'] for e in entries}!=set(range(expected)):raise ValueError('Formal source indices missing')


def write_trace(row,path):
    from .paths import save
    path=Path(path);save(row,path)
    save(dict(trace_sha256=sha(path),signature_sha256=row['signature_sha256'],index=row['index']),path.with_suffix('.sha.json'))


def read_trace(path,signature,index):
    import json
    path=Path(path);bound=json.loads(path.with_suffix('.sha.json').read_text())
    if bound['trace_sha256']!=sha(path) or bound['signature_sha256']!=signature or bound['index']!=index:raise ValueError('Episode trace checksum/identity changed')
    row=json.loads(path.read_text())
    if row['signature_sha256']!=signature or row['index']!=index:raise ValueError('Episode trace contents identity changed')
    return row
