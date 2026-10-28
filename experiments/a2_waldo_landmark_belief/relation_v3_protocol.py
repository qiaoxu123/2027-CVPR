"""Predeclared development selection; independent FIT/VERIFY are not inputs."""
import math
import numpy as np
from .direct_risk_protocol import operating_points,group_effects,point_key
from .relation_v3_loss import LOSS_WEIGHTS,FIXED_ACTION_WEIGHT,FEATURE_NAMES_V3

CONFIG=dict(version='a2_relation_v3',seed=0,batch_size=8,workers=2,lr=.0003,weight_decay=.01,
    rank_epochs=20,refine_epochs=5,gradient_clip=5.,real_source_weight=8.,
    warmstart='v1 selected base/category only; no A0 context heads',
    stage21='load best of all20 rank checkpoints; reset AdamW; jointly refine actor and risk for all5',
    fixed_action=FIXED_ACTION_WEIGHT,loss_weights=LOSS_WEIGHTS,risk_class_weights=[1.,8.,6.],
    risk_features=list(FEATURE_NAMES_V3),temperature=1.,
    rank_selection='real DEV proposal R1 then NE; teacher DEV proposal R1 then NE; earlier epoch',
    refinement_selection='safe real DEV operating point and teacher preservation first; same-checkpoint fresh Q/outcomes; earlier epoch',
    dev_gate=dict(minimum_supported_real_groups=30,minimum_gain_real_groups=5,maximum_broken_group_rate=.02,
                  real_positive_r1=True,real_nonincreasing_ne=True,teacher_nonnegative_r1=True,teacher_nonincreasing_ne=True),
    no_eligible_gate='diagnostic checkpoint only, no independent FIT collection or benchmark enable')


def safe_dev_points(real,teacher):
    """Fit threshold menu on real DEV only, check same choice on teacher DEV."""
    for population in (real,teacher):
        scores=np.asarray(population['scores'],dtype=float);available=np.asarray(population['available'])
        if (available.ndim!=1 or available.dtype!=np.bool_ or scores.shape!=(len(available),3)
            or not np.isfinite(scores).all() or (scores<0).any() or not np.allclose(scores.sum(-1),1.,atol=1e-6)):
            raise ValueError('Invalid real/teacher threeclass scores')
        # Validate axes/distances even if no threshold point is available.
        group_effects(population['identities'],population['before_ne'],population['after_ne'],available)
    points=operating_points(real['scores'],real['available'],real['identities'],real['before_ne'],real['after_ne'])
    for p in points:
        params=p['parameters'];scores=np.asarray(teacher['scores'])
        allow=np.asarray(teacher['available'],dtype=bool)&(scores[:,1]>=params['minimum_gain'])&(scores[:,2]<=params['maximum_harm'])
        p['teacher_summary']=group_effects(teacher['identities'],teacher['before_ne'],teacher['after_ne'],allow)
        r=p['summary'];t=p['teacher_summary']
        p['eligible']=bool(r['supported_groups']>=30 and r['gain_groups']>=5 and r['mean_r1_difference']>0
            and r['mean_ne_difference_m']<=0 and r['broken_group_rate'] is not None and r['broken_group_rate']<=.02
            and t['mean_r1_difference']>=0 and t['mean_ne_difference_m']<=0
            and t['broken_group_rate'] is not None and t['broken_group_rate']<=.02)
    chosen=max((p for p in points if p['eligible']),key=point_key,default=None)
    return dict(selected_point=chosen,operating_points=points,eligible=chosen is not None)


def selection_key(record):
    dev=record['dev'];real=dev['real'];teacher=dev['teacher'];epoch=record['epoch']
    rank=(real['proposal_r1'],-real['proposal_ne'],teacher['proposal_r1'],-teacher['proposal_ne'],-epoch)
    if epoch<=20:return rank
    point=dev['selected_point']
    return (point is not None,*(point_key(point) if point is not None else (0.,)*7),*rank)


def choose_stage(history,*,refine):
    records=[r for r in history if (r['epoch']>20)==refine]
    if not records:raise ValueError('No checkpoint in requested stage')
    return max(records,key=selection_key)


def verify_history(history):
    if len(history)!=25 or [r['epoch'] for r in history]!=list(range(1,26)):raise ValueError('All20+5 sequential epochs required')
    for record in history:
        epoch=record['epoch'];refine=epoch>20
        if record['stage']!=('refinement' if refine else 'ranking') or record['stage_epoch']!=(epoch-20 if refine else epoch):raise ValueError('Training stage changed')
        for role,count in (('train',33440),('dev',5647)):
            steps=math.ceil(count/CONFIG['batch_size']) if role=='train' else 0
            if record[role]['views']!=count or record[role]['optimizer_updates']!=steps:raise ValueError('Incomplete epoch population or updates')
