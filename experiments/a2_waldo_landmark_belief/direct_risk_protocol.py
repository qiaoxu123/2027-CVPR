"""Fixed score-quantile operating points; private metrics never enter forward."""
import numpy as np

GAIN_QUANTILES=(0.,.25,.5,.75,.9,.95)
HARM_QUANTILES=(.05,.1,.25,.5,.75)
TEMPERATURES=(.5,1.,2.,4.)


def group_effects(identities,before,after,allow):
    before=np.asarray(before,dtype=float);after=np.asarray(after,dtype=float);allow=np.asarray(allow,dtype=bool)
    if before.ndim!=1 or after.shape!=before.shape or allow.shape!=before.shape or len(identities)!=len(before) or not len(before):raise ValueError('Incomplete paired target observations')
    if not np.isfinite(before).all() or not np.isfinite(after).all() or (before<0).any() or (after<0).any():raise ValueError('Invalid private distances')
    groups={};indices=[]
    for identity in identities:
        key=(identity['map_name'],identity['object_id']);indices.append(groups.setdefault(key,len(groups)))
    indices=np.asarray(indices);counts=np.bincount(indices)
    actual=np.where(allow,after,before);gain=(before>20)&(actual<=20);harm=(before<=20)&(actual>20)
    g=np.bincount(indices,weights=gain);h=np.bincount(indices,weights=harm)
    baseline=np.bincount(indices,weights=before<=20)>0;broken=int((h>0).sum())
    return dict(groups=len(groups),views=len(before),corrected=int(gain.sum()),broken=int(harm.sum()),
        gain_groups=int((g>h).sum()),broken_groups=broken,a0_correct_groups=int(baseline.sum()),
        supported_groups=int((np.bincount(indices,weights=allow)>0).sum()),
        broken_group_rate=broken/int(baseline.sum()) if baseline.any() else None,
        mean_r1_difference=float(((g-h)/counts).mean()),
        mean_ne_difference_m=float((np.bincount(indices,weights=actual-before)/counts).mean()))


def operating_points(scores,available,identities,before,after):
    scores=np.asarray(scores,dtype=float);available=np.asarray(available,dtype=bool)
    if scores.shape!=(len(available),3) or not np.isfinite(scores).all() or (scores<0).any() or not np.allclose(scores.sum(-1),1.,atol=1e-6):raise ValueError('Invalid threeclass scores')
    if not available.any():return []
    points=[]
    for gq in GAIN_QUANTILES:
        minimum=float(np.quantile(scores[available,1],gq,method='linear'))
        for hq in HARM_QUANTILES:
            maximum=float(np.quantile(scores[available,2],hq,method='linear'))
            allow=available&(scores[:,1]>=minimum)&(scores[:,2]<=maximum)
            points.append(dict(index=len(points),gain_quantile=gq,harm_quantile=hq,
                parameters=dict(minimum_gain=minimum,maximum_harm=maximum),summary=group_effects(identities,before,after,allow)))
    return points


def point_key(point):
    s=point['summary']
    return (s['mean_r1_difference'],-s['broken_groups'],-s['mean_ne_difference_m'],-s['supported_groups'],
            point['gain_quantile'],-point['harm_quantile'],-point['index'])


def select_point(points,*,dev):
    def viable(p):
        s=p['summary']
        return (s['mean_r1_difference']>0 and s['mean_ne_difference_m']<=0 and s['supported_groups']>=(30 if dev else 10)
            and s['gain_groups']>=(5 if dev else 0) and s['broken_group_rate'] is not None and s['broken_group_rate']<=.02)
    return max((p for p in points if viable(p)),key=point_key,default=None)
