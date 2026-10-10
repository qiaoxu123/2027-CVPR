"""Paired state, family and map bootstrap summaries for the fixed Seen pilot."""
from collections import defaultdict
import numpy as np


def interval(values, *, groups=None, seed=20261010, repetitions=2000):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):return None
    if not np.isfinite(values).all():raise ValueError('nonfinite statistical observations')
    rng=np.random.default_rng(seed);estimates=[]
    if groups is None:
        for offset in range(0,repetitions,256):
            size=min(256,repetitions-offset);ids=rng.integers(len(values),size=(size,len(values)))
            estimates.extend(values[ids].mean(-1).tolist())
    else:
        if len(groups)!=len(values):raise ValueError('cluster alignment mismatch')
        sums=defaultdict(float);counts=defaultdict(int)
        for group,value in zip(groups,values):sums[str(group)]+=float(value);counts[str(group)]+=1
        keys=sorted(sums);a=np.array([sums[k] for k in keys]);n=np.array([counts[k] for k in keys])
        for offset in range(0,repetitions,256):
            size=min(256,repetitions-offset);ids=rng.integers(len(keys),size=(size,len(keys)))
            estimates.extend((a[ids].sum(-1)/n[ids].sum(-1)).tolist())
    return [float(x) for x in np.quantile(estimates,[.025,.975])]


def summarize_rows(rows, key, *, repetitions=2000, seed=20261010):
    chosen=[r for r in rows if r.get('metrics',{}).get(key) is not None]
    if not chosen:return dict(n=0,mean=None,iid_95ci=None,family_cluster_95ci=None,map_cluster_95ci=None)
    values=[r['metrics'][key] for r in chosen];families=[r['family_id'] for r in chosen];maps=[r['map_name'] for r in chosen]
    return dict(n=len(values),mean=float(np.mean(values)),independent_families=len(set(families)),maps=len(set(maps)),
                iid_95ci=interval(values,repetitions=repetitions,seed=seed),
                family_cluster_95ci=interval(values,groups=families,repetitions=repetitions,seed=seed),
                map_cluster_95ci=interval(values,groups=maps,repetitions=repetitions,seed=seed),
                cluster_warning='fewer than 5 maps; map uncertainty unstable' if len(set(maps))<5 else None)


def paired_difference(left,right,key, *, repetitions=2000,seed=20261010, identity='sample_id'):
    def index(rows):
        result={}
        for row in rows:
            if row[identity] in result:raise ValueError('duplicate paired identity '+str(row[identity]))
            result[row[identity]]=row
        return result
    a,b=index(left),index(right)
    if set(a)!=set(b):raise ValueError('paired populations differ')
    deltas=[]
    for sample in sorted(a):
        x,y=a[sample],b[sample]
        if (x['family_id'],x['map_name'])!=(y['family_id'],y['map_name']):raise ValueError('paired metadata differs')
        xv,yv=x.get('metrics',{}).get(key),y.get('metrics',{}).get(key)
        if (xv is None)!=(yv is None):raise ValueError('paired metric eligibility differs for '+key)
        if xv is not None:deltas.append(dict(sample_id=sample,family_id=x['family_id'],map_name=x['map_name'],metrics={key:float(yv)-float(xv)}))
    return summarize_rows(deltas,key,repetitions=repetitions,seed=seed)
