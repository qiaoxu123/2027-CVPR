"""Paired episode effects with target-cluster uncertainty; no model selection."""
import numpy as np
from scipy.stats import binomtest


def paired_statistics(a0,a2,*,draws=5000,seed=0):
    if not a0 or len(a0)!=len(a2):raise ValueError('Incomplete paired population')
    if len({r['index'] for r in a0})!=len(a0):raise ValueError('Duplicate paired episode')
    if draws<100:raise ValueError('At least100 bootstrap draws required')
    for before,after in zip(a0,a2):
        if any(before[k]!=after[k] for k in ('index','map_name','object_id')):
            raise ValueError('Paired episode/source target mismatch')
    keys=sorted(set(a0[0]['metrics'])&set(a2[0]['metrics']))
    before=np.asarray([[r['metrics'][k] for k in keys] for r in a0],dtype=np.float64)
    after=np.asarray([[r['metrics'][k] for k in keys] for r in a2],dtype=np.float64)
    if not np.isfinite(before).all() or not np.isfinite(after).all():raise ValueError('Nonfinite paired metrics')
    groups={}
    for i,r in enumerate(a0):groups.setdefault((r['map_name'],r['object_id']),[]).append(i)
    values=list(groups.values());delta=after-before
    sums=np.asarray([delta[g].sum(0) for g in values]);counts=np.asarray([len(g) for g in values])
    # Resample target instances, retaining all their paired episodes. The ratio
    # preserves the reported episode-weighted estimand with unequal group sizes.
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,draws,128):
        indices=rng.integers(len(values),size=(min(128,draws-start),len(values)))
        samples.append(sums[indices].sum(1)/counts[indices].sum(1)[:,None])
    interval=np.quantile(np.concatenate(samples),[.025,.975],axis=0)
    sr=keys.index('SR');gains=int(np.sum(after[:,sr]>before[:,sr]));losses=int(np.sum(after[:,sr]<before[:,sr]))
    p=float(binomtest(gains,gains+losses,p=.5).pvalue) if gains+losses else 1.
    return dict(episodes=len(a0),target_clusters=len(values),bootstrap_draws=draws,seed=seed,
        metrics={k:dict(A0=float(before[:,j].mean()),A2=float(after[:,j].mean()),difference=float(delta[:,j].mean()),
                        cluster_bootstrap95=interval[:,j].tolist()) for j,k in enumerate(keys)},
        SR_pairs=dict(corrected=gains,broken=losses,unchanged=len(a0)-gains-losses,
                      corrected_fraction=gains/len(a0),broken_fraction=losses/len(a0),mcnemar_exact_p=p),
        interpretation='A2 minus A0; target-instance cluster paired bootstrap95%; McNemar exact descriptive p treats episodes independent, so cluster CI is primary; one training seed, no across-seed uncertainty')
