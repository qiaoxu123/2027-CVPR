"""Paired resampling: identical draws across models, including map clusters."""
import numpy as np


def intervals(values, maps, seed=20261010, draws=2000):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return dict(n=0, mean=None, iid_95ci=None, map_cluster_95ci=None, maps=0)
    groups = [np.flatnonzero(np.asarray(maps) == m) for m in sorted(set(maps))]
    rng = np.random.default_rng(seed)
    iid, clustered = [], []
    for _ in range(draws):
        iid.append(x[rng.integers(len(x), size=len(x))].mean())
        ids = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        clustered.append(x[ids].mean())
    return dict(n=len(x), mean=float(x.mean()), iid_95ci=np.quantile(iid, [.025,.975]).tolist(),
                map_cluster_95ci=np.quantile(clustered, [.025,.975]).tolist(), maps=len(groups),
                cluster_warning='few maps: interval unstable' if len(groups)<10 else None)


def macro_f1(truth, pred):
    classes = sorted(set(truth))
    if not classes:
        return None
    scores=[]
    for c in classes:
        tp=sum(t==c and p==c for t,p in zip(truth,pred))
        fp=sum(t!=c and p==c for t,p in zip(truth,pred))
        fn=sum(t==c and p!=c for t,p in zip(truth,pred))
        scores.append(2*tp/max(1,2*tp+fp+fn))
    return float(np.mean(scores))
