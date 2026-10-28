"""Pure paired final-report statistics; never selects checkpoints or gates.

Input rows must first be independently rescored by navigation. Target-cluster
resampling retains complete paired episodes; query Recall deduplicates the first
source-index occurrence of map+instruction, matching the original evaluator.
"""
import numpy as np
from scipy.stats import binomtest

METRICS=('SR','OSR','SPL','NE','nDTW','sDTW','policy_stop','false_stop')
IDENTITY=('map_name','object_id','instruction')


def _align(a0,a2):
    if not a0 or len(a0)!=len(a2):raise ValueError('Incomplete paired population')
    indexed=[]
    for rows in (a0,a2):
        index={r['index']:r for r in rows}
        if len(index)!=len(rows):raise ValueError('Duplicate paired source index')
        indexed.append(index)
    if set(indexed[0])!=set(indexed[1]):raise ValueError('Different source episodes')
    left,right=([index[i] for i in sorted(index)] for index in indexed)
    recall_keys=set(left[0]['recall'])
    if not recall_keys:raise ValueError('Missing Recall measurements')
    for before,after in zip(left,right):
        if any(before[k]!=after[k] for k in IDENTITY):raise ValueError('Paired source query/target mismatch')
        for row in (before,after):
            if set(row['recall'])!=recall_keys:raise ValueError('Inconsistent Recall schema')
            try:values=np.asarray([row['metrics'][k] for k in METRICS]+list(row['recall'].values()),dtype=float)
            except (KeyError,TypeError,ValueError) as error:raise ValueError('Missing or invalid final metrics') from error
            if not np.isfinite(values).all():raise ValueError('Nonfinite final metrics')
            m=row['metrics']
            if any(m[k] not in (0,1) for k in ('SR','OSR','policy_stop','false_stop')) or m['false_stop']>m['policy_stop']:
                raise ValueError('Invalid binary success/stop measurements')
            if m['false_stop'] and m['SR']:raise ValueError('Successful episode cannot be a false stop')
            if not row['step_diagnostics'] or len(row['inference_seconds'])!=len(row['step_diagnostics']) or len(row['coverage'])!=len(row['step_diagnostics']):
                raise ValueError('Incomplete inference/coverage records')
    return left,right


def _bootstrap(numerators,denominators,rows,cluster,draws,seed):
    groups={}
    for i,row in enumerate(rows):
        key=(row['map_name'],row['object_id']) if cluster=='target' else row['map_name']
        groups.setdefault(key,[]).append(i)
    if cluster=='map' and len(groups)<2:return None
    sums=np.stack([numerators[indices].sum(0) for indices in groups.values()])
    counts=np.stack([denominators[indices].sum(0) for indices in groups.values()])
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,draws,64):
        selected=rng.integers(len(groups),size=(min(64,draws-start),len(groups)))
        n=sums[selected].sum(1);d=counts[selected].sum(1)
        with np.errstate(divide='ignore',invalid='ignore'):samples.append(n[:,:,1]/d[:,:,1]-n[:,:,0]/d[:,:,0])
    samples=np.concatenate(samples)
    result=[]
    for column in samples.T:
        valid=column[np.isfinite(column)]
        result.append((np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,len(valid)))
    return result


def _summaries(before,after,keys,rows,draws,seed,*,ratio=None):
    numerators=np.stack([before,after],axis=-1).astype(float)
    denominators=np.ones_like(numerators)
    if ratio is not None:
        column,den_before,den_after=ratio
        denominators[:,column,0]=den_before;denominators[:,column,1]=den_after
    totals=numerators.sum(0);counts=denominators.sum(0)
    target=_bootstrap(numerators,denominators,rows,'target',draws,seed)
    maps=_bootstrap(numerators,denominators,rows,'map',draws,seed)
    result={}
    for j,key in enumerate(keys):
        values=[float(totals[j,k]/counts[j,k]) if counts[j,k] else None for k in (0,1)]
        result[key]=dict(A0=values[0],A2=values[1],difference=values[1]-values[0] if None not in values else None,
            A0_denominator=int(counts[j,0]),A2_denominator=int(counts[j,1]),
            target_cluster_bootstrap95=target[j][0],target_cluster_valid_draws=target[j][1],
            map_cluster_bootstrap95=maps[j][0] if maps else None,map_cluster_valid_draws=maps[j][1] if maps else 0)
    return result


def _latency(rows):
    values=np.asarray([v for row in rows for v in row['inference_seconds']],dtype=float)*1000
    if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Finite positive synchronized inference times required')
    return dict(decisions=len(values),mean_ms=float(values.mean()),median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)))


def _mechanism(a0,a2):
    updated=[any(d.get('a2_updated',False) for d in row['step_diagnostics']) for row in a2]
    diagnostics=[d for row in a2 for d in row['step_diagnostics']]
    steps=sum(bool(d.get('a2_updated',False)) for d in diagnostics)
    gains=sum(after['metrics']['SR']>before['metrics']['SR'] for before,after in zip(a0,a2))
    losses=sum(after['metrics']['SR']<before['metrics']['SR'] for before,after in zip(a0,a2))
    subset=[(before,after) for before,after,selected in zip(a0,a2,updated) if selected]
    coverage=[c for row in a2 for c in row['coverage']]
    visible=sum(bool(c['target_visible']) for c in coverage)
    covered=sum(bool(c['candidate_point_covered']) for c in coverage)
    near=sum(bool(c['candidate_within20m']) for c in coverage)
    if covered>visible or near>visible:raise ValueError('Candidate coverage exceeds visible denominator')
    pairs=dict(corrected=int(gains),broken=int(losses),unchanged=len(a0)-gains-losses,
        corrected_fraction=gains/len(a0),broken_fraction=losses/len(a0),
        mcnemar_exact_descriptive_p=float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.)
    failures=sum(not row['metrics']['SR'] for row in a0);successes=len(a0)-failures
    pairs.update(A0_failures=failures,A0_successes=successes,corrected_fraction_among_A0_failures=gains/failures if failures else None,
        broken_fraction_among_A0_successes=losses/successes if successes else None)
    mechanism=dict(observed_steps=len(diagnostics),updated_steps=steps,updated_episodes=sum(updated),fallback_episodes=len(a2)-sum(updated),
        fallback_episode_fraction=(len(a2)-sum(updated))/len(a2),fallback_step_fraction=(len(diagnostics)-steps)/len(diagnostics),
        detector_cap_steps=sum(d.get('a2_candidates',0)>=300 for d in diagnostics),
        detector_failure_steps=sum(bool(d.get('a2_detection_error')) for d in diagnostics),
        updated_subset=dict(episodes=len(subset),descriptive_only=True,
            corrected=sum(b['metrics']['SR']>a['metrics']['SR'] for a,b in subset),broken=sum(b['metrics']['SR']<a['metrics']['SR'] for a,b in subset),
            mean_NE_difference_m=float(np.mean([b['metrics']['NE']-a['metrics']['NE'] for a,b in subset])) if subset else None,
            note='Selected by actual A2 behavior; descriptive subset, not an independent efficacy estimate.'))
    coverage=dict(visible_target_views=visible,point_covered_views=covered,within20m_views=near,
        point_coverage=covered/visible if visible else None,within20m_coverage=near/visible if visible else None,
        definition='Post-rollout target inside actual same-category WALDO polygon, or its20m buffer; visible RGB-valid target views form the denominator.')
    return pairs,mechanism,coverage


def _compare(a0,a2,draws,seed):
    keys=list(METRICS)+['False_Stop_Rate']
    before=np.asarray([[r['metrics'][k] for k in METRICS]+[r['metrics']['false_stop']] for r in a0])
    after=np.asarray([[r['metrics'][k] for k in METRICS]+[r['metrics']['false_stop']] for r in a2])
    navigation=_summaries(before,after,keys,a0,draws,seed,
        ratio=(len(keys)-1,[r['metrics']['policy_stop'] for r in a0],[r['metrics']['policy_stop'] for r in a2]))
    navigation['false_stop_episode_fraction']=navigation.pop('false_stop')
    queries=set();indices=[]
    for i,row in enumerate(a0):
        key=(row['map_name'],row['instruction'])
        if key not in queries:queries.add(key);indices.append(i)
    recall_keys=sorted(a0[0]['recall'])
    recall=_summaries(np.asarray([[a0[i]['recall'][k] for k in recall_keys] for i in indices]),
        np.asarray([[a2[i]['recall'][k] for k in recall_keys] for i in indices]),recall_keys,[a0[i] for i in indices],draws,seed)
    pairs,mechanism,coverage=_mechanism(a0,a2)
    latency={name:_latency(rows) for name,rows in (('A0',a0),('A2',a2))}
    latency['mean_difference_ms']=latency['A2']['mean_ms']-latency['A0']['mean_ms']
    return dict(episodes=len(a0),unique_queries=len(indices),target_clusters=len({(r['map_name'],r['object_id']) for r in a0}),
        maps=len({r['map_name'] for r in a0}),navigation=navigation,recall=recall,SR_pairs=pairs,mechanism=mechanism,candidate_coverage=coverage,latency=latency)


def compare_pairs(a0,a2,*,draws=5000,seed=0):
    if not isinstance(draws,int) or draws<100:raise ValueError('At least100 paired bootstrap draws required')
    a0,a2=_align(a0,a2);result=_compare(a0,a2,draws,seed)
    result['by_map']={name:_compare([r for r in a0 if r['map_name']==name],[r for r in a2 if r['map_name']==name],draws,seed)
        for name in sorted({r['map_name'] for r in a0})}
    result.update(bootstrap_draws=draws,seed=seed,
        interpretation='A2 minus A0. Primary95% intervals resample target instances, keeping all paired episodes. Map-cluster intervals are descriptive with few maps. McNemar p treats episodes as independent and is descriptive. One training seed; no across-seed claim.',
        ratio_note='False_Stop_Rate uses false policy stops / all policy stops in each arm. A zero denominator is undefined, never zero. Ratio intervals use only resampled draws with both denominators positive; valid draw counts are disclosed.',
        recall_note='One first source-index episode per map+instruction, matching original evaluator; final observed belief may precede the final commanded pose.',
        latency_note='All synchronized policy decisions; excludes PNG/PT/trace serialization. Different cache warmth/shared background speech traffic may affect times.')
    return result
