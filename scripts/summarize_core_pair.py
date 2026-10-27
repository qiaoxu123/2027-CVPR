"""Paired full-split reporting for the pre-frozen core comparison only."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.closedloop import aggregate

ARMS=('A0_original_residual','A1_joint_rgb_geometry')
METRICS=('SR','OSR','SPL','NE','nDTW','sDTW','False_Stop_Rate','false_stop_episode_fraction')


def format_metric(value,signed=False):
    if value is None:return 'N/A'
    return format(value,'+.6f' if signed else '.6f')


def paired_statistics(left,right,repetitions=10000):
    if len(left)!=len(right) or not left:raise ValueError('paired nonempty rows required')
    for a,b in zip(left,right):
        if (a['index'],a['episode_id'],a['map_name'],a['public_query_sha256'])!=(b['index'],b['episode_id'],b['map_name'],b['public_query_sha256']):
            raise ValueError('paired episode identity mismatch')
    result={};rng=np.random.default_rng(0)
    for key in METRICS:
        source='false_stop' if key in ('False_Stop_Rate','false_stop_episode_fraction') else key
        a=np.asarray([r['metrics'][source]for r in left]);b=np.asarray([r['metrics'][source]for r in right]);delta=b-a
        if key=='False_Stop_Rate':
            sa=np.asarray([r['metrics']['policy_stop']for r in left]);sb=np.asarray([r['metrics']['policy_stop']for r in right])
            estimate=(b.sum()/sb.sum()-a.sum()/sa.sum()) if sa.sum() and sb.sum() else None
        else:estimate=float(delta.mean())
        boot=[]
        for start in range(0,repetitions,100):
            ids=rng.integers(0,len(a),(min(100,repetitions-start),len(a)))
            if key=='False_Stop_Rate':
                da=sa[ids].sum(-1);db=sb[ids].sum(-1);valid=(da>0)&(db>0)
                boot.extend((b[ids].sum(-1)[valid]/db[valid]-a[ids].sum(-1)[valid]/da[valid]).tolist())
            else:boot.extend(delta[ids].mean(-1).tolist())
        result[key]=dict(A1_minus_A0=estimate,paired_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist() if boot else None)
    a=np.asarray([r['metrics']['SR']for r in left]);b=np.asarray([r['metrics']['SR']for r in right]);gain=int(((a==0)&(b==1)).sum());loss=int(((a==1)&(b==0)).sum())
    result['SR_McNemar_exact']=dict(A1_only_success=gain,A0_only_success=loss,discordant=gain+loss,two_sided_p=float(binomtest(gain,gain+loss,.5).pvalue)if gain+loss else 1.)
    # Cluster resampling keeps duplicate map+instruction episodes together.
    groups=defaultdict(list)
    for i,row in enumerate(left):groups[row['public_query_sha256']].append(i)
    sums=np.asarray([sum(b[i]-a[i]for i in ids)for ids in groups.values()]);sizes=np.asarray([len(ids)for ids in groups.values()]);boot=[]
    for _ in range(repetitions):
        ids=rng.integers(0,len(sums),len(sums));boot.append(float(sums[ids].sum()/sizes[ids].sum()))
    result['SR_query_cluster_bootstrap_95ci']=np.quantile(boot,[.025,.975]).tolist()
    return result


def paired_recall(left,right,repetitions=10000):
    unique=[]
    for rows in (left,right):
        entries={}
        for row in rows:entries.setdefault(row['public_query_sha256'],row['goal_retrieval'])
        unique.append(entries)
    if list(unique[0])!=list(unique[1]):raise ValueError('paired retrieval queries differ')
    rng=np.random.default_rng(0);result=dict(unique_queries=len(unique[0]))
    for key in next(iter(unique[0].values())):
        if 'R@' not in key and not key.endswith('NE_m'):continue
        delta=np.asarray([unique[1][q][key]-unique[0][q][key]for q in unique[0]])
        boot=[]
        for start in range(0,repetitions,100):
            ids=rng.integers(0,len(delta),(min(100,repetitions-start),len(delta)));boot.extend(delta[ids].mean(-1).tolist())
        result[key]=dict(A1_minus_A0=float(delta.mean()),paired_query_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
    return result


def retrieval_summary(rows):
    unique={}
    for row in rows:unique.setdefault(row['public_query_sha256'],row['goal_retrieval'])
    return dict(unique_queries=len(unique),**{k:float(np.mean([v[k]for v in unique.values()]))for k in next(iter(unique.values()))if 'R@' in k or k.endswith('NE_m')})


def summarize(root):
    root=Path(root);frozen=json.loads((root/'frozen_pair.json').read_text());result=dict(frozen_pair_sha256=file_sha256(root/'frozen_pair.json'),frozen_at_utc=frozen['frozen_at_utc'],seed=0,splits={},statistics='10000 paired episode bootstrap samples, seed 0; SR additionally query-cluster bootstrap and exact two-sided McNemar. Single training seed; episode uncertainty is not training-seed uncertainty.')
    tables=[]
    for split,count in (('val_seen',2470),('val_unseen',2697)):
        reports=[];rows=[]
        for arm in ARMS:
            folder=root/arm/('full_'+split);report=json.loads((folder/'metrics.json').read_text());reports.append(report)
            records=[json.loads(line)for line in (folder/'predictions.jsonl').read_text().splitlines()];rows.append(records)
            if len(records)!=count or len({r['index']for r in records})!=count or report['metrics']['episodes']!=count:raise ValueError('incomplete full evaluation')
            if report['checkpoint_sha256']!=frozen['arms'][arm]['checkpoint_sha256']:raise ValueError('evaluation changed frozen checkpoint')
        if reports[0]['episode_manifest_sha256']!=reports[1]['episode_manifest_sha256']:raise ValueError('paired manifests differ')
        statistics=paired_statistics(*rows)
        maps={}
        for name in sorted({r['map_name']for r in rows[0]}):
            mr=[[r for r in records if r['map_name']==name]for records in rows]
            maps[name]=dict(episodes=len(mr[0]),A0=aggregate([r['metrics']for r in mr[0]]),A1=aggregate([r['metrics']for r in mr[1]]),retrieval_A0=retrieval_summary(mr[0]),retrieval_A1=retrieval_summary(mr[1]),paired=paired_statistics(*mr,repetitions=2000))
        result['splits'][split]=dict(episodes=count,episode_manifest_sha256=reports[0]['episode_manifest_sha256'],A0=reports[0]['metrics'],A1=reports[1]['metrics'],retrieval_A0=retrieval_summary(rows[0]),retrieval_A1=retrieval_summary(rows[1]),paired=statistics,paired_recall=paired_recall(*rows),by_map=maps)
        for key in METRICS:
            tables.append(dict(split=split,metric=key,A0=reports[0]['metrics'][key],A1=reports[1]['metrics'][key],**statistics[key]))
    atomic_json(result,root/'core_pair_results.json')
    report_dir=Path('reports/core_pair_20261027');report_dir.mkdir(exist_ok=True,parents=True);atomic_json(result,report_dir/'results.json')
    with (report_dir/'metrics.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    map_rows=[]
    for split,entry in result['splits'].items():
        for name,record in entry['by_map'].items():
            for key in METRICS:
                map_rows.append(dict(split=split,map=name,episodes=record['episodes'],metric=key,A0=record['A0'][key],A1=record['A1'][key],**record['paired'][key]))
    with (report_dir/'by_map.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(map_rows[0]));w.writeheader();w.writerows(map_rows)
    lines=['# Core RGB fusion pair, seed 0','',f"Both A3/B3/C5 training runs completed. Checkpoints frozen at {frozen['frozen_at_utc']} before full evaluation.",'','| Split | Metric | A0 | A1 | A1−A0 | Paired 95% CI |','|---|---|---:|---:|---:|---|']
    for row in tables:
        lines.append(f"| {row['split']} | {row['metric']} | {format_metric(row['A0'])} | {format_metric(row['A1'])} | {format_metric(row['A1_minus_A0'],signed=True)} | {row['paired_bootstrap_95ci']} |")
    lines+=['','## Target localization recall (unique public queries)','','| Split | Belief | K | A0 | A1 | A1−A0 | Paired query 95% CI |','|---|---|---:|---:|---:|---:|---|']
    for split,entry in result['splits'].items():
        for belief in ('initial_static','dynamic_first','dynamic_final'):
            for k in (1,4,8,16):
                key=belief+f'_R@{k}/20m';paired=entry['paired_recall'][key]
                lines.append(f"| {split} | {belief} | {k} | {entry['retrieval_A0'][key]:.6f} | {entry['retrieval_A1'][key]:.6f} | {paired['A1_minus_A0']:+.6f} | {paired['paired_query_bootstrap_95ci']} |")
    lines+=['','## Per-map navigation','','| Split | Map | Episodes | A0 SR | A1 SR | SR Δ (pp) | A0 SPL | A1 SPL | A0 NE | A1 NE |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split,entry in result['splits'].items():
        for name,record in entry['by_map'].items():
            a,b=record['A0'],record['A1']
            lines.append(f"| {split} | {name} | {record['episodes']} | {a['SR']:.6f} | {b['SR']:.6f} | {100*(b['SR']-a['SR']):+.3f} | {a['SPL']:.6f} | {b['SPL']:.6f} | {a['NE']:.3f} | {b['NE']:.3f} |")
    lines+=['','Fractions are used for SR/OSR/SPL/nDTW/sDTW/False-Stop, NE is meters. False-Stop Rate divides false stops by policy stops; false_stop_episode_fraction divides by all episodes. All per-map metrics, retrieval R@1/4/8/16 within 20m, paired statistics and hashes are in results.json. Goal recall is unique map+instruction query macro-average, with initial static, first observation, final observation separately. Final observation time can differ across arms.','',result['statistics'],'','Server artifacts: '+str(root.resolve()),'','No checkpoint or inference control was selected from full Seen or Unseen results.']
    (report_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);summarize(p.parse_args().root)
