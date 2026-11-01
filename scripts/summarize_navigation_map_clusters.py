"""Completed paired navigation comparison with whole-map bootstrap robustness."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.summarize_native_navigation import canonical_data_hashes
from graphnav.checkpointing import atomic_json,file_sha256


def cluster_effect(difference,maps,repeats=20000,seed=0):
    difference=np.asarray(difference,float);names=sorted(set(maps));ids=np.array([names.index(name) for name in maps]);counts=np.bincount(ids);sums=np.bincount(ids,weights=difference);means=sums/counts
    sampled=np.random.default_rng(seed).integers(len(names),size=(repeats,len(names)))
    weighted=sums[sampled].sum(1)/counts[sampled].sum(1);equal=means[sampled].mean(1)
    return dict(episode_weighted_difference=float(difference.mean()),map_equal_difference=float(means.mean()),map_cluster_95ci_episode_weighted=np.quantile(weighted,[.025,.975]).tolist(),map_cluster_95ci_map_equal=np.quantile(equal,[.025,.975]).tolist(),by_map={name:dict(episodes=int(counts[i]),difference=float(means[i])) for i,name in enumerate(names)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    roots=list(map(Path,[a.baseline,a.candidate]));reports=[json.loads((r/'metrics.json').read_text()) for r in roots];rows=[[json.loads(s) for s in (r/'predictions.jsonl').read_text().splitlines()] for r in roots]
    if reports[0]['episode_manifest_sha256']!=reports[1]['episode_manifest_sha256'] or reports[0]['checkpoint_sha256']!=reports[1]['checkpoint_sha256'] or canonical_data_hashes(reports[0]['input_sha256'])!=canonical_data_hashes(reports[1]['input_sha256']) or reports[0]['protocol']!=reports[1]['protocol']:raise ValueError('Unmatched completed evaluation provenance')
    if any(len(x)!=r['metrics']['episodes'] for x,r in zip(rows,reports)):raise ValueError('Incomplete evaluation')
    if [(x['episode_id'],x['index'],x['map_name']) for x in rows[0]]!=[(x['episode_id'],x['index'],x['map_name']) for x in rows[1]]:raise ValueError('Unmatched paired episodes/maps')
    maps=[x['map_name'] for x in rows[0]];effects={}
    for metric,scale in [('SR',100),('OSR',100),('SPL',100),('NE',1),('steps',1)]:
        delta=[scale*(y['metrics'][metric]-x['metrics'][metric]) for x,y in zip(*rows)];effects[metric]=cluster_effect(delta,maps)
    result=dict(episodes=len(maps),maps=len(set(maps)),effects=effects,baseline_report_sha=file_sha256(roots[0]/'metrics.json'),candidate_report_sha=file_sha256(roots[1]/'metrics.json'),protocol='20000 paired whole-map bootstrap draws, seed0, preserve all episodes within sampled maps. Report both episode-weighted and map-equal effects. Pointwise intervals; few held-out maps limit geographic generalization.')
    atomic_json(result,Path(a.output));print(json.dumps({k:v for k,v in result.items() if k!='effects'}));print(json.dumps({k:{n:v for n,v in effect.items() if n!='by_map'} for k,effect in effects.items()},indent=2))


if __name__=='__main__':main()
