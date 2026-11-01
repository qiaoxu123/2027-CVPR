"""Paired complete-rollout comparison; rejects partial or mismatched episode sets."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from graphnav.workspace_paths import PROJECT,WORKSPACE
from graphnav.checkpointing import atomic_json


def canonical_data_hashes(hashes):
 result={}
 for path,digest in hashes.items():
  key=path.split('/sbfnav-data/',1)[1] if '/sbfnav-data/' in path else path
  if key in result:raise ValueError('Ambiguous dataset provenance path')
  result[key]=digest
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 dirs=[Path(a.baseline),Path(a.candidate)];reports=[json.loads((d/'metrics.json').read_text()) for d in dirs]
 rows=[[json.loads(line) for line in (d/'predictions.jsonl').read_text().splitlines()] for d in dirs]
 assert reports[0]['episode_manifest_sha256']==reports[1]['episode_manifest_sha256'],'Different rollout episode manifests'
 assert reports[0]['checkpoint_sha256']==reports[1]['checkpoint_sha256'],'Different original checkpoint'
 assert canonical_data_hashes(reports[0]['input_sha256'])==canonical_data_hashes(reports[1]['input_sha256']),'Different dataset source'
 assert reports[0]['protocol']==reports[1]['protocol'],'Different navigation protocol'
 for key in ['stop_threshold','head_execution','explore_candidates']:
  assert reports[0]['args'].get(key)==reports[1]['args'].get(key),'Different controller: '+key
 assert [[r['index'] for r in x] for x in rows][0]==[[r['index'] for r in x] for x in rows][1]
 for r,report in zip(rows,reports):assert len(r)==report['metrics']['episodes'],'Partial evaluation'
 sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
 from sbfnav.dataset import CityNavDataset
 split=reports[0]['args']['split'];assert reports[1]['args']['split']==split
 source=CityNavDataset(WORKSPACE/'DATA/sbfnav-data',split,version='processed_citynav',allow_test_unseen=split=='test_unseen')
 keys=[(source[r['index']].map_name,source[r['index']].instruction) for r in rows[0]];lookup={key:i for i,key in enumerate(sorted(set(keys)))};groups=np.array([lookup[key] for key in keys]);counts=np.bincount(groups);rng=np.random.default_rng(0)
 deltas={}
 for metric,scale in [('SR',100),('OSR',100),('SPL',100),('NE',1),('steps',1),('inference_ms',1)]:
  before=np.array([r['metrics'][metric] for r in rows[0]]);after=np.array([r['metrics'][metric] for r in rows[1]]);difference=after-before;sums=np.bincount(groups,weights=difference);draws=[]
  for _ in range(50):
   sample=rng.integers(len(counts),size=(100,len(counts)));draws.extend((scale*sums[sample].sum(1)/counts[sample].sum(1)).tolist())
  deltas[metric]=dict(difference=float(scale*difference.mean()),group_bootstrap_95ci=np.quantile(draws,[.025,.975]).tolist())
 before=np.array([r['metrics']['SR'] for r in rows[0]]);after=np.array([r['metrics']['SR'] for r in rows[1]])
 result=dict(episodes=len(before),groups=len(counts),baseline=reports[0]['metrics'],candidate=reports[1]['metrics'],deltas=deltas,helped=int(((after>0)&(before==0)).sum()),harmed=int(((after==0)&(before>0)).sum()),candidate_terminations={k:sum(r['termination']==k for r in rows[1]) for k in {r['termination'] for r in rows[1]}},note='Complete predefined paired rollout subset; subgroup/Seen diagnostics are not final held-out confirmation.')
 atomic_json(result,Path(a.output));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
