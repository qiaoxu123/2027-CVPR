"""Dev-only strength audit of saved foundation scores; no new Seen evaluation."""
import argparse
import json
from pathlib import Path
import torch
from .evaluate_features import load
from .goal_evidence import predict
from experiments.bidirectional_core.cache import CachedCoreDataset
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--features',required=True);p.add_argument('--scores',required=True);p.add_argument('--output',required=True);args=p.parse_args();output=Path(args.output)
    if output.exists():raise ValueError('refusing strength audit overwrite')
    rows={r['metadata']['sample_id']:r for r in load(args.features)};saved=torch.load(args.scores,map_location='cpu',weights_only=True);dataset=CachedCoreDataset(args.cache,'E1')
    records={r['sample_id']:dict(score=r['scores']['real'],confidence=rows[r['sample_id']]['public']['roi_valid'][None].float(),observed=rows[r['sample_id']]['public']['roi_valid'][None],row=rows[r['sample_id']]) for r in saved};trials=[]
    for history in [False,True]:
        for gain in [0,.025,.05,.1,.2]:
            pred=predict(dataset,records,gain,0,history);forward=[r['metrics']['r1'] for r in pred if r['direction']=='forward'];trials.append(dict(gain=gain,history=history,forward_goal_r1=sum(forward)/len(forward),net_recovery=sum(r['metrics']['net_recovery'] for r in pred),damage=sum(r['metrics']['damage'] or 0 for r in pred)))
    best=max(trials,key=lambda r:(r['forward_goal_r1'],r['net_recovery'],-r['damage'],-r['gain'],-int(r['history'])))
    result=dict(trials=trials,selected=dict(best),scope='internal Dev-only strength diagnostic across same pre-existing gain grid; no new Seen or Unseen evaluation, no prompt or representation changes',scores_sha256=file_sha256(args.scores),cache_manifest_sha256=file_sha256(Path(args.cache)/'manifest.json'))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
