"""Post-run exact equivalence for a manifest-preserving subset of a full rollout."""
import argparse
import json
from pathlib import Path

from scripts.summarize_closedloop import read_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('full','subset','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise ValueError('refusing equivalence overwrite')
    a,full,_=read_run(args.full);b,subset,_=read_run(args.subset)
    for key in ('protocol','checkpoint_sha256','head_checkpoint_sha256','dynamic_checkpoint_sha256','visual_checkpoint_sha256','input_sha256'):
        if a.get(key)!=b.get(key):raise ValueError('incompatible equivalence input: '+key)
    failures=[]
    for index,row in subset.items():
        baseline=full.get(index)
        if baseline is None:raise ValueError('subset episode absent from full denominator')
        exact=baseline['episode_id']==row['episode_id'] and baseline['poses']==row['poses'] and baseline['termination']==row['termination']
        exact=exact and {k:v for k,v in baseline['metrics'].items() if k!='inference_ms'}=={k:v for k,v in row['metrics'].items() if k!='inference_ms'}
        if not exact:failures.append(index)
    output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(full=args.full,subset=args.subset,episodes=len(subset),all_poses_termination_navigation_metrics_exact=not failures,
        changed_episode_indices=failures,subset_seconds=b['seconds'],
        note='Latency excluded from equality. Source/config/checkpoints and every subset episode/start must agree; full denominator is preserved. This is a performance-equivalence test, not an accuracy optimization.')
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
