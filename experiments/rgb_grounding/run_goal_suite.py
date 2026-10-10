"""Execute every pre-frozen residual diagnostic once, without test selection."""
import argparse
import json
from pathlib import Path
from .goal_evidence import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--cache',required=True);p.add_argument('--features',required=True);p.add_argument('--controls',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    suite=json.loads(Path(args.suite).read_text())
    if not suite['frozen']:raise ValueError('unfrozen comparison suite')
    root=Path(args.output)
    if root.exists():raise ValueError('refusing suite overwrite')
    root.mkdir(parents=True)
    configs={'selected_fallback':suite['production_config'],**suite['diagnostics']}
    for name,path in configs.items():
        print(json.dumps(dict(running=name,frozen_config=path)),flush=True)
        run(argparse.Namespace(mode='evaluate',cache=args.cache,features=args.features,controls=args.controls,config=path,output=str(root/name),device=args.device,heads=None))
    (root/'complete.json').write_text(json.dumps(dict(complete=True,configurations=list(configs),suite=suite),indent=2))


if __name__=='__main__':main()
