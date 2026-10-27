"""Fixed-candidate internal-dev ablations of the actually trained network."""
from common import *
import argparse
import torch
from train import load_groups,evaluate_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);model,state=load_branch(args.checkpoint,'cuda');groups=load_groups()
    validate_cache_signature(state,json.loads((ROOT/'data/manifest.json').read_text())['signature'])
    controls={}
    for control in ['real','no_language','no_depth','no_geometry','no_height','no_offset','shuffle_depth']:
        controls[control],_=evaluate_rows(model,groups['dev'],'cuda',control)
    save(dict(checkpoint_sha256=sha(args.checkpoint),controls=controls,
        scope='training internal-dev, fixed candidates; network ablations only, not navigation or full-pipeline language ablation'),ROOT/'reports/ablations.json')
    print(json.dumps(controls),flush=True)


if __name__=='__main__':main()
