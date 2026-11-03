"""TRAIN-fit frozen intervention; never modifies deployed inference or weights."""
import argparse
import math
import time
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_public_geometry_visual import load_fit
from scripts.train_local_chain_position import evaluate


def uniform_prior_feature(module, args):
    x = args[0].clone()
    # Only the final spatial-MLP feature changes. Actual A0 stays untouched.
    x[..., -1] = -math.log(784)
    return (x,)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    source = args.root / 'world_v3_envelope_s0/best.pt'
    model = WorldLikelihoodPositionHead().eval().requires_grad_(False)
    state = torch.load(source, map_location='cpu', weights_only=True)['model']
    model.load_state_dict(state)
    data = load_fit(args.root)
    started = time.perf_counter()
    results = {'original': evaluate(model, data, 'cpu', 8)}
    hook = model.spatial_score.register_forward_pre_hook(uniform_prior_feature)
    try:
        results['uniform_spatial_prior'] = evaluate(model, data, 'cpu', 8)
    finally:
        hook.remove()
    base = results['original']['rows']
    for name, result in results.items():
        result['paired_to_original'] = dict(
            corrected=sum(r['success'] and not b['success'] for r, b in zip(result['rows'], base)),
            harmed=sum(not r['success'] and b['success'] for r, b in zip(result['rows'], base)))
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    atomic_json(dict(results=results, seconds=time.perf_counter()-started,
        checkpoint_sha256=file_sha256(source), script_sha256=file_sha256(__file__),
        device='cpu', train_fit_only=True, inference_uses_gt=False,
        caveat='Existing loader opens TRAIN labels; labels only enter scoring, not model inputs. No new independent confirmation or SR.'), args.output/'results.json')
    print({k: dict(metrics=v['metrics'], paired=v['paired_to_original']) for k,v in results.items()}, flush=True)


if __name__ == '__main__':
    main()
