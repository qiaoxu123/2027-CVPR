"""Read-only original SigLIP2 audit; cached TRAIN features are NOT fixed-yaw data.

Optional backward probe uses an in-memory copy. No optimizer or checkpoint write.
No Val/Test samples are read; historical CSV aggregates are explicitly labelled.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from graphnav.appearance_matching import MultiScaleAppearanceMatcher


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--train-cache', required=True, type=Path)
    parser.add_argument('--historical-ablation', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest = json.loads((args.train_cache / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('complete TRAIN manifest required')
    if manifest['protocol']['cache_kind'] != 'stage1-current-and-prefix-v1':
        raise ValueError('unexpected feature provenance')
    chunk_path = args.train_cache / manifest['chunks'][0]['file']
    if sha256(chunk_path) != manifest['chunks'][0]['sha256']:
        raise ValueError('TRAIN chunk checksum mismatch')
    chunk = torch.load(chunk_path, map_location='cpu', weights_only=True)
    row = next(r for r in chunk['rows'] if any(
        f['step'] == r['cutoff_step'] and f['roi_valid'].any()
        for f in chunk['episodes'][r['episode_index']]))
    frame = next(f for f in chunk['episodes'][row['episode_index']]
                 if f['step'] == row['cutoff_step'])
    torch.manual_seed(0)
    device = torch.device(args.device)
    matcher = MultiScaleAppearanceMatcher().to(device)
    matcher.load_state_dict({k.removeprefix('matcher.'): v for k, v in state['model'].items()
                             if k.startswith('matcher.')}, strict=True)
    images = frame['roi_features'][None].float().to(device)
    valid = frame['roi_valid'][None].to(device)
    quality = frame['roi_quality'][None].float().to(device)
    text = F.normalize((row['native_text'].float() + row['appearance_text'].float())[None], dim=-1).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    scores, known, _ = matcher(text, images, valid, quality)
    # Nonconstant synthetic objective tests gradient plumbing only, NOT task learning.
    weights = torch.linspace(-1, 1, scores.shape[-1], device=device)
    loss = (scores * weights * known).sum() / known.sum().clamp_min(1)
    loss.backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    grads = {name: float(p.grad.norm()) if p.grad is not None else None
             for name, p in matcher.named_parameters()}
    with torch.no_grad():
        zero, _, _ = matcher(text, torch.zeros_like(images), valid, quality)
        shuffled = images.clone()
        for scale in range(valid.shape[-1]):
            ids = valid[0, :, scale].nonzero().flatten()
            shuffled[0, ids, scale] = images[0, ids.flip(0), scale]
        shuffle, _, _ = matcher(text, shuffled, valid, quality)
    gain = float(state['model']['visual_gain'].tanh())
    report = dict(
        scope='read-only code/checkpoint audit plus copied-head GPU gradient smoke; no training or navigation',
        checkpoint_sha256=sha256(args.checkpoint), train_manifest_sha256=sha256(args.train_cache/'manifest.json'),
        train_chunk_sha256=sha256(chunk_path), grid_size=state['a3_args']['grid_size'],
        protocol=state['protocol'], trained_gain_tanh=gain,
        max_abs_delta_bound=state['protocol']['residual_bound']*abs(gain),
        max_pairwise_logit_gap_reversible=2*state['protocol']['residual_bound']*abs(gain),
        cached_observation_yaw_rad=float(frame['pose'][3]),
        cached_features_are_new_fixed_yaw=False, episode_index=int(row['episode_index']),
        step=int(frame['step']), supported_cells=int(known.sum()),
        real_vs_zero_score_l1=float((scores-zero).detach().abs().mean()),
        real_vs_shuffle_score_l1=float((scores-shuffle).detach().abs().mean()),
        device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        forward_backward_seconds=elapsed, gradient_norms=grads,
        gradients_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in matcher.parameters()),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
        warning='Single cached TRAIN frame; gradient objective is synthetic. Not accuracy, not training improvement, not fixed-yaw validation.')
    if args.historical_ablation:
        report['historical_ablation_sha256'] = sha256(args.historical_ablation)
        report['historical_not_new_evaluation'] = list(csv.DictReader(args.historical_ablation.open()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('grid_size','max_abs_delta_bound','max_pairwise_logit_gap_reversible','supported_cells','gradients_finite','forward_backward_seconds')}, indent=2))


if __name__ == '__main__':
    main()
