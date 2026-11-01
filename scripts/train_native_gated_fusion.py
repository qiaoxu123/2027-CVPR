"""TRAIN-only gate fitting, TRAIN-group calibration, internal Dev diagnostic.

Frozen native object queries and old visual residual are never retrained.
No runtime admission, official Val-Seen/Unseen, or Test access is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.gated_belief_fusion import (
    ReliabilityGatedBelief, count_outcomes, training_gate_targets,
)
from graphnav.object_query_belief import ObjectQueryRanker
from scripts.train_object_query_belief import collate, load

THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0)


def checked_rows(directory, partition):
    manifest = json.loads((Path(directory) / 'manifest.json').read_text())
    if manifest.get('partition') != partition or not manifest.get('complete'):
        raise ValueError('Only completed explicit TRAIN/internal Dev data allowed')
    verified, rows = load(directory)
    if not rows or any('labels' not in row or 'target_world' not in row['labels']
                       or 'group' not in row for row in rows):
        raise ValueError('Incomplete TRAIN/Dev provenance')
    return verified, rows


def split_train_by_group(rows):
    fit, calibrate = [], []
    for row in rows:
        digest = hashlib.sha256(str(row['group']).encode('utf-8')).digest()
        (calibrate if digest[0] % 5 == 0 else fit).append(row)
    if not fit or not calibrate:
        raise ValueError('Group-hash TRAIN fit/calibration split is empty')
    if {r['group'] for r in fit} & {r['group'] for r in calibrate}:
        raise ValueError('TRAIN fit/calibration leakage')
    return fit, calibrate


def tensor_batch(batch, device, kernel_sigma):
    x = {k: (v.to(device).float() if v.is_floating_point() else v.to(device))
         for k, v in batch.items() if torch.is_tensor(v)}
    # Preserve actual observed support while allowing the same sigma used by
    # the deployed native checkpoint. Never use a target-centered kernel.
    x['kernel'] = torch.exp(
        -torch.cdist(x['grid_world'], x['query_world']).square()
        / (2 * kernel_sigma ** 2)
    ) * x['observed'][..., None] * x['valid'][:, None]
    return x


@torch.no_grad()
def frozen_scores(ranker, x, device):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        return ranker(x['features'], x['text'], x['text_mask'],
                      x['geometry'], x['valid'])


def gate_pass(gate, x, q, null, *, enabled=False, threshold=0.8, transfer=0.9):
    return gate(
        x['base_logits'], q, null, x['kernel'], x['observed'], x['valid'],
        x['geometry'], x['grid_world'],
        enabled=enabled, threshold=threshold, transfer=transfer,
    )


def _add_counts(acc, item):
    for name, value in item.items():
        acc[name] = acc.get(name, 0) + value


@torch.no_grad()
def evaluate(rows, gate, ranker, device, kernel_sigma, *,
             threshold, transfer, batch_size):
    gate.eval()
    aggregate = {}
    confidence = []
    positives = 0
    selected = 0
    visual_coverage = 0
    for batch in DataLoader(rows, batch_size=batch_size, collate_fn=collate,
                            shuffle=False, num_workers=0):
        x = tensor_batch(batch, device, kernel_sigma)
        q, null = frozen_scores(ranker, x, device)
        result = gate_pass(gate, x, q, null, enabled=True,
                           threshold=threshold, transfer=transfer)
        targets = training_gate_targets(
            x['base_logits'], result['visual_probability'], result['usable'],
            x['grid_world'], x['target_world'], transfer=transfer)
        old, _ = ranker.update_belief(
            x['base_logits'], q, null, x['kernel'], x['observed'], x['valid'])
        _add_counts(aggregate, {'switch_oracle_eligible': int(targets['oracle_eligible'].sum())})
        _add_counts(aggregate, {'supported_views': int(result['usable'].sum())})
        current = count_outcomes(
            x['base_logits'], result['logits'], x['grid_world'], x['target_world'])
        previous = count_outcomes(
            x['base_logits'], old, x['grid_world'], x['target_world'])
        _add_counts(aggregate, current)
        for key in ('corrected', 'damaged', 'updated_correct', 'updated_ne_sum'):
            aggregate['residual_' + key] = aggregate.get('residual_' + key, 0) + previous[key]
        confidence.extend(result['reliability'].cpu().tolist())
        positives += int(targets['target'].sum())
        selected += int(result['accepted'].sum())
    n = aggregate['count']
    aggregate.update(
        gate_threshold=threshold, transfer=transfer,
        gate_accepted=selected,
        net_corrections=aggregate['corrected'] - aggregate['damaged'],
        residual_net=aggregate['residual_corrected'] - aggregate['residual_damaged'],
        base_R1=aggregate['base_correct'] / n,
        gate_R1=aggregate['updated_correct'] / n,
        residual_R1=aggregate['residual_updated_correct'] / n,
        gate_NE=aggregate['updated_ne_sum'] / n,
        base_NE=aggregate['base_ne_sum'] / n,
        residual_NE=aggregate['residual_updated_ne_sum'] / n,
        positive_train_utility=positives,
        supported_coverage=aggregate['supported_views'] / n,
        reliability_mean=sum(confidence) / n,
    )
    return aggregate


def select_threshold(calibration, max_damage_fraction):
    """Freeze TRAIN-only decision rule before consulting internal Dev."""
    limit = math.floor(calibration[0]['count'] * max_damage_fraction)
    eligible = [record for record in calibration if record['damaged'] <= limit]
    if not eligible:
        raise ValueError('No safe calibration candidate')
    # In case of no positive gain, prefer the no-op threshold 1.0.
    return max(eligible, key=lambda rec: (
        rec['net_corrections'], -rec['damaged'], -rec['gate_accepted'],
        rec['gate_threshold'],
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True, help='Existing native TRAIN cache')
    p.add_argument('--dev', required=True, help='Existing internal grouped Dev cache')
    p.add_argument('--native-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--transfer', type=float, default=0.9)
    p.add_argument('--max-calibration-damage-fraction', type=float, default=0.02)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    a = p.parse_args()
    if not 1 <= a.epochs <= 12 or not 0 < a.lr <= .01 or not 0 < a.transfer <= 1:
        p.error('Invalid bounded experimental configuration')
    if not 0 <= a.max_calibration_damage_fraction <= .10:
        p.error('Calibration damage budget must be between 0 and 10%')
    out = Path(a.output)
    if out.exists():
        raise ValueError('Refusing to overwrite another experiment')
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    native_path = Path(a.native_checkpoint)
    if not (native_path.parent / 'complete.json').exists():
        raise ValueError('An explicitly completed frozen native checkpoint is required')
    native_sha = file_sha256(native_path)
    state = torch.load(native_path, map_location='cpu', weights_only=True)
    if state.get('kind') != 'native-aerialvg-object-query-belief-v1':
        raise ValueError('Not a trained Native AerialVG Query Ranker')
    config = state['config']
    if config.get('relation'):
        raise ValueError('This pilot requires the frozen plain Native Query Ranker')
    train_meta, train_rows = checked_rows(a.train, 'train')
    # Metadata-only Dev preflight. Do not deserialize Dev tensors until the
    # gate and threshold are irrevocably selected using TRAIN groups.
    dev_manifest_path = Path(a.dev) / 'manifest.json'
    dev_meta = json.loads(dev_manifest_path.read_text())
    if dev_meta.get('partition') != 'dev' or not dev_meta.get('complete'):
        raise ValueError('Only completed internal Dev role permitted')
    if (train_meta['base_sha'] != dev_meta['base_sha']
            or train_meta['base_sha'] != config['base_sha']):
        raise ValueError('Native/A0 baseline SHA mismatch')
    fit_rows, calibration_rows = split_train_by_group(train_rows)
    ranker = ObjectQueryRanker(relation=False).to(device)
    ranker.load_state_dict(state['model'], strict=True)
    ranker.eval().requires_grad_(False)
    gate = ReliabilityGatedBelief().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=a.lr, weight_decay=.01)
    kernel_sigma = float(config.get('kernel_sigma', 8.0))
    if not math.isfinite(kernel_sigma) or kernel_sigma <= 0:
        raise ValueError('Invalid frozen native kernel sigma')
    out.mkdir(parents=True)
    protocol = dict(
        kind='aerialvg-gated-belief-train-calibrate-dev-pilot-v1',
        admitted_for_runtime=False,
        frozen_native_sha=native_sha, base_sha=config['base_sha'],
        train_manifest_sha=file_sha256(Path(a.train) / 'manifest.json'),
        dev_manifest_sha=file_sha256(Path(a.dev) / 'manifest.json'),
        fit_rows=len(fit_rows), calibration_rows=len(calibration_rows),
        dev_rows=len(dev_meta.get('rows', [])),
        fit_groups=len({x['group'] for x in fit_rows}),
        calibration_groups=len({x['group'] for x in calibration_rows}),
        policy='Train group-hash fit; disjoint Train group calibration; internal Dev report only. Never select by Dev, Val-Seen, Unseen or Test.',
        labels='TRAIN-only spatial visual-argmax helps when A0 argmax wrong; not object identity.',
        epochs=a.epochs, seed=a.seed, lr=a.lr, transfer=a.transfer,
        kernel_sigma=kernel_sigma, max_calibration_damage_fraction=a.max_calibration_damage_fraction,
    )
    atomic_json(protocol, out / 'protocol.json')
    history = []
    for epoch in range(1, a.epochs + 1):
        gate.train()
        losses = []
        positives = 0
        rows = 0
        generator = torch.Generator().manual_seed(a.seed + epoch)
        loader = DataLoader(fit_rows, batch_size=a.batch_size, shuffle=True,
                            collate_fn=collate, num_workers=0, generator=generator)
        for raw in loader:
            x = tensor_batch(raw, device, kernel_sigma)
            q, null = frozen_scores(ranker, x, device)
            pred = gate_pass(gate, x, q, null)
            target = training_gate_targets(
                x['base_logits'], pred['visual_probability'], pred['usable'],
                x['grid_world'], x['target_world'], transfer=a.transfer)['target']
            loss = F.binary_cross_entropy_with_logits(
                pred['reliability_logit'], target,
                pos_weight=torch.tensor(4., device=device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()) * len(target))
            rows += len(target)
            positives += int(target.sum())
        entry = dict(epoch=epoch, fit_loss=sum(losses) / rows,
                     fit_positive_rate=positives / rows, fit_rows=rows)
        history.append(entry)
        atomic_json(history, out / 'history.json')
        print(json.dumps(entry), flush=True)

    # Do not select model/checkpoint on internal Dev. One final model only.
    calibration = [
        evaluate(calibration_rows, gate, ranker, device, kernel_sigma,
                 threshold=threshold, transfer=a.transfer, batch_size=a.batch_size)
        for threshold in THRESHOLDS
    ]
    chosen = select_threshold(calibration, a.max_calibration_damage_fraction)
    # Strictly independent of any Dev outcome; Dev read only after decision.
    atomic_json(dict(sweep=calibration, chosen=chosen,
                     objective='TRAIN calibration maximum net corrections subject to <=2% damage default'),
                out / 'train_calibration.json')
    atomic_save(dict(
        kind='native-aerialvg-gated-belief-pilot-v1',
        state_dict=gate.state_dict(), protocol=protocol,
        threshold=chosen['gate_threshold'], transfer=a.transfer,
        admitted_for_runtime=False,
    ), out / 'gate_pilot.pt')
    validated_dev_meta, dev_rows = checked_rows(a.dev, 'dev')
    if validated_dev_meta['base_sha'] != dev_meta['base_sha']:
        raise ValueError('Dev manifest changed after TRAIN calibration')
    if {r['group'] for r in train_rows} & {r['group'] for r in dev_rows}:
        raise ValueError('TRAIN/internal Dev target group overlap')
    dev = evaluate(dev_rows, gate, ranker, device, kernel_sigma,
                   threshold=chosen['gate_threshold'], transfer=a.transfer,
                   batch_size=a.batch_size)
    atomic_json(dev, out / 'internal_dev.json')
    if file_sha256(native_path) != native_sha:
        raise ValueError('Frozen native checkpoint changed during gate training')
    atomic_json(dict(
        completed=True, admitted_for_runtime=False,
        native_checkpoint_unchanged=True,
        selected_threshold=chosen['gate_threshold'],
        dev_only_evaluation=True, formal_navigation_not_run=True,
        checkpoint_sha=file_sha256(out / 'gate_pilot.pt'),
    ), out / 'complete.json')
    print(json.dumps(dict(calibration=chosen, dev=dev)), flush=True)


if __name__ == '__main__':
    main()
