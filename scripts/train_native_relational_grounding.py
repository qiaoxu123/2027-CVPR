"""Frozen AerialVG + trainable target/reference relation grounding + Gate pilot.

Only the new relation attention and reliability gate are trained. Named
CityRefer reference polygons and public instruction relations are loaded from
the existing Stage1 cache. TRAIN fit/calibration are disjoint by group.
Internal Dev is accessed exactly once after both models and threshold freeze.
No official Val-Seen / Val-Unseen / Test or navigation control is touched.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.gated_belief_fusion import (
    ReliabilityGatedBelief, count_outcomes, training_gate_targets,
)
from graphnav.native_relation_grounding import (
    RelationAwareQueryRanker, public_relation_descriptors,
    weak_relation_ranking_loss,
)
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.stage1_data import Stage1Dataset
from scripts.train_native_gated_fusion import (
    THRESHOLDS, select_threshold, split_train_by_group, tensor_batch,
)
from scripts.train_object_query_belief import collate, load


def augment_public_relations(rows, reference_dataset):
    """No target/GT use in relation feature preparation or forward.

    Graph preparation is shared with the existing A0 map geometry: it parses
    named reference objects from public CityRefer polygon footprints.
    """
    index = {}
    for i, record in enumerate(reference_dataset.static.records):
        key = (record['map_name'], record['instruction'])
        index.setdefault(key, i)
    out = []
    for row in rows:
        key = (row['map_name'], row['instruction'])
        if key not in index:
            raise ValueError('Native row lacks matching public map/instruction')
        graph = reference_dataset.static._prepared(index[key])[0]
        f, t, m = public_relation_descriptors(
            row['query_world'].float().numpy(), graph)
        aug = dict(row)
        aug['relation_features'] = f.half()
        aug['relation_types'] = t
        aug['relation_valid'] = m
        out.append(aug)
    return out


def collate_relation(rows):
    batch = collate(rows)
    batch['relation_features'] = torch.stack([
        item['relation_features'] for item in rows])
    batch['relation_types'] = torch.stack([
        item['relation_types'] for item in rows])
    batch['relation_valid'] = torch.stack([
        item['relation_valid'] for item in rows])
    return batch


@torch.no_grad()
def old_native(native, x, device):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                        enabled=device.type == 'cuda'):
        return native(x['features'], x['text'], x['text_mask'],
                      x['geometry'], x['valid'])


def relation_prediction(rel, x, q, null, *, enabled=True):
    return rel(
        x['features'], x['text'], x['text_mask'],
        x['geometry'], x['valid'],
        x['relation_features'], x['relation_types'], x['relation_valid'],
        q, null, relation_enabled=enabled,
    )


def gate_prediction(gate, x, pred, *, enabled=False, threshold=.8,
                    transfer=.9):
    return gate(
        x['base_logits'], pred['logits'], pred['null'],
        x['kernel'], x['observed'], x['valid'], x['geometry'],
        x['grid_world'], enabled=enabled,
        threshold=threshold, transfer=transfer,
    )


def accumulate(acc, item):
    for key, value in item.items():
        acc[key] = acc.get(key, 0) + value


def relation_data_summary(rows):
    named = [r for r in rows if r['relation_valid'].any()]
    supervised = []
    for row in named:
        distance = (row['query_world'].float() - row['labels']['target_world'][:2].float()).norm(dim=-1)
        good = (distance <= 20.) & row['valid'].bool()
        if good.any() and (row['valid'].bool() & ~good).any():
            supervised.append(row)
    return dict(states=len(rows), groups=len({r['group'] for r in rows}),
                named_states=len(named), named_groups=len({r['group'] for r in named}),
                supervised_states=len(supervised),
                supervised_groups=len({r['group'] for r in supervised}),
                reference_cardinality={str(k): sum(int(r['relation_valid'].sum()) == k for r in rows)
                                       for k in range(9)})


def shuffled_relation_inputs(x):
    """Reverse descriptors across valid queries, leaving Native evidence fixed.

    Permuting reference slots alone is not a meaningful attention ablation.
    This fixed, label-free permutation disrupts query/reference geometry.
    """
    result = dict(x)
    features = x['relation_features'].clone()
    for i in range(len(features)):
        ids = x['valid'][i].nonzero().flatten()
        features[i, ids] = x['relation_features'][i, ids.flip(0)]
    result['relation_features'] = features
    return result


@torch.no_grad()
def evaluate(rows, native, rel, gate, device, sigma, threshold, transfer,
             batch_size, *, ablations=False):
    rel.eval()
    gate.eval()
    stats, ranks, strata = {}, {}, {}
    max_no_relation_difference = 0.
    for raw in DataLoader(rows, batch_size=batch_size,
                          shuffle=False, collate_fn=collate_relation):
        x = tensor_batch(raw, device, sigma)
        q0, null = old_native(native, x, device)
        candidate = relation_prediction(rel, x, q0, null)
        off = relation_prediction(rel, x, q0, null, enabled=False)
        assert torch.equal(off['logits'], q0.float().masked_fill(~x['valid'], -1e4))
        predictions = {'relation': candidate}
        if ablations:
            predictions.update(relation_off=off, relation_shuffled=relation_prediction(
                rel, shuffled_relation_inputs(x), q0, null))
        old_logits, _ = native.update_belief(
            x['base_logits'], q0, null, x['kernel'], x['observed'], x['valid'], bound=2.)
        variants = {'native': old_logits}
        scores = {'native': q0}
        for prefix, pred in predictions.items():
            variants[prefix + '_residual'], _ = native.update_belief(
                x['base_logits'], pred['logits'], null, x['kernel'], x['observed'],
                x['valid'], bound=2.)
            fusion = gate_prediction(gate, x, pred, enabled=True,
                                     threshold=threshold, transfer=transfer)
            variants[prefix + '_gate'] = fusion['logits']
            scores[prefix] = pred['logits']
            accumulate(stats, {prefix + '_accepted': int(fusion['accepted'].sum())})
            if prefix == 'relation':
                targets = training_gate_targets(
                    x['base_logits'], fusion['visual_probability'], fusion['usable'],
                    x['grid_world'], x['target_world'], transfer=transfer)
                accumulate(stats, {'gate_recoverable': int(targets['oracle_eligible'].sum())})
        named = candidate['has_relation']
        for name, value in variants.items():
            accumulate(stats, {name + '_' + k: v for k, v in count_outcomes(
                x['base_logits'], value, x['grid_world'], x['target_world']).items()})
            for label, mask in [('named', named), ('no_named', ~named)]:
                if mask.any():
                    subset = strata.setdefault(label, {})
                    accumulate(subset, {name + '_' + k: v for k, v in count_outcomes(
                        x['base_logits'][mask], value[mask], x['grid_world'][mask],
                        x['target_world'][mask]).items()})
                    if name == 'native':
                        accumulate(subset, {'states': int(mask.sum())})
        distance = (x['query_world'] - x['target_world'][:, None, :2]).norm(dim=-1)
        covered = ((distance <= 20.) & x['valid']).any(-1)
        for name, score in scores.items():
            selected = score.masked_fill(~x['valid'], -1e4).argmax(-1)
            correct = (distance.gather(1, selected[:, None]).squeeze(-1) <= 20.) & covered
            accumulate(ranks, {name: int(correct.sum()), name + '_named': int((correct & named).sum())})
        if (~named).any():
            difference = (candidate['logits'][~named] - off['logits'][~named]).abs().max()
            max_no_relation_difference = max(max_no_relation_difference, float(difference))
        accumulate(stats, dict(states=len(named), relation_present=int(named.sum()),
                              candidate_covered=int(covered.sum()),
                              rel_delta_abs_sum=float(candidate['relation_delta'].abs().sum()),
                              valid_queries=int(x['valid'].sum())))
    n = stats['states']
    result = dict(states=n, relation_present=stats['relation_present'],
                  gate_accepted=stats['relation_accepted'], gate_recoverable=stats['gate_recoverable'],
                  candidate_covered=stats['candidate_covered'],
                  native_rank_correct_on_covered=ranks['native'],
                  relation_rank_correct_on_covered=ranks['relation'],
                  rank_counts=ranks,
                  mean_relation_delta_abs=stats['rel_delta_abs_sum']/max(stats['valid_queries'], 1),
                  no_named_max_logit_difference=max_no_relation_difference,
                  A0_R1=stats['native_base_correct']/n, A0_NE=stats['native_base_ne_sum']/n,
                  threshold=threshold, transfer=transfer,
                  data_summary=relation_data_summary(rows))
    def outcome(acc, name, count):
        return dict(R1=acc[name+'_updated_correct']/count,
                    NE=acc[name+'_updated_ne_sum']/count,
                    corrected=acc[name+'_corrected'], damaged=acc[name+'_damaged'],
                    net=acc[name+'_corrected']-acc[name+'_damaged'])
    for name in variants:
        result[name] = outcome(stats, name, n)
    result['strata'] = {label: dict(states=acc['states'],
                                   A0_R1=acc['native_base_correct']/acc['states'],
                                   **{name: outcome(acc, name, acc['states']) for name in variants})
                        for label, acc in strata.items()}
    return result


@torch.no_grad()
def gate_calibration(rows, native, rel, gate, device, sigma, threshold,
                     transfer, batch_size):
    report = evaluate(rows, native, rel, gate, device, sigma, threshold,
                      transfer, batch_size)
    stats = report['relation_gate']
    return {
        'count': report['states'], 'gate_threshold': threshold,
        'gate_accepted': report['gate_accepted'],
        'corrected': stats['corrected'],
        'damaged': stats['damaged'],
        'net_corrections': stats['net'],
        'gate_R1': stats['R1'],
        'gate_NE': stats['NE'],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--dev', required=True)
    p.add_argument('--stage1-cache', required=True,
                   help='GraphNav/data/rebuilt_stage1_20261025/frames')
    p.add_argument('--native-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--gate-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--transfer', type=float, default=.9)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    a = p.parse_args()
    if (not 1 <= a.epochs <= 10 or not 1 <= a.gate_epochs <= 10
            or not 0 < a.transfer <= 1):
        p.error('Invalid bounded pilot configuration')
    out = Path(a.output)
    if out.exists():
        raise ValueError('Use a new experiment output directory')
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.manual_seed(a.seed)
    source = Path(a.native_checkpoint)
    if not (source.parent / 'complete.json').exists():
        raise ValueError('Frozen native checkpoint is not completed')
    source_sha = file_sha256(source)
    frozen = torch.load(source, map_location='cpu', weights_only=True)
    if frozen['kind'] != 'native-aerialvg-object-query-belief-v1':
        raise ValueError('Incorrect Native checkpoint type')
    config = frozen['config']
    if config.get('relation'):
        raise ValueError('Use frozen native plain ranking as a paired reference')
    train_manifest, train_rows = load(a.train)
    # Dev metadata only, no Dev tensors or outcomes before model freeze.
    dev_manifest = json.loads((Path(a.dev) / 'manifest.json').read_text())
    if (not train_manifest.get('complete')
            or train_manifest.get('partition') != 'train'
            or not dev_manifest.get('complete')
            or dev_manifest.get('partition') != 'dev'
            or train_manifest['base_sha'] != dev_manifest['base_sha']
            or train_manifest['base_sha'] != config['base_sha']):
        raise ValueError('TRAIN/internal Dev/native provenance mismatch')
    stage1_path = Path(a.stage1_cache) / 'queries.jsonl'
    if not stage1_path.is_file():
        raise ValueError('Missing public CityNav static query cache')
    stage1_sha = file_sha256(stage1_path)
    static = Stage1Dataset(a.stage1_cache, verify=False)
    train_rows = augment_public_relations(train_rows, static)
    fit, calibration_rows = split_train_by_group(train_rows)

    native = ObjectQueryRanker(relation=False).to(device)
    native.load_state_dict(frozen['model'], strict=True)
    native.eval().requires_grad_(False)
    rel = RelationAwareQueryRanker().to(device)
    rel_optimizer = torch.optim.AdamW(rel.parameters(), lr=2e-4, weight_decay=.01)
    sigma = float(config['kernel_sigma'])
    if not 0 < sigma < 100 or not math.isfinite(sigma):
        raise ValueError('Invalid frozen Gaussian sigma')
    out.mkdir(parents=True)
    protocol = dict(
        kind='aerialvg-native-relational-gated-pilot-v1',
        admitted_for_runtime=False,
        base_commit='a14292c1677964e358b278c7e948f0abfecf929a',
        implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        implementation_sha={name: file_sha256(name) for name in (
            __file__, 'graphnav/native_relation_grounding.py', 'graphnav/semantic_geometry.py')},
        native_checkpoint_sha=source_sha,
        base_sha=config['base_sha'],
        train_manifest_sha=file_sha256(Path(a.train) / 'manifest.json'),
        dev_manifest_sha=file_sha256(Path(a.dev) / 'manifest.json'),
        stage1_public_queries_sha=stage1_sha,
        train_fit=len(fit), train_calibration=len(calibration_rows),
        fit_relation_summary=relation_data_summary(fit),
        calibration_relation_summary=relation_data_summary(calibration_rows),
        data_policy='fit/calibration disjoint TRAIN instruction groups; Dev once after freeze',
        supervision='20m spatial utility weak label; no target identity claim',
        unknown_reference='unmatched/ambiguous named map reference => no relation update',
        epochs=a.epochs, gate_epochs=a.gate_epochs,
        transfer=a.transfer, seed=a.seed,
    )
    atomic_json(protocol, out / 'protocol.json')

    initial_relation = {name: p.detach().clone() for name, p in rel.named_parameters()}
    gradient_max = {}
    history = []
    for epoch in range(a.epochs):
        rel.train()
        total = supervised = count = steps = 0
        loader = DataLoader(
            fit, batch_size=a.batch_size, shuffle=True,
            collate_fn=collate_relation,
            generator=torch.Generator().manual_seed(a.seed + epoch),
        )
        for raw in loader:
            x = tensor_batch(raw, device, sigma)
            q, null = old_native(native, x, device)
            pred = relation_prediction(rel, x, q, null)
            loss, detail = weak_relation_ranking_loss(
                pred['logits'], x['valid'], pred['has_relation'],
                x['query_world'], x['target_world'])
            rel_optimizer.zero_grad(set_to_none=True)
            if detail['supervised_rows']:
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite relation loss')
                loss.backward()
                for name, parameter in rel.named_parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise ValueError('Nonfinite relation gradient')
                        module = name.split('.')[0]
                        gradient_max[module] = max(gradient_max.get(module, 0.), float(parameter.grad.norm()))
                torch.nn.utils.clip_grad_norm_(rel.parameters(), 2.)
                rel_optimizer.step()
                steps += 1
            total += float(loss.detach()) * len(x['valid'])
            supervised += detail['supervised_rows']
            count += len(x['valid'])
        row = dict(epoch=epoch + 1, fit_loss=total / max(count, 1),
                   supervised_rows=supervised, states=count, optimizer_steps=steps)
        history.append(row)
        atomic_json(history, out / 'relation_history.json')
        print(json.dumps(row), flush=True)

    atomic_json(dict(gradient_max=gradient_max,
                     parameter_delta_norm={name: float((p.detach()-initial_relation[name]).norm())
                                           for name, p in rel.named_parameters()},
                     frozen_native_has_grad=any(p.grad is not None for p in native.parameters())),
                out / 'gradient_audit.json')
    # Second stage: freeze relation ranker and fit Gate on its new scores.
    rel.eval().requires_grad_(False)
    gate = ReliabilityGatedBelief().to(device)
    gate_optimizer = torch.optim.AdamW(gate.parameters(), lr=5e-4)
    gate_history = []
    for epoch in range(a.gate_epochs):
        gate.train()
        total = count = positives = 0
        loader = DataLoader(
            fit, batch_size=a.batch_size, shuffle=True,
            collate_fn=collate_relation,
            generator=torch.Generator().manual_seed(a.seed + 100 + epoch),
        )
        for raw in loader:
            x = tensor_batch(raw, device, sigma)
            q, null = old_native(native, x, device)
            with torch.no_grad():
                pred = relation_prediction(rel, x, q, null)
            prediction = gate_prediction(gate, x, pred)
            labels = training_gate_targets(
                x['base_logits'], prediction['visual_probability'],
                prediction['usable'], x['grid_world'], x['target_world'],
                transfer=a.transfer)
            loss = F.binary_cross_entropy_with_logits(
                prediction['reliability_logit'], labels['target'],
                pos_weight=torch.tensor(4., device=device),
            )
            gate_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gate_optimizer.step()
            total += float(loss.detach()) * len(x['valid'])
            count += len(x['valid'])
            positives += int(labels['target'].sum())
        row = dict(epoch=epoch + 1, loss=total / max(count, 1),
                   fit_positive=positives, states=count)
        gate_history.append(row)
        atomic_json(gate_history, out / 'gate_history.json')
        print(json.dumps(row), flush=True)

    # TRAIN-only calibration. Held-out internal Dev cannot influence
    # model, epoch or threshold selection.
    sweep = [
        gate_calibration(calibration_rows, native, rel, gate, device,
                         sigma, t, a.transfer, a.batch_size)
        for t in THRESHOLDS
    ]
    selected = select_threshold(sweep, .02)
    threshold = selected['gate_threshold']
    atomic_json(dict(sweep=sweep, selected=selected,
                     damage_budget_fraction=.02),
                out / 'train_calibration.json')
    atomic_save(dict(
        kind='aerialvg-native-relation-gate-pilot-v1',
        relation_state=rel.state_dict(), gate_state=gate.state_dict(),
        threshold=threshold, transfer=a.transfer, protocol=protocol,
        admitted_for_runtime=False,
    ), out / 'pilot.pt')

    if file_sha256(Path(a.dev) / 'manifest.json') != protocol['dev_manifest_sha']:
        raise ValueError('DEV changed after TRAIN calibration')
    dev_meta, dev_rows = load(a.dev)
    if dev_meta['partition'] != 'dev':
        raise ValueError('Dev role mismatch')
    if {x['group'] for x in train_rows} & {x['group'] for x in dev_rows}:
        raise ValueError('TRAIN/internal Dev group leakage')
    dev_rows = augment_public_relations(dev_rows, static)
    report = evaluate(dev_rows, native, rel, gate, device, sigma,
                      threshold, a.transfer, a.batch_size, ablations=True)
    atomic_json(report, out / 'internal_dev.json')
    if file_sha256(source) != source_sha or file_sha256(stage1_path) != stage1_sha:
        raise ValueError('Frozen checkpoint or public reference cache changed')
    atomic_json(dict(
        complete=True, admitted_for_runtime=False,
        model_frozen=True, formal_navigation_not_run=True,
        native_sha_verified=True, public_cache_sha_verified=True,
        checkpoint_sha=file_sha256(out / 'pilot.pt'),
    ), out / 'complete.json')
    print(json.dumps(dict(dev=report, calibration=selected)), flush=True)


if __name__ == '__main__':
    main()
