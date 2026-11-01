"""Read-only TRAIN/internal-Dev diagnostics of a completed relation/Gate pilot."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.gated_belief_fusion import ReliabilityGatedBelief, training_gate_targets
from graphnav.native_relation_grounding import RelationAwareQueryRanker
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.stage1_data import Stage1Dataset
from scripts.train_native_relational_grounding import (
    augment_public_relations, collate_relation, gate_prediction, old_native,
    relation_prediction, shuffled_relation_inputs,
)
from scripts.train_native_gated_fusion import split_train_by_group, tensor_batch
from scripts.train_object_query_belief import load


@torch.no_grad()
def diagnose(rows, native, relation, gate, device, sigma, threshold, transfer):
    counts = dict(states=0, named=0, rank_changed=0, named_rank_corrected=0,
                  named_rank_damaged=0, shuffled_rank_changed=0,
                  relation_top1_within_bound_of_native=0, oracle_gate_positive=0,
                  gate_accepted=0, relation_score_changed_queries=0,
                  shuffled_score_changed_queries=0)
    probabilities, deltas, margins, relation_probabilities = [], [], [], []
    positive_probabilities, negative_probabilities = [], []
    for raw in DataLoader(rows, batch_size=16, shuffle=False, collate_fn=collate_relation):
        x = tensor_batch(raw, device, sigma)
        native_q, null = old_native(native, x, device)
        pred = relation_prediction(relation, x, native_q, null)
        shuffled = relation_prediction(relation, shuffled_relation_inputs(x), native_q, null)
        fusion = gate_prediction(gate, x, pred, enabled=True, threshold=threshold, transfer=transfer)
        labels = training_gate_targets(x['base_logits'], fusion['visual_probability'],
                                      fusion['usable'], x['grid_world'], x['target_world'], transfer=transfer)
        named = pred['has_relation']
        native_rank = native_q.masked_fill(~x['valid'], -1e4).argmax(-1)
        rel_rank = pred['logits'].argmax(-1)
        shuffled_rank = shuffled['logits'].argmax(-1)
        distance = (x['query_world'] - x['target_world'][:, None, :2]).norm(dim=-1)
        native_good = distance.gather(1, native_rank[:, None]).squeeze(-1) <= 20.
        rel_good = distance.gather(1, rel_rank[:, None]).squeeze(-1) <= 20.
        sorted_native = native_q.float().masked_fill(~x['valid'], -1e4).topk(2, dim=-1).values
        margin = sorted_native[:, 0] - sorted_native[:, 1]
        probability = fusion['reliability_logit'].sigmoid()
        positives = labels['target'].bool()
        counts['states'] += len(named)
        counts['named'] += int(named.sum())
        counts['rank_changed'] += int((named & (native_rank != rel_rank)).sum())
        counts['named_rank_corrected'] += int((named & ~native_good & rel_good).sum())
        counts['named_rank_damaged'] += int((named & native_good & ~rel_good).sum())
        counts['shuffled_rank_changed'] += int((named & (shuffled_rank != rel_rank)).sum())
        counts['relation_top1_within_bound_of_native'] += int((named & (margin <= 2 * relation.bound)).sum())
        counts['oracle_gate_positive'] += int(positives.sum())
        counts['gate_accepted'] += int(fusion['accepted'].sum())
        counts['relation_score_changed_queries'] += int(((pred['logits'] != native_q.float()) & x['valid']).sum())
        counts['shuffled_score_changed_queries'] += int(((pred['logits'] != shuffled['logits']) & x['valid']).sum())
        probabilities.extend(probability.cpu().tolist())
        positive_probabilities.extend(probability[positives].cpu().tolist())
        negative_probabilities.extend(probability[~positives].cpu().tolist())
        relation_probabilities.extend(probability[named].cpu().tolist())
        margins.extend(margin[named].cpu().tolist())
        deltas.extend(pred['relation_delta'][named[:, None] & x['valid']].abs().cpu().tolist())
    def summary(values):
        if not values:
            return dict(count=0)
        v = torch.tensor(values)
        return dict(count=len(v), min=float(v.min()), mean=float(v.mean()),
                    median=float(v.median()), max=float(v.max()))
    return dict(counts=counts, gate_probability=summary(probabilities),
                oracle_positive_gate_probability=summary(positive_probabilities),
                oracle_negative_gate_probability=summary(negative_probabilities),
                named_gate_probability=summary(relation_probabilities),
                named_native_top2_margin=summary(margins), named_abs_relation_delta=summary(deltas))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'dev', 'stage1-cache', 'native-checkpoint', 'pilot', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    destination = Path(a.output)
    if destination.exists():
        raise ValueError('Do not overwrite an existing diagnostic')
    pilot_path, native_path = Path(a.pilot), Path(a.native_checkpoint)
    pilot_sha, native_sha = file_sha256(pilot_path), file_sha256(native_path)
    pilot = torch.load(pilot_path, map_location='cpu', weights_only=True)
    protocol = pilot['protocol']
    if not (pilot_path.parent / 'complete.json').exists():
        raise ValueError('Only completed frozen pilots allowed')
    if native_sha != protocol['native_checkpoint_sha']:
        raise ValueError('Frozen Native provenance mismatch')
    for directory, role in [(a.train, 'train'), (a.dev, 'dev')]:
        meta = json.loads((Path(directory) / 'manifest.json').read_text())
        if meta['partition'] != role or not meta['complete']:
            raise ValueError('Only TRAIN/internal Dev allowed')
        if file_sha256(Path(directory) / 'manifest.json') != protocol[role + '_manifest_sha']:
            raise ValueError('Pilot data provenance mismatch')
    static = Stage1Dataset(a.stage1_cache, verify=False)
    device = torch.device('cuda')
    frozen = torch.load(native_path, map_location='cpu', weights_only=True)
    native = ObjectQueryRanker(relation=False).to(device)
    native.load_state_dict(frozen['model'])
    relation = RelationAwareQueryRanker().to(device)
    relation.load_state_dict(pilot['relation_state'])
    gate = ReliabilityGatedBelief().to(device)
    gate.load_state_dict(pilot['gate_state'])
    for model in (native, relation, gate):
        model.eval().requires_grad_(False)
    _, train = load(a.train)
    _, dev = load(a.dev)
    if {r['group'] for r in train} & {r['group'] for r in dev}:
        raise ValueError('TRAIN/internal Dev overlap')
    train = augment_public_relations(train, static)
    dev = augment_public_relations(dev, static)
    fit, calibration = split_train_by_group(train)
    report = {role: diagnose(rows, native, relation, gate, device,
                             float(frozen['config']['kernel_sigma']),
                             pilot['threshold'], pilot['transfer'])
              for role, rows in [('train_fit', fit), ('train_calibration', calibration), ('internal_dev', dev)]}
    if file_sha256(pilot_path) != pilot_sha or file_sha256(native_path) != native_sha:
        raise ValueError('Read-only weights changed')
    atomic_json(dict(pilot_sha=pilot_sha, native_sha=native_sha, partitions=report), destination)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
