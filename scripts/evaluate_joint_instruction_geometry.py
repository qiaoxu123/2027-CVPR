"""TRAIN development audit of joint checkpoints; labels are scoring-only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from graphnav.candidate_geometry import collate_full
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.full_experiment import _batch_inputs
from graphnav.local_goal_decoder import decode_first_candidate
from graphnav.trainable_instruction_encoder import TrainableInstructionBERT
from scripts.train_v2_top1_geometry import Indexed
from scripts.unified_pgvb_v3_pilot import _load_base


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'source', 'trained', 'joint', 'bert-cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--component-ablation', action='store_true',
                        help='Diagnostic mismatched-weight controls, not deployable model selection')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cfg = _load_base(args.source, 'cuda')
    original = torch.load(args.trained, map_location='cpu', weights_only=True)
    original = {k[9:]: v for k, v in original['model'].items() if k.startswith('geometry.')}
    joint = torch.load(args.joint, map_location='cpu', weights_only=False)
    if joint['fingerprint']['geometry'] != file_sha256(args.trained):
        raise ValueError('Joint model does not match source Geometry')
    encoder = TrainableInstructionBERT(cfg.text_backbone, device='cuda', max_length=96)
    encoder.load_cache(args.bert_cache)
    original_bert = {k: v.detach().cpu().clone() for k, v in encoder.model.state_dict().items()}
    labels = json.loads((args.data / 'labels_private.json').read_text())
    indices = [row['index'] for row in labels if row['split'] == 'development']
    ds = Indexed(args.data / 'public_train.jsonl', grid_size=28,
                 max_landmarks=cfg.max_landmarks, max_roads=cfg.max_roads,
                 max_relations=cfg.max_relations, prior_sigma_m=cfg.prior_sigma_m,
                 rule_sigma_m=cfg.rule_sigma_m, deduplicate_queries=False)
    assert all(row['target_xy'] == [0., 0.] for row in ds.records)
    predictions = {}
    arms = [('original', original, original_bert),
            ('joint', joint['geometry'], joint['bert'])]
    if args.component_ablation:
        arms += [('adapted_geometry_only', joint['geometry'], original_bert),
                 ('adapted_bert_only', original, joint['bert'])]
    for arm, geometry, bert in arms:
        model.load_state_dict(geometry, strict=True)
        encoder.model.load_state_dict(bert, strict=True)
        model.eval(); encoder.model.eval()
        rows = []
        for batch in DataLoader(ds, batch_size=16, sampler=indices,
                                num_workers=0, collate_fn=collate_full):
            obs, tokens, mask, names = _batch_inputs(batch, encoder, 'cuda', 'geometry')
            field = model(obs, variant='geometry', tokens=tokens,
                          token_mask=mask, node_emb=names)
            soft = decode_first_candidate(field, obs['grid_xy'])
            for b, i in enumerate(batch['row_index'].tolist()):
                bounds = np.asarray(ds.records[i]['world_bounds'])
                # Private target is read only after both public predictions exist.
                target = np.asarray(labels[i]['target_world'])
                row = dict(index=i, episode=labels[i]['episode'])
                positions = dict(grid=obs['grid_xy'][b, field.belief_logits[b].argmax()],
                                 local5=soft.candidate_xy[b, 0])
                for decoder, position in positions.items():
                    xy = bounds[:2] + position.cpu().numpy() * (bounds[2:] - bounds[:2])
                    row[decoder] = dict(xy=xy.tolist(), NE=float(np.linalg.norm(xy-target)))
                rows.append(row)
        predictions[arm] = rows
    summary = {}
    for decoder in ('grid', 'local5'):
        a = np.array([r[decoder]['NE'] for r in predictions['original']])
        for arm in predictions:
            if arm == 'original':
                continue
            b = np.array([r[decoder]['NE'] for r in predictions[arm]])
            summary[f'{decoder}/{arm}'] = dict(n=len(a), original_R20=float((a<=20).mean()),
                                   adapted_R20=float((b<=20).mean()), original_NE=float(a.mean()),
                                   adapted_NE=float(b.mean()), corrections=int(((a>20)&(b<=20)).sum()),
                                   damage=int(((a<=20)&(b>20)).sum()))
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(predictions, args.output / 'predictions.json')
    atomic_json(dict(summary=summary, joint_sha256=file_sha256(args.joint),
                     split='TRAIN development only; no Val/Test or navigation',
                     checkpoint_epoch=joint['epoch']), args.output / 'summary.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
