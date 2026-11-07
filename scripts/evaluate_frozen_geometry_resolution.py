"""TRAIN-only query-density diagnostic; same frozen pointwise scoring model."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from graphnav.candidate_geometry import collate_full
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.full_experiment import _batch_inputs
from graphnav.local_goal_decoder import decode_first_candidate
from graphnav.token_language import FrozenBERTTokens
from scripts.train_v2_top1_geometry import Indexed
from scripts.unified_pgvb_v3_pilot import _load_base


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'source', 'trained', 'bert-cache', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='Engineering smoke only')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    model, cfg = _load_base(args.source, 'cuda')
    state = torch.load(args.trained, map_location='cpu', weights_only=True)
    model.load_state_dict({k[9:]: v for k, v in state['model'].items()
                           if k.startswith('geometry.')}, strict=True)
    model.eval().requires_grad_(False)
    encoder = FrozenBERTTokens(cfg.text_backbone, device='cuda', max_length=96)
    encoder.load_cache(args.bert_cache)
    labels = json.loads((args.data / 'labels_private.json').read_text())
    indices = [r['index'] for r in labels if r['split'] == 'development']
    if args.limit:
        indices = indices[:args.limit]
    predictions = {}
    for size in (28, 56):
        ds = Indexed(args.data / 'public_train.jsonl', grid_size=size,
                     max_landmarks=cfg.max_landmarks, max_roads=cfg.max_roads,
                     max_relations=cfg.max_relations, prior_sigma_m=cfg.prior_sigma_m,
                     rule_sigma_m=cfg.rule_sigma_m, deduplicate_queries=False)
        assert all(r['target_xy'] == [0., 0.] for r in ds.records)
        rows, fields = [], []
        start = time.perf_counter()
        compute_s = 0.
        torch.cuda.reset_peak_memory_stats()
        for batch in DataLoader(ds, batch_size=8, sampler=indices,
                                num_workers=0, collate_fn=collate_full):
            torch.cuda.synchronize(); tick = time.perf_counter()
            obs, tokens, mask, names = _batch_inputs(batch, encoder, 'cuda', 'geometry')
            field = model(obs, variant='geometry', tokens=tokens,
                          token_mask=mask, node_emb=names)
            # Preserve normalized/world window radius rather than changing it
            # opportunistically when doubling query density.
            soft = decode_first_candidate(field, obs['grid_xy'], radius=2*size//28)
            torch.cuda.synchronize(); compute_s += time.perf_counter()-tick
            fields.append(field.belief_logits.float().cpu().numpy())
            for b, i in enumerate(batch['row_index'].tolist()):
                bounds = np.asarray(ds.records[i]['world_bounds'])
                target = np.asarray(labels[i]['target_world'])  # scoring only
                row = dict(index=i, map_name=ds.records[i]['map_name'])
                for name, xy in [('grid', obs['grid_xy'][b, field.belief_logits[b].argmax()]),
                                 ('local', soft.candidate_xy[b, 0])]:
                    world = bounds[:2] + xy.cpu().numpy() * (bounds[2:]-bounds[:2])
                    row[name] = dict(xy=world.tolist(), NE=float(np.linalg.norm(world-target)))
                rows.append(row)
            if len(rows) % 256 == 0:
                print(json.dumps(dict(grid=size, completed=len(rows), elapsed_s=time.perf_counter()-start)), flush=True)
        predictions[str(size)] = rows
        np.savez_compressed(args.output / f'public_fields_{size}.npz',
                            logits=np.concatenate(fields), indices=np.asarray(indices))
        atomic_json(dict(rows=rows, wall_s=time.perf_counter()-start,
                         compute_s=compute_s, peak_GiB=torch.cuda.max_memory_allocated()/2**30),
                    args.output / f'rows_{size}.json')
        del ds
    summary = {}
    for decoder in ('grid', 'local'):
        a = np.array([r[decoder]['NE'] for r in predictions['28']])
        b = np.array([r[decoder]['NE'] for r in predictions['56']])
        summary[decoder] = dict(n=len(a), R20_28=float((a<=20).mean()),
                               R20_56=float((b<=20).mean()), NE_28=float(a.mean()),
                               NE_56=float(b.mean()), corrections=int(((a>20)&(b<=20)).sum()),
                               damage=int(((a<=20)&(b>20)).sum()))
    atomic_json(dict(summary=summary, checkpoint_sha256=file_sha256(args.trained),
                     data_sha256=file_sha256(args.data/'manifest.json'),
                     scope='TRAIN development; no training, navigation, Val or Test',
                     smoke=bool(args.limit)), args.output / 'summary.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
