"""Paired frozen SigLIP ROI-resolution probe; no GT-dependent public decisions."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graphnav.checkpointing import atomic_json, file_sha256


def public_choices(features, valid, text, index):
    """All arguments are inference-available. No target/label is accepted."""
    known = valid.any(-1)
    pooled = (features * valid[None, :, :, None]).sum(2) / valid.sum(-1).clamp_min(1)[None, :, None]
    scores = F.normalize(pooled, dim=-1) @ F.normalize(text, dim=-1).T
    permutation = torch.arange(len(valid), device=valid.device)
    rng = np.random.default_rng(42 + index)
    for pattern in torch.unique(valid, dim=0):
        ids = ((valid == pattern).all(-1) & known).nonzero().flatten()
        permutation[ids] = ids[torch.tensor(rng.permutation(len(ids)), device=valid.device)]
    choices = {'geometry': 0}
    for r, resolution in enumerate(('current384', 'native_roi')):
        for c, channel in enumerate(('full', 'appearance')):
            for shuffled in (False, True):
                key = f'{resolution}_{channel}' + ('_shuffle' if shuffled else '')
                if bool(known[0]) and int(known.sum()) >= 2:
                    s = scores[r, permutation, c] if shuffled else scores[r, :, c]
                    choices[key] = int(s.masked_fill(~known, -torch.inf).argmax())
                else:
                    choices[key] = 0
    return choices, scores


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'data', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((a.cache / 'manifest.json').read_text())
    assert manifest['public_queries_sha256'] == file_sha256(a.data / 'public_train.jsonl')
    labels = json.loads((a.data / 'labels_private.json').read_text())
    selected = [r for r in labels if r['split'] == 'development']
    rows = []
    sizes = []
    start = time.perf_counter()
    for label in selected:
        i = label['index']
        with np.load(a.cache / 'public_features' / f'{i:05d}.npz') as z:
            valid = torch.tensor(z['valid'], device='cuda')
            features = torch.tensor(z['features'].astype(np.float32), device='cuda')
            text = torch.tensor(z['text_features'].astype(np.float32), device='cuda')
            choices, scores = public_choices(features, valid, text, i)
            # Private labels become available only after every arm is sealed.
            distance = np.linalg.norm(z['candidate_world'] - label['target_world'], axis=1)
            supported = z['valid'].any(-1)
            metadata = json.loads((a.cache / 'public_features' / f'{i:05d}.json').read_text())
            sizes.extend({key: roi[key] for key in ('extent_m', 'native_size', 'low_size', 'source_gsd_m')}
                         for roi in metadata['roi_metadata'] if roi['valid'])
            rows.append(dict(index=i, supported=int(supported.sum()),
                             eligible=bool(supported[0] and supported.sum()>=2),
                             correct_supported=bool(((distance<=20)&supported).any()),
                             choice=choices, NE={k: float(distance[v]) for k,v in choices.items()}))
    torch.cuda.synchronize()
    summary = {}
    for subset in ('all', 'public_eligible', 'correct_supported_eligible'):
        rr = [r for r in rows if subset=='all' or
              (r['eligible'] and (subset=='public_eligible' or r['correct_supported']))]
        metrics = {}
        for arm in rows[0]['choice']:
            errors = np.array([r['NE'][arm] for r in rr])
            baseline = np.array([r['NE']['geometry'] for r in rr])
            metrics[arm] = dict(n=len(rr), R20=float((errors<=20).mean()) if len(rr) else None,
                                NE=float(errors.mean()) if len(rr) else None,
                                corrections=int(((errors<=20)&(baseline>20)).sum()),
                                damage=int(((errors>20)&(baseline<=20)).sum()))
        summary[subset] = metrics
    a.output.mkdir(parents=True, exist_ok=True)
    atomic_json(rows, a.output / 'rows.json')
    atomic_json(sizes, a.output / 'roi_sizes.json')
    atomic_json(dict(summary=summary, n=len(rows), seconds=time.perf_counter()-start,
                     inputs=dict(cache_manifest=file_sha256(a.cache/'manifest.json'),
                                 labels=file_sha256(a.data/'labels_private.json')),
                     scope='TRAIN development; centre-distance location proxy, not instance accuracy; cached real-start observations'),
                a.output / 'summary.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
