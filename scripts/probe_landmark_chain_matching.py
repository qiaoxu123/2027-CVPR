"""Frozen TRAIN role-text mechanism probe. Predict and score are separate commands.

No instance-GT boxes exist here: 20m world proximity is a spatial proxy only.
No global Belief/controller is evaluated by this script.
"""
import argparse
import json
from pathlib import Path
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder, MultiScaleAppearanceMatcher
from graphnav.checkpointing import atomic_json, file_sha256


def predict(a):
    if a.output.exists():
        raise FileExistsError('do not overwrite sealed predictions')
    meta = json.loads((a.features/'manifest.json').read_text())
    if not meta['complete'] or not meta['inputs_public_only']:
        raise ValueError('complete public-only observed features required')
    roles = json.loads(a.roles.read_text())
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    native = ObservedSigLIPMatchingEncoder(a.backbone, device=a.device)
    matcher = MultiScaleAppearanceMatcher().to(a.device).eval()
    matcher.load_state_dict({k.removeprefix('matcher.'):v for k,v in state['model'].items() if k.startswith('matcher.')}, strict=True)
    matcher.requires_grad_(False)
    target = {k: native.texts([v['target_text']]) for k,v in roles.items()}
    reference = {k: native.texts([v['reference_text']]) for k,v in roles.items()}
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for chunk in meta['chunks']:
            path = a.features/chunk['file']
            if file_sha256(path) != chunk['sha256']:
                raise ValueError('feature cache changed')
            for record in torch.load(path, map_location='cpu', weights_only=True)['records']:
                key = record['task_key']
                image = record['roi_features'][None].float().to(a.device)
                valid = record['roi_valid'][None].to(a.device)
                quality = record['roi_quality'][None].float().to(a.device)
                original = F.normalize((record['native_text'].float()+record['appearance_text'].float())[None], dim=-1).to(a.device)
                # Fixed, pre-specified scope probe: target at smallest supported ROI,
                # surrounding description at largest supported ROI. No world directions compiled.
                small = valid & (valid.long().cumsum(-1) == 1)
                large = valid & (valid.flip(-1).long().cumsum(-1).flip(-1) == 1)
                def score(text, features, support):
                    return matcher(text, features, support, quality)[0][0]
                baseline = score(original, image, valid)
                target_only = score(target[key], image, valid)
                chain = score(target[key], image, small) + .5*score(reference[key], image, large)
                shuffled = image.clone()
                for scale in range(valid.shape[-1]):
                    ids = valid[0,:,scale].nonzero().flatten()
                    shuffled[0,ids,scale] = image[0,ids.flip(0),scale]
                chain_shuffle = score(target[key], shuffled, small) + .5*score(reference[key], shuffled, large)
                baseline_shuffle = score(original, shuffled, valid)
                target_shuffle = score(target[key], shuffled, valid)
                baseline_zero = score(original, torch.zeros_like(image), valid)
                result = dict(frame_key=record['frame_key'], task_key=key, map_name=record['map_name'],
                              step=record['step'], supported_world=record['grid_world'][record['roi_ids']].tolist(), methods={})
                for method, scores in [('original_text',baseline), ('target_text',target_only), ('scope_multiscale',chain), ('scope_shuffled',chain_shuffle), ('original_shuffled',baseline_shuffle), ('target_shuffled',target_shuffle), ('original_zero',baseline_zero)]:
                    order = scores.argsort(descending=True)[:10].cpu()
                    result['methods'][method] = dict(top_world=record['grid_world'][record['roi_ids'][order]].tolist(),
                                                      top_scores=scores[order.to(a.device)].cpu().tolist())
                predictions.append(result)
    report = dict(scope='frozen local observed-cell spatial ranking only; NOT instance accuracy, NOT Goal Belief or SR',
                  checkpoint_sha256=file_sha256(a.checkpoint), features_manifest_sha256=file_sha256(a.features/'manifest.json'),
                  roles_sha256=file_sha256(a.roles), source_sha256=file_sha256(__file__),
                  predictor_reads_labels=False, frames=len(predictions), seconds=time.perf_counter()-started,
                  scope_weight=.5, scope_weight_selection='fixed before any scoring, mechanism development only', predictions=predictions)
    atomic_json(report, a.output)
    print(json.dumps(dict(sealed_prediction=str(a.output), sha256=file_sha256(a.output), frames=len(predictions))))


def score(a):
    if a.output.exists():
        raise FileExistsError('preserve prior scores')
    # Load and SHA predictions before labels. Scoring does not rewrite predictions.
    prediction_sha = file_sha256(a.predictions)
    prediction = json.loads(a.predictions.read_text())
    labels = {r['frame_key']:r for r in map(json.loads, a.labels.read_text().splitlines()) if r['status']=='captured'}
    rows = []
    for row in prediction['predictions']:
        label = labels[row['frame_key']]
        gt = np.array(label['target_world'][:2])
        support = np.array(row['supported_world'])
        eligible = len(support)>0 and np.linalg.norm(support-gt,axis=-1).min()<=20.
        out = dict(frame_key=row['frame_key'],task_key=row['task_key'],category=label['category'],
                   geometrically_supported=bool(eligible), true_visibility='unknown', instance_match='unknown', methods={})
        for method, values in row['methods'].items():
            distance = np.linalg.norm(np.array(values['top_world']).reshape(-1,2)-gt,axis=-1)
            out['methods'][method] = dict(top1=bool(len(distance) and distance[0]<=20),
                                         top5=bool(len(distance) and distance[:5].min()<=20),
                                         top1_ne_m=float(distance[0]) if len(distance) else None)
        rows.append(out)
    summary = {}
    methods = list(rows[0]['methods'])
    def aggregate(part):
        tasks = defaultdict(list)
        for r in part:
            tasks[r['task_key']].append(r)
        values = {}
        for method in methods:
            paired = [(r['methods']['original_text']['top1'],r['methods'][method]['top1']) for r in part]
            values[method] = dict(frames=len(part), physical_targets=len(tasks),
                spatial_top1=sum(r['methods'][method]['top1'] for r in part)/len(part) if part else None,
                spatial_top5=sum(r['methods'][method]['top5'] for r in part)/len(part) if part else None,
                top1_ne_m=np.mean([r['methods'][method]['top1_ne_m'] for r in part if r['methods'][method]['top1_ne_m'] is not None]).item() if part else None,
                task_balanced_top1=np.mean([np.mean([r['methods'][method]['top1'] for r in group]) for group in tasks.values()]).item() if tasks else None,
                spatial_corrections_vs_original=sum(not x and y for x,y in paired), spatial_harms_vs_original=sum(x and not y for x,y in paired))
        return values
    summary['all_captured'] = aggregate(rows)
    summary['geometric_support_only_not_verified_visible'] = aggregate([r for r in rows if r['geometrically_supported']])
    for category in sorted({r['category'] for r in rows}):
        summary[category] = aggregate([r for r in rows if r['category']==category])
    atomic_json(dict(scope=prediction['scope'], warning='20m world proximity does not prove correct-instance recognition; no human boxes or verified visibility. Development samples only.',
                     sealed_prediction_sha256=prediction_sha, labels_sha256=file_sha256(a.labels),
                     true_visibility_unknown=len(rows), summary=summary, rows=rows), a.output)
    if file_sha256(a.predictions) != prediction_sha:
        raise RuntimeError('scoring mutated sealed predictions')
    print(json.dumps(summary['geometric_support_only_not_verified_visible'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    infer = sub.add_parser('predict')
    for name in ('features','roles','checkpoint','output'):
        infer.add_argument('--'+name, required=True, type=Path)
    infer.add_argument('--backbone', required=True)
    infer.add_argument('--device', default='cuda')
    evaluate = sub.add_parser('score')
    for name in ('predictions','labels','output'):
        evaluate.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    (predict if a.mode=='predict' else score)(a)


if __name__ == '__main__':
    main()
