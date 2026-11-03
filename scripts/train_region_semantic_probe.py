"""Fixed three-epoch GPU linear probe, no Goal/Belief/controller modification.

Final TRAIN-dev is evaluated once after the predetermined last epoch. Smoke
runs skip evaluation entirely. Contours/IDs enter labels only, never forward.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256, rng_state, restore_rng
from graphnav.region_semantic_probe import RegionSemanticProbe, classification_metrics, shuffle_observed_features


def load_frames(cache, label_root, subsets):
    manifest = json.loads((cache/'manifest.json').read_text())
    proof = json.loads((label_root/'summary.json').read_text())
    if proof['feature_manifest_sha256'] != file_sha256(cache/'manifest.json'):
        raise ValueError('label/feature protocol differs')
    if proof['labels_sha256'] != file_sha256(label_root/'loss_labels_private.pt'):
        raise ValueError('weak labels changed')
    if manifest['split'] != 'train_seen' or manifest['camera_yaw_rad'] != 0:
        raise ValueError('only fixed-yaw TRAIN cache accepted')
    labels = torch.load(label_root/'loss_labels_private.pt', map_location='cpu', weights_only=True)
    output = {subset: [] for subset in subsets}
    for chunk in manifest['chunks']:
        path = cache/chunk['file']
        if file_sha256(path) != chunk['sha256']:
            raise ValueError('corrupt frozen image features')
        for row in torch.load(path, map_location='cpu', weights_only=True)['records']:
            if row['subset'] not in output or row['frame_key'] not in labels:
                continue
            # Explicitly discard all prior, target text and map fields. The
            # probe receives RGB descriptors only.
            output[row['subset']].append(dict(image_features=row['image_features'], local_valid=row['local_valid'],
                label=labels[row['frame_key']], frame_key=row['frame_key'], task_key=row['task_key'], map_name=row['map_name']))
    return output


@torch.no_grad()
def evaluate(model, frames, native_text, priors, pixel_priors):
    model.eval(); truth = []; multi_truth = []; methods = defaultdict(list); multi_methods = defaultdict(list)
    per_frame = []; sensitivity = 0; eligible_frames = 0
    for row in frames:
        image = row['image_features'].cuda(); valid = row['local_valid'].cuda()
        shuffled = shuffle_observed_features(image, valid)
        label = row['label'].numpy(); known = label >= 0
        normalized = F.normalize(image.float(), dim=-1).mean(-2)
        normalized_shuffle = F.normalize(shuffled.float(), dim=-1).mean(-2)
        predictions = {
            'trained': model(image_features=image).argmax(-1).cpu().numpy(),
            'trained-shuffle': model(image_features=shuffled).argmax(-1).cpu().numpy(),
            'native-text': (normalized@native_text.T).argmax(-1).cpu().numpy(),
            'native-text-shuffle': (normalized_shuffle@native_text.T).argmax(-1).cpu().numpy(),
            'class-prior': np.full(len(label), priors.argmax()),
            'pixel-prior': pixel_priors.argmax(-1),
        }
        multi = len(set(label[known])) > 1
        truth.extend(label[known]); eligible_frames += int(multi)
        sensitivity += int((predictions['trained'][known] != predictions['trained-shuffle'][known]).sum())
        if multi: multi_truth.extend(label[known])
        for name, prediction in predictions.items():
            methods[name].extend(prediction[known])
            if multi: multi_methods[name].extend(prediction[known])
        per_frame.append(dict(frame_key=row['frame_key'], task_key=row['task_key'], map_name=row['map_name'],
            labelled_patches=int(known.sum()), multi_class=multi,
            methods={name: classification_metrics(label[known], prediction[known]) for name, prediction in predictions.items()}))
    return dict(metrics={name: classification_metrics(truth, prediction) for name, prediction in methods.items()},
        multi_class_frames=eligible_frames, total_frames=len(frames),
        multi_class_metrics={name: classification_metrics(multi_truth, prediction) for name, prediction in multi_methods.items()},
        shuffle_predictions_changed=sensitivity, rows=per_frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'labels', 'native-text', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 3 or args.max_batches < 0 or (not args.max_batches and args.epochs != 3):
        raise ValueError('three fixed epochs, or explicitly bounded smoke only')
    if args.output.exists() and not args.resume:
        raise FileExistsError('preserve experiment, use exact resume')
    if args.resume and (args.output/'status.json').exists() and not args.max_batches:
        if json.loads((args.output/'status.json').read_text())['status'] == 'complete':
            raise ValueError('completed fixed probe; refuse repeated development evaluation')
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    fingerprint = {str(path): file_sha256(path) for path in
        (args.cache/'manifest.json', args.labels/'summary.json', args.labels/'loss_labels_private.pt',
         args.native_text, Path(__file__), Path('graphnav/region_semantic_probe.py'))}
    text_proof = json.loads(args.native_text.with_suffix('.provenance.json').read_text())
    if text_proof['vectors_sha256'] != file_sha256(args.native_text) or text_proof['captions'] != ['a building', 'a parking area', 'a ground area']:
        raise ValueError('native text changed')
    data = load_frames(args.cache, args.labels, ('fit',) if args.max_batches else ('fit', 'dev'))
    fit = data['fit']
    x = torch.cat([row['image_features'][row['label'] >= 0] for row in fit]).cuda()
    y = torch.cat([row['label'][row['label'] >= 0].long() for row in fit]).cuda()
    counts = y.bincount(minlength=3).float()
    if (counts < 100).any():
        raise ValueError('too little geometric class support for the probe')
    weights = (counts.sum()/(3*counts)).detach()
    model = RegionSemanticProbe().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    history = []; start_epoch = 0; first_gradient = None
    if args.resume:
        state = torch.load(args.output/'last.pt', map_location='cpu', weights_only=True)
        if state['fingerprint'] != fingerprint:
            raise ValueError('resume source/input SHA changed')
        for key in ('cache', 'labels', 'native_text', 'output', 'max_batches'):
            if state['config'][key] != config[key]: raise ValueError('resume config differs: '+key)
        if state['epoch'] > args.epochs: raise ValueError('resume beyond budget')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        start_epoch = state['epoch']; history = state['history']; first_gradient = state['first_gradient']
    args.output.mkdir(parents=True, exist_ok=True)
    def checkpoint(epoch):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(), epoch=epoch,
            history=history, first_gradient=first_gradient, fingerprint=fingerprint, config=config,
            architecture='vision-only-region-linear-probe', selection='fixed final epoch; no development selection')
    if not args.resume: atomic_save(checkpoint(0), args.output/'last.pt')
    atomic_json(dict(config=config, fingerprint=fingerprint, train_patches=len(y), class_counts=counts.cpu().tolist(),
        parameters=sum(p.numel() for p in model.parameters()), model_inputs=['image_features'],
        supervision='TRAIN contour region proxy only, not confirmed pixel-instance labels'), args.output/'config.json')
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train(); losses = []; steps = 0
        generator = torch.Generator().manual_seed(epoch)
        loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True, generator=generator)
        for step, (features, target) in enumerate(loader):
            if args.max_batches and step >= args.max_batches: break
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(image_features=features), target, weight=weights)
            if not torch.isfinite(loss): raise FloatingPointError('invalid proxy classification loss')
            loss.backward()
            if first_gradient is None:
                first_gradient = float(model.classifier.weight.grad.norm())
                if not np.isfinite(first_gradient) or first_gradient <= 0: raise RuntimeError('no visual gradient')
            optimizer.step(); losses.append(float(loss.detach())); steps += 1
        entry = dict(epoch=epoch+1, loss=float(np.mean(losses)), steps=steps,
                     seconds=time.perf_counter()-started)
        history.append(entry)
        atomic_save(checkpoint(epoch+1), args.output/'last.pt')
        atomic_json(dict(history=history, first_gradient=first_gradient,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30), args.output/'training.json')
        print(json.dumps(entry), flush=True)
    if args.max_batches:
        atomic_json(dict(status='smoke_complete', dev_evaluated=False, epoch=args.epochs), args.output/'status.json')
        return
    # Freeze before the single development pass. No checkpoint selection.
    model.eval().requires_grad_(False)
    atomic_save(checkpoint(args.epochs), args.output/'final.pt')
    native_text = F.normalize(torch.load(args.native_text, map_location='cuda', weights_only=True).float(), dim=-1)
    pixel_counts = np.ones((144, 3))
    for row in fit:
        labels = row['label'].numpy()
        for index in range(3): pixel_counts[:, index] += labels == index
    results = {subset: evaluate(model, data[subset], native_text, counts.cpu().numpy(), pixel_counts)
               for subset in ('fit', 'dev')}
    atomic_json(dict(status='complete', checkpoint_sha256=file_sha256(args.output/'final.pt'),
        results=results, fixed_last_epoch=3, navigation_run=False,
        warning='Correlated region proxy patches, not actual pixel-instance accuracy or navigation SR'), args.output/'evaluation.json')
    atomic_json(dict(status='complete', dev_evaluated_once_after_freeze=True), args.output/'status.json')
    print(json.dumps({subset: result['metrics'] for subset, result in results.items()}), flush=True)


if __name__ == '__main__': main()
