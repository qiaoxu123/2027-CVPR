"""Same-inference V3 warm-start control versus training-only region supervision.

Frozen image/A0 caches, fixed three epochs, no new controller or inference
fields. Private region labels and world positions are loss-side only.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256, rng_state, restore_rng
from graphnav.region_auxiliary_loss import RegionAuxiliaryHead, region_auxiliary_loss
from graphnav.visual_chain_position import WorldLikelihoodPositionHead, visual_chain_loss
from scripts.train_local_chain_position import records, collate as base_collate, batch_inputs, evaluate


def collate(rows):
    result = base_collate(rows)
    if 'private_region_labels' in rows[0]:
        result['private_region_labels'] = torch.stack([r['private_region_labels'] for r in rows])
    return result


def paired_changes(prediction, reference):
    expected = {r['frame_key']: r for r in reference['rows']}
    if set(expected) != {r['frame_key'] for r in prediction['rows']}:
        raise ValueError('same paired Goal evaluation frames required')
    corrections = sum(r['success'] and not expected[r['frame_key']]['success'] for r in prediction['rows'])
    harms = sum(not r['success'] and expected[r['frame_key']]['success'] for r in prediction['rows'])
    return dict(corrections=corrections, harms=harms, net=corrections-harms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'fit-labels', 'dev-labels', 'fit-observation-filter', 'role-vectors',
                'region-labels', 'warm-start', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--auxiliary-weight', type=float, choices=(0., .5), required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps-per-epoch', type=int, default=159)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    smoke = args.steps_per_epoch == 2
    if args.steps_per_epoch not in (2, 159) or not 1 <= args.epochs <= 3 or (not smoke and args.epochs != 3):
        raise ValueError('fixed three-epoch 477-update pilot or bounded smoke only')
    if args.output.exists() and not args.resume:
        raise FileExistsError('preserve earlier experiments, use exact resume')
    if args.resume and (args.output/'status.json').exists():
        if json.loads((args.output/'status.json').read_text())['status'] == 'complete':
            raise ValueError('terminal experiment, no repeated development evaluation')
    torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    fingerprint = {str(path): file_sha256(path) for path in
        (args.cache/'manifest.json', args.fit_labels, args.dev_labels, args.fit_observation_filter,
         args.role_vectors, args.region_labels/'summary.json', args.region_labels/'loss_labels_private.pt',
         args.warm_start, Path(__file__), Path('graphnav/region_auxiliary_loss.py'),
         Path('graphnav/visual_chain_position.py'), Path('graphnav/world_likelihood_evidence.py'),
         Path('graphnav/local_chain_raster.py'), Path('scripts/train_local_chain_position.py'))}
    source = torch.load(args.warm_start, map_location='cpu', weights_only=True)
    if source['architecture'] != 'world-likelihood-v3' or source['feature_manifest_sha256'] != file_sha256(args.cache/'manifest.json'):
        raise ValueError('frozen V3/cache protocol mismatch')
    if source['role_vectors_sha256'] != file_sha256(args.role_vectors) or source['config']['role_mode'] != 'reference':
        raise ValueError('frozen V3 public text protocol differs')
    region_proof = json.loads((args.region_labels/'summary.json').read_text())
    if region_proof['feature_manifest_sha256'] != source['feature_manifest_sha256'] or region_proof['labels_sha256'] != file_sha256(args.region_labels/'loss_labels_private.pt'):
        raise ValueError('weak region annotation provenance mismatch')
    if region_proof['filter_sha256'] != file_sha256(args.fit_observation_filter) or source['fit_observation_filter_sha256'] != file_sha256(args.fit_observation_filter):
        raise ValueError('warm-start/region/Goal observation envelope differs')
    model = WorldLikelihoodPositionHead().cuda(); model.load_state_dict(source['model'], strict=True)
    auxiliary = RegionAuxiliaryHead().cuda()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(list(model.parameters())+list(auxiliary.parameters()), lr=.001, weight_decay=.001)
    roles = torch.load(args.role_vectors, map_location='cpu', weights_only=True)
    allowed = set(json.loads(args.fit_observation_filter.read_text())['frame_keys'])
    fit = records(args.cache, args.fit_labels, 'fit', roles, 'reference', allowed)
    masks = torch.load(args.region_labels/'loss_labels_private.pt', map_location='cpu', weights_only=True)
    counts = torch.zeros(3, dtype=torch.long)
    for row in fit:
        mask = masks[row['frame_key']]
        if mask.shape != row['local_valid'].shape or not ((mask >= -1) & (mask < 3)).all():
            raise ValueError('invalid auxiliary label shape/classes')
        row['private_region_labels'] = mask
        known = mask[(mask >= 0) & row['local_valid']].long()
        counts += known.bincount(minlength=3)
    if (counts <= 0).any(): raise ValueError('all three TRAIN region classes required')
    class_weights = (counts.sum()/(3*counts.float())).cuda()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(role_mode='reference', seed=0, batch_size=8, lr=.001)
    history = []; start_epoch = 0; gradient_proof = None
    if args.resume:
        previous = torch.load(args.output/'last.pt', map_location='cpu', weights_only=True)
        if previous['fingerprint'] != fingerprint:
            raise ValueError('resume source/input SHA mismatch')
        for key in config:
            if key not in ('epochs', 'resume') and config[key] != previous['config'][key]:
                raise ValueError('resume configuration changed: '+key)
        if previous['epoch'] > args.epochs: raise ValueError('resume beyond budget')
        model.load_state_dict(previous['model']); auxiliary.load_state_dict(previous['region_auxiliary'])
        optimizer.load_state_dict(previous['optimizer']); restore_rng(previous['rng'])
        start_epoch = previous['epoch']; history = previous['history']; gradient_proof = previous['gradient_proof']
    args.output.mkdir(parents=True, exist_ok=True)
    def checkpoint(epoch):
        return dict(model=model.state_dict(), region_auxiliary=auxiliary.state_dict(),
            optimizer=optimizer.state_dict(), rng=rng_state(), epoch=epoch, history=history,
            gradient_proof=gradient_proof, fingerprint=fingerprint, config=config,
            architecture='world-likelihood-v3', experiment='region-auxiliary-goal-v1',
            feature_manifest_sha256=file_sha256(args.cache/'manifest.json'),
            role_vectors_sha256=file_sha256(args.role_vectors),
            warm_start_checkpoint_sha256=file_sha256(args.warm_start),
            selection='fixed final epoch; not selected by development outcomes')
    if not args.resume: atomic_save(checkpoint(0), args.output/'last.pt')
    atomic_json(dict(config=config, fingerprint=fingerprint, fit_frames=len(fit), class_counts=counts.tolist(),
        inference_parameters=sum(p.numel() for p in model.parameters()),
        training_only_parameters=sum(p.numel() for p in auxiliary.parameters()),
        normal_inference_changed=False), args.output/'config.json')
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train(); auxiliary.train(); losses = []; aux_losses = []; points = 0
        generator = torch.Generator().manual_seed(epoch)
        sampler = RandomSampler(fit, replacement=True, num_samples=args.steps_per_epoch*8, generator=generator)
        for batch in DataLoader(fit, batch_size=8, sampler=sampler, generator=generator, collate_fn=collate):
            optimizer.zero_grad(set_to_none=True)
            public = batch_inputs(batch, 'cuda')
            logits, visual = model(**public, return_visual=True)
            terms = visual_chain_loss(model, logits, visual, public, batch['candidate_world'].cuda(),
                batch['target_world'].cuda(), batch['category'], batch['target_phrase'])
            aux_logits = auxiliary(image_features=public['image_features'], projection=model.image)
            aux_loss, known = region_auxiliary_loss(aux_logits, batch['private_region_labels'].cuda(), public['local_valid'], class_weights)
            if gradient_proof is None and known:
                goal_gradient = torch.autograd.grad(terms['total'], model.image.weight, retain_graph=True)[0]
                region_gradient = torch.autograd.grad(aux_loss, model.image.weight, retain_graph=True)[0]
                gradient_proof = dict(goal_visual_norm=float(goal_gradient.norm()), region_shared_visual_norm=float(region_gradient.norm()),
                    cosine_similarity=float(F.cosine_similarity(goal_gradient.flatten(), region_gradient.flatten(), dim=0)),
                    effective_region_weight=args.auxiliary_weight)
                if not torch.isfinite(region_gradient).all() or gradient_proof['region_shared_visual_norm'] <= 0:
                    raise RuntimeError('auxiliary does not reach shared visual projection')
            loss = terms['total']+args.auxiliary_weight*aux_loss
            if not torch.isfinite(loss): raise FloatingPointError('non-finite joint loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(auxiliary.parameters()), 5.)
            optimizer.step(); losses.append(float(loss.detach())); aux_losses.append(float(aux_loss.detach())); points += known
        entry = dict(epoch=epoch+1, total_loss=float(np.mean(losses)), region_loss=float(np.mean(aux_losses)),
            region_point_presentations=points, steps=args.steps_per_epoch, seconds=time.perf_counter()-started)
        history.append(entry)
        atomic_save(checkpoint(epoch+1), args.output/'last.pt')
        atomic_json(dict(history=history, gradient_proof=gradient_proof,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            changed_visual_projection_max_abs=float((model.image.weight-initial['image.weight']).abs().max())), args.output/'training.json')
        print(json.dumps(entry), flush=True)
    if smoke:
        atomic_json(dict(status='smoke_complete', dev_evaluated=False), args.output/'status.json'); return
    model.eval(); auxiliary.eval()
    atomic_save(checkpoint(args.epochs), args.output/'final.pt')
    # Verify that the standard, unmodified deployment constructor can load it.
    fresh = WorldLikelihoodPositionHead(); fresh.load_state_dict(torch.load(args.output/'final.pt', map_location='cpu', weights_only=True)['model'], strict=True)
    dev = records(args.cache, args.dev_labels, 'dev', roles, 'reference')
    if {r['map_name'] for r in fit} & {r['map_name'] for r in dev}:
        raise ValueError('new Goal-head fitting/development map leakage')
    baseline = WorldLikelihoodPositionHead().cuda().eval(); baseline.load_state_dict(source['model'], strict=True)
    reference = evaluate(baseline, dev, 'cuda', 8)
    results = {control: evaluate(model, dev, 'cuda', 8, control) for control in ('real', 'zero', 'shuffle', 'no_geometry', 'full_text')}
    changes = {control: paired_changes(value, reference) for control, value in results.items()}
    atomic_json(dict(checkpoint_sha256=file_sha256(args.output/'final.pt'), frozen_baseline=reference,
        results=results, paired_vs_frozen=changes, fixed_final_epoch=3, navigation_run=False,
        warning='TRAIN development Goal results; regions are weak proxies, not pixel instance supervision'), args.output/'development.json')
    atomic_json(dict(status='complete', dev_evaluated_after_fixed_freeze=True,
        standard_deployment_state_dict_compatible=True), args.output/'status.json')
    print(json.dumps(dict(status='complete', baseline=reference['metrics'],
        results={control: value['metrics'] for control, value in results.items()}, paired_vs_frozen=changes)), flush=True)


if __name__ == '__main__': main()
