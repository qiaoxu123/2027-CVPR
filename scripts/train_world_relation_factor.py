"""TRAIN-only, seed-matched relation factor fitting on a frozen V3 baseline.

Public forward inputs are explicitly whitelisted. Private positions enter only
goal loss or posthoc evaluation. This script cannot load formal Val/Test.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from graphnav.checkpointing import atomic_json, atomic_save, file_sha256, rng_state, restore_rng
from graphnav.relation_factor_data import public_reference_inputs
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from graphnav.world_relation_factor import WorldRelationFactor
from scripts.train_local_chain_position import records, INPUTS

REFERENCE_INPUTS = ('reference_text', 'relation_ids', 'reference_valid',
                    'reference_geometry', 'reference_map_types', 'reference_support')


def load_data(args, subset, role_vectors, proposals, vectors, catalog):
    allowed = set(json.loads(args.fit_observation_filter.read_text())['frame_keys']) if subset == 'fit' else None
    rows = records(args.cache, getattr(args, subset+'_labels'), subset,
                   role_vectors, 'reference', allowed)
    for row in rows:
        role = proposals[row['task_key']]
        if role['map_name'] != row['map_name']:
            raise ValueError('role and feature map mismatch')
        row.update(public_reference_inputs(world_hypotheses=row['candidate_world'],
            global_cells=row['prior_logits'].numel(), quality=row['quality'], role=role,
            text_vectors=vectors[row['task_key']], map_objects=catalog[row['map_name']]))
    return rows


def collate(rows):
    batch = {key: torch.stack([r[key] for r in rows])
             for key in (*INPUTS, *REFERENCE_INPUTS, 'candidate_world', 'target_world')}
    for key in ('task_key', 'frame_key', 'category'):
        batch[key] = [row[key] for row in rows]
    return batch


def forward_inputs(batch, device):
    inputs = {key: batch[key].to(device) for key in (*INPUTS, *REFERENCE_INPUTS)}
    inputs['world_hypotheses'] = batch['candidate_world'].to(device)
    return inputs


def goal_loss(logits, base_logits, world_hypotheses, target_world):
    cells = base_logits.shape[-1]
    distances = (world_hypotheses[:, :cells]-target_world[:, None]).norm(dim=-1)
    positive = distances <= 20
    eligible = positive.any(-1)
    logprob = logits[:, :cells].log_softmax(-1)
    good = torch.logsumexp(logprob.masked_fill(~positive, float('-inf')), -1)
    goal = -good[eligible].mean() if eligible.any() else logits[:, :cells].sum()*0
    correct = logprob.masked_fill(~positive, float('-inf')).amax(-1)
    incorrect = logprob.masked_fill(positive, float('-inf')).amax(-1)
    rank = torch.relu(.2+incorrect-correct)[eligible].mean() if eligible.any() else goal*0
    base_prob = base_logits.detach().log_softmax(-1)
    preservation = (base_prob.exp()*(base_prob-logprob)).sum(-1).mean()
    return goal+.1*rank+.1*preservation


def metrics(rows):
    group = defaultdict(list)
    for row in rows:
        group[row['task_key']].append(row)
    result = dict(frames=len(rows), physical_targets=len(group),
        goal_r1=float(np.mean([r['success'] for r in rows])),
        goal_ne_m=float(np.mean([r['ne_m'] for r in rows])),
        task_balanced_goal_r1=float(np.mean([np.mean([r['success'] for r in members]) for members in group.values()])))
    for baseline in ('base', 'a0'):
        correct = sum(not r[baseline+'_success'] and r['success'] for r in rows)
        harm = sum(r[baseline+'_success'] and not r['success'] for r in rows)
        result[baseline+'_corrections'] = correct
        result[baseline+'_harms'] = harm
        result[baseline+'_net'] = correct-harm
    return result


@torch.no_grad()
def evaluate(model, rows, device, batch_size, mode, control='real'):
    model.eval(); result = []
    for batch in DataLoader(rows, batch_size=batch_size, collate_fn=collate):
        inputs = forward_inputs(batch, device)
        prediction, details = model(**inputs, relation_mode=mode, control=control, return_details=True)
        base = model.base(**{key: inputs[key] for key in INPUTS})
        cells = inputs['prior_logits'].shape[-1]
        choice = prediction[:, :cells].argmax(-1).cpu()
        base_choice = base[:, :cells].argmax(-1).cpu()
        a0_choice = batch['prior_logits'].argmax(-1)
        for i in range(len(choice)):
            world = batch['candidate_world'][i]; truth = batch['target_world'][i]
            ne, base_ne, a0_ne = [float((world[c]-truth).norm()) for c in (choice[i], base_choice[i], a0_choice[i])]
            active = details['active_references'][i]
            result.append(dict(task_key=batch['task_key'][i], frame_key=batch['frame_key'][i], category=batch['category'][i],
                ne_m=ne, success=ne <= 20, base_ne_m=base_ne, base_success=base_ne <= 20,
                a0_ne_m=a0_ne, a0_success=a0_ne <= 20, active_references=int(active.sum()),
                reference_mass=float(details['reference_mass'][i][active].mean()) if active.any() else None,
                evidence_range=float(details['evidence'][i].max()-details['evidence'][i].min()),
                predicted_world=world[choice[i]].tolist()))
    return dict(metrics=metrics(result), rows=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'fit-labels', 'dev-labels', 'fit-observation-filter', 'base-checkpoint',
                 'base-role-vectors', 'roles', 'reference-vectors', 'catalog', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--relation-mode', choices=('typed', 'hidden'), required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps-per-epoch', type=int, default=159)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 3 or not 1 <= args.steps_per_epoch <= 159:
        raise ValueError('bounded three-epoch TRAIN pilot only')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Preserve experiments; use exact resume')
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    provenance = json.loads(args.reference_vectors.with_suffix('.provenance.json').read_text())
    role_proof = json.loads((args.roles.parent/'summary.json').read_text())
    if role_proof['catalog_sha256'] != file_sha256(args.catalog):
        raise ValueError('role/catalog provenance mismatch')
    if provenance['roles_sha256'] != file_sha256(args.roles) or provenance['vectors_sha256'] != file_sha256(args.reference_vectors):
        raise ValueError('public text cache provenance mismatch')
    fingerprint = {name: file_sha256(getattr(args, name)) for name in
        ('fit_labels', 'dev_labels', 'fit_observation_filter', 'base_checkpoint',
         'base_role_vectors', 'roles', 'reference_vectors', 'catalog')}
    fingerprint['feature_manifest'] = file_sha256(args.cache/'manifest.json')
    for path in (Path(__file__), Path('graphnav/world_relation_factor.py'),
                 Path('graphnav/relation_factor_data.py'), Path('graphnav/relation_factor_roles.py'),
                 Path('graphnav/visual_chain_position.py'), Path('graphnav/world_likelihood_evidence.py'),
                 Path('scripts/train_local_chain_position.py')):
        fingerprint[str(path.resolve().relative_to(Path.cwd()))] = file_sha256(path)
    manifest = json.loads((args.cache/'manifest.json').read_text())
    if manifest['split'] != 'train_seen' or manifest['camera_yaw_rad'] != 0:
        raise ValueError('TRAIN fixed-yaw feature cache only')
    state = torch.load(args.base_checkpoint, map_location='cpu', weights_only=True)
    if state['architecture'] != 'world-likelihood-v3' or state['feature_manifest_sha256'] != fingerprint['feature_manifest']:
        raise ValueError('frozen base feature provenance mismatch')
    if state['role_vectors_sha256'] != fingerprint['base_role_vectors']:
        raise ValueError('frozen base role cache mismatch')
    base = WorldLikelihoodPositionHead(); base.load_state_dict(state['model'], strict=True)
    model = WorldRelationFactor(base).to(args.device)
    original_base = {k: value.clone() for k, value in model.base.state_dict().items()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=.001)
    role_vectors = torch.load(args.base_role_vectors, map_location='cpu', weights_only=True)
    vectors = torch.load(args.reference_vectors, map_location='cpu', weights_only=True)
    proposals = json.loads(args.roles.read_text()); catalog = json.loads(args.catalog.read_text())
    fit = load_data(args, 'fit', role_vectors, proposals, vectors, catalog)
    dev = load_data(args, 'dev', role_vectors, proposals, vectors, catalog)
    fit_groups = {r['task_key'] for r in fit}; dev_groups = {r['task_key'] for r in dev}
    if fit_groups & dev_groups or {r['map_name'] for r in fit} & {r['map_name'] for r in dev}:
        raise ValueError('new-head TRAIN fit/dev group leakage')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    start_epoch = 0; history = []; best = (-1., float('-inf')); first_gradient = None
    if args.resume:
        last = torch.load(args.output/'last.pt', map_location=args.device, weights_only=True)
        if last['fingerprint'] != fingerprint:
            raise ValueError('resume source/input hashes changed')
        for key in config:
            if key not in ('resume', 'epochs', 'device') and config[key] != last['config'][key]:
                raise ValueError('resume config changed: '+key)
        if last['epoch'] > args.epochs:
            raise ValueError('resume epoch beyond budget')
        model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer']); restore_rng(last['rng'])
        history = last['history']; best = tuple(last['best']); start_epoch = last['epoch']
        first_gradient = last['first_effective_gradient']
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_summary = {}
    for label, data in [('fit', fit), ('dev', dev)]:
        active = [r for r in data if (r['reference_support'] & r['local_valid'][None] & r['reference_valid'][:, None]).any()]
        dataset_summary[label] = dict(frames=len(data), tasks=len({r['task_key'] for r in data}),
            active_frames=len(active), active_tasks=len({r['task_key'] for r in active}),
            multi_reference_frames=sum(int((r['reference_support'] & r['local_valid'][None] & r['reference_valid'][:, None]).any(-1).sum()) > 1 for r in data))
    atomic_json(dict(config=config, fingerprint=fingerprint, dataset=dataset_summary,
                     trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                     supervision='TRAIN world goal loss only; no reference instance labels or pixel GT'), args.output/'config.json')
    print(json.dumps(dict(dataset=dataset_summary, status='training')), flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train(); losses = []; active_presentations = 0
        generator = torch.Generator().manual_seed(args.seed+epoch)
        sampler = RandomSampler(fit, replacement=True, num_samples=args.steps_per_epoch*args.batch_size, generator=generator)
        for batch in DataLoader(fit, batch_size=args.batch_size, sampler=sampler, generator=generator, collate_fn=collate):
            optimizer.zero_grad(set_to_none=True)
            inputs = forward_inputs(batch, args.device)
            logits, details = model(**inputs, relation_mode=args.relation_mode, return_details=True)
            with torch.no_grad():
                base_logits = model.base(**{key: inputs[key] for key in INPUTS})[:, :784]
            loss = goal_loss(logits, base_logits, inputs['world_hypotheses'], batch['target_world'].to(args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite relation goal loss')
            loss.backward()
            if first_gradient is None and details['active_references'].any():
                first_gradient = {key: float(p.grad.norm()) if p.grad is not None else None
                                  for key, p in model.named_parameters() if p.requires_grad}
                for key in ('reference_image.weight', 'pair_geometry.0.weight', 'relation_embedding.weight'):
                    if not first_gradient[key] or not np.isfinite(first_gradient[key]):
                        raise RuntimeError('missing effective gradient: '+key)
            if any(p.grad is not None for p in model.base.parameters()):
                raise RuntimeError('frozen original visual baseline received gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); losses.append(float(loss.detach()))
            active_presentations += int(details['active_references'].any(-1).sum())
        validation = evaluate(model, dev, args.device, args.batch_size, args.relation_mode)
        entry = dict(epoch=epoch+1, loss=float(np.mean(losses)), active_presentations=active_presentations,
                     elapsed_seconds=time.perf_counter()-started, dev=validation['metrics'])
        history.append(entry)
        key = (entry['dev']['goal_r1'], -entry['dev']['goal_ne_m']); improved = key > best
        if improved:
            best = key
        if any(not torch.equal(value, original_base[k]) for k, value in model.base.state_dict().items()):
            raise RuntimeError('frozen base tensor changed')
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(),
            epoch=epoch+1, config=config, fingerprint=fingerprint, history=history, best=best,
            first_effective_gradient=first_gradient, architecture='world-relation-factor-v1')
        atomic_save(checkpoint, args.output/'last.pt')
        if improved:
            atomic_save(checkpoint, args.output/'best.pt')
        atomic_json(dict(history=history, first_effective_gradient=first_gradient,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            frozen_base_unchanged=True), args.output/'training.json')
        print(json.dumps(entry), flush=True)
    selected = torch.load(args.output/'best.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(selected['model'])
    controls = {control: evaluate(model, dev, args.device, args.batch_size, args.relation_mode, control)
                for control in ('real', 'shuffle-references', 'no-map-score')}
    controls['hidden-relations'] = evaluate(model, dev, args.device, args.batch_size, 'hidden')
    atomic_json(dict(selected_epoch=selected['epoch'], checkpoint_sha256=file_sha256(args.output/'best.pt'),
        results=controls, total_seconds=time.perf_counter()-started,
        warning='TRAIN development diagnostics, not independent confirmation or navigation SR'), args.output/'development.json')
    print(json.dumps(dict(status='complete', selected_epoch=selected['epoch'],
        development={key: value['metrics'] for key, value in controls.items()})), flush=True)


if __name__ == '__main__':
    main()
