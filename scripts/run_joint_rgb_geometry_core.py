"""Run the ONE essential paired comparison: original residual vs joint RGB/geometry.

Single seed; same expert-prefix cache, verified A3/head, ORIGINAL A3/B3/C5
staged training protocol, original closed-loop evaluator and Seen128 episodes.
This script is intentionally NOT a multi-seed / shuffle / zero-RGB sweep.
CUDA, full CityNav data, pretrained model caches and original checkpoints
must be available on the training host.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


STAGES = (('A', 3), ('B', 3), ('C', 5))
ARMS = (('A0_original_residual', 'original', 'graphnav-v1'),
        ('A1_joint_rgb_geometry', 'joint-rgb-geometry', 'graphnav-joint-rgb'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def launch(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as out:
        out.write('$ ' + ' '.join(map(str, command)) + '\n')
        out.flush()
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache','sources','sbf-source','sbf-config','data-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--output', default='runs/joint_rgb_geometry_core_20261027')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('full paired training requires CUDA host with CityNav data')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(args.sources).read_text())
    if not {'a3','head','reference'} <= source.keys():
        raise ValueError('missing original verified training sources')
    for role in ('a3','head','reference'):
        path = Path(source[role]['path'])
        if not path.exists():
            raise FileNotFoundError(f'{role}: {path}')
    manifest = json.loads((Path(args.cache)/'manifest.json').read_text())
    if not manifest.get('complete') or manifest.get('smoke_only'):
        raise ValueError('full original Stage1 expert-prefix cache required')

    protocol = dict(
        experiment='one_seed_paired_original_vs_joint_rgb_geometry',
        model='graphnav-stage1',
        seed=args.seed, epochs=dict(STAGES), batch_size=args.batch_size,
        stage='original A3/B3/C5 schedule from verified A3 and original trajectory',
        feature_cache_sha256=sha(Path(args.cache)/'manifest.json'),
        original_sources_sha256=sha(args.sources),
        evaluation_split='val_seen',
        episode_count=128,
        other_controls='original GraphNav-V1; no extra instance detector',
    )
    identity = output/'paired_protocol.json'
    if identity.exists():
        if json.loads(identity.read_text()) != protocol:
            raise ValueError('refuse to reuse output with a different protocol')
    else:
        identity.write_text(json.dumps(protocol, indent=2))

    summary = {}
    for label, fusion, evaluation_model in ARMS:
        arm = output/label
        previous = None
        for stage, stage_epochs in STAGES:
            train = arm/'stages'/stage
            last = train/'last.pt'
            complete = train/'complete.json'
            if not complete.exists():
                if train.exists() and not last.exists():
                    raise ValueError(f'cannot safely resume partial run without last.pt: {train}')
                command = [
                    sys.executable, '-m', 'scripts.train_graphnav_stage1',
                    '--cache', args.cache, '--sources', args.sources,
                    '--stage', stage, '--epochs', str(stage_epochs),
                    '--fusion-kind', fusion,
                    '--sbf-source', args.sbf_source, '--sbf-config', args.sbf_config,
                    '--data-root', args.data_root, '--output', str(train),
                    '--batch-size', str(args.batch_size), '--workers', str(args.workers),
                    '--seed', str(args.seed),
                ]
                if previous is not None:
                    command.extend(['--init', str(previous)])
                if last.exists():
                    command.extend(['--resume', str(last)])
                launch(command, arm/f'train_{stage}_console.log')
            if not complete.exists() or not (train/'best.pt').exists():
                raise RuntimeError(f'{label} stage {stage} did not complete')
            completed = json.loads(complete.read_text())
            if completed['epochs'] != stage_epochs or completed['smoke']:
                raise ValueError(f'{label} stage {stage} incomplete or smoke')
            previous = train/'best.pt'
        candidates = []
        for stage in ('B', 'C'):
            checkpoint = arm/'stages'/stage/'best.pt'
            selected = arm/f'seen128_{stage}_candidate'
            metrics_path = selected/'metrics.json'
            if not metrics_path.exists():
                if selected.exists():
                    raise ValueError(f'partial eval exists, archive and inspect first: {selected}')
                launch([
                    sys.executable, '-m', 'scripts.evaluate_closedloop',
                    '--model', evaluation_model,
                    '--sbf-source', args.sbf_source, '--config', args.sbf_config,
                    '--data-root', args.data_root, '--checkpoint', str(checkpoint),
                    '--split', 'val_seen', '--max-episodes', '128',
                    '--stop-threshold', '.3', '--head-execution', 'guarded-blend',
                    '--output', str(selected),
                ], arm/f'eval_{stage}_console.log')
            evaluated = json.loads(metrics_path.read_text())
            if evaluated['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError(f'checkpoint/evaluation SHA mismatch: {label} {stage}')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            dev_recall = float(state['history'][-1]['dev']['belief_R@1/20m'])
            metric = evaluated['metrics']
            candidates.append(((metric['SR'], metric['SPL'], dev_recall), stage, checkpoint, evaluated))
        _, selected_stage, best, evaluated = max(candidates, key=lambda item: item[0])
        summary[label] = dict(
            selected_stage=selected_stage, checkpoint_sha256=sha(best),
            fusion_kind=fusion, completed_epochs=dict(STAGES),
            metrics=evaluated['metrics'],
            episode_manifest_sha256=evaluated['episode_manifest_sha256'],
        )
    a = summary[ARMS[0][0]]
    b = summary[ARMS[1][0]]
    if a['episode_manifest_sha256'] != b['episode_manifest_sha256']:
        raise ValueError('paired arms have different episode manifests')
    delta = {key: b['metrics'][key] - a['metrics'][key]
             for key in ('SR','SPL','OSR','NE','nDTW','sDTW')
             if key in a['metrics'] and key in b['metrics']}
    final = dict(protocol=protocol, arms=summary, joint_minus_original=delta,
                 scope='single seed Seen128 only: not heldout proof or full-split result')
    (output/'core_pair_results.json').write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
