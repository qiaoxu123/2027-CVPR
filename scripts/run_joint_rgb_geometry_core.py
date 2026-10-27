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
from datetime import datetime, timezone
from graphnav.checkpointing import atomic_json

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


def candidate_dev_recall(state):
    # best.pt can contain later history when resuming old runs. Use its saved selection key.
    return float(state['best_key'][2])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache','sources','sbf-source','sbf-config','data-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--bootstrap-checkpoint')
    p.add_argument('--bert-cache')
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
    if sum(c['rows'] for c in manifest['chunks'])!=87512 or len(manifest['protocol']['episode_indices'])!=21878:
        raise ValueError('core pair requires all 87512 original source states')
    if not manifest.get('complete') or manifest.get('smoke_only'):
        raise ValueError('full original Stage1 expert-prefix cache required')

    protocol = dict(
        experiment='one_seed_paired_original_vs_joint_rgb_geometry',
        model='graphnav-stage1',
        seed=args.seed, epochs=dict(STAGES), batch_size=args.batch_size,
        stage='original A3/B3/C5 schedule from verified A3 and original trajectory',
        feature_cache_sha256=sha(Path(args.cache)/'manifest.json'),
        original_sources_sha256=sha(args.sources),
        sbf_config_sha256=sha(args.sbf_config),
        sbf_source_sha256={str(p.relative_to(args.sbf_source)):sha(p) for p in sorted(Path(args.sbf_source).glob('sbfnav/**/*.py'))},
        data_root=str(Path(args.data_root).resolve()),
        selection_split='val_seen128',
        final_splits=dict(val_seen=2470,val_unseen=2697),
        bootstrap_sha256=sha(args.bootstrap_checkpoint) if args.bootstrap_checkpoint else None,
        bert_cache_sha256=sha(args.bert_cache) if args.bert_cache else None,
        other_controls='original GraphNav-V1; no extra instance detector',
    )
    identity = output/'paired_protocol.json'
    if identity.exists():
        if json.loads(identity.read_text()) != protocol:
            raise ValueError('refuse to reuse output with a different protocol')
    else:
        identity.write_text(json.dumps(protocol, indent=2))

    if args.seed != 0:raise ValueError('core pair requires seed 0')
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
                for flag in ('bootstrap_checkpoint','bert_cache'):
                    if getattr(args,flag):command.extend(['--'+flag.replace('_','-'),getattr(args,flag)])
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
    # Complete BOTH arms before selecting/freezing either final checkpoint.
    for label, fusion, evaluation_model in ARMS:
        arm = output/label
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
            dev_recall = candidate_dev_recall(state)
            metric = evaluated['metrics']
            candidates.append(((metric['SR'], metric['SPL'], dev_recall), stage, checkpoint, evaluated))
        _, selected_stage, best, evaluated = max(candidates, key=lambda item: item[0])
        summary[label] = dict(
            selected_stage=selected_stage, checkpoint=str(best.resolve()), checkpoint_sha256=sha(best),
            fusion_kind=fusion, completed_epochs=dict(STAGES),
            metrics=evaluated['metrics'],
            episode_manifest_sha256=evaluated['episode_manifest_sha256'],
        )
    a = summary[ARMS[0][0]]
    b = summary[ARMS[1][0]]
    if a['episode_manifest_sha256'] != b['episode_manifest_sha256']:
        raise ValueError('paired arms have different episode manifests')
    frozen_path=output/'frozen_pair.json'
    frozen=dict(arms=summary,completed_epochs=dict(STAGES),all_stages_complete=True,
                joint_training_complete=True,visual_control='real',history_control='checkpoint',
                stop_threshold=.3,head_execution='guarded-blend',active_observation=False,
                sbf_config_sha256=protocol['sbf_config_sha256'],sbf_source_sha256=protocol['sbf_source_sha256'],
                data_root=protocol['data_root'],protocol_sha256=sha(identity),selection='Seen128 SR then SPL then internal dev recall; no full split selection')
    if frozen_path.exists():
        old=json.loads(frozen_path.read_text())
        if {k:v for k,v in old.items() if k!='frozen_at_utc'}!=frozen:
            raise ValueError('refuse to change previously frozen pair')
    else:
        frozen['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
        atomic_json(frozen,frozen_path)
    for label,fusion,evaluation_model in ARMS:
        selected=summary[label];arm=output/label
        manifest=dict(frozen,**{k:v for k,v in selected.items() if k not in frozen},model=evaluation_model)
        arm_freeze=arm/'frozen_final_candidate.json'
        if not arm_freeze.exists():atomic_json(manifest,arm_freeze)
        if json.loads(arm_freeze.read_text())['checkpoint_sha256']!=selected['checkpoint_sha256']:
            raise ValueError('frozen arm checkpoint differs')
    # No full evaluation or Unseen access occurs until the pair is frozen above.
    for split,count in (('val_seen',2470),('val_unseen',2697)):
        for label,fusion,evaluation_model in ARMS:
            arm=output/label;selected=summary[label];destination=arm/('full_'+split)
            report=destination/'metrics.json'
            if not report.exists():
                if destination.exists():raise ValueError('partial full evaluation; inspect/archive before retry: '+str(destination))
                launch([sys.executable,'-m','scripts.evaluate_closedloop','--model',evaluation_model,
                        '--sbf-source',args.sbf_source,'--config',args.sbf_config,'--data-root',args.data_root,
                        '--checkpoint',selected['checkpoint'],'--split',split,
                        '--frozen-final-candidate',str(arm/'frozen_final_candidate.json'),
                        '--output',str(destination)],arm/('eval_full_'+split+'.log'))
            result=json.loads(report.read_text())
            if result['metrics']['episodes']!=count or result['checkpoint_sha256']!=selected['checkpoint_sha256']:
                raise ValueError('full evaluation count/checkpoint mismatch')
    from scripts.summarize_core_pair import summarize
    summarize(output)


if __name__ == '__main__':
    main()
