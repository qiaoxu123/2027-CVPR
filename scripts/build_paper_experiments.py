"""Offline, evidence-first PolyBelief paper tables. Never trains or runs inference.

Run from this worktree with the Baselines reproduction Python:
  python -m scripts.build_paper_experiments
Only docs/PAPER_EXPERIMENT_TODO.md and reports/paper_experiments are written.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
WORKSPACE = ROOT.parent
OUT = HERE / 'reports/paper_experiments'
MAIN = ROOT / 'GraphNav'
ORIGINAL = WORKSPACE / 'DATA/citynav/extracted/data/citynav'
REVISED = WORKSPACE / 'DATA/hett/release/refined_citynav/processed_citynav'
SPLITS = ('val_seen', 'val_unseen', 'test_unseen')
KEYS = ('NE', 'SR', 'OSR', 'SPL')
EXPECTED = dict(zip(SPLITS, (2470, 2697, 5281)))
METHODS = ('Random', 'Seq2Seq+GSM', 'CMA+GSM', 'MGP', 'AerialVLN+GSM',
           'FlightGPT', 'GeoNav', 'HTNav', 'HETT', 'SBFNav', 'PolyBelief')
METHOD_SOURCES = {
    'Random': ('https://arxiv.org/html/2609.05841v1', 'https://github.com/water-cookie/citynav'),
    'Seq2Seq+GSM': ('https://arxiv.org/html/2406.14240v3', 'https://github.com/water-cookie/citynav'),
    'CMA+GSM': ('https://arxiv.org/html/2406.14240v3', 'https://github.com/water-cookie/citynav'),
    'MGP': ('https://arxiv.org/html/2406.14240v3', 'https://github.com/water-cookie/citynav'),
    'AerialVLN+GSM': ('https://arxiv.org/html/2406.14240v3', 'https://github.com/AirVLN/AirVLN'),
    'FlightGPT': ('https://arxiv.org/html/2505.12835v1', 'https://github.com/Uavln/FlightGPT'),
    'GeoNav': ('https://arxiv.org/html/2504.09587', 'https://github.com/Xhtshr/geonav-official'),
    'HTNav': ('https://arxiv.org/html/2604.08883', 'https://github.com/tyrofanfan/HTNav'),
    'HETT': ('https://arxiv.org/html/2512.14222', 'https://github.com/crotonyl/HETT'),
    'SBFNav': ('https://arxiv.org/html/2609.05841v1', 'No author code URL in released v1; local SBFNav is independent reimplementation'),
    'PolyBelief': ('Local frozen full-model report; no external paper-reported row', 'Current GraphNav research branch'),
    'Human': ('https://arxiv.org/html/2406.14240v3', 'Reference upper bound; not a reproduced policy'),
}
SHA_CACHE = {}


def load(p):
    return json.loads(Path(p).read_text())


def sha(p):
    p = Path(p)
    if not p.is_file():
        return ''
    key = (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
    if key not in SHA_CACHE:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(b)
        SHA_CACHE[key] = h.hexdigest()
    return SHA_CACHE[key]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csvout(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def git(p, *args):
    r = subprocess.run(['git', '-C', str(p), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def ident(r):
    return (f"{r['area']}_block_{r['block']}", int(r['object_ids'][0]), int(r['ann_ids'][0]))


def eid(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return (str(value[0]), int(value[1]), int(value[2]))


def rawmap(p):
    rows = load(p)
    assert all(len(r['object_ids']) == len(r['ann_ids']) == 1 for r in rows), p
    result = {ident(r): r for r in rows}
    assert len(result) == len(rows), f'duplicate IDs: {p}'
    return result


def path_metrics(poses, target):
    xy = np.asarray(poses, dtype=np.float64)[:, :2]
    goal = np.asarray(target, dtype=np.float64)[:2]
    assert len(xy) and np.isfinite(xy).all() and np.isfinite(goal).all()
    dist = np.linalg.norm(xy - goal, axis=1)
    length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    optimal = dist[0]
    success = float(dist[-1] <= 20.)
    return dict(NE=float(dist[-1]), SR=success, OSR=float(dist.min() <= 20.),
                SPL=success * float(optimal / max(length, optimal, 1e-12)))


def aggregate(rows):
    assert rows
    vals = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    assert 0 <= vals['SPL'] <= vals['SR'] + 1e-10 <= vals['OSR'] + 1e-10
    return dict(episodes=len(rows), **{k: vals[k] * (1 if k == 'NE' else 100) for k in KEYS})


def mdtable(rows, cols):
    def cell(v):
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['|' + '|'.join(cols) + '|', '|' + '|'.join(['---'] * len(cols)) + '|'] +
                     ['|' + '|'.join(cell(r.get(k, '')) for k in cols) + '|' for r in rows]) + '\n'


def inventory():
    repos = [MAIN] + [p.parent for p in (ROOT / 'Baselines').rglob('.git')]
    snapshots = []
    for p in dict.fromkeys(repos):
        snapshots.append(dict(path=str(p), commit=git(p, 'rev-parse', 'HEAD'),
                              branches=git(p, 'branch', '-a'),
                              worktrees=git(p, 'worktree', 'list', '--porcelain'),
                              status=git(p, 'status', '--short')))
    save('repository_inventory.json', snapshots)
    roots = [MAIN, ROOT / 'Ablations/GraphNav', ROOT / 'Ablations/HETT', ROOT / 'Baselines']
    rows, checkpoints = [], []
    skip = {'.git', 'data', 'DATA', 'data_refined', 'external', 'site-packages',
            '.reproduction-env', '.venv', 'venv', '__pycache__', 'node_modules',
            'public_inputs', 'source_snapshot', 'source', 'sources', 'heatmaps', 'images'}
    skip.update({'cache', 'cache_fp16', 'feature_cache', 'token_cache', 'frames', 'observed_train'})
    for root in roots:
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in skip and Path(folder) / d != OUT]
            for f in files:
                p = Path(folder) / f
                if p.is_symlink() or not p.is_file():
                    continue
                if p.suffix in ('.pt', '.pth', '.safetensors') and any(
                        word in f.lower() for word in ('best', 'last', 'epoch', 'checkpoint', 'model',
                                                       'mgp', 'seq2seq', 'cma', 'gibson', 'resnet')):
                    checkpoints.append(dict(path=str(p), bytes=p.stat().st_size,
                                            sha256='see selected_checkpoint_inventory.csv; not all historical weights hashed'))
                if (f.endswith(('metrics.json', 'summary.json', 'comparison.csv')) or
                        f in ('predictions.jsonl', 'RESULTS.md', 'REPORT.md', 'complete.json', 'status.json')):
                    rows.append(dict(path=str(p), bytes=p.stat().st_size,
                                     category='per_episode' if f == 'predictions.jsonl' else 'report_or_status',
                                     existence_verified=True))
    csvout('asset_inventory.csv', rows)
    csvout('checkpoint_inventory.csv', checkpoints)
    save('inventory_scope.json', dict(roots=list(map(str, roots)), excluded_directories=sorted(skip),
                                    policy='Worktree reports/runs scanned; symlink assets indexed by authoritative targets, no DATA copies. Branch lists include unmounted branches; no checkout/mutation of historical branches.'))


def datasets():
    audits, originals, revised = [], {}, {}
    for split in ('train_seen', *SPLITS):
        op, rp = ORIGINAL / f'citynav_{split}.json', REVISED / f'citynav_{split}.json'
        a, b = rawmap(op), rawmap(rp)
        originals[split], revised[split] = a, b
        removed, added = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
        common = a.keys() & b.keys()
        audits.append(dict(split=split, original=len(a), revised=len(b), removed=len(removed), added=len(added),
                           instruction_changed=sum(a[k]['descriptions'] != b[k]['descriptions'] for k in common),
                           trajectory_changed=sum(a[k]['trajectory'] != b[k]['trajectory'] for k in common),
                           target_changed=sum(a[k]['target_positions'] != b[k]['target_positions'] for k in common),
                           maps=len({k[0] for k in b}), original_path=str(op), revised_path=str(rp),
                           original_sha256=sha(op), revised_sha256=sha(rp), removed_episode_ids=removed,
                           added_episode_ids=added,
                           removal_reason='upstream revised release omits ID; per-ID author reason unavailable; do not infer from metrics'))
        if split in EXPECTED:
            assert len(b) == EXPECTED[split]
    save('dataset_version_audit.json', audits)
    metadata_paths = [ORIGINAL.parent / 'cityrefer/processed_descriptions.json',
                      REVISED.parent / 'cityrefer/processed_descriptions.json']
    flattened = []
    for path in metadata_paths:
        flattened.append({(m, int(o), i): v for m, objs in load(path).items()
                          for o, descriptions in objs.items() for i, v in enumerate(descriptions)})
    ma, mb = flattened
    common_metadata = ma.keys() & mb.keys()
    save('landmark_annotation_audit.json', dict(
        files=[dict(path=str(p), sha256=sha(p)) for p in metadata_paths],
        original_entries=len(ma), revised_entries=len(mb),
        changed_entries=sum(ma[k] != mb[k] for k in common_metadata),
        changed_landmark_entries=sum(ma[k].get('landmarks') != mb[k].get('landmarks') for k in common_metadata),
        note='Processed target/landmark/surroundings metadata changes are distinct from raw navigation instruction text.'))
    manifests = {}
    for version, splits in [('original', originals), ('revised', revised)]:
        allrows = splits['test_unseen']
        membership, sources = {}, []
        for diff in ('easy', 'medium', 'hard'):
            path = ORIGINAL / f'citynav_test_unseen_{diff}.json'
            subset = rawmap(path)
            sources.append(dict(difficulty=diff, path=str(path), sha256=sha(path), episodes=len(subset),
                                published_distance_min=min(r['dist_start_to_target'] for r in subset.values()),
                                published_distance_max=max(r['dist_start_to_target'] for r in subset.values())))
            for k in subset:
                assert k not in membership, f'difficulty overlap: {k}'
                membership[k] = diff
        assert set(membership) == set(originals['test_unseen'])
        selected = {k: membership[k] for k in allrows}
        # Existing locally generated revised subsets must agree with original membership intersection.
        for diff in ('easy', 'medium', 'hard'):
            p = WORKSPACE / 'DATA/sbfnav-data/processed_citynav' / f'citynav_test_unseen_{diff}.json'
            assert set(rawmap(p)) == {k for k in revised['test_unseen'] if membership[k] == diff}
        manifests[version] = dict(episodes=len(selected), disjoint=True, exhaustive=True,
                                 sources=sources, counts={d: list(selected.values()).count(d) for d in ('easy', 'medium', 'hard')},
                                 membership=[dict(episode_id=k, difficulty=d) for k, d in sorted(selected.items())])
    save('difficulty_episode_manifest.json', dict(rule='Official released CityNav subset membership by (map_name, object_id, ann_id); revised = original membership intersect revised IDs. No new distance thresholds.',
                                                versions=manifests))
    return originals, revised, {v: {tuple(r['episode_id']): r['difficulty'] for r in x['membership']} for v, x in manifests.items()}, audits


def paper_tables():
    soup = BeautifulSoup((OUT / 'sources/sbfnav.html').read_text(), 'html.parser')
    tables = soup.find_all('figure', class_='ltx_table')
    output = []
    for number, splits in [(1, SPLITS), (2, ('easy', 'medium', 'hard'))]:
        rows = []
        for tr in tables[number - 1].find_all('tr'):
            cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
            if len(cells) != 13:
                continue
            try:
                vals = list(map(float, cells[1:]))
            except ValueError:
                continue
            label = cells[0]
            name = label.split(' (')[0].replace('*', '')
            for i, split in enumerate(splits):
                row = dict(method=name, paper_label=label, split='test_unseen' if number == 2 else split,
                           difficulty=split if number == 2 else 'all',
                           dataset='revised_paper_unspecified_file_hash' if '*' in label else 'original_CityNav',
                           **dict(zip(KEYS, vals[4*i:4*i+4])), episodes='',
                           result_type='paper_reported', status='Done',
                           source_url='https://arxiv.org/html/2609.05841v1', source_table=number,
                           source_sha256=sha(OUT / 'sources/sbfnav.html'),
                           checkpoint_sha256='', source_commit='',
                           note='Human is reference upper bound only' if name == 'Human' else
                           'SBFNav Table 2 Seq2Seq/CMA do not have +GSM; do not relabel' if number == 2 and name in ('Seq2Seq', 'CMA') else
                           'Reported by SBFNav; not independently rerun; SBFNav released supplement unavailable')
                row['original_method_paper_url'], row['official_code_url'] = METHOD_SOURCES.get(name, METHOD_SOURCES.get(name+'+GSM', ('','')))
                rows.append(row)
        if number == 1:
            for split in SPLITS:
                rows.append(dict(method='PolyBelief', split=split, difficulty='all', dataset='revised',
                                 result_type='not_in_SBFNav_paper', status='Pending',
                                 note='New method; actual local results are in table1_ours_unified.csv'))
        csvout('table1_paper_reported.csv' if number == 1 else 'table2_paper_reported.csv', rows)
        output.append(rows)
    csvout('method_sources.csv', [dict(method=m, method_paper=p, official_code=c,
                                      note='GSM adaptations refer to CityNav; parent Seq2Seq/CMA encoders additionally documented at https://github.com/jacobkrantz/VLN-CE and https://arxiv.org/abs/2004.02857. Parent weights do not prove CityNav+GSM replication.' if '+GSM' in m else '')
                                  for m,(p,c) in METHOD_SOURCES.items()])
    return output


def command_from_args(args):
    cmd = [sys.executable, '-m', 'scripts.evaluate_closedloop']
    for k, v in args.items():
        if v is None or v is False:
            continue
        cmd.append('--' + k.replace('_', '-'))
        if v is not True:
            cmd.append(str(v))
    return shlex.join(cmd)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory()
    original, revised, memberships, dataaudit = datasets()
    papers, paperdifficulty = paper_tables()
    registry, unified, local_other, difficulty, selected_weights, validations = [], [], [], [], [], []
    episode_metrics_export = []

    def register(method, variant, split, source, metrics, *, checkpoint='', checkpoint_sha='', commit='',
                 command='', predictions='', log='', protocol='', dataset='revised', status='Done',
                 comparable=False, missing='', kind='local_evaluation'):
        p = Path(source)
        cp = Path(checkpoint) if checkpoint else None
        cp_sha = sha(cp) if cp and cp.is_file() else checkpoint_sha
        if checkpoint_sha and cp and cp.is_file():
            assert cp_sha == checkpoint_sha, f'checkpoint mismatch {cp}'
        if cp:
            selected_weights.append(dict(method=method, variant=variant, checkpoint=str(cp),
                                         exists=cp.is_file(), bytes=cp.stat().st_size if cp.is_file() else '',
                                         checkpoint_sha256=cp_sha))
        row = dict(method=method, variant=variant, split=split, dataset=dataset, difficulty='all',
                   **metrics, checkpoint=str(cp or ''), checkpoint_sha256=cp_sha,
                   source_commit=commit, protocol=protocol, result_path=str(p), result_sha256=sha(p),
                   predictions=str(predictions), predictions_exist=bool(predictions) and Path(predictions).is_file(),
                   log=str(log), log_exists=bool(log) and Path(log).is_file(), command=command,
                   status=status, comparable=comparable, result_type=kind, missing=missing,
                   data_sha256=sha(REVISED / f'citynav_{split}.json') if dataset == 'revised' else
                   sha(ORIGINAL / f'citynav_{split}.json') if split in SPLITS and dataset == 'original' else '')
        row['experiment_id'] = hashlib.sha256(json.dumps([method, variant, split, dataset]).encode()).hexdigest()[:16]
        if status == 'Done':
            assert p.is_file() and metrics.get('episodes'), f'Done without real result: {row}'
        registry.append(row)
        return row

    def record_episode_metrics(base, values):
        for k, m in values.items():
            episode_metrics_export.append(dict(experiment_id=base['experiment_id'], method=base['method'],
                                               variant=base['variant'], split=base['split'], dataset=base['dataset'],
                                               map_name=k[0], object_id=k[1], ann_id=k[2],
                                               **{key: m[key] * (1 if key == 'NE' else 100) for key in KEYS}))

    def diffrows(base, episode_metrics, version):
        assert set(episode_metrics) == set(memberships[version]), (base['method'], 'incomplete difficulty IDs')
        for d in ('easy', 'medium', 'hard'):
            r = dict(base, difficulty=d, **aggregate([m for k, m in episode_metrics.items() if memberships[version][k] == d]))
            difficulty.append(r)

    for split, folder in [('val_seen', 'full_seen'), ('val_unseen', 'full_unseen'), ('test_unseen', 'full_test_unseen')]:
        path = MAIN / 'runs/graphnav_v1_stage1' / folder
        m = load(path / 'metrics.json')
        em = {}
        with (path / 'predictions.jsonl').open() as f:
            for line in f:
                pred = json.loads(line)
                k = eid(pred['episode_id'])
                assert k not in em
                em[k] = path_metrics(pred['poses'], revised[split][k]['target_positions'][0])
                for metric in KEYS:
                    assert np.isclose(em[k][metric], pred['metrics'][metric], rtol=1e-9, atol=1e-8)
        assert set(em) == set(revised[split]) and len(em) == EXPECTED[split]
        metrics = aggregate(list(em.values()))
        for key in KEYS:
            assert np.isclose(metrics[key], m['metrics'][key] * (1 if key == 'NE' else 100), atol=1e-9)
        canonical = next(r for r in csv.DictReader((MAIN / 'reports/graphnav_v1_stage1/GRAPHNAV_V1_METRICS.csv').open())
                         if r['split'].lower().replace('-', '_') == split)
        assert all(np.isclose(float(canonical[k]), m['metrics'][k], atol=1e-12) for k in KEYS)
        row = register('PolyBelief', 'frozen full model; C epoch1; 28x28', split, path / 'metrics.json', metrics,
                       checkpoint=MAIN / 'runs/graphnav_v1_stage1/best.pt', checkpoint_sha=m['checkpoint_sha256'],
                       commit=m['graphnav_commit'], command=command_from_args(m['args']),
                       predictions=path / 'predictions.jsonl', log=path.parent / (folder + '.log'),
                       protocol='benchmark_2d;20m;horizon20;guarded-blend;stop0.3;arrival1mm;observed-only', comparable=True)
        unified.append(row)
        record_episode_metrics(row, em)
        validations.append(dict(method='PolyBelief', split=split, exact_id_coverage=True,
                                per_episode_trace_metrics_match=True, source_aggregate_match=True, episodes=len(em)))
        save(f'polybelief_{split}_metrics.json', dict(metrics=metrics, source=str(path / 'metrics.json'),
                                                   source_sha256=sha(path / 'metrics.json'), status='Done'))
        if split == 'test_unseen':
            diffrows(row, em, 'revised')

    # SBFNav: the local project is an independent paper-based reimplementation, never official weights.
    sbf = ROOT / 'Baselines/SBFNav'
    sbfrun = sbf / 'runs/sbfnav_main_e20_bs16_s0_20260929'
    for split, epoch in [('val_seen', 12), ('val_unseen', 12), ('test_unseen', 12), ('val_seen', 14), ('val_unseen', 14)]:
        p = sbfrun / (f'eval_epoch{epoch}' if epoch == 12 else 'evaluation') / split
        m = load(p / 'metrics.json')
        cp = sbfrun / 'training/checkpoints' / Path(m['checkpoint']).name
        em = {}
        with (p / 'predictions.jsonl').open() as f:
            for line in f:
                pred = json.loads(line)
                k = eid(pred['episode_id'])
                assert k not in em
                em[k] = path_metrics(pred['trajectory_xyzyaw'], revised[split][k]['target_positions'][0])
        assert set(em) == set(revised[split])
        metrics = aggregate(list(em.values()))
        assert all(np.isclose(metrics[k], m['benchmark_2d'][k.lower()], atol=1e-8) for k in KEYS)
        row = register('SBFNav', f'paper-based reimplementation epoch{epoch}', split, p / 'metrics.json', metrics,
                       checkpoint=cp, checkpoint_sha=m['checkpoint_sha256'], commit=m['git_commit'], command=m['command'],
                       predictions=p / 'predictions.jsonl', protocol='benchmark_2d;20m;native bounded controller;observed-only',
                       missing='Historical selection used Val-Unseen; cannot satisfy new Seen-only selection. Epoch12 has all three splits; epoch14 validation is a separate configuration. Run log missing from this directory.',
                       comparable=False)
        local_other.append(row)
        record_episode_metrics(row, em)
        # Same revised inputs and common XY evaluator; native controllers and old selection remain labeled.
        unified.append(dict(row, result_type='common_metric_replay_historical_selection', comparable=False))
        if split == 'test_unseen':
            diffrows(row, em, 'revised')
            for d, x in m['difficulty_breakdown'].items():
                r = next(r for r in difficulty if r['method'] == 'SBFNav' and r['difficulty'] == d)
                assert all(np.isclose(r[k], x['benchmark_2d'][k.lower()], atol=1e-8) for k in KEYS)

    # Frozen local adaptations: replay recorded trajectories, preserving input-data identity.
    for method in ('HTNav', 'AerialVLN-GSM'):
        for run in ('reproduction_20261012', 'alignment_20261013'):
            base = ROOT / 'Baselines' / method / 'runs' / run / 'full'
            frozen = load(base / 'frozen_candidate.json') if (base / 'frozen_candidate.json').exists() else {}
            for split in SPLITS:
                p = base / 'evaluation' / (split + '.json')
                if not p.is_file():
                    continue
                data = load(p)
                version = 'original' if method == 'AerialVLN-GSM' else 'revised'
                gt = original[split] if version == 'original' else revised[split]
                traces = {eid(k): v for k, v in data['trajectories'].items()}
                assert set(traces) == set(gt) == {eid(k) for k in data['episode_ids']}
                em = {k: path_metrics(v, gt[k]['target_positions'][0]) for k, v in traces.items()}
                metrics = aggregate(list(em.values()))
                native = data['metrics']
                keys = dict(NE='mean_final_pos_to_goal_dist', SR='success_rate_final_pos_to_goal',
                            OSR='success_rate_oracle_pos_to_goal', SPL='success_rate_weighted_by_path_length')
                native_vals = {k: native.get(k, native.get(keys[k])) * (1 if k == 'NE' else 100) for k in KEYS}
                for k in ('NE', 'SR', 'OSR'):
                    assert np.isclose(metrics[k], native_vals[k], atol=1e-8), (method, split, k)
                commit = frozen.get('source_commit', '')
                source = ROOT / f'Baselines/comparison/runs/{run}/{method}_source.json'
                if source.exists() and not commit:
                    sd = load(source)
                    commit = sd.get('commit', sd.get('head', sd.get('git_commit', '')))
                row = register(method, ('local reimplementation' if method == 'AerialVLN-GSM' else 'repaired released code') + ';' + run,
                               split, p, metrics, checkpoint=base / 'best.pt', checkpoint_sha=frozen.get('checkpoint_sha256', ''),
                               commit=commit, predictions=p, dataset=version,
                               log=ROOT / f'Baselines/comparison/runs/{run}/{method}_{split}.log',
                               command=shlex.join([sys.executable, str(ROOT / 'Baselines' / method / 'reproduce.py'), '--phase', 'eval',
                                                   '--checkpoint', str(base / 'best.pt'), '--output', str(base), '--split', split]),
                               protocol='benchmark_2d;20m;native actions100;local adaptation;preserved RGB/depth proxy and GSM inputs',
                               missing='Not strict paper replication; see ALIGNMENT_AUDIT_20261013.md; native action/observation permissions differ; per-stage logs in comparison queue', comparable=False)
                local_other.append(dict(row, **native_vals, result_type='native_local_adaptation_metrics',
                                        protocol=row['protocol'] + ';native optimal-path definition retained'))
                record_episode_metrics(row, em)
                if version == 'revised':
                    unified.append(dict(row, result_type='common_metric_replay_local_adaptation', comparable=False))
                validations.append(dict(method=method, variant=run, split=split, episodes=len(em),
                                        native_NE_SR_OSR_match=True, native_SPL_percent=native_vals['SPL'],
                                        common_SPL_percent=metrics['SPL'],
                                        SPL_difference_pp=metrics['SPL']-native_vals['SPL'],
                                        note='HTNav native optimal length is generated discrete shortest teacher path; common evaluator uses Euclidean start-goal length. Both values retained.'))
                if split == 'test_unseen':
                    diffrows(row, em, version)

    # Historical baseline aggregates remain visible, never promoted to unified reproduction.
    old = list(csv.DictReader((ROOT / 'Baselines/comparison/local_results.csv').open()))
    for oldrow in old:
        if oldrow['method'] not in ('Random', 'Seq2Seq+GSM', 'CMA+GSM', 'MGP', 'HETT', 'GeoNav'):
            continue
        p = ROOT / oldrow['source']
        n = int(oldrow['episodes']) if oldrow['episodes'] else ''
        partial = oldrow['method'] == 'HETT'
        cfg = load(p).get('config', {}) if p.suffix == '.json' else {}
        cp = ROOT / 'Baselines/CityNav-GSM' / cfg['checkpoint'] if cfg.get('checkpoint') else ''
        row = register(oldrow['method'], oldrow['variant'], oldrow['split'], p,
                       dict(episodes=n, **{k: float(oldrow[k]) for k in KEYS}), checkpoint=cp,
                       dataset=oldrow['dataset'], kind=oldrow['evidence'], status='Blocked' if partial else 'Done',
                       protocol='historical native protocol; not unified input/controller',
                       missing='Document only; no episode trajectories to verify denominator' if partial else
                       'Historical aggregate: checkpoint/source commit/log/episode chain incomplete' if oldrow['method'] != 'GeoNav' else
                       'DeepSeek substitution for GPT-4o; original data; fixed-altitude/cache inputs; no strict original-model reproduction')
        local_other.append(row)
    geo = ROOT / 'Baselines/GeoNav/results/geonav_deepseek_flash_full_corrected/geonav_origin/test_unseen/episode_metrics_test_unseen.jsonl'
    em = {}
    for line in geo.open():
        r = json.loads(line)
        k = eid(r['episode_id'])
        assert k not in em
        em[k] = {key: r[key] for key in KEYS}
    assert set(em) == set(original['test_unseen'])
    georow = next(r for r in local_other if r['method'] == 'GeoNav')
    assert all(np.isclose(aggregate(list(em.values()))[k], georow[k], atol=1e-8) for k in KEYS)
    georow['predictions'] = str(geo)
    georow['predictions_exist'] = True
    diffrows(georow, em, 'original')
    # Table2 aggregates for GSM variants already exist; preserve provenance limits explicitly.
    for r in local_other:
        if r['method'] not in ('Random', 'Seq2Seq+GSM', 'CMA+GSM', 'MGP') or r['split'] != 'test_unseen':
            continue
        d = load(r['result_path'])
        for level, table in d.get('table2', {}).items():
            if level not in ('easy', 'medium', 'hard') or 'ours' not in table:
                continue
            vals = table['ours']
            difficulty.append(dict(r, difficulty=level, episodes=vals['n_episodes'],
                                   **{k: vals[k] * (1 if k == 'NE' else 100) for k in KEYS},
                                   result_type='historical_difficulty_aggregate_not_episode_reverified'))

    # Missing rows for all targets/splits, with explicit reasons; no historical values leak into unified table.
    reasons = {
        'Random': 'Historical native policies exist; no saved full-split trajectories with complete provenance for this frozen observed-only protocol.',
        'Seq2Seq+GSM': 'Historical aggregates exist; local checkpoint references broken; exact source/log/trajectory chain incomplete.',
        'CMA+GSM': 'Historical aggregates exist; local checkpoint references broken; exact source/log/trajectory chain incomplete.',
        'MGP': 'Official checkpoint recovered in isolated paper_baseline_assets directory (SHA in mgp_recovered_asset.json); not proven identical to missing historical weight. Saved per-episode historical traces/source chain unavailable for common-evaluator replay; native aggregates remain separate.',
        'AerialVLN+GSM': 'Local original-data adaptation completed; different data/action/imagery protocol; aligned full training incomplete.',
        'FlightGPT': 'Official public weight identified (~16.6GB); local four shards absent; native evaluator adaptation and model-load smoke not completed.',
        'GeoNav': 'Original-data DeepSeek Test completed; GPT-4o access/configuration not verified; Seen/Unseen absent. Paid service cannot be assumed.',
        'HTNav': 'Two local repaired configurations have validation results; official weights not found in released README; local adaptation differs from official method.',
        'HETT': 'Official checkpoint archive exists and SHA verified; HETT/main is a modified trajectory-belief research checkout, not a verified checkpoint-compatible upstream policy. Isolated --help passed after dependency fix; hardcoded paths/output and missing episode export require an upstream-compatible isolated evaluator. Historical document only.',
        'SBFNav': 'Independent reimplementation has trajectories; validation epoch14 vs Test epoch12; old selection used Val-Unseen, incompatible with new selection constraint.'}
    for method in METHODS:
        for split in SPLITS:
            if any(r['method'] == method and r['split'] == split for r in unified):
                continue
            unified.append(dict(method=method, split=split, dataset='revised', episodes=0,
                                NE='', SR='', OSR='', SPL='', status='Blocked', result_type='missing_unified',
                                comparable=False, missing=reasons[method], result_path='', checkpoint_sha256='',
                                source_commit='', protocol='PolyBelief frozen observed-only benchmark_2d target protocol',
                                data_sha256=sha(REVISED / f'citynav_{split}.json')))

    # Existing ablations retain raw source identity but publish concrete module names.
    abl = []
    def ablation(path, label):
        m = load(path)
        metrics = m.get('metrics')
        if not metrics or not all(k in metrics for k in KEYS):
            return
        row = register('PolyBelief', label, m.get('split', 'val_seen'), path,
                       dict(episodes=metrics['episodes'], **{k: metrics[k] * (1 if k == 'NE' else 100) for k in KEYS}),
                       checkpoint=m.get('args', {}).get('checkpoint', ''),
                       checkpoint_sha=m.get('checkpoint_sha256', ''), commit=m.get('graphnav_commit', ''),
                       command=command_from_args(m['args']) if 'args' in m else '',
                       predictions=path.parent / 'predictions.jsonl',
                       log=path.parent.parent / (path.parent.name + '.log'),
                       protocol=json.dumps(m.get('protocol', {}), ensure_ascii=False),
                       missing='Diagnostic/subset/alternate configuration; not formal main row')
        row['grid_size'] = m.get('args', {}).get('stage1_grid_size') or 28
        row['residual_checkpoint_sha256'] = m.get('calibrated_residual_sha256', '')
        row['residual_checkpoint'] = m.get('args', {}).get('stage1_residual_checkpoint') or ''
        if row['residual_checkpoint']:
            assert sha(row['residual_checkpoint']) == row['residual_checkpoint_sha256']
        row['episode_manifest_sha256'] = m.get('episode_manifest_sha256', '')
        row['inference_controls'] = json.dumps(m.get('stage1_inference_controls', {}))
        row['evaluator_sha256'] = m.get('evaluator_sha256', '')
        if '_full_unseen_' in path.parent.name:
            row['log'] = str(HERE / 'visual_residual_unseen_validation' / (path.parent.name.split('_full_unseen_')[-1] + '.log'))
            row['log_exists'] = Path(row['log']).is_file()
        abl.append(row)
    for path in sorted((HERE / 'visual_evaluations').glob('*/metrics.json')):
        if '_full_' in str(path):
            ablation(path, path.parent.name.replace('calibrated_residual', 'Calibrated Visual Residual').replace('absolute_residual', 'Non-centered Visual Residual'))
    for path in sorted((HERE / 'full_resolution_384').glob('*/metrics.json')):
        ablation(path, '384x384 goal grid; original frozen visual residual')
    for name, label in [('selected_seen128', 'Current RGB + causal history; frozen full model'),
                        ('current_only_seen128', 'Current RGB only; frozen full model'),
                        ('no_belief_seen128', 'Geometry-only belief; frozen full model'),
                        ('zero_rgb_seen128', 'Zero RGB; frozen full model'), ('shuffle_rgb_seen128', 'Shuffle RGB; frozen full model')]:
        ablation(MAIN / 'runs/graphnav_v1_stage1' / name / 'metrics.json', label)
    stoproot = ROOT / 'Ablations/GraphNav/stop-arrival-guard-20261016/stop_guard_results'
    for path in sorted(stoproot.glob('*/metrics.json')):
        ablation(path, 'Stop Head / Arrival Guard: ' + path.parent.name)
    # Historical controller comparisons are not frozen-model controlled contrasts.
    planning = MAIN / 'reports/trajectory_seed0_20261010/planning_controls.csv'
    for r in csv.DictReader(planning.open()):
        abl.append(dict(method='PolyBelief', variant='Historical controller: ' + r['control'], split='val_seen',
                        episodes=128, **{k: float(r[k]) * (1 if k == 'NE' else 100) for k in KEYS},
                        result_path=str(planning), status='Done', result_type='historical_controller_subset',
                        missing='Different earlier trajectory head; does not establish final frozen full-model controlled comparison',
                        checkpoint_sha256='', source_commit='', protocol='Seen128 earlier controller experiment'))
    csvout('ablation_results.csv', abl)
    csvout('common_episode_metrics.csv', episode_metrics_export)
    csvout('table1_ours_unified.csv', unified)
    csvout('table1_local_other_protocols.csv', local_other)
    csvout('table2_difficulty_comparison.csv', difficulty)
    csvout('experiment_registry.csv', registry + [r for r in unified if r['status'] != 'Done'])
    for name in ('mgp_recovered_asset.json', 'hett_asset_preflight.json'):
        if (OUT / name).exists():
            a = load(OUT / name)
            selected_weights.append(dict(method='MGP' if name.startswith('mgp') else 'HETT',
                                         variant='official asset; not newly evaluated', checkpoint=a.get('asset', a.get('checkpoint')),
                                         exists=True, bytes=a['bytes'], checkpoint_sha256=a['sha256']))
    csvout('selected_checkpoint_inventory.csv', list({str(r): r for r in selected_weights}.values()))
    save('verification.json', dict(checks=validations, all_passed=True, inference_runs_started=0,
                                   training_runs_started=0, official_weights_untouched=True))
    return dict(unified=unified, local=local_other, difficulty=difficulty, ablations=abl,
                dataaudit=dataaudit, reasons=reasons, paper=papers, paperdifficulty=paperdifficulty)


if __name__ == '__main__':
    result = build()
    # Narrative report generation lives in a separate small, inspectable module.
    from scripts.paper_experiment_report import write_reports
    write_reports(result)
    save('build_provenance.json', dict(command=shlex.join([sys.executable, '-m', 'scripts.build_paper_experiments']),
                                     cwd=str(HERE), commit=git(HERE, 'rev-parse', 'HEAD'),
                                     branch=git(HERE, 'branch', '--show-current'), timestamp=time.time(),
                                     script_sha256=sha(__file__), python=sys.version,
                                     numpy=np.__version__, metrics_units='NE meters; SR/OSR/SPL percent',
                                     source_sha256={p:sha(HERE/p) for p in ('scripts/build_paper_experiments.py', 'scripts/paper_experiment_report.py', 'scripts/count_polybelief_parameters.py', 'scripts/recover_paper_baseline_asset.py', 'tests/test_paper_experiments.py')},
                                     no_inference=True, no_training=True))
    print(json.dumps({k: len(v) for k, v in result.items() if isinstance(v, list)}, ensure_ascii=False))
