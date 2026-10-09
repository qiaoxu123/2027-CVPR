"""Executed audit of the matched G0/G1 heads; no model fitting or target input."""
import json
from pathlib import Path
import torch
from graphnav.checkpointing import file_sha256
from graphnav.trajectory_data import FEATURE_KEYS

RUNS = {
    'g0': ('runs/full_seed0_trajectory', 'runs/a3_replicate_seed0/best.pt'),
    'g1_v2': ('runs/p1_v2_matched_trajectory', 'runs/a3_p1_v2_seed0/best.pt'),
    'g1_v3': ('runs/p1_v3_matched_trajectory', 'runs/a3_p1_v3_hard_seed0/best.pt'),
}

baseline_args = baseline_fingerprints = baseline_schema = baseline_features = None
rows = {}
for name, (folder, a3) in RUNS.items():
    folder = Path(folder)
    report = json.loads((folder / 'report.json').read_text())
    args = {k: v for k, v in report['args'].items() if k not in ('cache', 'output')}
    fingerprints = {k: report['input_fingerprint'][k]['sha256'] for k in ('samples', 'queries')}
    manifest_path = Path(report['args']['cache']) / 'manifest.json'
    if file_sha256(manifest_path) != report['input_fingerprint']['cache_manifest']['sha256']:
        raise ValueError('training cache manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete'] or manifest['protocol']['checkpoint_sha256'] != file_sha256(a3):
        raise ValueError('head cache is not bound to its own A3 checkpoint')
    chunk = manifest['chunks'][0]
    chunk_path = manifest_path.parent / chunk['file']
    if file_sha256(chunk_path) != chunk['sha256']:
        raise ValueError('cache chunk changed')
    features = torch.load(chunk_path, map_location='cpu', weights_only=True)
    feature_schema = {k: dict(shape=list(features[k].shape[1:]), dtype=str(features[k].dtype)) for k in FEATURE_KEYS}
    head = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    schema = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in head['model'].items()}
    if len(report['history']) != 5 or report['training_samples'] != 115559 or report['development_samples'] != 12950:
        raise ValueError('training epochs or split sample counts differ')
    if baseline_args is None:
        baseline_args, baseline_fingerprints, baseline_schema, baseline_features = args, fingerprints, schema, feature_schema
    elif args != baseline_args or fingerprints != baseline_fingerprints or schema != baseline_schema or feature_schema != baseline_features:
        raise ValueError('unmatched structure, data, configuration or consumed feature schema')
    rows[name] = dict(a3_checkpoint=a3, a3_sha256=file_sha256(a3), head_checkpoint=str(folder / 'best.pt'),
        head_sha256=file_sha256(folder / 'best.pt'), cache_manifest=str(manifest_path),
        cache_manifest_sha256=file_sha256(manifest_path), cache_queries=manifest['total_queries'],
        unused_dense_cache_fields_present=manifest['protocol']['dense_context'],
        training_samples=report['training_samples'], development_samples=report['development_samples'],
        selected_head_epoch=head['epoch'], head_protocol=head['protocol'])
    del features, head

output = Path(__file__).with_name('fairness_audit.json')
if output.exists():
    raise ValueError('refusing audit overwrite')
result = dict(passed=True, matched_args=baseline_args, matched_input_sha256=baseline_fingerprints,
    consumed_feature_schema=baseline_features, head_parameter_schema=baseline_schema, models=rows,
    audit_source_sha256=file_sha256(__file__),
    note='Same trajectory/Stop architecture, optimizer settings, seed, split seed, training samples and five epochs. Each A3 is paired with its own cache/head. The loader consumes only FEATURE_KEYS and discards additional dense fields: their presence in G0 does not change head inputs. Actual tensors vary because of A3 initialization/training; they are not expected equal. P1 pretraining is additional computation reported separately. Head epoch selection is identical internal Train-family development-loss logic, so selected epochs may differ. No new Unseen result is read or used.')
output.write_text(json.dumps(result, indent=2))
print(json.dumps(dict(passed=True, models=list(rows), selected_epochs={k: v['selected_head_epoch'] for k, v in rows.items()})))
