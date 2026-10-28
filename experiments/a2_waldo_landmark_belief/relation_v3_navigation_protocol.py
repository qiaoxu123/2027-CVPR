"""Predeclared real pilot population and V3-only navigation admission checks.

Preparing this metadata neither loads calibration tensors nor compares their
performance. A positive calibration is necessary, not sufficient, for a real
navigation gain; benchmark execution additionally requires a successful pilot
and a final freeze.
"""
import argparse
import hashlib
import json
from .paths import ROOT,sha,save
from .collected_snapshot import BASE
from .relation_v3_confirmation_protocol import build_population,verify_opaque_seal,immutable
from .relation_v3_calibrate import validate_result_contract
from .relation_v3_gate_fit import validate_parameters


CONFIG=dict(version='relation_v3_real_navigation',seed=0,pilot_targets=64,
    pilot_source='Fixed209 independent TRAIN-source VERIFY targets; preserve original query',
    pilot_order='SHA256 relation-v3-navigation-pilot:0:<map_name>:<object_id>',
    full_counts=dict(val_seen=2470,val_unseen=2697),fixed_action=.25,temperature=1.,
    pilot_acceptance=dict(nonzero_updates=True,changed_real_trajectories=True,
        maximum_sr_losses_minus_gains=0,maximum_mean_ne_increase_m=0.),
    annotation_access='Public episode only during rollout; target coordinate/trajectory after complete rollout',
    benchmark_selection=False,no_val_unseen_tuning=True,device_execution='Serial shared lock; no parallel evaluator')


def build_pilot_population(roles,confirmation,source):
    names=('train','dev','calibration_fit','calibration_verify')
    if set(roles['target_roles'])!=set(names):raise ValueError('Four disjoint original roles required')
    key=lambda e:(e['map_name'],e['object_id'])
    groups={n:{key(e) for e in roles['target_roles'][n]} for n in names}
    if any(len(groups[n])!=len(roles['target_roles'][n]) for n in names):raise ValueError('Duplicate target roles')
    if any(groups[a]&groups[b] for i,a in enumerate(names) for b in names[i+1:]):raise ValueError('Pilot target role overlap')
    if len(groups['calibration_fit'])!=84 or len(groups['calibration_verify'])!=209:
        raise ValueError('Complete84FIT/209VERIFY inventory required')
    entries=confirmation['fresh_entries']+confirmation['sealed_entries']
    if len(entries)!=293 or len({key(e) for e in entries})!=293 or len({e['index'] for e in entries})!=293:
        raise ValueError('Complete unique confirmation source queries required')
    if len(confirmation['fresh_entries'])!=163 or len(confirmation['sealed_entries'])!=130:
        raise ValueError('Original fresh163 and sealed130 populations required')
    original={e['index']:e for e in source['entries']}
    if len(original)!=len(source['entries']):raise ValueError('Duplicate original source query')
    for e in entries:
        if e['fold'] not in ('fit','verify') or key(e) not in groups['calibration_'+e['fold']]:
            raise ValueError('Confirmation query changed role')
        row=original.get(e['index'])
        if row is None or any(e.get(k)!=v for k,v in row.items()):raise ValueError('Original confirmation query changed')
        if row['split'] not in ('dev','calibration'):raise ValueError('TRAIN-source reserved query required')
    if any(e['fold']!='verify' for e in confirmation['sealed_entries']):raise ValueError('Original sealed fold changed')
    verify=[e for e in entries if e['fold']=='verify']
    if {key(e) for e in verify}!=groups['calibration_verify']:raise ValueError('Complete VERIFY target inventory required')
    return sorted(verify,key=lambda e:hashlib.sha256(
        f"relation-v3-navigation-pilot:0:{e['map_name']}:{e['object_id']}".encode()).digest())[:64]


def require_candidate(candidate,decision):
    validate_result_contract(candidate,decision)
    if not candidate['enabled'] or not candidate['verify_gate']['passed']:
        raise ValueError('Nonzero independently verified V3 candidate required for real pilot')
    validate_parameters(candidate['fit_selected_parameters'])
    return candidate


def prepare_pilot(root=BASE/'revision_v3'):
    from pathlib import Path
    from .relation_v3_confirmation_pipeline import require_cpu_gates
    root=Path(root);require_cpu_gates(root)
    roles=json.loads((root/'roles.json').read_text());confirmation=json.loads((root/'confirmation_population.json').read_text())
    source=json.loads((BASE/'source_manifest.json').read_text());seal=json.loads((root/'sealed_verify130.json').read_text())
    if roles['lineage']['original_source_sha256']!=sha(BASE/'source_manifest.json'):
        raise ValueError('Original source manifest changed')
    verify_opaque_seal(seal)
    expected=build_population(roles,source,seal)
    expected.update(roles_sha256=sha(root/'roles.json'),source_manifest_sha256=sha(BASE/'source_manifest.json'),
        sealed_verify_sha256=sha(root/'sealed_verify130.json'),
        preparation_code_sha256=sha(ROOT/'experiments/a2_waldo_landmark_belief/relation_v3_confirmation_protocol.py'))
    if confirmation!=expected:raise ValueError('Original complete confirmation inventory changed')
    pilot=build_pilot_population(roles,confirmation,source)
    signature=dict(roles_sha256=sha(root/'roles.json'),confirmation_population_sha256=sha(root/'confirmation_population.json'),
        source_manifest_sha256=sha(BASE/'source_manifest.json'),sealed_verify_sha256=sha(root/'sealed_verify130.json'),
        preparation_code_sha256=sha(__file__),configuration=CONFIG)
    immutable(dict(signature=signature,entries=pilot),root/'navigation_pilot_population.json')
    immutable(CONFIG,root/'navigation_protocol.json')
    report=dict(status='v3_fixed64_real_navigation_pilot_metadata_prepared',
        population_sha256=sha(root/'navigation_pilot_population.json'),protocol_sha256=sha(root/'navigation_protocol.json'),
        roles_sha256=signature['roles_sha256'],pilot_groups=len(pilot),tensor_deserializations=0,
        new_observations=False,comparative_performance_consumed=False,real_pilot_completed=False,full_benchmark_completed=False)
    save(report,ROOT/'reports/a2_execution_20261027/RELATION_V3_NAVIGATION_PROTOCOL.json')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();print(json.dumps(prepare_pilot()),flush=True)
