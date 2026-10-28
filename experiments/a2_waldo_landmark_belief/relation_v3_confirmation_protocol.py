"""Post-training confirmation inventory; opaque seals, no tensor/metric reading."""
from collections import Counter
import hashlib,json
from pathlib import Path
from .paths import ROOT,sha,save
from .collected_snapshot import BASE
from .calibration_protocol import calibration_fold,CALIBRATION_CONFIG
from .direct_risk_protocol import GAIN_QUANTILES,HARM_QUANTILES

COUNTS=dict(fresh=163,fit=84,verify=209,sealed=130)
CONFIG=dict(version='relation_v3_independent_confirmation',seed=0,fresh_targets=163,fit_targets=84,verify_targets=209,
    sealed_existing_verify_targets=130,fresh_verify_targets=79,temperatures=[1.],fixed_action=.25,
    gain_quantiles=list(GAIN_QUANTILES),harm_quantiles=list(HARM_QUANTILES),quantile_method='linear',
    source='Original CityNav TRAIN only; complete frozen-A0 closed-loop horizon20, V3 shadowOFF',
    selection='One FIT-only score-threshold choice; no VERIFY runner-up or reselection',
    label_timing='First destination coordinate use after the whole A0 episode rollout',
    heldout_order='Do not decode sealed130 before model and FIT choice are frozen; exactly one VERIFY aggregate',
    fit_rule='R1>0, NE<=0, supported>=10, broken-group-rate<=.02; fixed30quantile menu',
    risk_criteria={k:v for k,v in CALIBRATION_CONFIG.items() if (k.startswith(('min_','max_','pilot_')) or k=='bootstrap_replicates') and not k.startswith('min_temperature')},
    benchmark_used=False,full_training_required='All20ranking+5refinement and all25 verified; eligible realDEV gate')


def key(row):return row['map_name'],row['object_id']


def build_population(roles,source,seal,*,expected=COUNTS):
    names=('train','dev','calibration_fit','calibration_verify')
    if set(roles['target_roles'])!=set(names):raise ValueError('Four explicit target roles required')
    groups={name:{key(e) for e in roles['target_roles'][name]} for name in names}
    if any(len(groups[n])!=len(roles['target_roles'][n]) for n in names):raise ValueError('Duplicate target roles')
    if any(groups[a]&groups[b] for i,a in enumerate(names) for b in names[i+1:]):raise ValueError('Supervision/confirmation target overlap')
    by_index={e['index']:e for e in source['entries']}
    if len(by_index)!=len(source['entries']):raise ValueError('Duplicate original source query')
    old=seal['targets'];old_keys={key(e) for e in old}
    if len(old)!=expected['sealed'] or len(old_keys)!=expected['sealed'] or not old_keys<=groups['calibration_verify']:
        raise ValueError('Sealed target population changed')
    for e in old:
        original=by_index[e['index']]
        if e['fold']!='verify' or calibration_fold(*key(e))!='verify' or any(e[k]!=original[k] for k in original):
            raise ValueError('Sealed original query/fold changed')
    fresh=[];seen=set()
    for item in roles['fresh_calibration']:
        target=key(item);fold=item['fold']
        if target in seen or target in old_keys or target in groups['train']|groups['dev']:raise ValueError('Fresh target reused or supervised')
        if fold not in ('fit','verify') or calibration_fold(*target)!=fold or target not in groups['calibration_'+fold]:raise ValueError('Fresh target fold changed')
        choices=[e['index'] for e in source['entries'] if key(e)==target and e['split'] in ('dev','calibration')]
        if sorted(choices)!=sorted(item['episode_indices']) or not choices:raise ValueError('Fixed source-query candidates changed')
        chosen=min(choices,key=lambda i:hashlib.sha256(f'relation-v3-query:0:{i}'.encode()).digest())
        if item['index']!=chosen:raise ValueError('Predeclared fresh query was reselected')
        original=by_index[chosen]
        if key(original)!=target or original['split']=='train':raise ValueError('Fresh source identity changed')
        fresh.append(dict(original,fold=fold));seen.add(target)
    counts=Counter(e['fold'] for e in fresh)
    if (len(fresh)!=expected['fresh'] or counts['fit']!=expected['fit'] or len(groups['calibration_fit'])!=expected['fit']
        or len(groups['calibration_verify'])!=expected['verify'] or counts['verify']+len(old)!=expected['verify']
        or {key(e) for e in fresh if e['fold']=='fit'}!=groups['calibration_fit']
        or {key(e) for e in fresh if e['fold']=='verify'}|old_keys!=groups['calibration_verify']):
        raise ValueError('Complete fixed confirmation population required')
    return dict(fresh_entries=fresh,sealed_entries=old,fresh_folds=dict(counts),target_counts=dict(expected))


def verify_opaque_seal(seal):
    """Hash bytes only; do not deserialize a single sealed tensor or metric."""
    original=BASE/'revision_v2'
    if seal['status']!='observed_preprocessed_VERIFY_not_used_for_candidate_performance_feedback':raise ValueError('Unexpected original seal status')
    for name,digest in seal['bound_sources'].items():
        if sha(Path(name))!=digest:raise ValueError('Opaque sealed source changed')
    targets={e['index'] for e in seal['targets']};rows=seal['rows'];commits=seal['episode_commits']
    if (len(targets)!=130 or len(rows)!=1034 or len(commits)!=130 or len({e['filename'] for e in rows})!=1034
        or {e['index'] for e in commits}!=targets or {e['episode'] for e in rows}!=targets):
        raise ValueError('Complete original seal required')
    population=json.loads((original/'confirmation_population.json').read_text())
    manifest=json.loads((original/'confirmation/manifest.json').read_text())
    if seal['targets']!=[e for e in population['entries'] if e['fold']=='verify']:
        raise ValueError('Sealed target queries differ from original population')
    expected=[(e['filename'],e['episode'],e['step'],e['sha256']) for e in manifest['entries'] if e['fold']=='verify']
    actual=[(e['filename'],e['episode'],e['step'],e['raw_sha256']) for e in rows]
    if actual!=expected or commits!=[e for e in manifest['episodes'] if e['index'] in targets]:
        raise ValueError('Sealed row/episode population differs from original committed source')
    for e in rows:
        name=e['filename']
        if Path(name).name!=name or e['episode'] not in targets:raise ValueError('Unknown sealed row path/target')
        if (sha(original/'confirmation'/name)!=e['raw_sha256'] or sha(original/'calibration/packets'/name)!=e['packet_sha256']
            or sha((original/'calibration/packets'/name).with_suffix('.sha.json'))!=e['packet_sidecar_sha256']):raise ValueError('Sealed row bytes changed')
    for e in commits:
        name=e['complete_file']
        if Path(name).name!=name or e['index'] not in targets:raise ValueError('Unknown sealed episode path')
        path=original/'confirmation'/name
        if sha(path)!=e['sha256'] or sha(path.with_suffix('.sha.json'))!=e['sidecar_sha256']:raise ValueError('Sealed episode bytes changed')


def immutable(value,path):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Immutable V3 confirmation metadata changed: '+str(path))
    else:save(value,path)


def prepare():
    root=BASE/'revision_v3';roles=json.loads((root/'roles.json').read_text());seal=json.loads((root/'sealed_verify130.json').read_text())
    source=json.loads((BASE/'source_manifest.json').read_text());cpu=json.loads((root/'cpu_gates.json').read_text())
    if roles['lineage']['original_source_sha256']!=sha(BASE/'source_manifest.json') or cpu['roles_sha256']!=sha(root/'roles.json'):
        raise ValueError('Frozen training role/source lineage changed')
    saved_seal=json.loads((ROOT/'reports/a2_execution_20261027/RELATION_V3_VERIFY_SEAL.json').read_text())
    if (saved_seal['seal_sha256']!=sha(root/'sealed_verify130.json')
        or seal['sealing_code_sha256']!=sha(ROOT/'scripts/a2_seal_unscored_verify.py')):raise ValueError('Original published holdout seal changed')
    verify_opaque_seal(seal);population=build_population(roles,source,seal)
    population.update(roles_sha256=sha(root/'roles.json'),source_manifest_sha256=sha(BASE/'source_manifest.json'),
        sealed_verify_sha256=sha(root/'sealed_verify130.json'),preparation_code_sha256=sha(Path(__file__)))
    immutable(population,root/'confirmation_population.json');immutable(CONFIG,root/'confirmation_protocol.json')
    report=dict(status='V3_confirmation_metadata_fixed_without_new_observation_or_sealed_tensor_decode',
        population_sha256=sha(root/'confirmation_population.json'),protocol_sha256=sha(root/'confirmation_protocol.json'),
        roles_sha256=population['roles_sha256'],fresh_targets=163,fit_targets=84,verify_targets=209,
        fresh_folds=population['fresh_folds'],sealed_tensor_deserializations=0,new_observations_collected=False,
        training_checkpoint_not_selected_here=True)
    save(report,ROOT/'reports/a2_execution_20261027/RELATION_V3_CONFIRMATION_PROTOCOL.json');print(json.dumps(report),flush=True)


if __name__=='__main__':prepare()
