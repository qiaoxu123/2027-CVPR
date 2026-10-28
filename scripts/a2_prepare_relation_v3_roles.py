"""Versioned metadata-only roles: real A0 training states and sealed holdout.

No VERIFY tensor, target coordinate, score or outcome is decoded by this script.
"""
from collections import defaultdict,Counter
import hashlib,json,re
from pathlib import Path
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE,cache_source
from experiments.a2_waldo_landmark_belief.revision_confirmation import immutable_json
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold


def verify_target_roles(roles):
    keys=list(roles)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            if roles[a]&roles[b]:raise ValueError(f'Target leakage between {a} and {b}')


def main():
    root=BASE/'revision_v3';root.mkdir(parents=True,exist_ok=True)
    candidate=BASE/'revision_v2/calibration/candidate_calibration.json';decision=json.loads(candidate.read_text())
    if decision['enabled'] or decision['verification'] is not None or decision['fit_selected_parameters'] is not None:
        raise ValueError('v2VERIFY has been used or FITcandidate existed; cannot declare it sealed')
    source=json.loads((BASE/'source_manifest.json').read_text());by_index={e['index']:e for e in source['entries']}
    roles={r:set() for r in ('train','dev','calibration_fit','calibration_verify')};rows=[];seen=set()
    def add(role,directory,entry,kind,target):
        key=(str(directory.resolve()),entry['filename'])
        if key in seen:raise ValueError('Duplicate source observation in v3roles')
        seen.add(key);roles[role].add(target)
        rows.append(dict(role=role,source_kind=kind,directory=str(directory.resolve()),filename=entry['filename'],sha256=entry['sha256'],
            episode=entry['episode'],step=entry['step'],map_name=target[0],object_id=target[1]))
    snapshot=BASE/'collected_snapshot';manifest=json.loads((snapshot/'manifest.json').read_text());original={e['index']:e for e in cache_source(snapshot)['entries']}
    for entry in manifest['entries']:
        if entry['split'] not in ('train','dev'):continue
        e=original[entry['episode']];add(entry['split'],snapshot,entry,'teacher_snapshot',(e['map_name'],e['object_id']))
    for version in ('revision_v1','revision_v2'):
        directory=BASE/version/'confirmation';manifest=json.loads((directory/'manifest.json').read_text())
        expected_status='independent_A0_confirmation_complete' if version=='revision_v1' else 'direct_risk_independent_A0_confirmation_complete'
        if manifest['status']!=expected_status or manifest['target_groups']!=256 or manifest['signature_sha256']!=sha(directory/'signature.json'):
            raise ValueError('Complete prior A0source required')
        population=json.loads((BASE/version/'confirmation_population.json').read_text());entries={e['index']:e for e in population['entries']}
        for entry in manifest['entries']:
            e=entries[entry['episode']];target=(e['map_name'],e['object_id'])
            if entry['fold']!=e['fold']:raise ValueError('Source fold changed')
            if version=='revision_v2' and entry['fold']=='verify':
                # Opaque SHA only: do not deserialize sealed raw/packet data.
                if sha(directory/entry['filename'])!=entry['sha256']:raise ValueError('Sealed VERIFY raw commit changed')
                roles['calibration_verify'].add(target)
                rows.append(dict(role='calibration_verify',source_kind='sealed_v2VERIFY',directory=str(directory.resolve()),**entry,map_name=target[0],object_id=target[1]))
            else:add('train' if entry['fold']=='fit' else 'dev',directory,entry,version+'_A0_closedloop',target)
    observed=set()
    for name in ('revision_confirmation_reserve.json','revision_v2_confirmation_reserve.json'):
        observed|={(e['map_name'],e['object_id']) for e in json.loads((BASE/name).read_text())['targets']}
    files=0
    for directory in BASE.glob('cache*'):
        if not directory.is_dir() or directory.name=='cache_query':continue
        for path in directory.rglob('*.pt'):
            match=re.fullmatch(r'(train|dev|calibration)_(\d+)_(\d+)\.pt',path.name)
            if not match:raise ValueError('Unknown oldview filename; audit before reserving')
            entry=by_index[int(match[2])];observed.add((entry['map_name'],entry['object_id']));files+=1
    available=defaultdict(list)
    for entry in source['entries']:
        key=(entry['map_name'],entry['object_id'])
        if entry['split']!='train' and key not in observed:available[key].append(entry['index'])
    if len(available)!=163:raise ValueError('Untouched source population changed; inspect before role allocation')
    fresh=[]
    for key in sorted(available,key=lambda g:hashlib.sha256(f'relation-v3-fresh-fit:0:{g}'.encode()).digest()):
        index=min(available[key],key=lambda i:hashlib.sha256(f'relation-v3-query:0:{i}'.encode()).digest())
        fold=calibration_fold(*key)
        fresh.append(dict(map_name=key[0],object_id=key[1],episode_indices=available[key],index=index,fold=fold))
        roles['calibration_'+fold].add(key)
    verify_target_roles(roles)
    expected=dict(train=33440,dev=5647,calibration_verify=1034)
    counts=Counter(r['role'] for r in rows)
    if any(counts[k]!=n for k,n in expected.items()) or len(roles['calibration_fit'])+len(roles['calibration_verify'])!=293:raise ValueError('Versioned role counts changed')
    lineage=dict(original_source_sha256=sha(BASE/'source_manifest.json'),snapshot_manifest_sha256=sha(snapshot/'manifest.json'),
        v1_manifest_sha256=sha(BASE/'revision_v1/confirmation/manifest.json'),v2_manifest_sha256=sha(BASE/'revision_v2/confirmation/manifest.json'),
        v2_failed_candidate_sha256=sha(candidate),v2_fit_decision_sha256=sha(BASE/'revision_v2/calibration/fit_decision.json'),
        preparation_code_sha256=sha(Path(__file__)))
    full=dict(status='v3_target_disjoint_roles_fixed_before_training',lineage=lineage,rows=rows,
        target_roles={role:[dict(map_name=m,object_id=o) for m,o in sorted(targets)] for role,targets in roles.items()},fresh_calibration=fresh,
        sealed_holdout_description='Observed/preprocessed v2VERIFY130, never aggregated or used for performance feedback; NOT unobserved data',
        forbidden_supervision='Exclude sealed130 and fresh163 identities from every rank/trust negative, spatial/relation/category label and training statistic; public map references remain public inputs')
    immutable_json(full,root/'roles.json')
    report=dict(status=full['status'],roles_sha256=sha(root/'roles.json'),lineage=lineage,observation_counts=dict(counts),
        target_counts={k:len(v) for k,v in roles.items()},old_view_files_scanned=files,fresh_observations_collected=False,
        fresh_targets_split_by_original_fixed_hash=dict(Counter(e['fold'] for e in fresh)),sealed_existing_verify_targets=130,
        sealed_verify_performance_aggregated=False,all_target_roles_disjoint=True,
        note='v1FIT127+v2FIT126 become TRAIN; v1VERIFY129 becomes DEV; prior versions unchanged; role audit does not prove future loss masking')
    save(report,ROOT/'reports/a2_execution_20261027/RELATION_V3_ROLE_AUDIT.json');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
