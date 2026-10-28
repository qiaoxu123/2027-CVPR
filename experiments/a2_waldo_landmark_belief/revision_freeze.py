"""Freeze a positive independently calibrated, real-pilot-checked revision."""
import argparse,json,math
from pathlib import Path
from datetime import datetime,timezone
from .paths import ROOT,DATA,RGBD,CHECKPOINT,A0_SHA,save,sha,bootstrap,verify_frozen_assets
from .collected_snapshot import BASE
from .calibration_protocol import verify_gate
from .revision_confirmation import immutable_json
from .revision_eval_protocol import evaluation_code_hashes,FULL_COUNTS


def require_candidate(candidate):
    if (candidate['status']!='revision_candidate_requires_real_closed_loop_pilot' or not candidate['enabled']
        or not candidate['no_verify_reselection'] or candidate['benchmark_used']):raise ValueError('Safe nonzero independent candidate required')
    if not candidate['verify_gate']['passed'] or not verify_gate(candidate['verification'])['passed']:
        raise ValueError('Independent verification conditions failed')
    if not 0<float(candidate['scale'])<=1 or not 0<=float(candidate['threshold'])<=1 or not math.isfinite(candidate['temperature']) or candidate['temperature']<=0:
        raise ValueError('Invalid frozen revision parameters')


def check_bound_digests(candidate,bindings):
    for field,path in bindings.items():
        if candidate.get(field)!=sha(path):raise ValueError(f'Calibrated source digest changed: {field}')


def validate_calibration_sources(candidate,root):
    root=Path(root)
    bindings=dict(analysis_signature_sha256=root/'calibration/analysis_signature.json',
                  fit_decision_sha256=root/'calibration/fit_decision.json',
                  confirmation_manifest_sha256=root/'confirmation/manifest.json',
                  packet_manifest_sha256=root/'calibration/packets/manifest.json',
                  training_verification_sha256=root/'training/artifact_verification.json')
    check_bound_digests(candidate,bindings)
    analysis=json.loads(bindings['analysis_signature_sha256'].read_text())
    if analysis['checkpoint_sha256']!=candidate['checkpoint_sha256'] or analysis['protocol_sha256']!=sha(root/'confirmation_protocol.json'):
        raise ValueError('Calibration analysis lineage changed')
    for name,digest in analysis['implementation'].items():
        if sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)!=digest:raise ValueError('Calibration code changed after fitting')
    fit=json.loads(bindings['fit_decision_sha256'].read_text())
    if fit['selected_index'] is None or fit['selected_parameters']!=candidate['fit_selected_parameters'] or fit['temperature']!=candidate['temperature_report']:
        raise ValueError('Unique FIT decision differs from frozen candidate')
    selected=fit['choices'][fit['selected_index']]['parameters']
    temperature=min(fit['temperature']['choices'],key=lambda r:r['nll'])['temperature'] if fit['temperature']['sufficient'] else 1.
    if selected!=fit['selected_parameters'] or any(candidate[k]!=selected[k] for k in ('scale','threshold')) or candidate['temperature']!=temperature:
        raise ValueError('Candidate parameters were reselected after heldout verification')
    confirmation=json.loads((root/'confirmation/manifest.json').read_text())
    signature=json.loads((root/'confirmation/signature.json').read_text())
    if confirmation['signature_sha256']!=sha(root/'confirmation/signature.json') or signature['checkpoint_sha256']!=candidate['checkpoint_sha256']:
        raise ValueError('Independent confirmation signature changed')
    from .revision_confirmation import confirmation_code_hashes
    if signature['code_sha256']!=confirmation_code_hashes() or signature['protocol_sha256']!=sha(root/'confirmation_protocol.json') or signature['population_sha256']!=sha(root/'confirmation_population.json'):
        raise ValueError('Independent confirmation source/implementation changed')
    for item in confirmation['episodes']:
        if sha(root/'confirmation'/item['complete_file'])!=item['sha256']:raise ValueError('Confirmation episode commit changed')
    for item in confirmation['entries']:
        if sha(root/'confirmation'/item['filename'])!=item['sha256']:raise ValueError('Confirmation raw row changed')
    packets=json.loads((root/'calibration/packets/manifest.json').read_text())
    if packets['signature_sha256']!=sha(root/'calibration/packets/signature.json'):raise ValueError('Packet signature changed')
    for item in packets['entries']:
        path=root/'calibration/packets'/item['filename']
        if sha(path)!=item['sha256'] or sha(path.with_suffix('.sha.json'))!=item['sidecar_sha256']:raise ValueError('Committed calibration packet changed')
    return list(bindings.values())+[root/'confirmation/signature.json',root/'calibration/packets/signature.json']


def evaluation_data_signature():
    bootstrap()
    from sbfnav.dataset import CityNavDataset
    annotations={};maps=set()
    for split,count in FULL_COUNTS.items():
        ds=CityNavDataset(DATA,split)
        if len(ds)!=count:raise ValueError('Full evaluation population changed')
        annotations[str(ds.annotation_path)]=sha(ds.annotation_path);maps.update(ds.map_names)
    files={(root/f'{name}.{ext}').resolve() for root in (RGBD,DATA/'rgbd') for name in maps for ext in ('png','tif')}
    return dict(annotations=annotations,rgbd={str(p):sha(p) for p in sorted(files)},
                public_objects_sha256=sha(DATA/'cityrefer/objects.json'),reference_links_sha256=sha(DATA/'cityrefer/processed_descriptions.json'))


def validate_frozen(path):
    frozen=json.loads(Path(path).read_text())
    if frozen['status']!='revision_frozen_after_independent_calibration_and_real_pilot' or not frozen['enabled']:
        raise ValueError('Final nonzero revision freeze required for formal evaluation')
    root=BASE/'revision_v1';checkpoint=root/'training/best.pt'
    if sha(CHECKPOINT)!=A0_SHA or frozen['a0_sha256']!=A0_SHA or sha(checkpoint)!=frozen['checkpoint_sha256']:
        raise ValueError('Final evaluation checkpoint changed')
    if evaluation_code_hashes()!=frozen['code_sha256']:raise ValueError('Evaluation/runtime implementation changed after freeze')
    if verify_frozen_assets()!=frozen['external_dependencies']:raise ValueError('External A0/SBF/vendor/weights changed after freeze')
    for name,digest in frozen['bound_files'].items():
        if sha(name)!=digest:raise ValueError('Frozen calibration/pilot/source artifact changed')
    if evaluation_data_signature()!=frozen['data_signature']:raise ValueError('Evaluation annotations/rasters changed after freeze')
    candidate=json.loads((root/'calibration/candidate_calibration.json').read_text());require_candidate(candidate)
    validate_calibration_sources(candidate,root)
    final=json.loads((root/'final_calibration.json').read_text())
    if not final['gate_final'] or any(final[k]!=candidate[k] for k in ('enabled','scale','threshold','temperature','checkpoint_sha256')):
        raise ValueError('Final inference parameters differ from unique calibrated candidate')
    return frozen


def main():
    parser=argparse.ArgumentParser();parser.parse_args();root=BASE/'revision_v1'
    from .revision_artifacts import verify_revision_training
    state,report=verify_revision_training()
    candidate_path=root/'calibration/candidate_calibration.json';candidate=json.loads(candidate_path.read_text());require_candidate(candidate)
    calibration_sources=validate_calibration_sources(candidate,root)
    from .revision_evaluation import verify_pilot_outputs
    pilot=verify_pilot_outputs()
    if (not pilot['decision']['passed'] or pilot['decision']['episodes']!=64
        or pilot['checkpoint_sha256']!=sha(root/'training/best.pt') or pilot['calibration_candidate_sha256']!=sha(candidate_path)):
        raise ValueError('A complete safe real64-pilot is required')
    if pilot['code_sha256']!=evaluation_code_hashes():raise ValueError('Real pilot ran different runtime/evaluation code')
    final=dict(candidate,status='revision_parameters_frozen_after_real_pilot',gate_final=True,pilot_report_sha256=sha(root/'pilot/report.json'))
    immutable_json(final,root/'final_calibration.json')
    paths=[root/'final_calibration.json',candidate_path,root/'pilot/report.json',root/'training/artifact_verification.json',
           root/'training/training_complete.json',root/'confirmation/manifest.json',root/'calibration/fit_decision.json',
           root/'calibration/packets/manifest.json',root/'confirmation_population.json',root/'confirmation_protocol.json']+calibration_sources
    frozen=dict(status='revision_frozen_after_independent_calibration_and_real_pilot',enabled=True,
                checkpoint_sha256=sha(root/'training/best.pt'),a0_sha256=A0_SHA,code_sha256=evaluation_code_hashes(),
                bound_files={str(p.resolve()):sha(p) for p in paths},data_signature=evaluation_data_signature(),external_dependencies=verify_frozen_assets(),
                episode_counts=FULL_COUNTS,model_selected_using='TRAIN internal dev only; independent calibration and real pilot',
                no_val_unseen_selection=True,frozen_at_utc=datetime.now(timezone.utc).isoformat())
    if (root/'frozen.json').exists():raise ValueError('Final freeze exists; inspect instead of refreezing after results')
    save(frozen,root/'frozen.json');print('Revision frozen; formal full evaluation authorized by verified internal evidence',flush=True)


if __name__=='__main__':main()
