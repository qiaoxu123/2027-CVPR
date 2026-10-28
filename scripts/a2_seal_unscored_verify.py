"""Opaque metadata/SHA seal; never deserialize any VERIFY tensor or metric."""
import json
from pathlib import Path
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.revision_confirmation import immutable_json


def main():
    root=BASE/'revision_v2';future=BASE/'revision_v3';future.mkdir(exist_ok=True)
    candidate=json.loads((root/'calibration/candidate_calibration.json').read_text());decision=json.loads((root/'calibration/fit_decision.json').read_text())
    if candidate['enabled'] or candidate['verification'] is not None or candidate['fit_selected_parameters'] is not None or decision['selected_index'] is not None:
        raise ValueError('VERIFY already eligible for feedback; cannot assert this no-candidate sealed path')
    population=json.loads((root/'confirmation_population.json').read_text());targets=[e for e in population['entries'] if e['fold']=='verify'];indices={e['index'] for e in targets}
    if len(targets)!=130 or len(indices)!=130:raise ValueError('Expected original130sealedtargets')
    manifest=json.loads((root/'confirmation/manifest.json').read_text());packets=json.loads((root/'calibration/packets/manifest.json').read_text())
    by_name={e['filename']:e for e in packets['entries']};records=[]
    for e in manifest['entries']:
        if e['fold']!='verify':continue
        if e['episode'] not in indices:raise ValueError('Unknown sealed episode')
        raw=root/'confirmation'/e['filename'];packet=root/'calibration/packets'/e['filename'];p=by_name[e['filename']]
        if sha(raw)!=e['sha256'] or sha(packet)!=p['sha256'] or sha(packet.with_suffix('.sha.json'))!=p['sidecar_sha256']:raise ValueError('Sealed opaque file changed')
        records.append(dict(filename=e['filename'],episode=e['episode'],step=e['step'],raw_sha256=e['sha256'],packet_sha256=p['sha256'],packet_sidecar_sha256=p['sidecar_sha256']))
    if len(records)!=1034:raise ValueError('Sealed observation count changed')
    commits=[]
    for e in manifest['episodes']:
        if e['index'] not in indices:continue
        path=root/'confirmation'/e['complete_file']
        if sha(path)!=e['sha256'] or sha(path.with_suffix('.sha.json'))!=e['sidecar_sha256']:raise ValueError('Sealed episode commit changed')
        commits.append(e)
    paths=[root/'confirmation/manifest.json',root/'confirmation/signature.json',root/'calibration/packets/manifest.json',root/'calibration/analysis_signature.json',
           root/'calibration/fit_decision.json',root/'calibration/candidate_calibration.json',root/'confirmation_population.json',root/'confirmation_protocol.json',
           ROOT/'experiments/a2_waldo_landmark_belief/direct_risk_calibrate.py',ROOT/'scripts/a2_direct_risk_fit_diagnosis.py']
    seal=dict(status='observed_preprocessed_VERIFY_not_used_for_candidate_performance_feedback',targets=targets,rows=records,episode_commits=commits,
        bound_sources={str(p.resolve()):sha(p) for p in paths},sealing_code_sha256=sha(Path(__file__)),
        fixed_machine_preprocessing_includes=['A0 per-episode metrics','rawGT integrity echo','per-row before/action grid distances'],
        not_claimed=['unobserved images','GT never read by machine','forensic guarantee over all external processes'],
        evidence='SelectedFITpointNone; immutable candidate.verificationNone; fixed source condition never invokes evaluate_risk(verify) when noselectedFITpoint; independent code/FIT-only whitelist review',
        allowed_next_use='Exclude from all fitting/training statistics; freeze next candidate and independentFIT-calibrated parameters, then exactly one VERIFY performance aggregate; no feedback-driven reselection')
    immutable_json(seal,future/'sealed_verify130.json')
    report=dict(status=seal['status'],targets=130,observations=1034,seal_sha256=sha(future/'sealed_verify130.json'),
        sealed_tensor_or_metric_deserializations_by_this_script=0,candidate_verification=None,no_model_performance_feedback_recorded=True,
        scope_note='Machine fixed preprocessing and per-row metrics already exist; this is an unscored-for-selection sealedholdout, not unobserved data')
    save(report,ROOT/'reports/a2_execution_20261027/RELATION_V3_VERIFY_SEAL.json');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
