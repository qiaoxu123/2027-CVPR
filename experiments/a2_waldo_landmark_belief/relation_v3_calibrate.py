"""Separate FIT sealing and one heldout aggregate; no benchmark access."""
import argparse,fcntl,json,time
from pathlib import Path
import torch
from .paths import ROOT,sha,save
from .collected_snapshot import BASE
from .relation_v3_confirmation_protocol import CONFIG,immutable
from .relation_v3_calibration_data import FreshConfirmation,calibration_code_hashes,load_frozen_choice,validate_fit_cache
from .relation_v3_gate_fit import choose_fit,evaluate_verify
from .calibration_protocol import verify_gate
from .direct_risk_protocol import select_point


def analysis_signature(source):
    from .relation_v3_confirmation_pipeline import require_cpu_gates
    require_cpu_gates(source.root)
    output=source.root/'calibration';output.mkdir(parents=True,exist_ok=True)
    signature=dict(checkpoint_sha256=source.checkpoint_sha256,roles_sha256=source.roles_sha256,
        population_sha256=sha(source.root/'confirmation_population.json'),protocol_sha256=sha(source.root/'confirmation_protocol.json'),
        confirmation_manifest_sha256=sha(source.directory/'manifest.json'),confirmation_signature_sha256=source.signature_sha256,
        sealed_verify_sha256=sha(source.root/'sealed_verify130.json'),training_verification_sha256=sha(source.root/'training/artifact_verification.json'),
        implementation=calibration_code_hashes(),posttraining_cpu_gate_sha256=sha(source.root/'posttraining_cpu_gates.json'),
        temperature=1.,fit_groups=84,verify_groups=209,protocol=CONFIG)
    immutable(signature,output/'analysis_signature.json');return signature


def seal_fit_decision(root,analysis,choices,chosen):
    root=Path(root);output=root/'calibration';path=output/'fit_decision.json'
    decision=dict(checkpoint_sha256=analysis['checkpoint_sha256'],roles_sha256=analysis['roles_sha256'],fit_groups=84,
        analysis_signature_sha256=sha(output/'analysis_signature.json'),choices=choices,
        selected_index=chosen['index'] if chosen else None,selected_parameters=chosen['parameters'] if chosen else None,
        temperature=1.,temperature_fitting=False,no_VERIFY_feedback_used=True,
        selection='Complete84FIT only, fixed30quantile menu and declared tie-break; no VERIFY runner-up')
    immutable(decision,path)
    restore_fit_seal(root,analysis,decision)
    return decision


def restore_fit_seal(root,analysis,decision):
    root=Path(root);output=root/'calibration';path=output/'fit_decision.json'
    validate_fit_cache(root)
    if (decision['checkpoint_sha256']!=analysis['checkpoint_sha256'] or decision['roles_sha256']!=analysis['roles_sha256']
        or decision['fit_groups']!=84 or decision['analysis_signature_sha256']!=sha(output/'analysis_signature.json')):
        raise ValueError('Existing FIT decision provenance changed')
    chosen=select_point(decision['choices'],dev=False)
    expected=chosen['parameters'] if chosen else None
    if decision['selected_parameters']!=expected or decision['selected_index']!=(chosen['index'] if chosen else None):
        raise ValueError('Existing FIT decision differs from unique menu selection')
    immutable(dict(status='v3_FIT_choice_frozen_before_VERIFY',fit_decision_sha256=sha(path),
        analysis_signature_sha256=sha(output/'analysis_signature.json'),
        fit_packet_manifest_sha256=sha(output/'packets_fresh_fit/manifest.json')),output/'fit_decision_seal.json')


def result_record(source,analysis,decision,stats,*,seconds):
    root=source.root;output=root/'calibration';selected=decision['selected_parameters']
    risk=verify_gate(stats) if stats is not None else dict(passed=False,checks=dict(fit_candidate_exists=False))
    enabled=bool(selected is not None and risk['passed'])
    result=dict(status='v3_candidate_requires_real_closed_loop_pilot' if selected is not None else 'v3_FIT_no_candidate_VERIFY_not_consumed',
        enabled=enabled,gate_final=False,fixed_action=.25,temperature=1.,temperature_fitting=False,
        minimum_gain=selected['minimum_gain'] if enabled else 1.,maximum_harm=selected['maximum_harm'] if enabled else 0.,
        checkpoint_sha256=source.checkpoint_sha256,roles_sha256=source.roles_sha256,training_code_sha256=source.state['code_sha256'],
        fit_selected_parameters=selected,fit_groups=84,verify_groups=209,verification_performance_evaluated=stats is not None,
        verification=stats,verify_gate=risk,no_verify_reselection=True,benchmark_used=False,
        analysis_signature_sha256=sha(output/'analysis_signature.json'),fit_decision_sha256=sha(output/'fit_decision.json'),
        fit_decision_seal_sha256=sha(output/'fit_decision_seal.json'),fit_packet_manifest_sha256=sha(output/'packets_fresh_fit/manifest.json'),
        confirmation_manifest_sha256=analysis['confirmation_manifest_sha256'],sealed_verify_sha256=analysis['sealed_verify_sha256'],
        training_verification_sha256=analysis['training_verification_sha256'],seconds=seconds,
        metric_scope='Internal grid localization on frozen-A0 observations, not navigation SR or final benchmark improvement')
    if stats is not None:
        result.update(fresh_verify_packet_manifest_sha256=sha(output/'packets_fresh_verify/manifest.json'),
            sealed_replay_manifest_sha256=sha(root/'sealed_replay/manifest.json'),
            sealed_packet_manifest_sha256=sha(output/'packets_sealed_verify/manifest.json'))
    validate_result_contract(result,decision)
    immutable(result,output/'candidate_calibration.json')
    commit_result(output)
    save(result,ROOT/'reports/a2_execution_20261027/RELATION_V3_CALIBRATION_FINAL.json')
    return result


def validate_result_contract(result,decision):
    selected=decision['selected_parameters'];stats=result['verification']
    if (result['fit_selected_parameters']!=selected or (stats is None)!=(selected is None)
        or result['fit_groups']!=84 or result['verify_groups']!=209 or (stats is not None and stats['groups']!=209)
        or result['verification_performance_evaluated']!=(stats is not None)):
        raise ValueError('Stored calibration choice/VERIFY scope changed')
    risk=verify_gate(stats) if stats is not None else dict(passed=False,checks=dict(fit_candidate_exists=False))
    enabled=bool(selected is not None and risk['passed'])
    expected=dict(status='v3_candidate_requires_real_closed_loop_pilot' if selected is not None else 'v3_FIT_no_candidate_VERIFY_not_consumed',
        enabled=enabled,gate_final=False,fixed_action=.25,temperature=1.,temperature_fitting=False,benchmark_used=False,no_verify_reselection=True,
        verify_gate=risk,minimum_gain=selected['minimum_gain'] if enabled else 1.,maximum_harm=selected['maximum_harm'] if enabled else 0.)
    if any(result[k]!=v for k,v in expected.items()):raise ValueError('Stored calibration enable/threshold/risk contract changed')


def commit_result(output):
    immutable(dict(candidate_sha256=sha(output/'candidate_calibration.json'),analysis_signature_sha256=sha(output/'analysis_signature.json'),
        fit_decision_sha256=sha(output/'fit_decision.json')),output/'candidate_calibration.sha.json')


def validate_existing(source,analysis):
    output=source.root/'calibration';result=json.loads((output/'candidate_calibration.json').read_text())
    validate_fit_cache(source.root)
    if (result['checkpoint_sha256']!=source.checkpoint_sha256 or result['roles_sha256']!=source.roles_sha256 or result['training_code_sha256']!=source.state['code_sha256']
        or result['analysis_signature_sha256']!=sha(output/'analysis_signature.json')
        or result['fit_decision_sha256']!=sha(output/'fit_decision.json')
        or result['fit_decision_seal_sha256']!=sha(output/'fit_decision_seal.json')
        or result['fit_packet_manifest_sha256']!=sha(output/'packets_fresh_fit/manifest.json')):raise ValueError('Existing unique calibration decision changed')
    decision=json.loads((output/'fit_decision.json').read_text());validate_result_contract(result,decision)
    if result['fit_selected_parameters'] is not None:
        load_frozen_choice(source.root,source.checkpoint_sha256,source.roles_sha256)
    if result['verification'] is not None and (result['verification']['groups']!=209 or result['verify_gate']!=verify_gate(result['verification'])):
        raise ValueError('Existing VERIFY report changed')
    if result['verification'] is not None:
        if (result['fresh_verify_packet_manifest_sha256']!=sha(output/'packets_fresh_verify/manifest.json')
            or result['sealed_replay_manifest_sha256']!=sha(source.root/'sealed_replay/manifest.json')
            or result['sealed_packet_manifest_sha256']!=sha(output/'packets_sealed_verify/manifest.json')):
            raise ValueError('Existing VERIFY input manifests changed')
        fresh=json.loads((output/'packets_fresh_verify/manifest.json').read_text());sealed=json.loads((output/'packets_sealed_verify/manifest.json').read_text())
        if result['verification']['views']!=len(fresh['entries'])+len(sealed['entries']):raise ValueError('Existing VERIFY observation count changed')
    commit_result(output)
    save(result,ROOT/'reports/a2_execution_20261027/RELATION_V3_CALIBRATION_FINAL.json')
    return result


def fit_stage():
    start=time.perf_counter();source=FreshConfirmation();analysis=analysis_signature(source);output=source.root/'calibration'
    if (output/'candidate_calibration.json').exists():
        result=validate_existing(source,analysis);print(json.dumps(dict(status='retained_unique_calibration',enabled=result['enabled'])),flush=True);return
    path=output/'fit_decision.json'
    if path.exists():
        decision=json.loads(path.read_text())
        restore_fit_seal(source.root,analysis,decision)
        if decision['selected_parameters'] is not None:
            load_frozen_choice(source.root,source.checkpoint_sha256,source.roles_sha256)
            print('Retained unique frozen FIT choice; VERIFY not rerun',flush=True);return
    else:
        packets=source.packets('fit');choices,chosen=choose_fit(packets,source.groups('fit'))
        decision=seal_fit_decision(source.root,analysis,choices,chosen)
    if decision['selected_parameters'] is None:
        result=result_record(source,analysis,decision,None,seconds=time.perf_counter()-start)
        print(json.dumps(dict(status=result['status'],enabled=False,verification=None)),flush=True)
    else:print(json.dumps(dict(status='unique_FIT_choice_frozen_VERIFY_not_consumed',selected_parameters=decision['selected_parameters'])),flush=True)


def verify_stage():
    start=time.perf_counter();source=FreshConfirmation();analysis=analysis_signature(source);output=source.root/'calibration'
    if (output/'candidate_calibration.json').exists():
        result=validate_existing(source,analysis);print(json.dumps(dict(status='retained_unique_calibration',enabled=result['enabled'])),flush=True);return
    decision=load_frozen_choice(source.root,source.checkpoint_sha256,source.roles_sha256)
    # The marker binds retries to this SAME fixed choice. A crash may reproduce
    # an unfinished aggregate, never inspect a runner-up or change thresholds.
    immutable(dict(fit_decision_sha256=sha(output/'fit_decision.json'),analysis_signature_sha256=sha(output/'analysis_signature.json'),
        checkpoint_sha256=source.checkpoint_sha256,roles_sha256=source.roles_sha256),output/'verification_started.json')
    from .relation_v3_sealed_replay import load_replayed_packets
    sealed=load_replayed_packets(source);fresh=source.packets('verify');all_packets=sealed+fresh
    expected={(e['map_name'],e['object_id']) for e in source.roles['target_roles']['calibration_verify']}
    stats=evaluate_verify(all_packets,decision['selected_parameters'],expected)
    result=result_record(source,analysis,decision,stats,seconds=time.perf_counter()-start)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('fit','verify'),required=True);args=parser.parse_args()
    torch.set_num_threads(2);output=BASE/'revision_v3/calibration';output.mkdir(parents=True,exist_ok=True)
    with (output/'calibration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (fit_stage if args.stage=='fit' else verify_stage)()


if __name__=='__main__':main()
