"""Observe the original three units without starting, stopping or changing jobs."""
import os
import subprocess
import json
from pathlib import Path


UNITS = (
    'graphnav-a2-reference-v4-pipeline-20261029.service',
    'graphnav-a2-reference-v4-confirmation-20261029.service',
    'graphnav-a2-reference-v4-navigation-20261029.service',
)
FIELDS = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'ExecMainPID',
          'ExecMainStatus', 'Result', 'InvocationID')
LIVE_IDENTITY_SHA256 = 'e1410c820e7bb7b3e6a958e4b1ab390a82efd37296cbdec18cbc69246e825dde'


class ObservationUnavailable(RuntimeError):
    """An unavailable observation cannot establish process completion."""


def parse_service_output(output):
    result = {}
    for block in output.strip().split('\n\n'):
        record = dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
        name = record.pop('Id', None)
        if name not in UNITS or name in result or any(k not in record for k in FIELDS):
            raise ObservationUnavailable('Incomplete or duplicate native service observation')
        result[name] = record
    if set(result) != set(UNITS):
        raise ObservationUnavailable('Not all original services were observed')
    return result


def read_services():
    command = ['systemctl', '--user', 'show', *UNITS]
    for field in ('Id', *FIELDS):
        command.extend(('-p', field))
    try:
        process = subprocess.run(command, text=True, capture_output=True, timeout=20,
                                 env=dict(os.environ, XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}'))
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ObservationUnavailable(str(error)) from error
    if process.returncode:
        raise ObservationUnavailable('Native service observation failed: ' + process.stderr.strip())
    return parse_service_output(process.stdout)


def require_terminal_services(services, identities):
    """False means verified waiting; malformed observation never implies ready."""
    if set(services) != set(UNITS) or set(identities) != set(UNITS):
        raise ObservationUnavailable('Complete original service identities required')
    ready = True
    for name in UNITS:
        current, original = services[name], identities[name]
        if any(k not in current for k in FIELDS):
            raise ObservationUnavailable('Incomplete native lifecycle observation')
        if (current['LoadState'] != 'loaded' or current['ActiveState'] == 'failed'
                or current['Result'] != 'success' or current['ExecMainStatus'] != '0'):
            raise ValueError('Original experiment service missing or failed')
        if (current['InvocationID'] != original['InvocationID']
                or current['ExecMainPID'] != original['MainPID']):
            raise ValueError('Observed service is another execution')
        try:
            pid = int(current['MainPID'])
        except (ValueError, TypeError) as error:
            raise ObservationUnavailable('Unreadable native PID') from error
        if pid > 0:
            if str(pid) != original['MainPID']:
                raise ValueError('Original native process identity changed')
            ready = False
        elif current['SubState'] in ('start', 'start-pre', 'start-post', 'running',
                                     'stop', 'stop-sigterm', 'stop-sigkill'):
            ready = False
        elif (current['ActiveState'], current['SubState']) not in (
                ('active', 'exited'), ('inactive', 'dead')):
            raise ValueError('Unknown original experiment lifecycle')
    return ready


def validate_checkpoint_header(state, proof):
    """Allow an ineligible final state for diagnosis without changing eligibility."""
    from experiments.a2_waldo_landmark_belief.paths import A0_SHA
    keys = ('code_sha256', 'roles_sha256', 'feature_manifest_sha256',
            'feature_signature_sha256', 'feature_build_status_sha256', 'config')
    if (state.get('a0_sha256') != A0_SHA
            or state.get('config', {}).get('version') != 'a2_relation_v4_public_roads'
            or state.get('scope') == 'TRAIN_functional_only'
            or state.get('epoch') not in range(21, 26)
            or state.get('epoch') != proof.get('selected_epoch')
            or type(state.get('checkpoint_eligible')) is not bool
            or state['checkpoint_eligible'] != proof.get('checkpoint_eligible')
            or any(state.get(key) != proof.get(key) for key in keys)):
        raise ValueError('Final verified V4 diagnostic checkpoint binding changed')


def admit(root):
    """Read no observation tensors until original jobs and all25 proofs finish.

    Benchmark performance is deliberately not parsed, even if the original
    navigation job completed; its fixed code, native outcome and status bind
    lifecycle only. This function never calls the final performance reporter.
    """
    from experiments.a2_waldo_landmark_belief.paths import sha, ROOT
    root = Path(root)
    identity_path = root/'navigation_service_identity.json'
    if sha(identity_path) != LIVE_IDENTITY_SHA256:
        raise ValueError('Original predeclared native identity bytes changed')
    identity = json.loads(identity_path.read_text())
    services = read_services()
    if not require_terminal_services(services, identity['units']):
        return None

    from experiments.a2_waldo_landmark_belief.reference_v4_confirmation_pipeline import admitted_training
    from experiments.a2_waldo_landmark_belief.reference_v4_navigation_sources import require_cpu_gates, cpu_signature
    from experiments.a2_waldo_landmark_belief.reference_v4_navigation_pipeline import (
        predecessor_outcome, POST_UNIT, NAV_UNIT, ADMITTED, REJECTED, WRAPPER,
    )
    require_cpu_gates(root)
    if json.loads((root/'navigation_pipeline_signature.json').read_text()) != cpu_signature(root):
        raise ValueError('Original navigation queue sources changed')
    confirmation_identity = json.loads((root/'confirmation_service_identity.json').read_text())
    post = json.loads((root/'post_pipeline_status.json').read_text())
    outcome = predecessor_outcome(services[POST_UNIT], post, confirmation_identity)
    nav = json.loads((root/'navigation_pipeline_status.json').read_text())
    permitted = ({'full_benchmarks_and_report_complete', 'real_pilot_rejected_no_benchmark'}
                 if outcome == ADMITTED else {'internal_confirmation_not_admitted'} if outcome in REJECTED else set())
    if (nav.get('state') not in permitted or nav.get('phase') != 'finished'
            or nav.get('confirmation_outcome') != outcome
            or nav.get('native_main_pid') != int(identity['units'][NAV_UNIT]['MainPID'])
            or nav.get('systemd_invocation_id') != services[NAV_UNIT]['InvocationID']
            or nav.get('execution_script_sha256') != sha(ROOT/WRAPPER)
            or nav.get('signature_sha256') != sha(root/'navigation_pipeline_signature.json')
            or nav.get('full_benchmark_completed') is not (nav.get('state') == 'full_benchmarks_and_report_complete')):
        raise ValueError('Original navigation lifecycle not fully concluded')
    proof = admitted_training(root)  # Rehash all25 artifacts and actual curves.
    checkpoint = root/'training/best.pt'
    if proof['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('Selected checkpoint differs from verified training')
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint_header(state, proof)
    return dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha(checkpoint),
                selected_epoch=proof['selected_epoch'], checkpoint_eligible=proof['checkpoint_eligible'],
                roles_sha256=proof['roles_sha256'], feature_manifest_sha256=proof['feature_manifest_sha256'],
                verification_sha256=sha(root/'training/artifact_verification.json'),
                curve_manifest_sha256=sha(root/'training/curve_manifest.json'),
                native_identity_sha256=LIVE_IDENTITY_SHA256,
                original_navigation_state=nav['state'], original_confirmation_outcome=outcome,
                diagnostic_only=True, no_runtime_enablement=True)
