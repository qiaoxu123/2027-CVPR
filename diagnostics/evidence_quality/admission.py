"""Observe the original three units without starting, stopping or changing jobs."""
import os
import subprocess


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
