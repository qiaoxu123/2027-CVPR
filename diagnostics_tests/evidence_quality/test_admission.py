"""Do not confuse live, missing or restarted units with completed experiments."""
import importlib

import pytest


def api():
    return importlib.import_module('diagnostics.evidence_quality.admission')


def observations():
    names = ('graphnav-a2-reference-v4-pipeline-20261029.service',
             'graphnav-a2-reference-v4-confirmation-20261029.service',
             'graphnav-a2-reference-v4-navigation-20261029.service')
    services, identities = {}, {}
    for i, name in enumerate(names, 1):
        identities[name] = dict(MainPID=str(100+i), InvocationID=f'{i:032x}')
        services[name] = dict(LoadState='loaded', ActiveState='inactive', SubState='dead',
                              MainPID='0', ExecMainPID=str(100+i), ExecMainStatus='0',
                              Result='success', InvocationID=f'{i:032x}')
    return services, identities


def test_all_three_exact_executions_must_have_finished_successfully():
    services, identities = observations()
    assert api().require_terminal_services(services, identities) is True
    services[next(iter(services))].update(ActiveState='active', SubState='exited')
    assert api().require_terminal_services(services, identities) is True


def test_running_training_prevents_new_diagnostic():
    services, identities = observations(); name = next(iter(services))
    services[name].update(MainPID='101', ActiveState='active', SubState='running')
    assert api().require_terminal_services(services, identities) is False


@pytest.mark.parametrize('field,value', [('InvocationID','f'*32), ('ExecMainPID','999'),
                                       ('ExecMainStatus','1'), ('LoadState','not-found'),
                                       ('Result','exit-code')])
def test_missing_failed_or_replaced_execution_rejected(field, value):
    services, identities = observations(); services[next(iter(services))][field] = value
    with pytest.raises(ValueError):
        api().require_terminal_services(services, identities)


def test_downstream_waiter_still_blocks_after_training_finished():
    services, identities = observations(); name = list(services)[2]
    services[name].update(MainPID='103', ActiveState='active', SubState='running')
    assert api().require_terminal_services(services, identities) is False


def test_incomplete_observation_is_not_terminal_evidence():
    services, identities = observations(); services[next(iter(services))].pop('InvocationID')
    with pytest.raises(api().ObservationUnavailable):
        api().require_terminal_services(services, identities)


def test_service_output_requires_all_three_explicit_unit_identities():
    services, _ = observations()
    blocks = ['\n'.join([f'Id={name}', *(f'{k}={v}' for k,v in values.items())])
              for name, values in services.items()]
    assert api().parse_service_output('\n\n'.join(blocks)) == services
    with pytest.raises(api().ObservationUnavailable):
        api().parse_service_output('\n\n'.join(blocks[:2]))
