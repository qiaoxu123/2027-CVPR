import hashlib
import pytest
from experiments.a2_waldo_landmark_belief.gpu_execution_policy import classify,validate_policy


def policy():
    return dict(mode='approved_background_services',approval_reference='Explicit user approval fixture',services=[
        dict(uid=1000,process_name='/voice/python',cmdline_sha256=hashlib.sha256(b'/voice/python\0service.py\0').hexdigest(),maximum_instances=1)])


def identity():return dict(uid=1000,process_name='/voice/python',cmdline_sha256=policy()['services'][0]['cmdline_sha256'])


def test_default_exclusive_policy_still_blocks_every_external_GPU_process():
    assert classify([10,11],dict(mode='exclusive',services=[],approval_reference=None),lambda p:identity(),self_pid=10)==[10,11]
    assert classify([10,11],dict(mode='exclusive',services=[],approval_reference=None),lambda p:identity(),self_pid=10,allow_self=True)==[11]


def test_only_exact_approved_service_is_exempt_and_A2_still_blocks():
    ids={10:identity(),11:dict(uid=1004,process_name='/a2/python',cmdline_sha256='0'*64)}
    assert classify([10,11],policy(),ids.get,self_pid=99)==[11]


@pytest.mark.parametrize('field,value',[('uid',1004),('process_name','/other/python'),('cmdline_sha256','f'*64)])
def test_changed_service_identity_is_not_authorized(field,value):
    record=identity();record[field]=value
    assert classify([10],policy(),lambda p:record,self_pid=99)==[10]


def test_duplicate_service_and_unreadable_process_are_blocked():
    assert classify([10,11],policy(),lambda p:identity(),self_pid=99)==[11]
    assert classify([10],policy(),lambda p:None,self_pid=99)==[10]


@pytest.mark.parametrize('bad',[dict(mode='anything',services=[]),dict(mode='approved_background_services',approval_reference='',services=[]),
    dict(mode='exclusive',services=[dict(uid=1)],approval_reference=None)])
def test_absent_approval_or_invalid_policy_cannot_allow_a_process(bad):
    with pytest.raises(ValueError):validate_policy(bad)
