import hashlib,json
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


def test_an_untrusted_PATH_shadow_cannot_hide_busy_processes(tmp_path,monkeypatch):
    from experiments.a2_waldo_landmark_belief import gpu_execution_policy as module
    fake=tmp_path/'bin';fake.mkdir();shadow=fake/'nvidia-smi'
    shadow.write_text('#!/bin/sh\nexit 0\n');shadow.chmod(0o755)
    trusted=tmp_path/'trusted-nvidia-smi';trusted.write_text('#!/bin/sh\nprintf "1234\\n"\n');trusted.chmod(0o755)
    policy_path=tmp_path/'policy.json';policy_path.write_text(json.dumps(dict(mode='exclusive',services=[],approval_reference=None)))
    monkeypatch.setenv('PATH',str(fake));monkeypatch.setattr(module,'NVSMI',str(trusted),raising=False);monkeypatch.setattr(module,'POLICY',policy_path)
    assert module.blocking_gpu_pids()==[1234]


def test_shared_host_policy_stays_exclusive_on_the_other_machine():
    from experiments.a2_waldo_landmark_belief.gpu_execution_policy import effective_policy
    shared=dict(policy(),approved_host_machine_id_sha256='a'*64)
    assert classify([10],effective_policy(shared,'a'*64),lambda p:identity(),self_pid=99)==[]
    assert classify([10],effective_policy(shared,'b'*64),lambda p:identity(),self_pid=99)==[10]
    assert classify([10],effective_policy(shared,None),lambda p:identity(),self_pid=99)==[10]


@pytest.mark.parametrize('host',[1234,'invalid'])
def test_invalid_host_binding_is_rejected(host):
    with pytest.raises(ValueError):validate_policy(dict(policy(),approved_host_machine_id_sha256=host))
