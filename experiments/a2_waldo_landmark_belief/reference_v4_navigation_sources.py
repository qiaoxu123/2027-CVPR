"""V4 navigation implementation, original episode and combined-trace binding."""
import json
import os
import re
import shutil
import subprocess
import sys
import torch
from pathlib import Path
from .paths import ROOT,sha,save
from .collected_snapshot import BASE
from .reference_v4_calibration_data import calibration_code_hashes
from .reference_v4_navigation_protocol import prepare_pilot
from .reference_v4_navigation_population import source_entries
from .reference_v4_confirmation_preparation import require_cpu_gates as require_post_cpu_gates
from .revision_eval_protocol import require_population,FULL_COUNTS


def code_hashes():
    names=('reference_v4_navigation_pipeline.py','reference_v4_navigation.py','reference_v4_navigation_sources.py','reference_v4_navigation_freeze.py',
        'reference_v4_navigation_admission.py','reference_v4_navigation_protocol.py','reference_v4_navigation_population.py',
        'relation_v3_navigation_trace.py','relation_v3_navigation_protocol.py','relation_v3_navigation_sources.py',
        'relation_v3_navigation_freeze.py','reference_v4_results.py','relation_v3_results.py','relation_v3_reporting.py','revision_evaluation.py','revision_eval_protocol.py','revision_freeze.py','revision_paired_stats.py')
    return dict(calibration_code_hashes(),**{n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names},
                **{'scripts/a2_reference_v4_remote_navigation.sh':sha(ROOT/'scripts/a2_reference_v4_remote_navigation.sh')})


def test_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for folder in ('tests_navigation_v3','tests_navigation_v4','tests_reporting_v3','tests_reporting_v4','tests_handoff_v4') for p in sorted((ROOT/folder).rglob('*.py'))}


def cpu_signature(root):
    root=Path(root)
    return dict(implementation=code_hashes(),test_source_sha256=test_hashes(),
        pilot_population_sha256=sha(root/'navigation_pilot_population.json'),
        protocol_sha256=sha(root/'navigation_protocol.json'),
        posttraining_cpu_gate_sha256=sha(root/'posttraining_cpu_gates.json'),
        confirmation_identity_sha256=sha(root/'confirmation_service_identity.json'),
        confirmation_signature_sha256=sha(root/'post_pipeline_signature.json'))


def require_cpu_gates(root=None):
    root=Path(root) if root is not None else BASE/'revision_v4'
    proof=json.loads((root/'navigation_cpu_gates.json').read_text())
    if (proof.get('status')!='v4_real_navigation_CPU_gates_passed'
        or any(proof.get(key)!=value for key,value in cpu_signature(root).items())
        or proof.get('test_log_sha256')!=sha(root/'navigation_cpu_tests.log')
        or proof.get('no_GPU_used') is not True or proof.get('no_navigation_observations') is not True):
        raise ValueError('Current actual-log-bound V4 real-navigation CPU proof required')
    require_post_cpu_gates(root)
    return proof


def cpu_gates(*,root=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized() or torch.cuda.device_count()!=0:
        raise ValueError('Navigation CPU gates require CUDA_VISIBLE_DEVICES empty')
    root=Path(root) if root is not None else BASE/'revision_v4'
    require_post_cpu_gates(root);prepare_pilot(root)
    signature=cpu_signature(root);path=root/'navigation_cpu_gates.json';log=root/'navigation_cpu_tests.log'
    if path.exists():
        try:
            proof=require_cpu_gates(root);print('Retained current V4 navigation CPU proof',flush=True);return proof
        except (ValueError,KeyError,FileNotFoundError):
            pass
    started=any(any((root/folder).rglob('signature.json')) for folder in ('pilot','evaluation'))
    if (root/'frozen.json').exists() or started:
        raise ValueError('Cannot replace navigation CPU proof after execution or final freeze')
    archive=root/'navigation_cpu_archive'
    for old in (path,log):
        if old.exists():
            archive.mkdir(exist_ok=True);shutil.copyfile(old,archive/(old.stem+'_'+sha(old)+old.suffix))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests','tests_confirmation_v3','tests_confirmation_v4',
            'tests_navigation_v3','tests_navigation_v4','tests_reporting_v3','tests_reporting_v4','tests_handoff_v4'],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    summary=re.search(r'\d+ passed[^\n]*',log.read_text())
    if result.returncode or summary is None or signature!=cpu_signature(root) or torch.cuda.is_initialized():
        raise ValueError('V4 navigation CPU tests failed or sources changed')
    proof=dict(signature,status='v4_real_navigation_CPU_gates_passed',test_log_sha256=sha(log),
        unit_test_summary=summary.group(),no_GPU_used=True,no_navigation_observations=True)
    save(proof,path);require_cpu_gates(root)
    print(json.dumps(dict(status=proof['status'],unit_test_summary=proof['unit_test_summary'])),flush=True)
    return proof


def check_combined(path,rows):
    with Path(path).open() as stream:
        for row in rows:
            line=next(stream,None)
            if line is None or json.loads(line)!=row:raise ValueError('Combined navigation trace is incomplete or changed')
        if next(stream,None) is not None:raise ValueError('Extra combined navigation trace episodes')


if __name__=='__main__':
    cpu_gates()
