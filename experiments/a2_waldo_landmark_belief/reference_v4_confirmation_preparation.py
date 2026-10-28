"""CPU-only preparation for the separate V4 independent-confirmation stage."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import torch

from .paths import ROOT,sha,save
from .reference_overlay_context import BASE
from .reference_v4_train import code_hashes as training_code_hashes
from .reference_v4_preparation import test_hashes as training_test_hashes
from .reference_v4_calibration_data import calibration_code_hashes


def post_test_hashes():
    return {str(path.relative_to(ROOT)):sha(path)
            for name in ('tests_confirmation_v3','tests_confirmation_v4')
            for path in sorted((ROOT/name).rglob('*.py'))}


def cpu_signature(root):
    root=Path(root)
    return dict(implementation=calibration_code_hashes(),post_test_source_sha256=post_test_hashes(),
        training_code_sha256=training_code_hashes(),training_test_source_sha256=training_test_hashes(),
        training_cpu_gate_sha256=sha(root/'cpu_gates.json'),
        roles_sha256=sha(BASE/'revision_v3/roles.json'),sealed_verify_sha256=sha(BASE/'revision_v3/sealed_verify130.json'),
        source_manifest_sha256=sha(BASE/'source_manifest.json'),
        population_sha256=sha(root/'confirmation_population.json'),protocol_sha256=sha(root/'confirmation_protocol.json'))


def require_cpu_gates(root=None):
    root=Path(root) if root is not None else BASE/'revision_v4'
    proof=json.loads((root/'posttraining_cpu_gates.json').read_text());expected=cpu_signature(root)
    if (proof.get('status')!='v4_posttraining_CPU_gates_passed'
        or any(proof.get(key)!=value for key,value in expected.items())
        or proof.get('no_CUDA_used') is not True or proof.get('no_new_confirmation_data_consumed') is not True
        or proof.get('test_log_sha256')!=sha(root/'posttraining_cpu/cpu_tests.log')):
        raise ValueError('Current actual-log-bound V4 post-training CPU gate required')
    return proof


def cpu_gates():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized() or torch.cuda.device_count()!=0:
        raise ValueError('V4 post-training CPU gates require CUDA_VISIBLE_DEVICES empty')
    root=BASE/'revision_v4';signature=cpu_signature(root);path=root/'posttraining_cpu_gates.json'
    if path.exists():
        try:
            proof=require_cpu_gates(root);print(json.dumps(dict(status='retained_current_V4_posttraining_CPU_gate',
                unit_test_summary=proof['unit_test_summary'])),flush=True);return proof
        except (ValueError,FileNotFoundError):
            pass
    if (root/'confirmation/signature.json').exists() or (root/'calibration/fit_decision.json').exists():
        raise ValueError('Cannot replace post-training source/proof after independent observation or FIT choice')
    directory=root/'posttraining_cpu';directory.mkdir(parents=True,exist_ok=True)
    log=directory/'cpu_tests.log';archive=directory/'archive'
    for old in (path,log):
        if old.exists():
            archive.mkdir(exist_ok=True);shutil.copyfile(old,archive/(old.stem+'_'+sha(old)+old.suffix))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests','tests_confirmation_v3','tests_confirmation_v4'],
                              cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    summary=re.search(r'\d+ passed[^\n]*',log.read_text())
    if result.returncode or summary is None or signature!=cpu_signature(root) or torch.cuda.is_initialized():
        raise ValueError('V4 post-training CPU tests failed or sources changed')
    proof=dict(signature,status='v4_posttraining_CPU_gates_passed',test_log_sha256=sha(log),
        unit_test_summary=summary.group(),no_CUDA_used=True,no_new_confirmation_data_consumed=True)
    save(proof,path);require_cpu_gates(root)
    print(json.dumps(dict(status=proof['status'],unit_test_summary=proof['unit_test_summary'])),flush=True)
    return proof


if __name__=='__main__':
    cpu_gates()
