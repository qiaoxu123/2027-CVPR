"""Durable CPU-only diagnostic after the entire original experiment concludes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import torch

from experiments.a2_waldo_landmark_belief.paths import ROOT, sha
from experiments.a2_waldo_landmark_belief.reference_overlay_context import BASE
from .admission import ObservationUnavailable
from .export import export_packets, source_binding
from .report import assess_and_report


def _cpu_environment():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Evidence-quality diagnostic must remain CPU-only')


def validate_cpu_proof(proof, signature, log):
    text = Path(log).read_text()
    summary = proof.get('unit_test_summary', '')
    if (proof.get('status') != 'evidence_quality_actual_CPU_tests_passed'
            or proof.get('signature') != signature or proof.get('returncode') != 0
            or proof.get('no_gpu_used') is not True
            or proof.get('test_log_sha256') != sha(log)
            or re.match(r'[1-9][0-9]* passed', summary) is None or summary not in text):
        raise ValueError('Current actual-log-bound diagnostic CPU proof required')


def _save(value, path):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def require_cpu(directory):
    directory = Path(directory)
    proof = json.loads((directory/'cpu_gates.json').read_text())
    validate_cpu_proof(proof, source_binding(None), directory/'cpu_tests.log')
    return proof


def prepare_cpu(directory):
    _cpu_environment()
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path, log = directory/'cpu_gates.json', directory/'cpu_tests.log'
    if path.exists():
        try:
            return require_cpu(directory)
        except (ValueError, KeyError, FileNotFoundError):
            pass
    if (directory/'fit').exists() or (directory/'report').exists() or any((directory/'packets').rglob('manifest.json')):
        raise ValueError('Cannot replace CPU proof after diagnostic data/model artifacts exist')
    archive = directory/'cpu_archive'
    for old in (path, log):
        if old.exists():
            archive.mkdir(exist_ok=True)
            shutil.copyfile(old, archive/(old.stem+'_'+sha(old)+old.suffix))
    signature = source_binding(None)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with log.open('w') as stream:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'diagnostics_tests/evidence_quality'],
                                cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    summary = re.search(r'[1-9][0-9]* passed[^\n]*', log.read_text())
    if result.returncode or summary is None or source_binding(None) != signature:
        raise ValueError('Diagnostic CPU tests failed or sources changed while testing')
    proof = dict(status='evidence_quality_actual_CPU_tests_passed', signature=signature,
                 test_log_sha256=sha(log), unit_test_summary=summary.group(), returncode=0, no_gpu_used=True)
    _save(proof, path)
    require_cpu(directory)
    return proof


def run_once(root, directory):
    _cpu_environment()
    original = source_binding(None)
    exported = export_packets(root, Path(directory)/'packets')
    if exported is None:
        return None
    if exported['binding'] != source_binding(exported['binding']['admission']):
        raise ValueError('Exported diagnostic belongs to different source/actor')
    result = assess_and_report(directory, exported)
    if source_binding(None) != original or exported['binding'] != source_binding(exported['binding']['admission']):
        raise ValueError('Diagnostic sources changed before completion')
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-cpu', action='store_true')
    mode.add_argument('--wait', action='store_true')
    mode.add_argument('--once', action='store_true')
    parser.add_argument('--root', type=Path, default=BASE/'revision_v4')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.output or args.root/'diagnostics/evidence_quality'
    _cpu_environment(); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_cpu:
            print(json.dumps(prepare_cpu(directory)), flush=True)
            return
        expected = require_cpu(directory)['signature']
        def status(state, **extra):
            value = dict(state=state, pid=os.getpid(), timestamp=time.time(), objective_achieved=False,
                         no_runtime_enablement=True, no_gpu_used=True, **extra)
            _save(value, directory/'pipeline_status.json')
            print(json.dumps(value), flush=True)
        try:
            while True:
                if require_cpu(directory)['signature'] != expected:
                    raise ValueError('Queued diagnostic source snapshot changed')
                try:
                    result = run_once(args.root, directory)
                except ObservationUnavailable as error:
                    status('original_service_observation_retry', reason=str(error))
                    if not args.wait:
                        return
                    time.sleep(300)
                    continue
                if result is not None:
                    status('complete_internal_cpu_diagnostic', result_status=result['status'],
                           promising_for_formal_iteration=result['promising_for_formal_iteration'])
                    return
                status('waiting_for_original_experiment_to_finish')
                if not args.wait:
                    return
                time.sleep(300)
        except Exception as error:
            status('failed', reason=str(error))
            raise


if __name__ == '__main__':
    main()
