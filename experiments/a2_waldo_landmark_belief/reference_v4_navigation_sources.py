"""V4 navigation implementation, original episode and combined-trace binding."""
import json
from pathlib import Path
from .paths import ROOT,sha
from .collected_snapshot import BASE
from .reference_v4_calibration_data import calibration_code_hashes
from .reference_v4_navigation_protocol import prepare_pilot
from .reference_v4_navigation_population import source_entries
from .revision_eval_protocol import require_population,FULL_COUNTS


def code_hashes():
    names=('reference_v4_navigation.py','reference_v4_navigation_sources.py','reference_v4_navigation_freeze.py',
        'reference_v4_navigation_admission.py','reference_v4_navigation_protocol.py','reference_v4_navigation_population.py',
        'relation_v3_navigation_trace.py','relation_v3_navigation_protocol.py','relation_v3_navigation_sources.py',
        'relation_v3_navigation_freeze.py','revision_evaluation.py','revision_eval_protocol.py','revision_freeze.py','revision_paired_stats.py')
    return dict(calibration_code_hashes(),**{n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names})


def test_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for folder in ('tests_navigation_v3','tests_navigation_v4') for p in sorted((ROOT/folder).rglob('*.py'))}


def require_cpu_gates(root=BASE/'revision_v4'):
    root=Path(root);proof=json.loads((root/'navigation_cpu_gates.json').read_text())
    if (proof['status']!='v4_real_navigation_CPU_gates_passed' or proof['implementation']!=code_hashes()
        or proof['test_source_sha256']!=test_hashes() or proof['test_log_sha256']!=sha(root/'navigation_cpu_tests.log')
        or proof['pilot_population_sha256']!=sha(root/'navigation_pilot_population.json')
        or proof['protocol_sha256']!=sha(root/'navigation_protocol.json')):
        raise ValueError('Current real-navigation CPU proof required')
    from .reference_v4_confirmation_preparation import require_cpu_gates as require_post_cpu_gates
    require_post_cpu_gates(root)
    return proof


def check_combined(path,rows):
    with Path(path).open() as stream:
        for row in rows:
            line=next(stream,None)
            if line is None or json.loads(line)!=row:raise ValueError('Combined navigation trace is incomplete or changed')
        if next(stream,None) is not None:raise ValueError('Extra combined navigation trace episodes')
