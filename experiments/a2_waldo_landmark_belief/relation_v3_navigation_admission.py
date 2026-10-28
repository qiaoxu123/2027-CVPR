"""Root-bound committed V3 calibration admission, before any device/runtime use.

The numerical require_candidate precheck alone is deliberately insufficient.
FreshConfirmation validates the complete eligible25-epoch proof and all source
observations; analysis_signature and validate_existing bind the unique choice,
roles, current implementations, seals and packet manifests. An absent candidate
commit is rejected before those validators can repair/create a commit sidecar.
"""
import json
from pathlib import Path
from .paths import sha
from .collected_snapshot import BASE
from .relation_v3_calibration_data import FreshConfirmation
from .relation_v3_calibrate import analysis_signature,validate_existing
from .relation_v3_navigation_protocol import require_candidate


def committed_candidate(root):
    directory=Path(root)/'calibration';path=directory/'candidate_calibration.json'
    actual=json.loads((directory/'candidate_calibration.sha.json').read_text())
    expected=dict(candidate_sha256=sha(path),analysis_signature_sha256=sha(directory/'analysis_signature.json'),
                  fit_decision_sha256=sha(directory/'fit_decision.json'))
    if actual!=expected:raise ValueError('Committed candidate/analysis/FIT choice changed')
    return json.loads(path.read_text()),actual


def load_candidate(root=BASE/'revision_v3'):
    root=Path(root);candidate,bound=committed_candidate(root)
    source=FreshConfirmation(root)
    analysis=analysis_signature(source)
    verified=validate_existing(source,analysis)
    retained,after=committed_candidate(root)
    if verified!=candidate or retained!=candidate or after!=bound:
        raise ValueError('Unique committed calibration changed during navigation admission')
    decision=json.loads((root/'calibration/fit_decision.json').read_text())
    require_candidate(candidate,decision)
    return candidate,source.state
