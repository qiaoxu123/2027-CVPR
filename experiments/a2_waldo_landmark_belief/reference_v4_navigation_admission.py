"""Committed V4-only calibration admission before runtime/device allocation."""
import json
from pathlib import Path

from .paths import sha
from .reference_overlay_context import BASE
from .reference_v4_calibration_data import FreshConfirmation,FEATURE_KEYS
from .reference_v4_calibrate import analysis_signature,validate_existing
from .reference_v4_navigation_protocol import require_candidate


def committed_candidate(root):
    directory=Path(root)/'calibration';path=directory/'candidate_calibration.json'
    bound=json.loads((directory/'candidate_calibration.sha.json').read_text())
    expected=dict(candidate_sha256=sha(path),analysis_signature_sha256=sha(directory/'analysis_signature.json'),
                  fit_decision_sha256=sha(directory/'fit_decision.json'))
    if bound!=expected:
        raise ValueError('Committed V4 candidate/analysis/FIT choice changed')
    return json.loads(path.read_text()),bound


def load_candidate(root=None):
    root=Path(root) if root is not None else BASE/'revision_v4'
    candidate,bound=committed_candidate(root)
    source=FreshConfirmation(root);analysis=analysis_signature(source)
    verified=validate_existing(source,analysis)
    retained,after=committed_candidate(root)
    if verified!=candidate or retained!=candidate or after!=bound:
        raise ValueError('Unique committed V4 calibration changed during navigation admission')
    if (candidate.get('checkpoint_sha256')!=source.checkpoint_sha256
        or any(candidate.get(key)!=source.state.get(key) or key not in source.state for key in FEATURE_KEYS)):
        raise ValueError('V4 navigation candidate belongs to another model/feature lineage')
    decision=json.loads((root/'calibration/fit_decision.json').read_text())
    require_candidate(candidate,decision)
    return candidate,source.state
