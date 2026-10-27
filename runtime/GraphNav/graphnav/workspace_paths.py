"""Resolve legacy experiment paths without rewriting historical provenance."""
from pathlib import Path
import os

PROJECT = next(p for p in Path(__file__).resolve().parents if p.name == '2027-CVPR')
WORKSPACE = PROJECT.parent
ALIASES = {
    '2027-CVPR-relational-visual-belief': PROJECT / 'GraphNav',
    'SBFNav-CityNav-Reproduction': PROJECT / 'Baselines/SBFNav',
    **{name: PROJECT / 'Ablations/GraphNav' / name for name in (
        '2027-CVPR-static-geometry', '2027-CVPR-full-token-rgb',
        '2027-CVPR-bidirectional', '2027-CVPR-closedloop-sbf-parity')},
    'Baselines': PROJECT / 'Baselines',
    'Ablations': PROJECT / 'Ablations',
}

def current_paths(value):
    """Translate absolute path fields in a freshly loaded JSON value in memory."""
    if isinstance(value, dict):
        return {key: current_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [current_paths(item) for item in value]
    # Host relocation: preserve historical records and translate only on read.
    if isinstance(value, str):
        for legacy_root in ('/home/rental/20260922_1/Workspace', '/home/20260922_1/Workspace'):
            if value == legacy_root or value.startswith(legacy_root + '/'):
                value = str(WORKSPACE) + value[len(legacy_root):]
                break
    if isinstance(value, str) and value.startswith(str(WORKSPACE) + '/'):
        normalized = Path(os.path.normpath(value))
        for name, destination in ALIASES.items():
            for source in (WORKSPACE / name, PROJECT / name):
                if source == destination:
                    continue
                if normalized == source or source in normalized.parents:
                    return str(destination / normalized.relative_to(source))
    return value
