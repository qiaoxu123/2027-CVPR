"""Atomic, exact-population CPU packet storage, separate TRAIN/DEV files."""
import json
from numbers import Integral
from pathlib import Path
import tempfile

import numpy as np

from experiments.a2_waldo_landmark_belief.paths import sha


COUNTS = dict(train=2025, dev=1029)
SOURCES = {'revision_v1_A0_closedloop', 'revision_v2_A0_closedloop'}
ARRAY_KEYS = {'features', 'available', 'before', 'after', 'outcomes'}


def validate_identities(role, identities):
    """Metadata checks precede any tensor deserialization."""
    if role not in COUNTS or len(identities) != COUNTS[role]:
        raise ValueError('Exact autonomous TRAIN/internalDEV population required')
    seen = set()
    for row in identities:
        if row.get('role') != role or row.get('source_kind') not in SOURCES:
            raise ValueError('Foreign role or source in diagnostic observations')
        for field in ('episode', 'step', 'object_id'):
            if (not isinstance(row.get(field), Integral) or isinstance(row[field], bool)
                    or row[field] < 0):
                raise ValueError('Invalid diagnostic observation identity')
        if not isinstance(row.get('map_name'), str) or not row['map_name'] or '\0' in row['map_name']:
            raise ValueError('Invalid public map identity')
        name, directory = row.get('filename'), row.get('directory')
        if (not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..')
                or not isinstance(directory, str) or not Path(directory).is_absolute()):
            raise ValueError('Unsafe diagnostic source path')
        for field in ('sha256', 'overlay_sha256'):
            digest = row.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Missing source/overlay content binding')
        key = (str(Path(directory).resolve()), name)
        if key in seen:
            raise ValueError('Duplicated diagnostic observation')
        seen.add(key)


def validate_arrays(role, arrays):
    if role not in COUNTS or set(arrays) != ARRAY_KEYS:
        raise ValueError('Unexpected diagnostic packet schema')
    n = COUNTS[role]
    for name, value in arrays.items():
        shape = (n, 22) if name == 'features' else (n,)
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype.kind not in 'fiub' or not np.isfinite(value).all()):
            raise ValueError('Invalid diagnostic array axes or values')
    x, available = arrays['features'], arrays['available']
    before, after, outcomes = arrays['before'], arrays['after'], arrays['outcomes']
    if (x.dtype != np.float32 or available.dtype != np.bool_
            or before.dtype.kind != 'f' or after.dtype.kind != 'f'
            or outcomes.dtype.kind not in 'iu' or (before < 0).any() or (after < 0).any()
            or not np.array_equal(x[:, 14], available.astype(np.float32))
            or not np.array_equal(before[~available], after[~available])
            or (x[~available, 15:] != 0).any()):
        raise ValueError('Inconsistent public availability or private distances')
    derived = ((before > 20) & (after <= 20)).astype(np.int64)
    derived += 2 * ((before <= 20) & (after > 20)).astype(np.int64)
    if not np.array_equal(derived, outcomes):
        raise ValueError('Outcome labels disagree with actual fixed-action distances')


def write_packets(directory, *, role, identities, arrays, binding):
    validate_identities(role, identities)
    validate_arrays(role, arrays)
    directory = Path(directory)
    destination = directory/role
    if destination.exists():
        raise ValueError('Diagnostic packets already exist; never overwrite')
    directory.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=f'.{role}-pending-', dir=directory))
    (pending/'identities.json').write_text(json.dumps(identities, indent=2) + '\n')
    with (pending/'packets.npz').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    manifest = dict(status='complete_cpu_diagnostic_packets', role=role, views=COUNTS[role],
                    binding=binding, files_sha256={name:sha(pending/name)
                        for name in ('identities.json', 'packets.npz')})
    (pending/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pending.rename(destination)
    return manifest


def load_packets(directory, *, role, expected_entries, binding):
    validate_identities(role, expected_entries)  # Refuse sealed roles before IO.
    directory = Path(directory)/role
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest.get('status') != 'complete_cpu_diagnostic_packets'
            or manifest.get('role') != role or manifest.get('views') != COUNTS[role]
            or manifest.get('binding') != binding
            or set(manifest.get('files_sha256', {})) != {'identities.json', 'packets.npz'}
            or any(sha(directory/name) != digest for name,digest in manifest['files_sha256'].items())):
        raise ValueError('Diagnostic cache commitment or source/model binding changed')
    identities = json.loads((directory/'identities.json').read_text())
    validate_identities(role, identities)
    if identities != expected_entries:
        raise ValueError('Diagnostic observation order/source changed')
    with np.load(directory/'packets.npz', allow_pickle=False) as archive:
        arrays = {key:archive[key].copy() for key in archive.files}
    validate_arrays(role, arrays)
    return dict(identities=identities, arrays=arrays, manifest=manifest,
                manifest_sha256=sha(directory/'manifest.json'))
