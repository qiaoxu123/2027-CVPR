"""CPU export only after original V4 completion; never an evaluator or policy."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.a2_waldo_landmark_belief.paths import ROOT, sha
from experiments.a2_waldo_landmark_belief.reference_overlay import BINDING_KEYS
from experiments.a2_waldo_landmark_belief.reference_overlay_data import ReferenceOverlayViews
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network
from .admission import admit
from .data import public_forward, private_outcomes
from .packets import COUNTS, validate_identities, write_packets, load_packets


def select_population(datasets, roles):
    """Verify exact ordered source membership before reading any observation."""
    if set(datasets) != set(COUNTS):
        raise ValueError('Both exact diagnostic roles required')
    result, targets = {}, {}
    for role in COUNTS:
        entries = [dict(e) for e in datasets[role].entries if e['source_kind'] != 'teacher_snapshot']
        expected = [e for e in roles['rows'] if e['role'] == role and e['source_kind'] != 'teacher_snapshot']
        validate_identities(role, entries)
        if ([{k:e[k] for k in BINDING_KEYS} for e in entries]
                != [{k:e[k] for k in BINDING_KEYS} for e in expected]):
            raise ValueError('Diagnostic source/order differs from frozen role inventory')
        result[role] = entries
        targets[role] = {(e['map_name'], e['object_id']) for e in entries}
    if targets['train'] & targets['dev']:
        raise ValueError('Diagnostic TRAIN/internalDEV target overlap')
    return result


def source_binding(admission):
    """Bind the isolated code/tests and predeclared design before fitting."""
    files = [*sorted(Path(__file__).parent.glob('*.py')),
             *sorted((ROOT/'diagnostics_tests/evidence_quality').glob('*.py')),
             ROOT/'docs/superpowers/specs/2026-10-29-a2-evidence-quality-diagnostic-design.md']
    return dict(admission=admission, implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
                scope='autonomous_TRAIN2025_internalDEV1029_CPU_only', version=1)


def export_packets(root, directory):
    """Return None while original jobs run; never allocate data/model early."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Diagnostic export requires explicitly hidden CUDA')
    root, directory = Path(root), Path(directory)
    admission = admit(root)
    if admission is None:
        return None
    binding = source_binding(admission)
    roles_path = root.parent/'revision_v3/roles.json'
    if sha(roles_path) != admission['roles_sha256']:
        raise ValueError('Admitted target roles changed')
    roles = json.loads(roles_path.read_text())
    datasets = {role:ReferenceOverlayViews(role) for role in COUNTS}
    entries = select_population(datasets, roles)
    if any(data.manifest_sha256 != admission['feature_manifest_sha256'] for data in datasets.values()):
        raise ValueError('Admitted feature manifest changed')
    checkpoint = Path(admission['checkpoint'])
    if sha(checkpoint) != admission['checkpoint_sha256']:
        raise ValueError('Admitted checkpoint changed before CPU inference')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = RelationV3Network(**state['model_config'])
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False); model.eval(); torch.set_num_threads(2)
    manifests = {}
    for role, dataset in datasets.items():
        if (directory/role).exists():
            manifests[role] = load_packets(directory, role=role, expected_entries=entries[role], binding=binding)['manifest']
            continue
        rows = {name:[] for name in ('features', 'available', 'before', 'after', 'outcomes')}
        actual_entries = []
        for index, entry in enumerate(dataset.entries):
            if entry['source_kind'] == 'teacher_snapshot':
                continue
            row = dataset[index]
            public = public_forward(model, row['public'])
            # Targets enter only the private outcome stage, never inference.
            private = private_outcomes(public, row['labels']['target_xy'])
            rows['features'].append(public['features'].numpy())
            rows['available'].append(bool(public['available']))
            rows['before'].append(private['before']); rows['after'].append(private['after'])
            rows['outcomes'].append(private['outcome']); actual_entries.append(dict(entry))
            if len(actual_entries) % 250 == 0:
                print(json.dumps(dict(role=role, exported=len(actual_entries), total=COUNTS[role])), flush=True)
        if actual_entries != entries[role] or source_binding(admission) != binding:
            raise ValueError('Diagnostic source population/code changed during export')
        if sha(checkpoint) != admission['checkpoint_sha256'] or sha(roles_path) != admission['roles_sha256']:
            raise ValueError('Admitted model/roles changed during export')
        arrays = dict(features=np.asarray(rows['features'], dtype=np.float32),
                      available=np.asarray(rows['available'], dtype=bool),
                      before=np.asarray(rows['before'], dtype=np.float64),
                      after=np.asarray(rows['after'], dtype=np.float64),
                      outcomes=np.asarray(rows['outcomes'], dtype=np.int64))
        manifests[role] = write_packets(directory, role=role, identities=actual_entries,
                                       arrays=arrays, binding=binding)
    # Reused packets take the fast branch too: verify commitments at exit.
    if (source_binding(admission) != binding or sha(checkpoint) != admission['checkpoint_sha256']
            or sha(roles_path) != admission['roles_sha256']):
        raise ValueError('Admitted diagnostic sources changed at export completion')
    return dict(binding=binding, manifests=manifests, entries=entries,
                no_gpu_used=True, diagnostic_only=True)
