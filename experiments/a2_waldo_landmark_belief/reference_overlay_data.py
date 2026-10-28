"""Complete TRAIN/DEV derived reference views over immutable source caches."""
import json
from pathlib import Path

from .paths import sha
from .reference_overlay_context import DEFAULT_OUTPUT, feature_signature
from .reference_overlay import read_reference_overlay, apply_reference_overlay, verify_overlay_population, overlay_filename
from .reference_training import build_reference_training_row
from .relation_v3_data import RelationV3Views, read_training_source


class ReferenceOverlayViews:
    def __init__(self, role, *, directory=DEFAULT_OUTPUT):
        # Reject a sealed role before constructing any source dataset/reading tensors.
        if role not in ('train','dev'):
            raise ValueError('Only TRAIN/DEV reference overlay datasets are permitted')
        self.parent = RelationV3Views(role)
        self.directory = Path(directory)
        self.signature_sha256 = sha(self.directory/'signature.json')
        signature = json.loads((self.directory/'signature.json').read_text())
        if signature != feature_signature():
            raise ValueError('Reference feature signature changed')
        self.manifest_path = self.directory/'manifest.json'
        self.manifest_sha256 = sha(self.manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get('status') != 'all_train_dev_reference_overlays_complete' or manifest['signature_sha256'] != self.signature_sha256:
            raise ValueError('Reference cache is incomplete or foreign')
        records = manifest['entries']
        expected = [entry for entry in self.parent.roles['rows'] if entry['role'] in ('train','dev')]
        verify_overlay_population(records,expected)
        by_name = {overlay_filename(entry):entry for entry in records}
        self.entries = [by_name[overlay_filename(entry)] for entry in self.parent.entries]
        self.role = role

    def __len__(self):
        return len(self.entries)

    def __getitem__(self,index):
        entry = self.entries[index]
        raw = read_training_source(entry,self.role,self.parent.allowed,self.parent.contexts[entry['source_kind']]['signature'])
        overlay = read_reference_overlay(self.directory/'views',entry,role=self.role,allowed_targets=self.parent.allowed)
        rebuilt = apply_reference_overlay(raw,overlay,entry,self.signature_sha256)
        result = build_reference_training_row(rebuilt,target_kind=self.parent.target_kinds[(entry['map_name'],entry['object_id'])],
                                              real=entry['source_kind']!='teacher_snapshot')
        result['target_instance'] = [entry['map_name'],entry['object_id']]
        return result
