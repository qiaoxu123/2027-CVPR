"""Frozen original A0/controller with separately admitted V4 public references."""
import json
from pathlib import Path

import numpy as np

from .paths import sha
from .policy import A2Policy
from .relation_v3_model import RelationV3Network
from .relation_v3_policy import RelationV3Policy
from .reference_overlay_context import BASE
from .reference_perception import ReferencePerception,bind_public_road_references
from .reference_v4_provenance import checked_state,validate_gate


class ReferenceV4Policy(A2Policy):
    # Same model-to-belief-to-original-controller operation as the tested V3
    # engine; only checkpoint admission and public reference extraction differ.
    _refine = RelationV3Policy._refine

    def __init__(self,directory,checkpoint,calibration=None,*,control='real',functional_only=False):
        state = checked_state(checkpoint,functional_only=functional_only)
        parameters = dict(enabled=False,minimum_gain=1.,maximum_harm=0.,temperature=1.)
        if calibration:
            parameters = json.loads(Path(calibration).read_text())
            if parameters['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('V4 gate/checkpoint mismatch')
            if functional_only:
                if parameters.get('scope') != 'TRAIN_functional_only' or parameters.get('gate_final') is not False:
                    raise ValueError('V4 functional override must be TRAIN-only and nonfinal')
            else:
                validate_gate(parameters,state)
        self.functional_episodes = None
        if functional_only:
            preflight = BASE/'revision_v4/preflight'
            signature = json.loads((preflight/'signature.json').read_text())
            public = preflight/'public_episodes.json'
            if (signature['initial_checkpoint_sha256'] != sha(checkpoint)
                or signature['code_sha256'] != state['code_sha256']
                or signature['roles_sha256'] != state['roles_sha256']
                or signature['feature_manifest_sha256'] != state['feature_manifest_sha256']
                or signature['public_episodes_sha256'] != sha(public)):
                raise ValueError('Pinned V4 functional TRAIN episodes required')
            self.functional_episodes = json.loads(public.read_text())
            if len(self.functional_episodes) != 32:
                raise ValueError('All32 V4 functional TRAIN episodes required')
        super().__init__(directory,collect=True,control=control)
        self.perception = ReferencePerception(self.perception)
        self.branch = RelationV3Network(**state['model_config']).to(self.device).eval().requires_grad_(False)
        self.branch.load_state_dict(state['model'],strict=True)
        self.v3_parameters = parameters

    def set_episode(self,episode,transform,catalog):
        if self.functional_episodes is not None:
            matched = any(record['map_name'] == episode.map_name and record['instruction'] == episode.instruction
                and np.array_equal(np.asarray(record['start_pose']),np.asarray(episode.start_pose))
                for record in self.functional_episodes)
            if not matched:
                raise ValueError('V4 functional policy is limited to pinned TRAIN episodes')
        super().set_episode(episode,transform,catalog)
        bind_public_road_references(self,episode,catalog)
