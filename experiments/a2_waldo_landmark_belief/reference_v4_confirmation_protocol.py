"""V4 independent inventory: immutable metadata, no heldout tensor decoding."""
from copy import deepcopy
import json
import re

from .paths import ROOT, sha, save
from .reference_overlay_context import BASE
from .relation_v3_confirmation_protocol import (
    COUNTS, CONFIG as V3_CONFIG, build_population, immutable, verify_opaque_seal,
)


DIRECTORY = BASE/'revision_v4'
BINDING_KEYS = ('code_sha256','roles_sha256','feature_manifest_sha256',
                'feature_signature_sha256','feature_build_status_sha256','initial_checkpoint_sha256')
CONFIG = deepcopy(V3_CONFIG)
CONFIG.update(version='reference_v4_independent_confirmation',
              source='Original CityNav TRAIN only; complete frozen-A0 closed-loop horizon20, V4 shadowOFF',
              full_training_required='All20ranking+5refinement and all25 verified; eligible V4 realDEV gate',
              public_features='Actual V4 connected public roads and occurrence mentions; no V3 prediction reuse')


def protocol_metadata(roles, source, seal, *, binding, source_sha256, seal_sha256, expected=COUNTS):
    """Revalidate predeclared identities and attach complete V4 feature lineage."""
    def digest(value):
        return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None
    if not isinstance(binding,dict) or any(key not in binding for key in BINDING_KEYS):
        raise ValueError('Complete V4 source/feature/initial-state binding required')
    codes=binding['code_sha256']
    if (not isinstance(codes,dict) or not codes
        or any(not isinstance(k,str) or not k or not digest(v) for k,v in codes.items())
        or any(not digest(binding[k]) for k in BINDING_KEYS if k!='code_sha256')
        or not digest(source_sha256) or not digest(seal_sha256)):
        raise ValueError('Malformed V4 source/feature binding')
    population=build_population(roles,source,seal,expected=expected)
    return dict(population,**deepcopy({key:binding[key] for key in BINDING_KEYS}),
                source_manifest_sha256=source_sha256,sealed_verify_sha256=seal_sha256)


def prepare():
    # This only fixes metadata and hashes opaque bytes. checked_state admission
    # in the collector is separately required before any new observations.
    from .reference_v4_preparation import current_binding
    binding=current_binding();old=BASE/'revision_v3'
    role_path=old/'roles.json';seal_path=old/'sealed_verify130.json'
    source_path=BASE/'source_manifest.json'
    roles=json.loads(role_path.read_text());seal=json.loads(seal_path.read_text())
    source=json.loads(source_path.read_text())
    if (sha(role_path)!=binding['roles_sha256']
        or roles['lineage']['original_source_sha256']!=sha(source_path)):
        raise ValueError('Frozen V4 role/source identity changed')
    published=json.loads((ROOT/'reports/a2_execution_20261027/RELATION_V3_VERIFY_SEAL.json').read_text())
    if (published['seal_sha256']!=sha(seal_path)
        or seal['sealing_code_sha256']!=sha(ROOT/'scripts/a2_seal_unscored_verify.py')):
        raise ValueError('Original opaque holdout seal changed')
    verify_opaque_seal(seal)
    value=protocol_metadata(roles,source,seal,binding=binding,
                            source_sha256=sha(source_path),seal_sha256=sha(seal_path))
    previous=json.loads((old/'confirmation_population.json').read_text())
    if any(previous[key]!=value[key] for key in ('fresh_entries','sealed_entries','fresh_folds','target_counts')):
        raise ValueError('V4 must retain the original predeclared independent queries')
    value['preparation_code_sha256']=sha(__file__)
    immutable(value,DIRECTORY/'confirmation_population.json')
    immutable(CONFIG,DIRECTORY/'confirmation_protocol.json')
    report=dict(status='V4_confirmation_metadata_fixed_without_observations_or_heldout_tensor_decode',
                population_sha256=sha(DIRECTORY/'confirmation_population.json'),
                protocol_sha256=sha(DIRECTORY/'confirmation_protocol.json'),
                roles_sha256=binding['roles_sha256'],feature_manifest_sha256=binding['feature_manifest_sha256'],
                fresh_targets=163,fit_targets=84,verify_targets=209,sealed_tensor_deserializations=0,
                new_observations_collected=False,training_checkpoint_not_selected_here=True)
    save(report,DIRECTORY/'confirmation_preparation.json')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    prepare()
