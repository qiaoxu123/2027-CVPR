"""Fixed V4 pilot/full populations; metadata preparation consumes no performance."""
from copy import deepcopy
import json
from pathlib import Path

from .paths import ROOT,sha,save
from .reference_overlay_context import BASE
from .reference_v4_confirmation_protocol import protocol_metadata,immutable,BINDING_KEYS,verify_opaque_seal
from .reference_v4_confirmation_preparation import require_cpu_gates
from .reference_v4_preparation import current_binding
from .reference_v4_calibrate import validate_result_contract
from .relation_v3_gate_fit import validate_parameters
from .relation_v3_navigation_protocol import CONFIG as V3_CONFIG,build_pilot_population


CONFIG=deepcopy(V3_CONFIG)
CONFIG['version']='reference_v4_real_navigation'


def validate_navigation_config():
    if any(CONFIG.get(key)!=value for key,value in V3_CONFIG.items() if key!='version'):
        raise ValueError('V4 requires original64 pilot and full2470/2697 navigation configuration')


def require_candidate(candidate,decision):
    validate_result_contract(candidate,decision)
    if not candidate['enabled'] or not candidate['verify_gate']['passed']:
        raise ValueError('Independently verified nonzero V4 candidate required for real pilot')
    validate_parameters(candidate['fit_selected_parameters'])
    return candidate


def prepare_pilot(root=None):
    validate_navigation_config();root=Path(root) if root is not None else BASE/'revision_v4'
    require_cpu_gates(root);binding=current_binding()
    roles_path=BASE/'revision_v3/roles.json';seal_path=BASE/'revision_v3/sealed_verify130.json'
    source_path=BASE/'source_manifest.json'
    roles=json.loads(roles_path.read_text());seal=json.loads(seal_path.read_text());source=json.loads(source_path.read_text())
    if roles['lineage']['original_source_sha256']!=sha(source_path) or sha(roles_path)!=binding['roles_sha256']:
        raise ValueError('Original V4 navigation role/source lineage changed')
    verify_opaque_seal(seal)
    expected=protocol_metadata(roles,source,seal,binding=binding,source_sha256=sha(source_path),seal_sha256=sha(seal_path))
    expected['preparation_code_sha256']=sha(ROOT/'experiments/a2_waldo_landmark_belief/reference_v4_confirmation_protocol.py')
    confirmation=json.loads((root/'confirmation_population.json').read_text())
    if confirmation!=expected:
        raise ValueError('Complete feature-bound V4 confirmation population changed')
    pilot=build_pilot_population(roles,confirmation,source)
    signature=dict({key:binding[key] for key in BINDING_KEYS},
        confirmation_population_sha256=sha(root/'confirmation_population.json'),source_manifest_sha256=sha(source_path),
        sealed_verify_sha256=sha(seal_path),preparation_code_sha256=sha(__file__),configuration=CONFIG)
    immutable(dict(signature=signature,entries=pilot),root/'navigation_pilot_population.json')
    immutable(CONFIG,root/'navigation_protocol.json')
    report=dict(status='v4_fixed64_real_navigation_pilot_metadata_prepared',
        population_sha256=sha(root/'navigation_pilot_population.json'),protocol_sha256=sha(root/'navigation_protocol.json'),
        roles_sha256=binding['roles_sha256'],feature_manifest_sha256=binding['feature_manifest_sha256'],pilot_groups=len(pilot),
        tensor_deserializations=0,new_observations=False,comparative_performance_consumed=False,
        real_pilot_completed=False,full_benchmark_completed=False)
    save(report,root/'navigation_protocol_preparation.json')
    return report


if __name__=='__main__':
    print(json.dumps(prepare_pilot()),flush=True)
