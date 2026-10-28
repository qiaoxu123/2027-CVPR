"""V4 actual-inference packets and frozen-FIT barrier; no old prediction reuse."""
from collections import Counter
import json
from pathlib import Path
import torch
from .paths import ROOT,sha,save
from .collected_snapshot import BASE,verify_cache_origin
from .reference_v4_confirmation_protocol import CONFIG,immutable,protocol_metadata,BINDING_KEYS
from .reference_v4_confirmation import collection_code_hashes,collection_signature
from .reference_v4_provenance import checked_state
from .relation_v3_model import RelationV3Network
from .relation_v3_packet import make_packet as numerical_packet
from .relation_v3_calibration_data import read_fresh_row,validate_fit_cache as validate_numeric_fit_cache
from .relation_v3_gate_fit import validate_parameters
from .direct_risk_protocol import select_point
from .train import atomic_checkpoint


FEATURE_KEYS=('feature_manifest_sha256','feature_signature_sha256','feature_build_status_sha256')


def calibration_code_hashes():
    names=('reference_v4_calibration_data.py','relation_v3_calibration_data.py','relation_v3_gate_fit.py',
           'relation_v3_packet.py','direct_risk_calibrate.py','revision_calibrate.py')
    return dict(collection_code_hashes(),**{name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name) for name in names})


def make_packet(row,model):
    packet=numerical_packet(row,model)
    packet['inference_authority']='saved actual V4 GPU T1 scores and fixed.25 action; CPU integrity checked'
    return packet


def validate_fit_cache(root,*,binding):
    root=Path(root);analysis=json.loads((root/'calibration/analysis_signature.json').read_text())
    if any(key not in binding or analysis.get(key)!=binding[key] for key in BINDING_KEYS):
        raise ValueError('V4 FIT analysis feature/model binding changed')
    signature=json.loads((root/'calibration/packets_fresh_fit/signature.json').read_text())
    if (analysis.get('checkpoint_sha256')!=sha(root/'training/best.pt')
        or signature.get('checkpoint_sha256')!=analysis['checkpoint_sha256']):
        raise ValueError('V4 FIT packet checkpoint differs from admitted analysis model')
    if any(signature.get(key)!=binding[key] for key in FEATURE_KEYS):
        raise ValueError('V4 FIT packet feature binding changed')
    return validate_numeric_fit_cache(root)


def load_frozen_choice(root,checkpoint_sha256,roles_sha256,*,binding):
    root=Path(root);directory=root/'calibration';path=directory/'fit_decision.json';decision=json.loads(path.read_text())
    selected=decision['selected_parameters']
    if selected is None:raise ValueError('No FIT candidate; heldout tensors and performance must remain unconsumed')
    validate_parameters(selected)
    validate_fit_cache(root,binding=binding)
    seal=json.loads((directory/'fit_decision_seal.json').read_text());analysis=directory/'analysis_signature.json'
    if (seal['fit_decision_sha256']!=sha(path) or seal['analysis_signature_sha256']!=sha(analysis)
        or decision['analysis_signature_sha256']!=sha(analysis) or decision['checkpoint_sha256']!=checkpoint_sha256
        or decision['roles_sha256']!=roles_sha256 or decision['fit_groups']!=84
        or seal['fit_packet_manifest_sha256']!=sha(directory/'packets_fresh_fit/manifest.json')):
        raise ValueError('Frozen FIT decision/source lineage changed')
    source=json.loads(analysis.read_text())
    if source['implementation']!=calibration_code_hashes() or source['checkpoint_sha256']!=checkpoint_sha256 or source['roles_sha256']!=roles_sha256:
        raise ValueError('Frozen FIT model/implementation changed')
    if any(source.get(key)!=binding.get(key) or key not in binding for key in BINDING_KEYS):
        raise ValueError('Frozen V4 FIT feature lineage changed')
    chosen=select_point(decision['choices'],dev=False)
    if chosen is None or chosen['index']!=decision['selected_index'] or chosen['parameters']!=selected:
        raise ValueError('FIT choice differs from predeclared complete-menu selection')
    return decision


class FreshConfirmation:
    def __init__(self,root=BASE/'revision_v4'):
        self.root=Path(root);self.directory=self.root/'confirmation';self.checkpoint=self.root/'training/best.pt'
        self.state=checked_state(self.checkpoint);self.checkpoint_sha256=sha(self.checkpoint)
        role_path=BASE/'revision_v3/roles.json';seal_path=BASE/'revision_v3/sealed_verify130.json'
        self.roles=json.loads(role_path.read_text());self.roles_sha256=sha(role_path)
        seal=json.loads(seal_path.read_text());source=json.loads((BASE/'source_manifest.json').read_text())
        self.population=json.loads((self.root/'confirmation_population.json').read_text())
        expected=protocol_metadata(self.roles,source,seal,binding=self.state,
            source_sha256=sha(BASE/'source_manifest.json'),seal_sha256=sha(seal_path))
        expected['preparation_code_sha256']=sha(ROOT/'experiments/a2_waldo_landmark_belief/reference_v4_confirmation_protocol.py')
        if self.population!=expected or self.state['roles_sha256']!=self.roles_sha256:
            raise ValueError('Fixed V4 confirmation/feature population changed')
        if json.loads((self.root/'confirmation_protocol.json').read_text())!=CONFIG:
            raise ValueError('Pre-observation V4 protocol changed')
        verify_cache_origin(BASE/'collected_snapshot')
        self.signature=json.loads((self.directory/'signature.json').read_text());self.signature_sha256=sha(self.directory/'signature.json')
        if self.signature!=collection_signature(self.root,self.state,self.checkpoint):
            raise ValueError('Fresh V4 collection model/source/feature signature changed')
        self.manifest=json.loads((self.directory/'manifest.json').read_text());self.entries={e['index']:e for e in self.population['fresh_entries']}
        if (self.manifest['status']!='v4_fresh163_independent_A0_confirmation_complete' or self.manifest['target_groups']!=163
            or self.manifest['signature_sha256']!=self.signature_sha256 or len(self.manifest['episodes'])!=163
            or {e['index'] for e in self.manifest['episodes']}!=set(self.entries)):raise ValueError('Complete fresh163 collection required')
        committed=[]
        # Only JSON source/commit identity fields are inspected here. Original
        # A0 metrics are not used for candidate feedback; VERIFY tensors stay closed.
        for e in self.manifest['episodes']:
            name=e['complete_file']
            if name!=f"episode_{e['index']}_complete.json":raise ValueError('Unsafe episode commit path')
            path=self.directory/name;sidecar=path.with_suffix('.sha.json')
            if sha(path)!=e['sha256'] or sha(sidecar)!=e['sidecar_sha256']:raise ValueError('Fresh episode commit changed')
            bound=json.loads(sidecar.read_text());info=json.loads(path.read_text())
            if (bound!=dict(complete_sha256=sha(path),signature_sha256=self.signature_sha256,index=e['index'])
                or info['entry']!=self.entries[e['index']] or info['signature_sha256']!=self.signature_sha256):raise ValueError('Fresh committed query/signature mismatch')
            if not info['rows'] or [r['step'] for r in info['rows']]!=list(range(len(info['rows']))) or len(info['rows'])>20:
                raise ValueError('Incomplete fresh episode observations')
            committed.extend(info['rows'])
        identity=lambda e:(e['filename'],e['sha256'],e['episode'],e['step'],e['fold'])
        if Counter(map(identity,committed))!=Counter(map(identity,self.manifest['entries'])):raise ValueError('Fresh manifest differs from committed observations')
        if len({(e['episode'],e['step']) for e in committed})!=len(committed):raise ValueError('Duplicate fresh observation')
        for item in committed:
            query=self.entries[item['episode']]
            if item['fold']!=query['fold'] or item['filename']!=f"view_{query['index']}_{item['step']}.pt" or sha(self.directory/item['filename'])!=item['sha256']:
                raise ValueError('Fresh raw observation/fold source changed')
        self.model=RelationV3Network(**self.state['model_config']).eval().requires_grad_(False);self.model.load_state_dict(self.state['model'],strict=True)

    def groups(self,fold):return {(e['map_name'],e['object_id']) for e in self.population['fresh_entries'] if e['fold']==fold}

    def packets(self,fold):
        if fold not in ('fit','verify'):raise ValueError('Explicit calibration fold required')
        decision=load_frozen_choice(self.root,self.checkpoint_sha256,self.roles_sha256,binding=self.state) if fold=='verify' else None
        output=self.root/'calibration'/('packets_fresh_'+fold);output.mkdir(parents=True,exist_ok=True)
        analysis=self.root/'calibration/analysis_signature.json'
        signature=dict(analysis_signature_sha256=sha(analysis),source_manifest_sha256=sha(self.directory/'manifest.json'),
            checkpoint_sha256=self.checkpoint_sha256,fold=fold,packet_code_sha256=sha(ROOT/'experiments/a2_waldo_landmark_belief/reference_v4_calibration_data.py'),
            **{key:self.state[key] for key in FEATURE_KEYS})
        if decision is not None:signature['fit_decision_sha256']=sha(self.root/'calibration/fit_decision.json')
        immutable(signature,output/'signature.json');digest=sha(output/'signature.json');packets=[];entries=[]
        for item in self.manifest['entries']:
            if item['fold']!=fold:continue
            query=self.entries[item['episode']];saved=output/item['filename'];bound_path=saved.with_suffix('.sha.json')
            if saved.exists() and bound_path.exists():
                bound=json.loads(bound_path.read_text())
                if bound!=dict(packet_sha256=sha(saved),signature_sha256=digest,source_row_sha256=item['sha256']):raise ValueError('Fresh derived packet commit changed')
                packet=torch.load(saved,map_location='cpu',weights_only=False)
            else:
                row=read_fresh_row(self.directory,item,query,self.signature_sha256,fold,verify_authorized=decision is not None)
                packet=make_packet(row,self.model);packet.update(signature_sha256=digest,source_row_sha256=item['sha256'])
                atomic_checkpoint(packet,saved);save(dict(packet_sha256=sha(saved),signature_sha256=digest,source_row_sha256=item['sha256']),bound_path)
            if (packet['group']!=[query['map_name'],query['object_id']] or packet['fold']!=fold or packet['episode']!=query['index']
                or packet['step']!=item['step'] or packet['signature_sha256']!=digest or packet['source_row_sha256']!=item['sha256']):raise ValueError('Fresh packet identity/fold changed')
            packets.append(packet);entries.append(dict(filename=item['filename'],sha256=sha(saved),sidecar_sha256=sha(bound_path),source_row_sha256=item['sha256']))
        expected=84 if fold=='fit' else 79
        if len({tuple(p['group']) for p in packets})!=expected:raise ValueError('Incomplete fresh fold packet population')
        immutable(dict(signature_sha256=digest,fold=fold,target_groups=expected,entries=entries),output/'manifest.json')
        return packets
