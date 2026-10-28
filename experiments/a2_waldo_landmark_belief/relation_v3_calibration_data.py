"""Role-filtered fresh records and explicit frozen-FIT barrier for heldout data."""
from collections import Counter
import json,re
from pathlib import Path
import torch
from .paths import ROOT,sha,save
from .collected_snapshot import BASE,verify_cache_origin
from .relation_v3_confirmation_protocol import build_population,CONFIG,immutable
from .relation_v3_confirmation import collection_code_hashes
from .relation_v3_provenance import checked_state
from .relation_v3_model import RelationV3Network
from .relation_v3_packet import make_packet
from .relation_v3_gate_fit import validate_parameters
from .direct_risk_protocol import select_point
from .train import atomic_checkpoint


def calibration_code_hashes():
    names=('relation_v3_calibration_data.py','relation_v3_calibrate.py','relation_v3_sealed_replay.py',
        'relation_v3_gate_fit.py','relation_v3_packet.py','direct_risk_calibrate.py','revision_calibrate.py','relation_v3_confirmation_pipeline.py')
    return dict(collection_code_hashes(),**{n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names})


def read_fresh_row(directory,item,query,signature,fold,*,verify_authorized=False):
    if (fold not in ('fit','verify') or item['fold']!=fold or query['fold']!=fold
        or (fold=='verify' and not verify_authorized)):raise ValueError('Cross-role or unsealed VERIFY tensor forbidden')
    name=item['filename']
    if name!=f"view_{query['index']}_{item['step']}.pt" or Path(name).name!=name or item['episode']!=query['index']:
        raise ValueError('Fresh source observation identity changed')
    path=Path(directory)/name
    if sha(path)!=item['sha256']:raise ValueError('Fresh raw record changed')
    row=torch.load(path,map_location='cpu',weights_only=False)
    if (row['signature_sha256']!=signature or row['episode']!=query['index'] or row['step']!=item['step'] or row['fold']!=fold
        or row['target_instance']!=[query['map_name'],query['object_id']]):raise ValueError('Fresh raw identity/fold mismatch')
    return row


def load_frozen_choice(root,checkpoint_sha256,roles_sha256):
    root=Path(root);directory=root/'calibration';path=directory/'fit_decision.json';decision=json.loads(path.read_text())
    selected=decision['selected_parameters']
    if selected is None:raise ValueError('No FIT candidate; heldout tensors and performance must remain unconsumed')
    validate_parameters(selected)
    validate_fit_cache(root)
    seal=json.loads((directory/'fit_decision_seal.json').read_text());analysis=directory/'analysis_signature.json'
    if (seal['fit_decision_sha256']!=sha(path) or seal['analysis_signature_sha256']!=sha(analysis)
        or decision['analysis_signature_sha256']!=sha(analysis) or decision['checkpoint_sha256']!=checkpoint_sha256
        or decision['roles_sha256']!=roles_sha256 or decision['fit_groups']!=84
        or seal['fit_packet_manifest_sha256']!=sha(directory/'packets_fresh_fit/manifest.json')):
        raise ValueError('Frozen FIT decision/source lineage changed')
    source=json.loads(analysis.read_text())
    if source['implementation']!=calibration_code_hashes() or source['checkpoint_sha256']!=checkpoint_sha256 or source['roles_sha256']!=roles_sha256:
        raise ValueError('Frozen FIT model/implementation changed')
    chosen=select_point(decision['choices'],dev=False)
    if chosen is None or chosen['index']!=decision['selected_index'] or chosen['parameters']!=selected:
        raise ValueError('FIT choice differs from predeclared complete-menu selection')
    return decision


def validate_fit_cache(root):
    root=Path(root);directory=root/'calibration/packets_fresh_fit';manifest=json.loads((directory/'manifest.json').read_text())
    signature_path=directory/'signature.json';digest=sha(signature_path);signature=json.loads(signature_path.read_text())
    source_path=root/'confirmation/manifest.json';source=json.loads(source_path.read_text())
    if (manifest['signature_sha256']!=digest or manifest['fold']!='fit' or manifest['target_groups']!=84
        or signature['fold']!='fit' or signature['source_manifest_sha256']!=sha(source_path)
        or signature['analysis_signature_sha256']!=sha(root/'calibration/analysis_signature.json')):raise ValueError('FIT cache source signature changed')
    expected={e['filename']:e for e in source['entries'] if e['fold']=='fit'};entries=manifest['entries']
    if len(expected)!=sum(e['fold']=='fit' for e in source['entries']) or len(entries)!=len(expected) or {e['filename'] for e in entries}!=set(expected):
        raise ValueError('Incomplete or duplicate FIT cached observations')
    episodes=set()
    for e in entries:
        match=re.fullmatch(r'view_(\d+)_(\d+)\.pt',e['filename'])
        if match is None:raise ValueError('Unsafe FIT cache path')
        original=expected[e['filename']];episodes.add(int(match[1]))
        if (original['episode'],original['step'])!=(int(match[1]),int(match[2])) or e['source_row_sha256']!=original['sha256']:
            raise ValueError('FIT packet rebound to another source observation')
        path=directory/e['filename'];sidecar=path.with_suffix('.sha.json')
        if sha(path)!=e['sha256'] or sha(sidecar)!=e['sidecar_sha256']:raise ValueError('FIT packet/sidecar bytes changed')
        if json.loads(sidecar.read_text())!=dict(packet_sha256=sha(path),signature_sha256=digest,source_row_sha256=original['sha256']):
            raise ValueError('FIT packet sidecar source changed')
    if len(episodes)!=84:raise ValueError('All84 fixed FIT queries required')
    return manifest


class FreshConfirmation:
    def __init__(self,root=BASE/'revision_v3'):
        self.root=Path(root);self.directory=self.root/'confirmation';self.checkpoint=self.root/'training/best.pt'
        self.state=checked_state(self.checkpoint);self.checkpoint_sha256=sha(self.checkpoint)
        self.roles=json.loads((self.root/'roles.json').read_text());self.roles_sha256=sha(self.root/'roles.json')
        seal=json.loads((self.root/'sealed_verify130.json').read_text());source=json.loads((BASE/'source_manifest.json').read_text())
        self.population=json.loads((self.root/'confirmation_population.json').read_text())
        expected=build_population(self.roles,source,seal)
        expected.update(roles_sha256=self.roles_sha256,source_manifest_sha256=sha(BASE/'source_manifest.json'),
            sealed_verify_sha256=sha(self.root/'sealed_verify130.json'),preparation_code_sha256=sha(ROOT/'experiments/a2_waldo_landmark_belief/relation_v3_confirmation_protocol.py'))
        if self.population!=expected or self.state['roles_sha256']!=self.roles_sha256:raise ValueError('Fixed confirmation/role population changed')
        if json.loads((self.root/'confirmation_protocol.json').read_text())!=CONFIG:raise ValueError('Pre-observation protocol changed')
        verify_cache_origin(BASE/'collected_snapshot')
        self.signature=json.loads((self.directory/'signature.json').read_text());self.signature_sha256=sha(self.directory/'signature.json')
        checks=dict(population_sha256=sha(self.root/'confirmation_population.json'),protocol_sha256=sha(self.root/'confirmation_protocol.json'),
            roles_sha256=self.roles_sha256,sealed_verify_sha256=sha(self.root/'sealed_verify130.json'),checkpoint_sha256=self.checkpoint_sha256,
            training_verification_sha256=sha(self.root/'training/artifact_verification.json'),code_sha256=collection_code_hashes(),
            sensor_signature_sha256=sha(self.root/'confirmation_sensor_signature.json'),observation_source=CONFIG['source'])
        if self.signature!=checks:raise ValueError('Fresh collection model/source signature changed')
        self.manifest=json.loads((self.directory/'manifest.json').read_text());self.entries={e['index']:e for e in self.population['fresh_entries']}
        if (self.manifest['status']!='v3_fresh163_independent_A0_confirmation_complete' or self.manifest['target_groups']!=163
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
        decision=load_frozen_choice(self.root,self.checkpoint_sha256,self.roles_sha256) if fold=='verify' else None
        output=self.root/'calibration'/('packets_fresh_'+fold);output.mkdir(parents=True,exist_ok=True)
        analysis=self.root/'calibration/analysis_signature.json'
        signature=dict(analysis_signature_sha256=sha(analysis),source_manifest_sha256=sha(self.directory/'manifest.json'),
            checkpoint_sha256=self.checkpoint_sha256,fold=fold,packet_code_sha256=sha(ROOT/'experiments/a2_waldo_landmark_belief/relation_v3_packet.py'))
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
