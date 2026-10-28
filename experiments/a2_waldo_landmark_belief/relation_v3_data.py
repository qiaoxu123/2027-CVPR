"""Role-checked training-only GT labels; no sealed calibration tensor loading."""
import json
from pathlib import Path
import torch
from .paths import DATA,ROOT,sha
from .collected_snapshot import BASE,cache_source,verify_cache_origin
from .core import _center
from .relation_v3_fusion import spatial_utility_labels

NETWORK_KEYS={'instruction_tokens','token_valid','mention_mask','landmark_name_tokens','name_valid','link_valid',
              'box_rgb','box_valid','pair_geometry','pair_valid','offset_limit'}


def observation_keys(entries):
    keys=[(str(Path(e['directory']).resolve()),e['filename']) for e in entries]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate V3 observation, including path aliases')
    return set(keys)


def read_training_source(reference,role,allowed_targets,signature):
    if role not in ('train','dev') or reference['role']!=role or (reference['map_name'],reference['object_id']) not in allowed_targets:
        raise ValueError('Calibration/foreign target forbidden before tensor deserialization')
    name=reference['filename']
    if Path(name).name!=name:raise ValueError('Unsafe source filename')
    path=Path(reference['directory'])/name
    if sha(path)!=reference['sha256']:raise ValueError('Training raw source changed')
    row=torch.load(path,map_location='cpu',weights_only=False)
    if (row['signature_sha256']!=signature or row['episode']!=reference['episode'] or row['step']!=reference['step']
        or row['target_instance']!=[reference['map_name'],reference['object_id']]):raise ValueError('Training source identity changed')
    return row


def build_training_row(raw,*,target_kind,real):
    import numpy as np
    source=raw['public']['network'];unknown=set(source)-NETWORK_KEYS-{'a0_context','box_kind'}
    if unknown or not NETWORK_KEYS<=set(source):raise ValueError('Unexpected/missing public network input')
    network={k:source[k] for k in NETWORK_KEYS}
    if any(not torch.is_tensor(v) for v in network.values()):raise ValueError('Network inputs must be public tensors')
    n=len(network['box_rgb']);l=len(network['link_valid']);candidates=raw['meta']['candidates'];references=raw['meta']['references']
    if len(candidates)>n or len(references)>l:raise ValueError('Public candidate/reference axes changed')
    kind=torch.zeros(n,dtype=torch.long)
    for i,candidate in enumerate(candidates):
        if candidate['kind'] not in (0,1):raise ValueError('Unsupported public detector category')
        kind[i]=candidate['kind']
    network['box_kind']=kind
    target=raw['labels']['target_xy'].float().clone()
    if target.shape!=(2,) or not torch.isfinite(target).all():raise ValueError('Invalid private TRAIN target')
    mentions=network['mention_mask'].bool()&network['token_valid'][None].bool()
    ref_valid=network['link_valid'].bool()&mentions.any(-1)&network['name_valid'].bool().any(-1)
    if ref_valid[len(references):].any():raise ValueError('Verified reference has no public geometry')
    reliable=network['box_valid'].bool()&(network['pair_valid'].bool()&ref_valid[None]).any(-1)
    spatial=raw['public']['spatial']
    usable=reliable&spatial['candidate_support'].bool().any(-1)
    geometrical=spatial_utility_labels({k:v[None] for k,v in spatial.items()},target[None])[0]
    compatible=kind==int(target_kind) if target_kind in (0,1) else torch.ones(n,dtype=torch.bool)
    utility=geometrical&compatible&usable
    angle=torch.zeros(l,2);distance=torch.zeros(l);valid=torch.zeros(l,dtype=torch.bool)
    for i,ref in enumerate(references):
        contour=np.asarray(ref['contour'],np.float32)
        if contour.ndim!=2 or contour.shape[-1]!=2 or not np.isfinite(contour).all():raise ValueError('Invalid public reference geometry')
        delta=target-torch.as_tensor(_center(contour));length=delta.norm()
        angle[i]=delta/length.clamp_min(1e-6);distance[i]=torch.log1p(length)/torch.log1p(torch.tensor(40.));valid[i]=bool(length>1 and ref_valid[i])
    labels=dict(target_xy=target,target_kind=torch.tensor(int(target_kind)),angle=angle,angle_valid=valid,distance=distance,
        localization_utility=utility,utility_valid=usable,geometric_utility=geometrical&usable,
        identity_label_mask=torch.zeros(n,dtype=torch.bool),sample_weight=torch.tensor(8. if real else 1.),source_is_real=torch.tensor(bool(real)))
    # Cached private associations are intentionally unused. Spatial suitability
    # for this TRAIN target is not a declaration of another object's identity.
    return dict(public=dict(network=network,spatial=spatial),labels=labels)


class RelationV3Views(torch.utils.data.Dataset):
    def __init__(self,role,*,roles_file=BASE/'revision_v3/roles.json'):
        if role not in ('train','dev'):raise ValueError('Only TRAIN/dev may enter this dataset')
        self.role=role;self.path=Path(roles_file);self.roles_sha256=sha(self.path);self.roles=json.loads(self.path.read_text());lineage=self.roles['lineage']
        from scripts.a2_prepare_relation_v3_roles import verify_target_roles
        sets={r:{(e['map_name'],e['object_id']) for e in values} for r,values in self.roles['target_roles'].items()};verify_target_roles(sets)
        self.allowed=sets[role];self.forbidden=sets['calibration_fit']|sets['calibration_verify']
        bindings=dict(original_source_sha256=BASE/'source_manifest.json',snapshot_manifest_sha256=BASE/'collected_snapshot/manifest.json',
            v1_manifest_sha256=BASE/'revision_v1/confirmation/manifest.json',v2_manifest_sha256=BASE/'revision_v2/confirmation/manifest.json',
            v2_failed_candidate_sha256=BASE/'revision_v2/calibration/candidate_calibration.json',v2_fit_decision_sha256=BASE/'revision_v2/calibration/fit_decision.json',
            preparation_code_sha256=ROOT/'scripts/a2_prepare_relation_v3_roles.py')
        if any(lineage.get(k)!=sha(p) for k,p in bindings.items()):raise ValueError('Versioned role source changed')
        candidate=json.loads(bindings['v2_failed_candidate_sha256'].read_text())
        if (candidate['confirmation_manifest_sha256']!=lineage['v2_manifest_sha256'] or candidate['fit_decision_sha256']!=lineage['v2_fit_decision_sha256']
            or candidate['packet_manifest_sha256']!=sha(BASE/'revision_v2/calibration/packets/manifest.json') or candidate['enabled']
            or candidate['verification'] is not None):raise ValueError('Prior failed decision/sealed status changed')
        self.entries=[e for e in self.roles['rows'] if e['role']==role]
        if len(self.entries)!=(33440 if role=='train' else 5647):raise ValueError('V3 population must be complete')
        actual_observations=observation_keys(self.entries)
        self.contexts={}
        snapshot=BASE/'collected_snapshot';signature=verify_cache_origin(snapshot)
        teacher=json.loads((snapshot/'manifest.json').read_text());source={e['index']:e for e in cache_source(snapshot)['entries']}
        self.contexts['teacher_snapshot']=dict(directory=snapshot.resolve(),signature=signature['parent_signature_sha256'],
            entries={e['filename']:dict(e,expected_role=e['split'],target=(source[e['episode']]['map_name'],source[e['episode']]['object_id'])) for e in teacher['entries']})
        for version in ('revision_v1','revision_v2'):
            directory=BASE/version/'confirmation';manifest=json.loads((directory/'manifest.json').read_text());saved=json.loads((directory/'signature.json').read_text())
            popfile=BASE/version/'confirmation_population.json';population=json.loads(popfile.read_text());source={e['index']:e for e in population['entries']}
            if manifest['signature_sha256']!=sha(directory/'signature.json') or saved['population_sha256']!=sha(popfile):raise ValueError('Onpolicy source signature/population changed')
            entries={}
            for e in manifest['entries']:
                q=source[e['episode']];expected='train' if e['fold']=='fit' else ('dev' if version=='revision_v1' else 'calibration_verify')
                if q['fold']!=e['fold']:raise ValueError('Onpolicy source fold changed')
                entries[e['filename']]=dict(e,expected_role=expected,target=(q['map_name'],q['object_id']))
            self.contexts[version+'_A0_closedloop']=dict(directory=directory.resolve(),signature=manifest['signature_sha256'],entries=entries)
        expected_observations={(str(c['directory']),name) for c in self.contexts.values() for name,e in c['entries'].items() if e['expected_role']==role}
        if actual_observations!=expected_observations:raise ValueError('V3 exact source population changed')
        if {(e['map_name'],e['object_id']) for e in self.entries}!=self.allowed:raise ValueError('V3 role target population changed')
        for e in self.entries:
            context=self.contexts.get(e['source_kind'])
            if context is None or Path(e['directory']).resolve()!=context['directory']:raise ValueError('Unregistered training source')
            original=context['entries'][e['filename']];target=(e['map_name'],e['object_id'])
            if (original['expected_role']!=role or original['target']!=target or target not in self.allowed or target in self.forbidden
                or any(e[k]!=original[k] for k in ('sha256','episode','step'))):raise ValueError('Role target/fold/observation source mismatch')
        objects_path=DATA/'cityrefer/objects.json'
        parent_signature=json.loads((Path(signature['parent_cache'])/'signature.json').read_text())
        if sha(objects_path)!=parent_signature['public_objects_sha256']:raise ValueError('Frozen category/public-map annotations changed')
        objects=json.loads(objects_path.read_text())
        # Only the allowed role's target CATEGORY is retained. No calibration
        # target coordinate/identity association is passed to supervision.
        self.target_kinds={(m,o):{'Car':0,'Building':1}.get(objects[m][str(o)]['object_type'],-1) for m,o in self.allowed}
        self.objects_sha256=sha(objects_path)
        if sha(self.path)!=self.roles_sha256:raise ValueError('V3 roles changed during dataset initialization')

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        reference=self.entries[index];context=self.contexts[reference['source_kind']]
        raw=read_training_source(reference,self.role,self.allowed,context['signature'])
        result=build_training_row(raw,target_kind=self.target_kinds[(reference['map_name'],reference['object_id'])],real=reference['source_kind']!='teacher_snapshot')
        result['target_instance']=[reference['map_name'],reference['object_id']]
        return result
