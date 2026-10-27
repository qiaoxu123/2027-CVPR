"""Strict pretrained initialization and complete Stage1 checkpoint construction."""
import hashlib
from types import SimpleNamespace
import torch
from .checkpointing import file_sha256,resolve_checkpoint_resource
from .full_experiment import new_model
from .stage1 import GraphNavStage1
from .static_transfer_runtime import validate_static_transfer

EXPECTED=dict(a3='9d9be861bf7e9cac986d747867990de3d409ccb9bc1545862d71a620448380d7',head='51d8af7f05a9a718a9dc0ed12721a24ca6dd252e2714cb25948f22cb88cd2dbd',reference='2008085175d3da5f84fa1fc880d3ef38ba3c3fdd7595bd33a3c856f22d6baf2c')


def module_sha(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize_model(paths,device,init=None,model_type='original'):
    for key,sha in EXPECTED.items():
        if file_sha256(paths[key])!=sha:raise ValueError('required pretrained SHA mismatch: '+key)
    a3=torch.load(paths['a3'],map_location='cpu',weights_only=True)
    head=torch.load(paths['head'],map_location='cpu',weights_only=True)
    reference=torch.load(paths['reference'],map_location='cpu',weights_only=True)
    validate_static_transfer(a3['args'],reference['args'])
    cache=resolve_checkpoint_resource(head['args']['cache'],paths['head'])/'manifest.json'
    if file_sha256(cache)!=head['input_fingerprint']['cache_manifest']['sha256']:raise ValueError('original head cache provenance differs')
    import json
    if json.loads(cache.read_text())['protocol']['checkpoint_sha256']!=EXPECTED['reference']:raise ValueError('head original reference mismatch')
    if head['protocol']['stop_positive_weight']!=20 or head['protocol']['observed_fraction']!=0:raise ValueError('original head input/calibration mismatch')
    if model_type not in ('original', 'joint-rgb-geometry'):
        raise ValueError('unknown Stage1 model_type')
    if model_type == 'joint-rgb-geometry':
        from .joint_rgb_geometry import JointRGBGeometryStage1
        cls = JointRGBGeometryStage1
    else:
        cls = GraphNavStage1
    model=cls(new_model(SimpleNamespace(**a3['args']),device)).to(device)
    model.a3.load_state_dict(a3['model'],strict=True);model.trajectory.load_state_dict(head['model'],strict=True)
    if init:
        previous=torch.load(init,map_location='cpu',weights_only=True)
        expected_name = 'graphnav-joint-rgb-geometry-v1' if model_type == 'joint-rgb-geometry' else 'graphnav-v1-stage1'
        if previous['protocol']['name']!=expected_name or previous['source_sha256']!=EXPECTED:raise ValueError('Stage1 checkpoint initialization lineage differs')
        model.load_state_dict(previous['model'],strict=True)
    return model,a3['args']


def protocol(backbone_revision,stage,model_type='original'):
    if model_type not in ('original', 'joint-rgb-geometry'):raise ValueError('invalid experiment model_type')
    name='graphnav-joint-rgb-geometry-v1' if model_type=='joint-rgb-geometry' else 'graphnav-v1-stage1'
    return dict(name=name,backbone='google/siglip2-large-patch16-384',backbone_revision=backbone_revision,
        stage=stage,use_history=stage in ('B','C'),active_observation=False,new_gat=False,relation_loss=False,
        residual_bound=.5,stop_threshold=.3,stop_positive_weight=20.,execution='guarded-blend',horizon=20,
        horizontal_step_m=50.,vertical_step_m=10.,freeze_bert=True,freeze_siglip=True,
        observed_fraction_to_legacy_controller=0.,initialization='Verified A3 Seed1 Epoch14 and original pretrained Trajectory/Stop; all later stages initialize from prior selected complete checkpoint',
        selection='Seen128 SR then SPL, then internal Train Dev belief Recall; Unseen never used')
