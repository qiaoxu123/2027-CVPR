"""Paths, reproducibility and immutable configuration; no new dependencies."""
import os
import sys
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.environ.get('CITYNAV_PROJECT_ROOT',str(ROOT.parents[1])))
SOURCE=PROJECT/'GraphNav' if (PROJECT/'GraphNav/graphnav').is_dir() else ROOT/'runtime/GraphNav'
DATA=Path(os.environ.get('CITYNAV_DATA_ROOT',str(PROJECT.parent/'DATA/sbfnav-data')))
RGBD=Path(os.environ.get('CITYNAV_RGBD_ROOT',str(PROJECT.parent/'DATA/rgbd')))
CHECKPOINT=SOURCE/'runs/graphnav_v1_stage1/best.pt'
if not CHECKPOINT.exists():CHECKPOINT=ROOT/'assets/main_checkpoint.pt'
SBF=PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'
if not (SBF/'sbfnav').is_dir():SBF=ROOT/'runtime/SBFNav'
PREVIOUS=PROJECT/'Experiments/BoxRelationCorrection'
WALDO=PROJECT/'Experiments/Height3DGrounding/Repaired3D_R10/reports/waldo_pilot'
if not WALDO.exists():WALDO=ROOT/'assets/waldo'
VENDOR=WALDO.parent/'obb_pilot/vendor'
if not VENDOR.exists():VENDOR=ROOT/'runtime/waldo_vendor'
sys.path[:0]=[str(SOURCE),str(SBF),str(VENDOR)]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_CONFIG_DIR']=str(ROOT/'configs/yolo')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def save(value,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)


def implementation_hashes():return {p.name:sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))}


def configuration():
    p=ROOT/'configs/frozen.json'
    if p.exists():return json.loads(p.read_text())
    import numpy as np
    from sbfnav.dataset import CityNavDataset
    previous=json.loads((PREVIOUS/'configs/frozen.json').read_text());ds=CityNavDataset(DATA,'train_seen',version=previous['data_version'])
    groups=sorted({(ds[e['index']].map_name,int(ds[e['index']].object_id)) for e in previous['train']})
    rng=np.random.default_rng(17);cal={groups[i] for i in rng.permutation(len(groups))[:max(1,round(len(groups)*.15))]}
    chosen={'train':[],'calibration':[],'dev':previous['dev']}
    for e in previous['train']:
        rec=ds[e['index']];chosen['calibration' if (rec.map_name,int(rec.object_id)) in cal else 'train'].append(e)
    c=dict(version=1,seed=17,grid_size=28,radius=40.,max_peaks=3,max_candidates=32,max_references=16,
        quantiles=[.25,.75,1.],hidden=256,lr=.0003,weight_decay=.01,batch_size=8,rank_epochs=20,offset_epochs=5,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),waldo_sha256=previous['waldo_sha256'],
        source_sha256=previous['source_sha256'],eval_manifests=previous['eval_manifests'],data_version=previous['data_version'],
        annotation_sha256=sha(ds.annotation_path),splits=chosen,surface_semantics='weak observed interior DSM median, NOT object top or extent')
    assert c['checkpoint_sha256']==previous['checkpoint_sha256']
    save(c,p);return c


def resolve_source_path(path):
    p=Path(path)
    if '/GraphNav/graphnav/' in str(path):return SOURCE/'graphnav'/p.name
    if '/sbfnav/' in str(path):return SBF/'sbfnav'/p.name
    return p


def resolve_row_path(path):
    local=ROOT/'data'/Path(path).name
    return local if local.exists() else Path(path)


def evaluation_manifest(c,split):
    p=Path(c['eval_manifests'][split])
    return p if p.exists() else ROOT/'configs'/f'{split}_paired128.json'


def verify_sources(c,annotations=True):
    if sha(CHECKPOINT)!=c['checkpoint_sha256']:raise ValueError('Original checkpoint changed')
    for path,digest in c['source_sha256'].items():
        if sha(resolve_source_path(path))!=digest:raise ValueError(f'Original source changed: {path}')
    if sha(WALDO/'WALDO30_yolov8l-p2_1024x1024.pt')!=c['waldo_sha256']:raise ValueError('WALDO checkpoint changed')
    if not annotations:return
    annotation=c.get('annotation_path')
    if annotation is None:
        from sbfnav.dataset import CityNavDataset
        annotation=CityNavDataset(DATA,'train_seen',version=c['data_version']).annotation_path
    if sha(annotation)!=c['annotation_sha256']:raise ValueError('Training annotation changed')


def validate_cache_signature(state,signature):
    if state['cache_signature']!=signature:raise ValueError('Checkpoint/cache label or extraction provenance differs')


def load_branch(path,device='cpu'):
    import torch
    from model import LocalRelationNetwork
    from schema import SCHEMA_VERSION
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['schema_version']!=SCHEMA_VERSION:raise ValueError('Checkpoint schema mismatch')
    if state['config_sha256']!=sha(ROOT/'configs/frozen.json'):raise ValueError('Checkpoint config mismatch')
    # Runtime-critical code must match training, while reporting tools may evolve.
    for name in ['model.py','geometry.py','schema.py','perception.py','fusion.py','policy.py']:
        if state['implementation'][name]!=sha(ROOT/'scripts'/name):raise ValueError(f'Checkpoint implementation mismatch: {name}')
    model=LocalRelationNetwork(**state['model_config']).to(device).eval();model.load_state_dict(state['model'])
    return model,state
