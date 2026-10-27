"""Explicit local frozen dependencies; no modification of A0/A1 source trees."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(os.environ.get('CITYNAV_PROJECT_ROOT', '/home/20261024_1/Workspace/2027-CVPR'))
DATA = Path(os.environ.get('CITYNAV_DATA_ROOT', str(PROJECT.parent / 'DATA/sbfnav-data')))
RGBD = Path(os.environ.get('CITYNAV_RGBD_ROOT', str(PROJECT.parent / 'DATA/rgbd')))
SOURCE = PROJECT / 'GraphNav'
SBF = PROJECT / 'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'
CHECKPOINT = SOURCE / 'runs/graphnav_v1_stage1/best.pt'
A0_SHA = '00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a'
WALDO = PROJECT / 'Experiments/Height3DGrounding/Repaired3D_R10/reports/waldo_pilot/WALDO30_yolov8l-p2_1024x1024.pt'
VENDOR = PROJECT / 'Experiments/LocalSemanticRelationRefinement/runtime/waldo_vendor'
WALDO_SHA = '1a5ac7a28a00ec5ab98385fa3088dfdf77a91e714e9736fcf2ce5c47d57c129a'
BERT_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'


def bootstrap():
    sys.path[:0] = [str(SOURCE), str(SBF), str(VENDOR)]
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'runs/a2_waldo_landmark_20261027/yolo')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def verify_frozen_assets():
    for path, expected in [(CHECKPOINT, A0_SHA), (WALDO, WALDO_SHA)]:
        if sha(path) != expected:
            raise ValueError(f'Frozen asset mismatch: {path}')
    files = list((SOURCE/'graphnav').rglob('*.py')) + list((SBF/'sbfnav').rglob('*.py')) + list(VENDOR.rglob('*.py'))
    return {'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': A0_SHA,
            'waldo': str(WALDO), 'waldo_sha256': WALDO_SHA,
            'bert_revision': BERT_REVISION,
            'source_sha256': {str(p): sha(p) for p in sorted(files)}}
