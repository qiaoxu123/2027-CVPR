"""Read-only input helpers for the P0 evidence extension."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

FINAL=Path(__file__).resolve().parents[1]
ASSETS=FINAL.parent
ROOT=ASSETS.parents[1]
sys.path.insert(0,str(ROOT))
SELECTED=['formal_val_unseen_success','visual_corrected','visual_degraded']
BASE_SHA='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a'
HEAD_SHA='5fc1a3e115b71d14fe1e8fbaf3ed29c1863d8b2bc080455d89f7a4e503d25f11'


def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def cases():return [json.loads(s) for s in (ASSETS/'cases.jsonl').read_text().splitlines() if s.strip()]
def fields(name):
    with np.load(ASSETS/'fields'/f'{name}.npz',allow_pickle=False) as a:
        return {k:a[k] for k in a.files}
def probability(x):
    v=np.asarray(x,np.float64);v=np.exp(v-v.max());return v/v.sum()
def entropy(p):return float(-(p*np.log(np.maximum(p,1e-300))).sum())
def csvwrite(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def protected_snapshot():
    """Original assets, code and formal aggregates are never rewritten."""
    files=[p for p in ASSETS.rglob('*') if p.is_file() and FINAL not in p.parents and 'local_only' not in p.parts and '__pycache__' not in p.parts]
    files+=list((ROOT/'graphnav').glob('*.py'))
    files+=list((ROOT/'visual_evaluations').glob('*/metrics.json'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
