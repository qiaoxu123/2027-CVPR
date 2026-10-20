"""Read-only source discovery for the 2026-10-20 paper asset audit."""
import csv, hashlib, json, os
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get('GEOMBELIEF_WORKSPACE', '/home/rental/20260922_1/Workspace'))
PROJECT = WORKSPACE / '2027-CVPR'
ARCHIVE = Path(os.environ.get('GEOMBELIEF_ARCHIVE', str(PROJECT / 'Ablations/GraphNav/visual-residual-20261013')))
MAIN = PROJECT / 'GraphNav'
BASELINES = PROJECT / 'Baselines'
OLD = ARCHIVE / 'reports/paper_experiments'
GEO = OLD / 'geombelief_remaining'
PAPER = Path(os.environ.get('GEOMBELIEF_PAPER', str(PROJECT / 'GeomBelief-paper')))

def read(path):
    return json.loads(Path(path).read_text())

def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def csvwrite(path, rows, fields=None):
    rows = list(rows)
    if not fields:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def current(path):
    """Translate historical paths in memory; never edit source manifests."""
    s = str(path)
    aliases = {'2027-CVPR-relational-visual-belief': MAIN,
               'SBFNav-CityNav-Reproduction': BASELINES / 'SBFNav',
               '2027-CVPR-closedloop-sbf-parity': PROJECT / 'Ablations/GraphNav/2027-CVPR-closedloop-sbf-parity',
               '2027-CVPR-full-token-rgb': PROJECT / 'Ablations/GraphNav/2027-CVPR-full-token-rgb'}
    for name, dest in aliases.items():
        for prefix in (WORKSPACE / name, PROJECT / name):
            if s == str(prefix) or s.startswith(str(prefix) + '/'):
                return Path(str(dest) + s[len(str(prefix)):])
    return Path(s)
