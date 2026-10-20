"""Inventory existing source assets without moving data or rerunning experiments."""
from common import *
import subprocess

def main():
    categories = [
        ('Frozen formal metrics', MAIN / 'reports/graphnav_v1_stage1', 'Existing'),
        ('Formal checkpoint', MAIN / 'runs/graphnav_v1_stage1/best.pt', 'Existing'),
        ('Formal full episode predictions', MAIN / 'runs/graphnav_v1_stage1', 'Existing'),
        ('Paper experiments and ablations', OLD, 'Existing'),
        ('OSM-derived CityRefer landmark polygons', WORKSPACE / 'DATA/sbfnav-data/cityrefer', 'Existing'),
        ('Original CityNav episodes', WORKSPACE / 'DATA/citynav/extracted/data/citynav', 'Existing'),
        ('Revised CityNav episodes', WORKSPACE / 'DATA/hett/release/refined_citynav/processed_citynav', 'Existing'),
        ('Real orthophoto and surface-height rasters', WORKSPACE / 'DATA/rgbd', 'Existing'),
        ('Observed crop renderer', ARCHIVE / 'graphnav/renderer.py', 'Existing'),
        ('Archived dense belief fields', MAIN / 'reports/graphnav_v1_stage1/heatmaps', 'Existing'),
        ('Baseline results/provenance', BASELINES / 'comparison', 'Existing'),
        ('GeomBelief full Unseen profiling', PROJECT / 'Ablations/GraphNav/profiling-val-unseen-20261016/profiling_val_unseen/full', 'Existing'),
        ('Official FlightGPT locally executable full evaluation', BASELINES / 'comparison/flightgpt/preflight.json', 'Need Evaluation'),
        ('Strict common-protocol official HETT rerun', OLD / 'hett_asset_preflight.json', 'Need Evaluation'),
        ('Verified official trained HTNav checkpoint', BASELINES / 'HTNav/official_trained_checkpoint.pt', 'Missing'),
        ('Perspective onboard camera captures', WORKSPACE / 'DATA/perspective_onboard', 'Missing'),
    ]
    rows = []
    for label,p,status in categories:
        exists = p.exists()
        rows.append(dict(asset=label,path=str(p),status=status if exists else 'Missing',exists=exists,
                         bytes=p.stat().st_size if p.is_file() else '',sha256=sha(p) if p.is_file() else ''))
    csvwrite(OUT / 'asset_inventory.csv',rows)
    protected = []
    for p in sorted((MAIN / 'reports/graphnav_v1_stage1').glob('*')):
        if p.is_file(): protected.append(dict(path=str(p),sha256=sha(p)))
    for p in [MAIN/'runs/graphnav_v1_stage1/best.pt',*(MAIN/'runs/graphnav_v1_stage1'/s/'metrics.json' for s in ('full_seen','full_unseen','full_test_unseen'))]:
        protected.append(dict(path=str(p),sha256=sha(p)))
    if (OUT/'protected_sources.json').exists():
        assert read(OUT/'protected_sources.json')==protected,'Protected sources changed'
    else:dump(OUT/'protected_sources.json',protected)
    repos=[]
    for p in (ARCHIVE,MAIN,PAPER,OUT.parents[1]):
        repos.append(dict(path=str(p),head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True)))
    if not (OUT/'repository_state_before.json').exists():dump(OUT/'repository_state_before.json',repos)
    lines=['# Real experiment asset inventory — 2026-10-20','',
        'Read-only audit. Existing = asset found, not automatic evidence of comparable evaluation. Need Evaluation = assets/protocol still require execution or verification. Missing = not found in the declared search scope. No training requested or performed.','',
        '|Asset|Status|Local source|','|---|---|---|']
    lines += [f"|{r['asset']}|{r['status']}|`{r['path']}`|" for r in rows]
    lines += ['','The detailed pre-existing recursive inventory is indexed at `'+str(OLD/'asset_inventory.csv')+'` and `checkpoint_inventory.csv`; current selected source hashes are revalidated by this bundle.',
        '', 'The paper repository is private and was accessed with existing GitHub credentials. Local `2027-CVPR/paper` is an older non-Git draft and is not the requested current manuscript.',
        '', 'Source experiment branch local HEAD and published branch HEAD differ; both commits are recorded. This bundle uses local saved outputs and versioned source on the requested branch, never silently substitutes the main branch.',
        '', 'No entire orthophoto, CityNav dataset, checkpoint or private credential is included in this bundle. Observed RGB is simulator-rendered orthophoto, not a perspective onboard photo.']
    (OUT/'ASSET_MANIFEST.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
