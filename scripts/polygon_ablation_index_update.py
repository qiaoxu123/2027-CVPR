"""Keep existing paper index up to date without replacing prior experiments."""
import csv
import json
from scripts.polygon_ablation_plan import HERE,OUT,BASE,canonical_job
from graphnav.checkpointing import file_sha256
from pathlib import Path


def update_index(local,navigation,partial):
    path=HERE/'reports/paper_experiments/experiment_registry.csv'
    old=list(csv.DictReader(path.open()))
    provenance=OUT/'implementation_provenance.json'
    commit=json.loads(provenance.read_text())['implementation_commit'] if provenance.exists() else '47ab082b412493567a9d7738ea8256a462a656b3'
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    navigation_sources=json.loads((OUT/'navigation_data_audit.json').read_text())['files']
    rows=[r for r in old if not r.get('result_path','').startswith(str(OUT))]
    for endpoint,values in [('localization',local),('navigation',navigation)]:
        for r in values:
            checkpoint=BASE if r['phase']=='frozen' else OUT/'training'/canonical_job(r['scope'],r['representation'],r['seed'])/'best.pt'
            log=OUT/'logs'/(Path(r['result_path']).parent.name+'.log') if endpoint=='localization' else OUT/'logs'/(Path(r['result_path']).parent.parent.name+'.log')
            data_sha=plan['input_sha256'][plan['paths'][r['split']]] if endpoint=='localization' else next(v['sha256'] for v in navigation_sources if v['path'].endswith('/citynav_'+r['split']+'.json'))
            rows.append(dict(method='PolyBelief',variant=f"Polygon {r['phase']} / {r['scope']} / {r['representation']} / seed{r['seed']}",
                             split=r['split'],dataset='revised',episodes=r['samples'] if endpoint=='navigation' else '',queries=r['samples'] if endpoint=='localization' else '',
                             endpoint=endpoint,status=r['status'],checkpoint_sha256=r['checkpoint_sha256'],
                             checkpoint=str(checkpoint),data_sha256=data_sha,grid_size=28,
                             NE=r['NE'],**{k:100*r[k] for k in ('SR','OSR','SPL') if k in r},
                             source_commit=r.get('source_commit',commit),
                             evaluator_sha256=file_sha256(HERE/('graphnav/closedloop.py' if endpoint=='navigation' else 'scripts/polygon_ablation_engine.py')),
                             episode_manifest_sha256=file_sha256(Path(r['result_path']).parent/'episodes.json') if endpoint=='navigation' else '',
                             protocol='Polygon pre-registered 28x28; frozen vs retrained and two geometry scopes separated; '+endpoint,
                             result_path=r['result_path'],predictions=r['predictions'],command='python -m scripts.run_polygon_ablation --resume',
                             result_sha256=file_sha256(r['result_path']),predictions_exist=Path(r['predictions']).exists(),log=str(log),log_exists=log.exists(),
                             comparable=False,result_type='controlled_geometry_ablation; navigation SR/OSR/SPL percent in registry, ratios in experiment CSVs',
                             experiment_family='polygon_ablation'))
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
    path=HERE/'reports/paper_experiments/todo_status.csv'
    tasks=list(csv.DictReader(path.open()))
    for r in tasks:
        if r['experiment']=='Polygon vs Centroid vs Bounding Box闭环':
            r.update(status='Running' if partial else 'Done',source='New fixed Train-Dev and 3-seed Polygon experiment',result_path='reports/paper_experiments/polygon_ablation/POLYGON_ABLATION_REPORT.md',
                     missing='Queue in progress; no unfinished condition marked Done' if partial else 'No missing predeclared condition; preserve negative and subgroup results')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=tasks[0].keys(),lineterminator='\n');w.writeheader();w.writerows(tasks)
