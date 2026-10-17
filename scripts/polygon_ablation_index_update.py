"""Keep existing paper index up to date without replacing prior experiments."""
import csv
from scripts.polygon_ablation_plan import HERE,OUT


def update_index(local,navigation,partial):
    path=HERE/'reports/paper_experiments/experiment_registry.csv'
    old=list(csv.DictReader(path.open()))
    rows=[r for r in old if not r.get('result_path','').startswith(str(OUT))]
    for endpoint,values in [('localization',local),('navigation',navigation)]:
        for r in values:
            rows.append(dict(method='PolyBelief',variant=f"Polygon {r['phase']} / {r['scope']} / {r['representation']} / seed{r['seed']}",
                             split=r['split'],dataset='revised',episodes=r['samples'] if endpoint=='navigation' else '',queries=r['samples'] if endpoint=='localization' else '',
                             endpoint=endpoint,status=r['status'],checkpoint_sha256=r['checkpoint_sha256'],
                             source_commit='See polygon_ablation/launch_record.json and training source snapshot SHA',
                             protocol='Polygon pre-registered 28x28; frozen vs retrained and two geometry scopes separated; '+endpoint,
                             result_path=r['result_path'],predictions=r['predictions'],command='python -m scripts.run_polygon_ablation --resume',
                             experiment_family='polygon_ablation'))
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    path=HERE/'reports/paper_experiments/todo_status.csv'
    tasks=list(csv.DictReader(path.open()))
    for r in tasks:
        if r['experiment']=='Polygon vs Centroid vs Bounding Box闭环':
            r.update(status='Running' if partial else 'Done',source='New fixed Train-Dev and 3-seed Polygon experiment',result_path='reports/paper_experiments/polygon_ablation/POLYGON_ABLATION_REPORT.md',
                     missing='Queue in progress; no unfinished condition marked Done' if partial else 'No missing predeclared condition; preserve negative and subgroup results')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=tasks[0].keys());w.writeheader();w.writerows(tasks)
