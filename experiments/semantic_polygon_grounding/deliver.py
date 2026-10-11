"""Package compact P4 diagnostics; full source-dataset audits remain local."""
import csv
import gzip
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .audit import DATA, OUT, save, sha

ASSETS=Path('docs/assets/semantic_polygon_grounding')


def read(path):return json.loads(Path(path).read_text())


def run():
    ASSETS.mkdir(parents=True,exist_ok=True)
    for name in ['map_audit.json','homonyms.json','annotation_schema.json','raw_split_audit.json',
                 'metrics.json','execution.json','decision.json','oracle_input_equivalence.json',
                 'rgb_binding_opportunities.json','same_named_building_cases.json','independence_and_model.json']:
        shutil.copy2(OUT/name,ASSETS/name)
    shutil.copy2(Path(__file__).with_name('protocol.json'),ASSETS/'protocol.json')
    shutil.copy2(OUT/'regression_tests.log',ASSETS/'regression_tests.txt')
    metrics=read(OUT/'metrics.json')
    with (ASSETS/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['split','families','method','goal_r1','goal_r4','goal_r8','ne','recovery','damage','net_recovery'],lineterminator='\n');writer.writeheader()
        for split,x in metrics.items():
            for method in ['A0_geometry','A1_known_name_set']:
                b=x['primary_family'][method]
                writer.writerow(dict(split=split,families=b['goal_r1']['n'],method=method,
                                     **{k:b[k]['mean'] for k in ['goal_r1','goal_r4','goal_r8','ne']},
                                     recovery='' if method=='A0_geometry' else 0,
                                     damage='' if method=='A0_geometry' else 0,
                                     net_recovery='' if method=='A0_geometry' else 0))
            for method in ['A1_ambiguous_instance','A2_geometry_joint','A3_rgb_attribute','A4_rgb_geometry','A5_black','A5_shuffled']:
                writer.writerow(dict(split=split,families=x['primary_family']['A0_geometry']['goal_r1']['n'],method=method))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=['Train','Internal Dev','Seen'];splits=['train','dev','seen'];x=np.arange(3)
    for i,key in enumerate(['strict_unique_reference_oracle_eligible','named_binding_sensitive','multiple_credible_same_named_candidates']):
        axes[0].bar(x+(i-1)*.25,[metrics[s]['primary_family']['subsets'][key]['families'] for s in splits],.25,label=['Unique valid name binding','Nonunique/unresolved/capped','Multiple credible same-name ROIs'][i])
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Independent families',title='Binding opportunities; instance truth absent');axes[0].legend(fontsize=7)
    for i,key in enumerate(['goal_r1','goal_r4','goal_r8']):
        axes[1].bar(x+(i-1)*.25,[100*metrics[s]['primary_family']['A0_geometry'][key]['mean'] for s in splits],.25,label=key)
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Goal accuracy (%)',title='Frozen A0 = known-name-set intervention');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(ASSETS/'audit_summary.png',dpi=160);plt.close(fig)
    public=read(DATA/'cityrefer/landmarks.json');examples=[]
    for split in ['dev','seen']:
        rows=json.loads(gzip.open(OUT/f'{split}_family_predictions.json.gz','rt').read())
        plan=torch.load(Path('runs/instance_grounding')/f'plan_{split}/plan.pt',weights_only=True)
        lookup={s['metadata']['sample_id']:s for s in plan['states']}
        groups=[('same_named_buildings',lambda r:any(sum(c['object_type']=='Building' for c in b['candidates'])>=2 for b in r['bindings'])),
                ('unique_anchor_error',lambda r:r['strict_unique_reference_oracle_eligible'] and not r['metrics']['goal_r1'])]
        for kind,predicate in groups:
            candidates=sorted([r for r in rows if predicate(r)],key=lambda r:r['family_id'])
            if not candidates:continue
            r=candidates[0];s=lookup[r['sample_id']];p=s['public'];b=np.asarray(p['bounds']);grid=b[:2]+np.array(p['grid_xy'])*(b[2:]-b[:2]);logits=np.array(p['static_logits']);fig,ax=plt.subplots(figsize=(7,6));im=ax.scatter(grid[:,0],grid[:,1],c=logits,s=12,cmap='Blues');fig.colorbar(im,ax=ax,label='Frozen Geometry logit')
            for j,binding in enumerate(r['bindings']):
                for c in binding['candidates']:
                    xy=np.array(public[r['map_name']][c['object_id']]['contour']);xy=np.vstack([xy,xy[0]]);ax.plot(xy[:,0],xy[:,1],color=plt.cm.tab10(j%10),linewidth=1.2)
                    if c['centroid_xy'] is not None:ax.text(*c['centroid_xy'],c['object_id'],fontsize=6)
            target=plan['labels'][r['sample_id']]['target_world'];pred=grid[r['geometry_ranking'][0]];ax.scatter(*target,c='green',marker='x',s=100,label='GT target: offline evaluation only');ax.scatter(*pred,c='red',marker='*',s=100,label='Frozen Geometry Top1');ax.set(xlabel='East (m)',ylabel='North (m)',title='\n'.join(textwrap.wrap(r['instruction'],75)));ax.legend(fontsize=7);fig.tight_layout();name=f'{split}_{kind}.png';fig.savefig(ASSETS/name,dpi=150);plt.close(fig)
            examples.append(dict(split=split,kind=kind,figure=name,sample_id=r['sample_id'],family_id=r['family_id'],instruction=r['instruction'],bindings=r['bindings'],goal_metrics=r['metrics'],parsed_relations=r['parsed_relations'],private_target_polygon_used=False))
    save(ASSETS/'failure_cases.json',examples)
    # Compact per-family outputs retain all candidate probabilities/rankings;
    # all repeated-prefix and 24k raw-instruction tables stay local.
    for split in splits:shutil.copy2(OUT/f'{split}_family_predictions.json.gz',ASSETS/f'{split}_family_predictions.json.gz')
    save(ASSETS/'sha256.json',{p.name:sha(p) for p in sorted(ASSETS.iterdir()) if p.name!='sha256.json'})
    print(json.dumps(dict(packaged_files=len(list(ASSETS.iterdir())),training=False)),flush=True)


if __name__=='__main__':run()
