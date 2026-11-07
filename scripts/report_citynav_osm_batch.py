"""Report actual batch outputs; unknown geographic errors remain null, never zero."""
import json,shutil,hashlib,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.osm_map.auto_registration import render_preview
root=Path('/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/osm_batch34_20261107');out=Path('reports/citynav_osm_batch34');store=Path('/home/20261024_1/.local/share/citynav-osm-controls/data');d=json.load(open(out/'summary.json'));rows=d['rows'];reasons=Counter()
for r in rows:
 raw=json.load(open(root/r['map_name']/'result.json'));flags=[]
 if raw['building_proxy_polygons']<3:flags.append('insufficient_building_proxies')
 if raw['edge_samples']<100:flags.append('insufficient_edge_samples')
 if raw['structure_cost_final']>10:flags.append('high_structural_cost')
 if r['local_max_corner_shift_m']>30:flags.append('unstable_local_correction')
 r['risk_flags']=flags;reasons.update(flags)
json.dump(d,open(out/'summary.json','w'),indent=2)
fig,ax=plt.subplots(1,2,figsize=(13,4));labels=['verified','pending_validation','unreliable'];x=np.arange(2)
for j,g in enumerate(labels):ax[0].bar(x+(j-1)*.25,[sum(r['city']==c and r['grade']==g for r in rows) for c in ['birmingham','cambridge']],width=.25,label=g)
ax[0].set(xticks=x,xticklabels=['Birmingham (12)','Cambridge (22)'],ylabel='Scenes',title='Independent validation status, not completion rate');ax[0].legend()
ax[1].bar(np.arange(34),[r['local_max_corner_shift_m'] for r in rows]);ax[1].axhline(15,color='orange',ls='--',label='15m selection limit');ax[1].axhline(30,color='red',ls='--',label='30m risk flag');ax[1].set(ylabel='Maximum corner shift (m)',xlabel='Scene order in scenes.csv',title='Rejected/local proposal motion, NOT geographic error');ax[1].legend();fig.tight_layout();fig.savefig(out/'overview.png',dpi=140);plt.close(fig)
cases=['birmingham_block_3','cambridge_block_3','birmingham_block_11','cambridge_block_33'];(out/'cases').mkdir(exist_ok=True)
for name in cases:
 shutil.copy2(store/'auto_proposals'/name/'alignment.png',out/'cases'/(name+'_selected.png'))
 if name=='cambridge_block_33':
  raw=json.load(open(root/name/'result.json'));raw['initialization_source']='frozen city seed';render_preview('/home/20261024_1/Workspace/DATA/sbfnav-data',raw,out/'cases'/(name+'_rejected_local.png'))
cp=json.load(open(root/'birmingham_block_3_public_polygon_check.json'));shutil.copy2(root/'birmingham_block_3_validated_osm.json',out/'birmingham_block_3_osm_local_xy.json');json.dump(cp,open(out/'public_polygon_proxy_check.json','w'),indent=2)
manifest=json.load(open(root/'manifest.json'));assert manifest['optimizer_sha256']==hashlib.sha256(subprocess.check_output(['git','show','71bd7e3d:tools/osm_map/auto_registration.py'])).hexdigest();assert manifest['source_script_sha256']==hashlib.sha256(subprocess.check_output(['git','show','71bd7e3d:scripts/batch_citynav_osm_registration.py'])).hexdigest()
json.dump(dict(run_source_commit='71bd7e3d',manifest=manifest,control_snapshots_sha256={c:hashlib.sha256((root/(c+'_controls_snapshot.json')).read_bytes()).hexdigest() for c in ['birmingham','cambridge']},risk_flags=dict(reasons),optimizer_used_target_gt=False,formal_dataset_task_labels_loaded=False,weights_modified=False),open(out/'provenance.json','w'),indent=2)
lines=['|场景|计算|质量等级|独立点|RMSE/P95/最大(m)|选择|风险标志|','|---|---|---|---:|---|---|---|']
for r in rows:
 v=r.get('validation');err=f"{v['rmse_m']:.3f}/{v['p95_m']:.3f}/{v['max_m']:.3f}" if v else '未知';grade={'verified':'已验证≤5m','pending_validation':'待验证','unreliable':'不可靠'}[r['grade']];lines.append(f"|{r['map_name']}|完成|{grade}|{v['n'] if v else 0}|{err}|城市统一|{', '.join(r['risk_flags']) or '—'}|")
(out/'scenes_zh.md').write_text('\n'.join(lines)+'\n');print(dict(reasons));print('report artifacts ready')
