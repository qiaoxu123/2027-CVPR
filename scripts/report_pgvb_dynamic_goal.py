"""Package sealed TRAIN controller-interface experiments; no new predictions."""
import argparse,csv,gzip,html,json,shutil
from pathlib import Path
import cv2,numpy as np,shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import file_sha256
from graphnav.landmark_navigation_diagnostic import public_region
from graphnav.landmark_projection import world_to_frame
from scripts.report_pgvb_closedloop import summarize
METHODS=['E4','Stability','Arrival','Both']


def main():
 p=argparse.ArgumentParser()
 for k in ['develop','confirm','selections','execution','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--strict',type=Path);p.add_argument('--strict-selection',type=Path)
 a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True);allgroups={};allpairs={};allrows={};case_lists={};checks={}
 freeze=json.loads((o/'freeze_before_confirmation.json').read_text())
 for path,sha in freeze['code_sha256'].items():assert file_sha256(path)==sha
 cohorts=[('develop',a.develop,24),('confirm',a.confirm,48)]
 if a.strict:cohorts.append(('strict_confirm',a.strict,24))
 for subset,run,n in cohorts:
  rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()];allrows[subset]=rows;assert len(rows)==4*n and {(x['index'],x['method']) for x in rows}=={(i,m) for i in range(n) for m in METHODS}
  assert json.loads((run/'metrics.json').read_text())['complete'];tasks={t['index']:t for t in json.loads(((a.strict_selection if subset=='strict_confirm' else a.selections/subset)/'tasks_public.json').read_text())};lookup={(x['index'],x['method']):x for x in rows};folder=o/subset;folder.mkdir(exist_ok=True)
  summaries={};pairs={};checks[subset]=dict(total_rollouts=len(rows),arrival_trajectory_prefix_exact=0,paired_start_equal=True)
  for m in METHODS:
   group=[x for x in rows if x['method']==m];s=summarize(group);s['correct_stops']=sum(int(x['metrics']['SR'] and x['termination']=='policy_stop') for x in group);s['false_stops']=sum(int(x['metrics']['false_stop']) for x in group);s['jumps_per100_transitions']=100*s['goal_jumps_over20']/max(s['goal_transition_count'],1);s['pending_decisions']=sum(d.get('goal_commitment',{}).get('pending',False) for x in group for d in x['step_diagnostics']);s['quantized_stop_releases']=sum(d.get('actual_should_stop',False) and not d.get('original_mm_guard_should_stop',False) for x in group for d in x['step_diagnostics']);summaries[m]=s
   if m!='E4':pairs[m]=dict(SR_corrections=[i for i in range(n) if not lookup[(i,'E4')]['metrics']['SR'] and lookup[(i,m)]['metrics']['SR']],SR_damages=[i for i in range(n) if lookup[(i,'E4')]['metrics']['SR'] and not lookup[(i,m)]['metrics']['SR']],correct_stop_gains=[i for i in range(n) if not (lookup[(i,'E4')]['metrics']['SR'] and lookup[(i,'E4')]['termination']=='policy_stop') and lookup[(i,m)]['metrics']['SR'] and lookup[(i,m)]['termination']=='policy_stop'])
  for i in range(n):
   base=lookup[(i,'E4')];arr=lookup[(i,'Arrival')];assert all(lookup[(i,m)]['poses'][0]==base['poses'][0] for m in METHODS)
   aa=np.asarray(arr['poses']);bb=np.asarray(base['poses']);assert len(aa)<=len(bb);np.testing.assert_allclose(aa,bb[:len(aa)],atol=0,rtol=0);checks[subset]['arrival_trajectory_prefix_exact']+=1
  allgroups[subset]=summaries;allpairs[subset]=pairs
  with gzip.open(folder/'predictions.jsonl.gz','wb') as f:f.write((run/'predictions.jsonl').read_bytes())
  for name in ['run_manifest.json','metrics.json']:shutil.copyfile(run/name,folder/name)
  with (folder/'per_task.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['index','map','name','type','method','SR','OSR','NE','steps','termination','correct_stop','goal_jumps_gt20'])
   for r in rows:
    ds=r['step_diagnostics'];j=sum(np.linalg.norm(np.asarray(y['selected_goal_xy'])-np.asarray(x['selected_goal_xy']))>20 for x,y in zip(ds[:-1],ds[1:]));w.writerow([r['index'],r['map_name'],r['landmark_name'],r['category'],r['method'],r['metrics']['SR'],r['metrics']['OSR'],r['metrics']['NE'],r['metrics']['steps'],r['termination'],bool(r['metrics']['SR'] and r['termination']=='policy_stop'),j])
  # Show both benefits and harms if any; never select only favorable cases.
  chosen=[]
  def add(i):
   if i not in chosen:chosen.append(i)
  for i in pairs['Arrival']['SR_damages']+pairs['Arrival']['SR_corrections']+pairs['Arrival']['correct_stop_gains']:add(i)
  for r in rows:
   if r['method']=='E4' and not r['metrics']['SR']:add(r['index'])
  for i in range(n):
   if len(chosen)>=4:break
   add(i)
  cases=[]
  for i in chosen[:6]:
   t=tasks[i];region=public_region(t['public_polygons'],t['world_bounds']);case=folder/'cases'/f'{i:02d}';case.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(4,3,figsize=(15,13))
   for rowno,m in enumerate(METHODS):
    r=lookup[(i,m)];ds=r['step_diagnostics'];d=ds[-1];z=np.load(d['audit_artifact']);png=Path(d['audit_artifact']).with_suffix('.png');rgb=cv2.cvtColor(cv2.imread(str(png)),cv2.COLOR_BGR2RGB);shutil.copyfile(png,case/(m+'.png'));shutil.copyfile(d['audit_artifact'],case/(m+'.npz'))
    ax=axes[rowno,0];ax.imshow(rgb)
    for poly in t['public_polygons']:
     xy=world_to_frame(np.asarray(poly),z['footprint_world'],rgb.shape);ax.plot(xy[:,0],xy[:,1],'c-',lw=.7)
    ax.set_xlim(0,383);ax.set_ylim(383,0);ax.set_title(f'{m}: SR={r["metrics"]["SR"]:.0f}, NE={r["metrics"]["NE"]:.2f}m; {r["termination"]}',fontsize=9)
    ax=axes[rowno,1]
    for other,color in zip(METHODS,['gray','orange','blue','green']):
     pp=np.asarray(lookup[(i,other)]['poses']);ax.plot(pp[:,0],pp[:,1],'-o',ms=2,lw=1,color=color,label=other)
    for poly in t['public_polygons']:
     xy=np.asarray(poly);ax.plot(xy[:,0],xy[:,1],'c-',lw=1)
    ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6);ax.set_title('World XY paired real closed loops')
    ax=axes[rowno,2];actual=[float(shapely.distance(shapely.Point(r['poses'][k][:2]),region)) for k in range(len(ds))];ax.plot(actual,label='Actual NE');ax.plot([x['public_landmark_goal_distance_m'] for x in ds],label='Goal NE');ax.plot([x['predicted_goal_distance'] for x in ds],label='To current Goal');ax.axhline(20,color='gray',ls='--');ax.legend(fontsize=6,loc='upper left');ax.set_xlabel('Decision step');ax.set_ylabel('m');twin=ax.twinx();twin.plot([x['stop_probability'] for x in ds],color='purple',ls=':',label='Stop p');twin.axhline(.3,color='purple',ls='--',alpha=.4);twin.set_ylim(0,1);twin.set_ylabel('Stop probability',color='purple')
   fig.suptitle(f'{subset}: {i} {t["landmark_name"]} | fixed Horizon20, frozen weights; public Polygon task');fig.tight_layout();fig.savefig(case/'comparison.png',dpi=120);plt.close(fig);cases.append(dict(index=i,name=t['landmark_name'],file=f'{subset}/cases/{i:02d}/comparison.png'))
  case_lists[subset]=cases
  # Map-cluster paired intervals describe small-set uncertainty, not significance.
  maps=sorted({x['map_name'] for x in tasks.values()});rng=np.random.default_rng(0);boot=[]
  for _ in range(2000):
   ids=[i for mp in rng.choice(maps,len(maps),replace=True) for i,t in tasks.items() if t['map_name']==mp]
   boot.append([np.mean([lookup[(i,'Arrival')]['metrics']['SR']-lookup[(i,'E4')]['metrics']['SR'] for i in ids]),np.mean([int(lookup[(i,'Arrival')]['metrics']['SR'] and lookup[(i,'Arrival')]['termination']=='policy_stop')-int(lookup[(i,'E4')]['metrics']['SR'] and lookup[(i,'E4')]['termination']=='policy_stop') for i in ids])])
  allgroups[subset]['Arrival_vs_E4_map_bootstrap_CI95']=dict(SR=np.quantile(np.asarray(boot)[:,0],[.025,.975]).tolist(),correct_stop_fraction=np.quantile(np.asarray(boot)[:,1],[.025,.975]).tolist(),maps=len(maps))
 # Check fully sealed and map/name disjoint before reporting either group.
 dev_tasks=json.loads((a.selections/'develop/tasks_public.json').read_text());con_tasks=json.loads((a.selections/'confirm/tasks_public.json').read_text());assert not ({t['map_name'] for t in dev_tasks}&{t['map_name'] for t in con_tasks});assert not ({t['task_id'] for t in dev_tasks}&{t['task_id'] for t in con_tasks})
 (o/'summary.json').write_text(json.dumps(dict(metrics=allgroups,paired=allpairs,verification=checks),indent=2,allow_nan=False));(o/'cases.json').write_text(json.dumps(case_lists,indent=2))
 logs=o/'logs';logs.mkdir(exist_ok=True)
 for f in a.execution.iterdir():
  if f.suffix in ['.log','.exit','.sh']:shutil.copyfile(f,logs/f.name)
 fig,axes=plt.subplots(1,3,figsize=(12,4));labels=['E4','Stable','Arrival','Both'];s=allgroups['strict_confirm' if a.strict else 'confirm']
 for ax,k,title in zip(axes,['SR','correct_stops','steps'],['Fresh TRAIN SR',f'Correct Policy Stops /{s["E4"]["episodes"]}','Mean steps (Horizon fixed20)']):ax.bar(labels,[s[m][k] for m in METHODS]);ax.set_title(title)
 fig.tight_layout();fig.savefig(o/'confirmation_summary.png',dpi=160);plt.close(fig)
 page='<meta charset="utf-8"><title>PGVB动态Goal与Stop接口</title><style>body{max-width:1500px;margin:auto;font:16px sans-serif}img{max-width:100%}</style><h1>动态Goal / 到达接口：TRAIN独立确认</h1><a href="REPORT_zh.md">中文报告</a> | <a href="summary.json">完整汇总</a><img src="confirmation_summary.png">'
 for subset,cases in case_lists.items():
  page+=f'<h2>{subset}</h2><a href="{subset}/per_task.csv">逐任务CSV</a>'
  for c in cases:
   page+=f'<h3>{c["index"]}: {html.escape(c["name"])}</h3><a href="{c["file"]}"><img src="{c["file"]}"></a>'
   for m in METHODS:page+=f'<a href="{subset}/cases/{c["index"]:02d}/{m}.png">{m} raw RGB</a> '
 page+='<h2>原64条只读诊断</h2>'
 for i in [1,13,34,39]:page+=f'<img src="historical_audit/task_{i:02d}.png">'
 (o/'index.html').write_text(page);print(json.dumps(dict(metrics=allgroups,paired=allpairs,verification=checks),indent=2))
if __name__=='__main__':main()
