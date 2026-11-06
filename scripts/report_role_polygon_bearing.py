"""Posthoc report only: never changes frozen inference or model selection."""
import json,argparse,collections,html,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.relation_bearing_strata import anchor_strata

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();r=a.run;o=Path('reports/relation_bearing');o.mkdir(exist_ok=True)
 records=json.load(open(a.manifest))['records'];parsed=[json.loads(x) for x in (r/'data/parsed.jsonl').read_text().splitlines()];s=json.load(open(r/'confirmation/summary.json'));models=s['results'];rows={k:[json.loads(x) for x in (r/'confirmation'/f'{k}_rows.jsonl').read_text().splitlines()] for k in models}
 controls=json.load(open(r/'controls_confirm/summary.json'))['results'];rows.update({k:[json.loads(x) for x in (r/'controls_confirm'/f'{k}_rows.jsonl').read_text().splitlines()] for k in controls})
 def means(rr):
  q=collections.defaultdict(list)
  for x in rr:q[x['index']].append(x['top1'])
  return {i:np.mean(v) for i,v in q.items()}
 def paired(left,right):
  l=means(rows[left]);z=means(rows[right]);ids=sorted(l);d=np.array([l[i]-z[i] for i in ids]);groups={}
  for name,key in [('physical_target',lambda i:':'.join(records[i]['episode_id'].split(':')[:2])),('map',lambda i:records[i]['map_name'])]:
   gg=collections.defaultdict(list)
   for i,v in zip(ids,d):gg[key(i)].append(v)
   values=list(gg.values());rng=np.random.default_rng(42);sums=np.array([sum(x) for x in values]);ns=np.array([len(x) for x in values]);ix=rng.integers(len(values),size=(10000,len(values)));boot=sums[ix].sum(1)/ns[ix].sum(1);groups[name]=dict(groups=len(values),CI95_pp=(np.percentile(boot,[2.5,97.5])*100).tolist())
  return dict(delta_pp=float(d.mean()*100),query_improved=int((d>0).sum()),query_harmed=int((d<0).sum()),query_equal=int((d==0).sum()),bootstrap=groups)
 comparisons={f'{l} minus {z}':paired(l,z) for l,z in [('relations_angle','old'),('relations_angle','matching_angle'),('relations_angle','public_type_prior')]}
 strata={};maps={}
 for name,rr in rows.items():
  g=collections.defaultdict(list);mg=collections.defaultdict(list)
  for x in rr:
   i=x['index'];ref=records[i]['landmarks'][x['anchor']];kind=anchor_strata(parsed[i],ref)['group'];g[kind].append(x);mg[records[i]['map_name']].append(x)
  strata[name]={k:dict(anchor_pairs=len(v),queries=len(means(v)),macro_Top1=float(np.mean(list(means(v).values()))),Top1=float(np.mean([x['top1'] for x in v])),MAE=float(np.mean([x['error'] for x in v]))) for k,v in g.items()}
  maps[name]={k:dict(queries=len(means(v)),macro_Top1=float(np.mean(list(means(v).values())))) for k,v in mg.items()}
 uncertainty={}
 for name in models:
  rr=rows[name];p8=np.array([x['probabilities'] for x in rr]);y=np.array([x['true_bin'] for x in rr]);c=p8.max(1);correct=np.array([x['top1'] for x in rr]);ece=0
  for lo in np.arange(0,1,.1):
   m=(c>=lo)&(c<lo+.1)
   if m.any():ece+=m.mean()*abs(correct[m].mean()-c[m].mean())
  uncertainty[name]=dict(ECE=float(ece),Brier=float(np.square(p8-np.eye(8)[y]).sum(1).mean()),mean_entropy=float(np.mean([x['entropy'] for x in rr])))
 result=dict(results=models,controls=controls,paired=comparisons,strata=strata,maps=maps,uncertainty=uncertainty,instance_ID_spatial_audit=json.load(open(o/"target_identity_audit.json")),confirmation_queries_total=4117,scorable_queries=3599,unscorable_no_defined_anchor=518)
 (o/'results.json').write_text(json.dumps(result,indent=2));(o/'FROZEN_BEFORE_CONFIRM.json').write_text((r/'FROZEN_BEFORE_CONFIRM.json').read_text())
 with (o/'confirmation_predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['method','index','map','anchor','true_bin','pred','top1','error','NLL','stratum'])
  for name,rr in rows.items():
   for x in rr:
    i=x['index'];w.writerow([name,i,records[i]['map_name'],x['anchor'],x['true_bin'],x['pred'],x['top1'],x['error'],x['NLL'],anchor_strata(parsed[i],records[i]['landmarks'][x['anchor']])['group']])
 fig,ax=plt.subplots(1,2,figsize=(12,4));names=list(models);ax[0].bar(range(len(names)),[models[k]['macro_Top1']*100 for k in names]);ax[0].set_xticks(range(len(names)),names,rotation=25,ha='right');ax[0].set_ylabel('Query macro Top-1 (%)')
 for name,sub in [('old','old'),('matching_angle','angle_only/matching'),('relations_angle','angle_only/relations')]:
  h=json.load(open(r/sub/'history.json'));ax[1].plot(range(1,len(h)+1),[x['dev']['macro_Top1']*100 for x in h],label=name)
 ax[1].legend();ax[1].set(xlabel='Epoch',ylabel='Development macro Top-1 (%)');fig.tight_layout();fig.savefig(o/'training_and_confirmation.png',dpi=150);plt.close(fig)
 # Actual public-map examples, with TRAIN GT used only in this posthoc plotting block.
 cat=json.load(open('/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json'));base={(x['index'],x['anchor']):x for x in rows['old']};chosen=[]
 for status in ['gain','loss','both_wrong','both_correct']:
  candidates=[x for x in rows['relations_angle'] if ('gain' if x['top1'] and not base[x['index'],x['anchor']]['top1'] else 'loss' if not x['top1'] and base[x['index'],x['anchor']]['top1'] else 'both_correct' if x['top1'] else 'both_wrong')==status]
  seen=set()
  for x in candidates:
   key=x['physical_id']
   if key in seen:continue
   seen.add(key);chosen.append((status,x))
   if len(seen)>=3:break
 cards=[]
 for j,(status,x) in enumerate(chosen):
  i=x['index'];row=records[i];ref=row['landmarks'][x['anchor']];objects=cat[row['map_name']];poly=np.array(objects[str(ref['id'])]['contour']);anchor=poly.mean(0);b=np.array(row['world_bounds']);gt=b[:2]+np.array(row['target_xy'])*(b[2:]-b[:2]);fig,ax=plt.subplots(1,2,figsize=(11,4))
  for obj in objects.values():
   pts=np.array(obj.get('contour',[]))
   if pts.ndim==2 and len(pts)>2 and np.linalg.norm(pts.mean(0)-anchor)<250:ax[0].plot(*np.vstack([pts,pts[0]]).T,color='silver',lw=.5)
  ax[0].fill(*poly.T,alpha=.5,color='orange');ax[0].scatter(*anchor,c='orange',label='Public anchor vertex mean');ax[0].scatter(*gt,c='green',marker='*',s=100,label='TRAIN GT: posthoc only');ang=x['pred']*np.pi/4;ax[0].arrow(*anchor,70*np.cos(ang),70*np.sin(ang),color='blue',width=1);ax[0].set_aspect('equal');ax[0].legend(fontsize=6);ax[0].set_title(row['map_name']+' / '+status);ax[0].set_xlabel('Map X (not asserted geographic east)')
  old=base[i,x['anchor']];ax[1].bar(np.arange(8)-.2,old['probabilities'],width=.4,label='Old');ax[1].bar(np.arange(8)+.2,x['probabilities'],width=.4,label='Relations angle');ax[1].axvline(x['true_bin'],color='green',ls='--');ax[1].set(xlabel='Map angle bin: 0 = +X; CCW',ylabel='Probability');ax[1].legend();fig.tight_layout();fn=f'case_{j:02d}.png';fig.savefig(o/fn,dpi=140);plt.close(fig)
  cards.append(f'<article><h3>{status}: {i}</h3><p>{html.escape(row["instruction"])}</p><pre>{html.escape(json.dumps(parsed[i]["paths"],ensure_ascii=False,indent=2))}</pre><a href="{fn}"><img width="900" src="{fn}"></a></article>')
 (o/'index.html').write_text('<meta charset="utf-8"><title>Language–Polygon bearing</title><h1>Frozen TRAIN map-heldout confirmation</h1><p>No RGB, no navigation; map GT overlays are posthoc.</p><img width="1000" src="training_and_confirmation.png">'+''.join(cards))
 print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
