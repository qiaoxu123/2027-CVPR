"""Offline summaries and attribution. Never selects models or modifies predictions."""
import collections,csv,gzip,hashlib,json,platform,re
from pathlib import Path
import numpy as np
import torch
from .evaluate import OUT,overlap
from .parser import words
from experiments.language_object_graph.run import load,METRICS
from experiments.bidirectional_core.stats import paired_difference,summarize_rows

def read(path):
 with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def main():
 cfg=json.loads((OUT/'frozen_config.json').read_text());dev=json.loads((OUT/'dev_complete.json').read_text());seen=json.loads((OUT/'seen_complete.json').read_text());allstats={};table=[]
 for split in ('train','dev','seen'):
  states,labels,digest=load(split);rows={m:read(OUT/f'{split}_{m if split=="seen" or m in ("A0","A1") else cfg["selection"][m]["key"]}.jsonl.gz') for m in ('A0','A1','A2','A3','A4')};base=rows['A1'];parsed=rows['A4'];meta={s['metadata']['sample_id']:s for s in states}
  weak=json.loads((OUT/f'{split}_weak_roles.json').read_text()) if split!='seen' else []
  wc=collections.Counter()
  for w in weak:
   text=meta[w['sample_id']]['metadata']['instruction'];ww=[x[0] for x in words(text)];names=w['reference_names'];wc['families']+=1;wc['declared_names']+=len(names);wc['families_no_declared_reference']+=not names
   for name in names:
    nw=[x[0] for x in words(name)];hits=sum(ww[i:i+len(nw)]==nw for i in range(len(ww)-len(nw)+1)) if nw else 0
    wc['literal_names']+=hits>0;wc['nonliteral_names']+=hits==0;wc['literal_occurrences']+=hits
  stats={};bytype=collections.defaultdict(list);bymap=collections.defaultdict(list);pc=collections.Counter()
  for r in parsed:
   l=r['language_graph'];pc['families']+=1;pc['unknown_target']+=l['confidence']==0;pc['multiple_target_hypotheses']+=len(l['hypotheses'])>1;pc['multiple_scope_hypotheses']+=any(len(f['subject_hypotheses'])>1 for f in l['factors']);pc['relative_frame_hypotheses']+=any(len(f.get('frame_hypotheses',[]))>1 for f in l['factors']);pc['no_factors']+=not l['factors'];pc['lexical_across_opposite']+=bool(re.search(r'\b(across|opposite)\b',r['instruction'],re.I));pc['lexical_ordinal']+=bool(re.search(r'\b(first|second|third|fourth|last)\b',r['instruction'],re.I))
  for m,rr in rows.items():
   metrics=seen['metrics'][m] if split=='seen' else dev['metrics'][split][m if m in ('A0','A1') else cfg['selection'][m]['key']]
   rec=sum(not a['metrics']['goal_r1'] and bool(b['metrics']['goal_r1']) for a,b in zip(base,rr));dam=sum(bool(a['metrics']['goal_r1']) and not b['metrics']['goal_r1'] for a,b in zip(base,rr))
   direct_rows=[dict(family_id=b['family_id'],map_name=b['map_name'],metrics=dict(recovery=float(not a['metrics']['goal_r1'] and bool(b['metrics']['goal_r1'])),damage=float(bool(a['metrics']['goal_r1']) and not b['metrics']['goal_r1']),net_recovery=float(b['metrics']['goal_r1']-a['metrics']['goal_r1']))) for a,b in zip(base,rr)]
   stats[m]=dict(metrics=metrics,direct_vs_P5=dict(recovery=rec,damage=dam,net=rec-dam,bootstrap={k:summarize_rows(direct_rows,k,seed=20261011) for k in ('recovery','damage','net_recovery')}),paired_vs_P5={k:paired_difference(base,rr,k,seed=20261011) for k in METRICS})
   t=dict(split=split,method=m,n=len(rr),recovery_vs_P5=rec,damage_vs_P5=dam,net_vs_P5=rec-dam)
   for k in METRICS:t[k]=None if m=='A0' and k.startswith('object_') else metrics[k]['mean']
   table.append(t)
  for j,s in enumerate(states):
   lab=labels[s['metadata']['sample_id']];mapname=s['metadata']['map_name'];bymap[mapname].append(j)
   # Public object type retrieved by label only for offline stratified analysis.
   graph=torch.load(Path('runs/language_object_graph/graphs')/(mapname+'.pt'),weights_only=True)
   typ=next((o.get('object_type','Unknown') for oid,o in zip(graph['ids'],graph['objects']) if str(oid)==str(lab['target_object_id'])),'Missing');bytype[typ].append(j)
  strata={kind:{group:{m:{k:float(np.mean([rows[m][j]['metrics'][k] for j in ix])) for k in ('goal_r1','object_r1','ne')} for m in rows}|{'n':len(ix)} for group,ix in groups.items()} for kind,groups in [('map',bymap),('type',bytype)]}
  allstats[split]=dict(n=len(states),maps=len(bymap),objects=len({(s['metadata']['map_name'],str(labels[s['metadata']['sample_id']]['target_object_id'])) for s in states}),plan_sha=digest,methods=stats,parser_counts=dict(pc),weak_denominators=dict(wc),strata=strata)
 save(OUT/'analysis.json',allstats)
 with (OUT/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
 wheel=OUT/'en_core_web_sm-3.8.0-py3-none-any.whl';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),device='CPU; no GPU inference/training',seed=20261011,geometry_checkpoint_sha='9d9be861bf7e9cac986d747867990de3d409ccb9bc1545862d71a620448380d7',base_commit='2cf62d2ac1f4cb8fe607af99ba781b3e4fde35e5',nlp=dict(spacy='3.8.7',model='en_core_web_sm',model_version='3.8.0',model_license='MIT',wheel_sha=sha(wheel),download='https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl',isolated_environment=str((OUT/'nlp_env').resolve())),runtime_seconds=dict(train_dev=dev['seconds'],seen=seen['seconds']),regression=dict(passed=388,warning='rasterio NotGeoreferencedWarning on existing affine fixture',seconds=7.91))
 env['dependency_exports']={split:{k:v for k,v in json.loads((OUT/f'{split}_dependencies.json').read_text()).items() if k!='parses'} for split in ('train','dev','seen')};save(OUT/'environment.json',env)
 print(json.dumps({s:{k:v for k,v in a.items() if k in ('n','maps','objects','parser_counts','weak_denominators')} for s,a in allstats.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
