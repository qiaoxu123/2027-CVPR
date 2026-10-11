"""Role/scope and localization evaluation. GT keys stay outside inference."""
import gzip,json,time,hashlib,collections
from pathlib import Path
import numpy as np
import torch
from .parser import parse
from .inference import forward
from experiments.language_object_graph.run import load,row,aggregate,dump_rows,METRICS
from experiments.language_object_graph.objects import ObjectGraph
from experiments.language_object_graph.pipeline import forward as old_forward
from experiments.semantic_polygon_grounding.audit import save,sha
from experiments.bidirectional_core.stats import paired_difference
OUT=Path('runs/role_aware_object_graph')

def dependency_lookup(split):
 raw=json.loads((OUT/f'{split}_dependencies.json').read_text());return raw['parses']
def dependency(text,cache):return cache[hashlib.sha256(text.encode()).hexdigest()]
def overlap(a,b):return a[0]<b[1] and a[1]>b[0]
def audit(parses,labels):
 rows=[];role=0;scope=0;scope_count=0;unknown=0;hyp=0
 for label in labels:
  lang=parses[label['sample_id']];entities=lang['entities'];pred=entities[0] if entities else None;gold=label['target_head_span'];is_unknown=lang['confidence']==0;unknown+=is_unknown;hyp+=len(lang['hypotheses'])>1 or any(len(f['subject_hypotheses'])>1 for f in lang['factors'])
  good=bool(is_unknown if gold is None else pred is not None and not is_unknown and overlap(pred['head_span'],gold));role+=good
  cases=[]
  for truth in label['relations']:
   matched=False
   if truth['kind'].endswith('unknown'):
    matched=not any(f['kind'] not in ('near','along') and overlap(entities[f['subject']]['head_span'],truth['subject_span']) for f in lang['factors'])
   else:
    for f in lang['factors']:
     if f['kind']!=truth['kind'] or not overlap(entities[f['subject']]['head_span'],truth['subject_span']):continue
     predicted=[entities[i]['head_span'] for i in f['anchors']]
     if all(any(overlap(p,g) for p in predicted) for g in truth['anchor_spans']):matched=True
   scope+=matched;scope_count+=1;cases.append(dict(truth=truth,primary_scope_correct=matched))
  rows.append(dict(sample_id=label['sample_id'],role_correct=good,unknown=is_unknown,multiple_hypotheses=len(lang['hypotheses'])>1,cases=cases,language=lang))
 return dict(n=len(labels),role_accuracy=role/max(len(labels),1),scope_n=scope_count,scope_accuracy=scope/max(scope_count,1),unknown=unknown,multiple_scope_or_role=hyp,label_status='Single assistant raw-text adjudication, no independent human verification. Not a certified stage1 pass.'),rows

def evaluate(states,labels,graphs,method,weight,backend,cache):
 rows=[];parses={};times=[]
 for state in states:
  m=state['metadata'];graph=graphs[m['map_name']];start=time.perf_counter();dep=dependency(m['instruction'],cache) if backend=='dependency' else None
  if method in ('A0','A1'):
   result=old_forward(graph,m['instruction'],state['public'],method='A2',weight=0 if method=='A0' else .5)
   logits=np.asarray(state['public']['static_logits']) if method=='A0' else result['logits']
  else:
   result=forward(graph,m['instruction'],state['public'],method=method,weight=weight,backend=backend,dependency=dep);logits=result['logits'];parses[m['sample_id']]=result['language']
  times.append(time.perf_counter()-start);r=row(state,labels[m['sample_id']],graph,result,logits);r.update(seconds=times[-1],parse_hypotheses=result.get('parse_hypotheses'),reference_posteriors=result.get('reference_posteriors'),supported=result.get('supported'),object_metric_source='P5 unary shadow, not native A0' if method=='A0' else 'public object posterior')
  r['goal_logits']=logits.tolist();rows.append(r)
 return rows,parses,dict(seconds=sum(times),mean_ms=1000*np.mean(times),p95_ms=1000*np.quantile(times,.95))

def weak_metrics(states,labels,graphs,parses,weak):
 truth={r['sample_id']:r for r in weak};rows=[];totals=collections.Counter()
 for state in states:
  m=state['metadata'];g=graphs[m['map_name']];l=parses[m['sample_id']];target=l['entities'][0] if l['entities'] else None;tid=g.byid[str(labels[m['sample_id']]['target_object_id'])];w=truth[m['sample_id']];names=w['reference_names'];totals['families']+=1
  valid=target is not None and l['confidence']>0;totals['type_correct']+=bool(valid and target['object_type']==g.types[tid]);totals['unknown']+=not valid;totals['target_named']+=bool(target and target['named_ids']);totals['target_name_identity_correct']+=bool(target and tid in target['named_ids'])
  reference_spans=[]
  from .parser import words
  text_words=words(m['instruction'])
  for name in names:
   nw=[x[0] for x in words(name)]
   for i in range(len(text_words)-len(nw)+1):
    if nw and [x[0] for x in text_words[i:i+len(nw)]]==nw:reference_spans.append([text_words[i][1],text_words[i+len(nw)-1][2]])
  for span in reference_spans:
   totals['literal_weak_references']+=1;totals['literal_reference_role_correct']+=any(e['role']=='reference' and overlap(e['head_span'],span) for e in l['entities'])
  rows.append(dict(sample_id=m['sample_id'],target_type_correct=bool(valid and target['object_type']==g.types[tid]),unknown=not valid,weak_reference_spans=reference_spans,language=l))
 return dict(counts=dict(totals),target_type_accuracy=totals['type_correct']/totals['families'],reference_role_accuracy=totals['literal_reference_role_correct']/max(totals['literal_weak_references'],1),status='Offline GT type + literal processed reference names, weak automatic supervision; missing/nonliteral references reported, not semantic scope truth'),rows

def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--phase',choices=['dev','seen'],required=True);args=a.parse_args();torch.set_num_threads(1)
 graphs={p.stem:ObjectGraph.restore(torch.load(p,weights_only=True)) for p in Path('runs/language_object_graph/graphs').glob('*.pt')};start=time.time()
 if args.phase=='dev':
  if (OUT/'dev_complete.json').exists():raise RuntimeError('preserve completed Dev')
  metrics={};allrows={};selection={};audits={};weak_stats={};timing={};text_labels=json.loads(Path(__file__).with_name('text_audit.json').read_text())
  for split in ('train','dev'):
   states,labels,_=load(split);cache=dependency_lookup(split);metrics[split]={};allrows[split]={};timing[split]={};weak_stats[split]={}
   for method in ('A0','A1'):
    key=method;r,_,t=evaluate(states,labels,graphs,method,.5,'rule',cache);metrics[split][key]=aggregate(r);allrows[split][key]=r;timing[split][key]=t;dump_rows(OUT/f'{split}_{key}.jsonl.gz',r)
   for backend in ('rule','dependency'):
    for method in ('A2','A3','A4'):
     for weight in (0,.25,.5):
      key=f'{method}_{backend}_w{weight:g}';r,parses,t=evaluate(states,labels,graphs,method,weight,backend,cache);metrics[split][key]=aggregate(r);allrows[split][key]=r;timing[split][key]=t;dump_rows(OUT/f'{split}_{key}.jsonl.gz',r)
      if weight==.5:
       weak,wr=weak_metrics(states,labels,graphs,parses,json.loads((OUT/f'{split}_weak_roles.json').read_text()));weak_stats[split][key]=weak;dump_rows(OUT/f'{split}_{key}_weak_parse.jsonl.gz',wr)
       if split=='dev':
        audit_stats,ar=audit(parses,text_labels);audits[key]=audit_stats;dump_rows(OUT/f'audit_{key}.jsonl.gz',ar)
      print(split,key,metrics[split][key]['goal_r1']['mean'],flush=True)
  # Backend chosen once with Dev weak type/reference role, then method weights
  # selected from the pre-registered three values. No Seen outcomes available.
  def parser_score(b):
   w=weak_stats['dev'][f'A2_{b}_w0.5'];return w['target_type_accuracy']+w['reference_role_accuracy']
  backend=max(('rule','dependency'),key=lambda b:(parser_score(b),b=='rule'))
  for method in ('A2','A3','A4'):
   keys=[k for k in allrows['dev'] if k.startswith(f'{method}_{backend}_')];best=max(keys,key=lambda k:(metrics['dev'][k]['goal_r1']['mean'],-metrics['dev'][k]['damage']['mean'],-float(k.split('_w')[1])));selection[method]=dict(backend=backend,weight=float(best.split('_w')[1]),key=best)
  comparisons={split:{key:{m:paired_difference(rows['A1'],r,m,seed=20261011) for m in METRICS} for key,r in rows.items() if key!='A0'} for split,rows in allrows.items()}
  result=dict(metrics=metrics,selection=selection,audits=audits,weak=weak_stats,comparisons=comparisons,timing=timing,seconds=time.time()-start,phase='Train/Dev only')
  save(OUT/'dev_complete.json',result);save(OUT/'frozen_config.json',dict(selection=selection,protocol_sha=sha(Path(__file__).with_name('protocol.json')),dev_result_sha=sha(OUT/'dev_complete.json'),frozen_unix=time.time(),stage1_status='Independent human role/scope verification unavailable; no GAT/RGB/closedloop training. Seen is one fixed diagnostic confirmation, not further model selection.',prior_seen_exposure=True))
 else:
  if (OUT/'seen_complete.json').exists():raise RuntimeError('Seen already evaluated')
  frozen=json.loads((OUT/'frozen_config.json').read_text());states,labels,_=load('seen');cache=dependency_lookup('seen');metrics={};rows={};timing={}
  for method in ('A0','A1','A2','A3','A4'):
   cfg=frozen['selection'].get(method,dict(weight=.5,backend='rule'));r,_,t=evaluate(states,labels,graphs,method,cfg['weight'],cfg['backend'],cache);rows[method]=r;metrics[method]=aggregate(r);timing[method]=t;dump_rows(OUT/f'seen_{method}.jsonl.gz',r);print('seen',method,metrics[method]['goal_r1']['mean'],flush=True)
  # Exact parity with P5 fixed outcomes; no change to success or source inputs.
  with gzip.open('runs/language_object_graph/seen_A2_w0.5.jsonl.gz','rt') as f:original=[json.loads(x) for x in f]
  for a,b in zip(original,rows['A1']):
   assert a['sample_id']==b['sample_id']
   for m in ('goal_r1','goal_r4','goal_r8','ne'):assert abs(a['metrics'][m]-b['metrics'][m])<1e-5
  comparisons={key:{m:paired_difference(rows['A1'],r,m,seed=20261011) for m in METRICS} for key,r in rows.items() if key!='A0'}
  multi={m:paired_difference(rows['A3'],rows['A4'],m,seed=20261011) for m in METRICS}
  save(OUT/'seen_complete.json',dict(metrics=metrics,comparisons_vs_P5=comparisons,multi_vs_single=multi,timing=timing,seconds=time.time()-start,frozen_sha=sha(OUT/'frozen_config.json'),prior_exploratory_seen_influence='P5 Seen previously exposed; P5.1 design and weights based only on Train/Dev; this confirmation is not pristine blind validation.'))
if __name__=='__main__':main()
