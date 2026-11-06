"""Language-only graph contract and independent-review-gated parsing metrics."""
import collections,re

ROLES={'target','landmark','intermediate','reference'}

def public_record(row):
 return dict(index=row['index'],text=row.get('text',row.get('instruction')),landmark_catalog=list(row['landmark_catalog']))

def relation_key(s):
 s=' '.join(str(s).lower().replace('_',' ').split())
 return {'beside':'next to','adjacent to':'next to','in':'inside','in front of':'front of','opposite':'across from'}.get(s,s)

def validate_graph(text,graph):
 errors=[];entities=[];ids=set()
 for e in graph.get('entities',[]):
  try:
   a,b=int(e['start']),int(e['end']);eid=str(e['id']);role=e['role'];value=str(e['text'])
   if not 0<=a<b<=len(text) or text[a:b]!=value or eid in ids or role not in ROLES:raise ValueError()
   entities.append(dict(id=eid,text=value,start=a,end=b,role=role,type=str(e.get('type','unknown'))));ids.add(eid)
  except (KeyError,ValueError,TypeError):errors.append(dict(kind='invalid_entity_span_or_role',raw=e))
 edges=[]
 for e in graph.get('relations',[]):
  if str(e.get('subject')) not in ids or str(e.get('object')) not in ids or e.get('subject')==e.get('object') or not str(e.get('relation','')).strip():errors.append(dict(kind='invalid_relation_endpoint',raw=e));continue
  edges.append(dict(subject=str(e['subject']),object=str(e['object']),relation=str(e['relation']),evidence=str(e.get('evidence','')),reference_frame=str(e.get('reference_frame','unspecified'))))
 return dict(entities=entities,relations=edges,validation_errors=errors,ambiguities=graph.get('ambiguities',[]),target_description=graph.get('target_description',''))

def relation_paths(graph,max_hops=4):
 nodes={e['id']:e for e in graph['entities']};adj=collections.defaultdict(list)
 for edge in graph['relations']:
  a,b=edge['subject'],edge['object'];adj[a].append((b,edge,1));adj[b].append((a,edge,-1))
 paths=[]
 for root,e in nodes.items():
  if e['role']!='target':continue
  queue=[([root],[])]
  while queue:
   history,steps=queue.pop(0);last=history[-1]
   if len(history)>1 and nodes[last]['role']=='landmark':paths.append(dict(nodes=history,steps=steps,hops=len(steps)));continue
   if len(steps)>=max_hops:continue
   for nxt,edge,direction in adj[last]:
    if nxt not in history:queue.append((history+[nxt],steps+[dict(**edge,orientation=direction)]))
 return paths

def graph_sets(g):
 lookup={e['id']:(e['start'],e['end']) for e in g['entities']}
 entities={(e['start'],e['end'],e['role']) for e in g['entities']};triples={(lookup[x['subject']],relation_key(x['relation']),lookup[x['object']]) for x in g['relations'] if x['subject'] in lookup and x['object'] in lookup}
 return entities,triples

def score_reviewed(predictions,reviews):
 gold={r['index']:r for r in reviews if r.get('reviewed') and r.get('independent') and r.get('reviewer') and r.get('graph') and r.get('source')=='human_editor'};pred={r['index']:r for r in predictions};counts={k:[0,0,0] for k in ['entities','target','landmark','intermediate','triples']};desc=chains=classes=chain_tasks=0
 for i,row in gold.items():
  a=pred.get(i,{}).get('graph',dict(entities=[],relations=[]));b=row['graph'];ea,ta=graph_sets(a);eb,tb=graph_sets(b)
  for key,sa,sb in [('entities',ea,eb),('triples',ta,tb)]+[(role,{x for x in ea if x[2]==role},{x for x in eb if x[2]==role}) for role in ['target','landmark','intermediate']]:
   counts[key][0]+=len(sa&sb);counts[key][1]+=len(sa);counts[key][2]+=len(sb)
  desc+=bool(a.get('target_description')) and a.get('target_description')==b.get('target_description');ca=relation_paths(a);cb=relation_paths(b)
  def canon_paths(paths,g):
   spans={e['id']:(e['start'],e['end']) for e in g['entities']}
   return {tuple((spans[s['subject']],relation_key(s['relation']),spans[s['object']],s['orientation']) for s in p['steps']) for p in paths}
  if cb:
   chain_tasks+=1;chains+=canon_paths(ca,a)==canon_paths(cb,b)
  classes+=({min(x['hops'],2) for x in ca}=={min(x['hops'],2) for x in cb})
 result=dict(reviewed_count=len(gold))
 for key,(tp,npred,ngold) in counts.items():
  precision=tp/npred if npred else 0.;recall=tp/ngold if ngold else 0.;result[key]=dict(Precision=precision,Recall=recall,F1=2*precision*recall/(precision+recall) if precision+recall else 0.,TP=tp,predicted=npred,gold=ngold) if gold else None
 result.update(description_exact=desc/len(gold) if gold else None,complete_chain_accuracy=chains/chain_tasks if chain_tasks else None,gold_chain_tasks=chain_tasks,direct_indirect_accuracy=classes/len(gold) if gold else None)
 return result
