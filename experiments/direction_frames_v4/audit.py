"""Phase 0: text-only binding extensions; separate public input from private GT."""
from shared import *
from shapely.geometry import Polygon
import copy,pandas as pd
from road_language import extract

def text_extension(row,catalog):
 text=row['instruction'];s=row['span_start'];w=row['semantic_word']
 if row['status']=='bound' or w not in MAIN or not CUE.search(text):return None
 # Only direct static directional relations, never anonymous nested referents.
 m=PATTERNS[row['word']].match(text[s:])
 if not m:return None
 after=s+m.end();hits=catalog.spans(text);hs=[h for h in hits if h['start']>=after and re.fullmatch(r'\s*(?:the\s+)?',text[after:h['start']],re.I)]
 if len(hs)!=1:return None
 h=hs[0];ids,res,ec=catalog.resolve(h)
 if not ids:return None
 tail=text[h['end']:]
 if re.match(r'\s+(?:building\s+)?(?:and|or|parking|yard|car\s+park)\b',tail,re.I):return None
 # A sentence-initial target NP immediately owns on the LEFT/RIGHT OF L.
 prefix=text[:s];prefix=re.sub(r'\b(?:directly\s+)?(?:on|to|at|in)\s+(?:the\s+)?$','',prefix,flags=re.I)
 direct=re.fullmatch(r'(?:Coming from [^.]+?\s+)?(?:The|A|An|This is the)\s+(?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|twelfth|last|grey|gray|white|black|brown|red|blue|roof|roofed|corner|large|small|big|concrete|that|is|located|situated)\s+)*(?:building|house|car|vehicle)\s*(?:(?:that\s+)?is\s+)?',prefix,re.I)
 rule=None
 if direct and not any(z['start']<s for z in hits):rule='simple_initial_target_NP_direction'
 # Textual discourse chain: every later sentence up to this relation starts It is,
 # with no intervening named or anonymous grammatical subject. Object mentions do
 # not introduce new SUBJECTS. The initial target NP must explicitly be a car.
 a=max(text.rfind('.',0,s),text.rfind('!',0,s))+1;local=text[a:s]
 if re.fullmatch(r'\s*It\s+is\s+(?:(?:on|to|at)\s+)?(?:the\s+)?',local,re.I):
  before=text[:a];parts=[z.strip() for z in before.split('.') if z.strip()]
  if parts and re.match(r'^(?:The|A|An)\s+(?:(?:black|white|grey|gray|blue|red|light|dark)\s+)*car\b',parts[0],re.I) and all(re.match(r'^It\s+is\b',z,re.I) for z in parts[1:]):rule='sentence_initial_It_is_target_discourse_chain'
 if not rule:return None
 r=copy.deepcopy(row);r.update(status='bound',reason='',scope='direct',reference_ids=ids,reference_text=text[h['start']:h['end']],reference_span=[h['start'],h['end']],entity_resolution=res,subject_role='annotated_target',subject_ids=[str(row['target_id'])],binding_rule=rule,subject_binding_rule=rule,parse_confidence=min(ec,.95 if rule=='simple_initial_target_NP_direction' else .9),confidence_tier='high' if min(ec,.95 if rule=='simple_initial_target_NP_direction' else .9)>=.95 else 'medium',origin='v4_text_scope_extension')
 return r

def run():
 OUT.mkdir(exist_ok=True);R=readlines(V3/'results/relations.jsonl.gz');oldX={r['occurrence_id']:r for r in readlines(V3/'results/public_inputs.jsonl.gz')};train=readj(DATA/'original_citynav/citynav_train_seen.json');O=readj(DATA/'cityrefer_original/objects.json');catalogs={m:TypedRoadCatalog([Entity(str(i),o['name'],o['object_type']) for i,o in obs.items() if o['name']]) for m,obs in O.items()};new=[];X=[];ledger=[]
 existing={r['occurrence_id'] for r in R}
 for i,ep in enumerate(train):
  text=ep['descriptions'][0];m=f"{ep['area']}_block_{ep['block']}";tid=str(ep['object_ids'][0]);ann=ep['ann_ids'][0]
  for z in extract(text,catalogs[m]):
   rid=f"{m}:{tid}:{ann}:{z['span_start']}"
   desc=O[m][tid]['descriptions'];assert ann<len(desc) and text_key(desc[ann])==text_key(text),'new relation annotation/target alignment mismatch'
   if rid in existing:continue
   R.append({**z,'semantic_word':z['word'],'occurrence_id':rid,'instruction':text,'episode':i,'map':m,'target_id':tid,'ann_id':ann,'target_category':O[m][tid]['object_type'],'cluster':f'{m}:{tid}','subject_ids':[tid],'subject_role':'annotated_target','scope':'road_travel_relative','status':'bound','reason':'','origin':'v4_explicit_road_anchor','annotation_alignment':'TRAIN_original_exact','entity_resolution':'independent_named_roads','reference_candidate_ids':z['reference_ids']});existing.add(rid)
 for prev in R:
  r=text_extension(prev,catalogs[prev['map']]) or copy.deepcopy(prev);ledger.append(r)
  if r['status']!='bound':continue
  rid=r['occurrence_id']
  if rid in oldX:X.append(oldX[rid]);continue
  obs=O[r['map']];ep=train[r['episode']];p={'cluster':r['cluster']};hits=catalogs[r['map']].spans(r['instruction']);mentioned=[]
  for h in hits:
   ids,res,conf=catalogs[r['map']].resolve(h);mentioned.append({**h,'resolved_ids':ids or [],'resolution':res,'confidence':conf})
  scene={i:{'id':i,'name':o['name'],'category':o['object_type'],'position':o['position'][:2],'contour':o['contour']} for i,o in obs.items() if i not in set(r['subject_ids']+[str(r['target_id'])]) and (o['name'] or o['object_type']=='Water')}
  x={'occurrence_id':rid,'cluster':p['cluster'],'episode':r['episode'],'map':r['map'],'origin':r['origin'],'instruction':r['instruction'],'word':r['semantic_word'],'direction_span':[r['span_start'],r['span_end']],'reference_ids':r['reference_ids'],'reference_text':r['reference_text'],'reference_span':r['reference_span'],'subject_role':r['subject_role'],'subject_ids':r['subject_ids'],'parse_tier':r['confidence_tier'],'parse_confidence':r['parse_confidence'],'binding_rule':r['binding_rule'],'start_xy':ep['trajectory'][0][:2],'mentioned':mentioned,'scene':scene};X.append(x)
  new.append({'occurrence_id':rid,'instruction':r['instruction'],'old_reason':prev.get('v2_reason',prev['reason']),'binding_rule':r['binding_rule'],'reference_ids':r['reference_ids'],'parse_confidence':r['parse_confidence']})
 scan=[]
 for i,e in enumerate(train):
  t=e['descriptions'][0];m=f"{e['area']}_block_{e['block']}";cues=[{'span':list(z.span()),'text':z[0]} for z in CUE.finditer(t)];hits=catalogs[m].spans(t)
  if cues:scan.append({'episode':i,'map':m,'instruction':t,'semantic_cues':cues,'named_mentions':hits,'direction_words':[{'span':list(z.span()),'text':z[0]} for z in TOKEN.finditer(t)],'ledger_relations':[r['occurrence_id'] for r in R if r['episode']==i]})
 writelines(OUT/'relations.jsonl.gz',ledger);writelines(OUT/'public_inputs.jsonl.gz',X);writelines(OUT/'train_semantic_scan.jsonl.gz',scan);writej(OUT/'binding_extensions.json',new);pd.DataFrame(new).to_csv(OUT/'binding_extensions.csv',index=False)
 writej(OUT/'dataset.json',{'train_episodes':len(train),'train_maps':len({(e['area'],e['block']) for e in train}),'candidate_maps':len({r['map'] for r in R}),'candidate_relations':len(R),'v3_candidate_relations':4659,'new_relation_syntax':len(R)-4659,'all_bound':len(X),'v3_bound_preserved':len(oldX),'new_bound':len(new),'semantic_scan_episodes':len(scan),'semantic_named_direction_episodes':sum(bool(r['named_mentions'] and r['direction_words']) for r in scan),'source_files':[{'path':str(f),'sha256':sha(f)} for f in [DATA/'original_citynav/citynav_train_seen.json',DATA/'cityrefer_original/objects.json']],'v3_ledger_sha256':sha(V3/'results/relations.jsonl.gz'),'public_input_sha256':sha(OUT/'public_inputs.jsonl.gz'),'binding_GT_used':False})
 print(readj(OUT/'dataset.json'));print('NEW',new)
if __name__=='__main__':run()
