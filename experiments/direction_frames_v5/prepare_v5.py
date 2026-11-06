"""Phase 1: extend roles from TRAIN text only, construct fields without target GT."""
from v5core import *
from language_v5 import *
import copy

def run():
 lock=readj(RESULT/'v4_lock.json');assert all(sha(V4/k)==v for k,v in lock['files'].items())
 R=readlines(V4/'results/relations.jsonl.gz');X=readlines(V4/'results/public_inputs.jsonl.gz');oldP=readlines(V4/'results/frozen_frames.jsonl.gz');oldF=readlines(V4/'results/road_fields.jsonl.gz');bx={x['occurrence_id']:x for x in X};br={r['occurrence_id']:r for r in R};bp={p['occurrence_id']:p for p in oldP};fields={f['key']:{k:v for k,v in f.items() if k!='key'} for f in oldF};T=readj(DATA/'original_citynav/citynav_train_seen.json');O=readj(DATA/'cityrefer_original/objects.json');maps={f"{e['area']}_block_{e['block']}" for e in T};C={m:TypedRoadCatalog([Entity(i,o['name'],o['object_type']) for i,o in O[m].items() if o['name']]) for m in maps}
 oldunique=[x for x in X if bp[x['occurrence_id']]['state']=='unique_field'];knownmaps={x['map'] for x in oldunique};knownobjects={x['cluster'] for x in oldunique};knownroads={(x['map'],tuple(x['reference_ids'])) for x in oldunique};knownpairs={bp[x['occurrence_id']]['road_field_key'] for x in oldunique};normalized=[template_text(x['instruction'],x['mentioned']) for x in oldunique];scan=[];candidate=[];upgrades=[]
 for i,e in enumerate(T):
  text=e['descriptions'][0]
  if not SEARCH.search(text):continue
  m=f"{e['area']}_block_{e['block']}";roles,hits=role_pairs(text,C[m]);tid=str(e['object_ids'][0]);ann=e['ann_ids'][0]
  scan.append({'episode':i,'map':m,'instruction':text,'role_pairs':roles,'named_hits':hits,'direction_spans':[{'word':z[0].lower(),'span':list(z.span())} for z in re.finditer(r'\b(?:left|right)\b',text,re.I)],'raw_cues':[z[0] for z in SEARCH.finditer(text)]})
  for w in re.finditer(r'\b(?:left|right)\b',text,re.I):
   rid=f'{m}:{tid}:{ann}:{w.start()}'
   if rid in bx:continue # Entire v4 bound population and predictions immutable.
   possible=[]
   for role in roles:
    ok,why,population=side_scope(text,w.start(),role,hits)
    if population:possible.append((role,ok,why))
   if not possible:continue
   pairs={(tuple(r['A']['resolved_ids']),tuple(r['B']['resolved_ids'])) for r,_,_ in possible};role,ok,why=possible[0]
   if len(pairs)>1:ok=False;why='multiple_language_role_pairs'
   if role['template'].endswith('UNDERSPECIFIED') or role['template'].endswith('OBSERVER'):ok=False;why='approach_or_observer_not_directed_road_entry'
   desc=O[m][tid]['descriptions'];aligned=ann<len(desc) and text_key(desc[ann])==text_key(text)
   if not aligned:ok=False;why='annotation_alignment_mismatch'
   A=role['A'];B=role['B'];key=m+':'+','.join(A['resolved_ids'])+'<-'+','.join(B['resolved_ids']);conf=min(A['confidence'],B['confidence'],.95);mask=template_text(text,hits);sim=max((difflib.SequenceMatcher(None,mask,old).ratio() for old in normalized),default=0.)
   supported=role['template'] in {'coming_from_deictic','coming_from_road_side','prefix_coming_from_road_side'}
   novelty='duplicate_or_highly_similar' if sim>=.9 else 'old_rule_semantics_missed_binding' if supported else 'previously_uncovered_expression'
   record={'occurrence_id':rid,'episode':i,'map':m,'target_id':tid,'ann_id':ann,'target_category':O[m][tid]['object_type'],'cluster':f'{m}:{tid}','instruction':text,'word':w[0].lower(),'semantic_word':w[0].lower(),'span_start':w.start(),'span_end':w.end(),'subject_ids':[tid] if ok else [],'subject_role':'annotated_target' if ok else 'unresolved','reference_ids':A['resolved_ids'],'reference_span':[A['start'],A['end']],'reference_text':text[A['start']:A['end']],'source_ids':B['resolved_ids'],'source_span':[B['start'],B['end']],'scope':'road_travel_relative' if ok else 'unknown','status':'bound' if ok else 'excluded','reason':'' if ok else why,'parse_confidence':conf if ok else 0.,'confidence_tier':'high' if ok and conf>=.95 else 'medium' if ok else 'excluded','origin':'v5_language_extension','binding_rule':why,'template':role['template'],'novelty':novelty,'similarity_to_v4_unique_max':sim,'masked_template':mask,'new_map_vs_v4_unique':m not in knownmaps,'new_road_vs_v4_unique':(m,tuple(A['resolved_ids'])) not in knownroads,'new_road_pair_vs_v4_unique':key not in knownpairs,'new_subject_vs_v4_unique':f'{m}:{tid}' not in knownobjects,'road_field_key':key,'old_relation_candidate':rid in br,'GT_used_for_binding':False}
   candidate.append(record)
   if not ok:continue
   scene={j:{'id':j,'name':o['name'],'category':o['object_type'],'position':o['position'][:2],'contour':o['contour']} for j,o in O[m].items() if j!=tid and (o['name'] or o['object_type']=='Water')};mentioned=[]
   for h in hits:
    ids,res,cf=C[m].resolve(h);mentioned.append({**h,'resolved_ids':ids or [],'resolution':res,'confidence':cf})
   x={'occurrence_id':rid,'map':m,'cluster':f'{m}:{tid}','episode':i,'instruction':text,'word':w[0].lower(),'reference_ids':A['resolved_ids'],'reference_text':record['reference_text'],'reference_span':record['reference_span'],'subject_ids':[tid],'subject_role':'annotated_target','direction_span':[w.start(),w.end()],'start_xy':e['trajectory'][0][:2],'mentioned':mentioned,'scene':scene,'parse_confidence':conf,'parse_tier':record['confidence_tier'],'origin':record['origin']};bx[rid]=x
   # Old geometry and ALL thresholds untouched; new language adapter exposes roles.
   if key not in fields:fields[key]=directed_field(x,A['resolved_ids'],B['resolved_ids'])
   f=fields[key];frozen_end_to_end=derive(x,{k:v for k,v in fields.items()})
   upgrades.append({'occurrence_id':rid,'road_field_key':key,'state':f['state'],'reason':f['reason'],'frame_type':'road','origin_xy':f.get('origin_xy'),'source_ids':B['resolved_ids'],'reference_ids':A['resolved_ids'],'word':w[0].lower(),'unique_scalar_heading':False,'geometry_algorithm':'frozen_v4_directed_field','GT_used':False,'language_evidence':[{'span':record['reference_span'],'text':record['reference_text'],'role':'travel_road_A'},{'span':record['source_span'],'text':text[B['start']:B['end']],'role':'source_road_B'},{'span':[w.start(),w.end()],'text':w[0],'role':'target_side'}],'frozen_v4_end_to_end_state_on_new_input':frozen_end_to_end['state']})
 # Preserve all original candidates; unknown old ones can have a separately listed
 # language upgrade, never silently overwrite historical v4 labels.
 writelines(RESULT/'semantic_scan.jsonl.gz',scan);writelines(RESULT/'new_candidates.jsonl.gz',candidate);writelines(RESULT/'public_inputs.jsonl.gz',list(bx.values()));writelines(RESULT/'new_frames.jsonl.gz',upgrades);writelines(RESULT/'road_fields.jsonl.gz',[{'key':k,**v} for k,v in fields.items()]);savecsv('new_relation_audit',candidate)
 writej(RESULT/'phase1_manifest.json',{'train_episodes':len(T),'v4_candidates':len(R),'new_candidate_ids':len({r['occurrence_id'] for r in candidate if r['occurrence_id'] not in br}),'old_candidates_reaudited':sum(r['occurrence_id'] in br for r in candidate),'new_bound':len(upgrades),'new_states':dict(collections.Counter(p['state'] for p in upgrades)),'inputs_sha256':sha(RESULT/'public_inputs.jsonl.gz'),'new_frames_sha256':sha(RESULT/'new_frames.jsonl.gz'),'fields_sha256':sha(RESULT/'road_fields.jsonl.gz'),'parser_code_sha256':sha(HERE/'language_v5.py'),'prepare_code_sha256':sha(HERE/'prepare_v5.py'),'TRAIN_only':True,'target_GT_or_future_used':False,'protocol':'retrospective; language-role adapter new; geometry frozen_v4'})
 print(readj(RESULT/'phase1_manifest.json'))
if __name__=='__main__':run()
