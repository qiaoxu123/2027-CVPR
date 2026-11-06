"""Real local parser inference; receives public_inputs only, never diagnostic labels."""
import argparse,json,time,hashlib,re,os
from pathlib import Path
import torch
from graphnav.parser_benchmark import validate_graph,relation_paths

RELATIONS=['north of','south of','east of','west of','near','next to','behind','front of','left of','right of','inside','outside','between','across from','along','on','end of','surrounded by','connected to','facing']
SYSTEM='''You are a navigation instruction semantic parser. Use ONLY the original English instruction and public landmark NAME catalog. No map coordinates, target identity, images, or outside facts are available. Return one compact JSON object, no commentary.
Extract the navigation TARGET (the object to reach), every mentioned NAMED LANDMARK, and unnamed INTERMEDIATE/REFERENCE objects. Preserve target color, shape, material and other descriptive attributes in target_description. Do not replace a target with its landmark. Scope relations correctly: a car inside a parking lot next to a named building requires car -> lot -> building, not an invented car -> building direct edge. A relation can have the target as its object (e.g. the cafe behind the target building); preserve grammatical subject/object. Do not invent links to connect disconnected objects. Preserve uncertainty. Do not interpret front/behind/left/right as geographic north/south/east/west.
Schema: {"target_description":"exact original target noun phrase including attributes, excluding external relational clauses", "entities":[{"id":"e0","text":"EXACT contiguous substring of instruction","role":"target|landmark|intermediate|reference","type":"building|car|parking|road|ground|other"}], "relations":[{"subject":"entity id","relation":"original relation phrase, e.g. inside/next to/across from","object":"entity id","evidence":"exact source relation phrase","reference_frame":"world_unspecified|relative_unspecified|not_applicable"}],"ambiguities":["brief uncertain scope"]}. Use separate entity IDs for distinct objects. Named landmarks must actually be mentioned in the instruction, not just listed in the catalog. Do not add facts.''' 

def infer_type(text):
 low=text.lower()
 for kind,pattern in [('parking',r'\b(parking|lot)\b'),('car',r'\b(car|cars|van|truck|vehicle|bus)\b'),('road',r'\b(road|street|lane|drive|footpath)\b'),('ground',r'\b(ground|grass|field|lawn|garden)\b'),('building',r'\b(building|house|hall|library|church|chapel|warehouse|cafe|facility|museum|store)\b')]:
  if re.search(pattern,low):return kind
 return 'other'

def aligned_entity(text,e,number):
 value=e.get('text','');start=e.get('start');end=e.get('end');changed=False
 if not isinstance(value,str) or not value:return None,'empty entity'
 if not isinstance(start,int) or not isinstance(end,int) or text[start:end]!=value:
  hits=[m.start() for m in re.finditer(re.escape(value),text)]
  if len(hits)!=1:return None,'nonunique/nonliteral entity: '+value
  start=hits[0];end=start+len(value);changed=True
 return dict(id=str(e.get('id',f'e{number}')),text=value,start=start,end=end,role=e.get('role','reference'),type=e.get('type','unknown')),'span_aligned_from_unique_literal' if changed else None

def convert_spacy(parsed,mentions):
 target=parsed['target_span'];tid=target['node'] if target else None;es={}
 for entity in parsed['entities']+([target] if target else []):
  x=entity.copy();role='target' if x['node']==tid else 'landmark' if x['kind']=='named_landmark' else 'intermediate';es[str(x['node'])]=dict(id=str(x['node']),text=x['text'],start=x['start'],end=x['end'],role=role,type='unknown')
 # Preserve the current public gazetteer discovery even if no grammatical edge reaches it.
 for m in mentions:
  if not any(e['start']==m['start'] and e['end']==m['end'] for e in es.values()):
   eid='public_'+str(m['start']);es[eid]=dict(id=eid,text=m['text'],start=m['start'],end=m['end'],role='landmark',type='unknown')
 relations={}
 for p in parsed['paths']:
  for edge in p['steps']:
   key=(str(edge['subject']['node']),edge['relation'],str(edge['object']['node']));relations[key]=dict(subject=key[0],relation=key[1],object=key[2],evidence=edge['evidence'],reference_frame=edge['reference_frame'])
 # Current edges not on a target-landmark path still matter for entity/relation evaluation.
 for e in parsed['edges']:
  if str(e['a']) in es and str(e['b']) in es:
   if not any(k[0]==str(e['a']) and k[2]==str(e['b']) for k in relations):relations[(str(e['a']),e['relation'],str(e['b']))]=dict(subject=str(e['a']),object=str(e['b']),relation=e['relation'],evidence=e.get('evidence',''))
 return dict(entities=list(es.values()),relations=list(relations.values()),target_description=target['text'] if target else '',ambiguities=['scope_uncertain'] if parsed['scope_uncertain'] else [])

def gliner_graph(raw,text,names):
 entities=[];warnings=[];lookup={}
 def add(item,role):
  item=dict(text=item) if isinstance(item,str) else dict(item);span,warn=aligned_entity(text,{**item,'id':f'e{len(entities)}','role':role},len(entities))
  if span is None:warnings.append(warn);return None
  key=(span['start'],span['end'],role)
  if key in lookup:return lookup[key]
  lookup[key]=span['id'];entities.append(span);return span['id']
 for label,es in raw.get('entities',{}).items():
  role={'navigation target':'target','named landmark':'landmark','intermediate reference object':'intermediate'}.get(label,'reference')
  for e in es:add(e,role)
 # Multiple target outputs remain visible; do not pick a GT-matching target.
 def endpoint(item):
  value=item.get('text','') if isinstance(item,dict) else item
  if not isinstance(value,str):return None
  hits=[e for e in entities if e['text']==value]
  if hits:
   if len(hits)>1:warnings.append('multiple_role_predictions_for_endpoint: '+value)
   return hits[0]['id']
  starts=[m.start() for m in re.finditer(re.escape(value),text)] if value else []
  if len(starts)==1:
   a=starts[0];b=a+len(value);overlap=[(min(e['end'],b)-max(e['start'],a),e['id']) for e in entities if min(e['end'],b)>max(e['start'],a)]
   if overlap:return max(overlap)[1]
  return add(item,'landmark' if value in names else 'intermediate')
 edges=[]
 for rel,values in raw.get('relation_extraction',{}).items():
  for pair in values:
   if isinstance(pair,dict):h,t=pair.get('head'),pair.get('tail')
   else:h,t=pair
   a,b=endpoint(h),endpoint(t)
   if a is not None and b is not None and a!=b:edges.append(dict(subject=a,object=b,relation=rel,evidence=rel))
 targets=[e['text'] for e in entities if e['role']=='target'];return dict(entities=entities,relations=edges,target_description=targets[0] if len(targets)==1 else '',ambiguities=warnings)

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parser',choices=['A','B','C','D'],required=True);p.add_argument('--model');p.add_argument('--batch',type=int,default=8);p.add_argument('--limit',type=int,default=0);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42);a.output.mkdir(parents=True,exist_ok=True);records=json.load(open(a.inputs));records=records[:a.limit] if a.limit else records
 config=dict(parser=a.parser,model=a.model,batch=a.batch,limit=a.limit,inputs_SHA=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),code_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prompt_SHA=hashlib.sha256(SYSTEM.encode()).hexdigest(),GT_input=False,seed=42)
 if (a.output/'config.json').exists():assert json.load(open(a.output/'config.json'))==config,'Resume configuration differs'
 else:
  (a.output/'config.json').write_text(json.dumps(config,indent=2));(a.output/'SOURCE.py').write_text(Path(__file__).read_text())
 path=a.output/'predictions.jsonl';done={json.loads(s)['index'] for s in path.read_text().splitlines()} if path.exists() else set();records=[r for r in records if r['index'] not in done];load=time.perf_counter();gpu=a.parser!='A'
 if a.parser in ['A','B']:
  import spacy
  from scripts.analyze_instruction_landmark_association import PublicNames
  from graphnav.role_relation_language import parse_doc
  if gpu:spacy.require_gpu()
  nlp=spacy.load('en_core_web_sm' if a.parser=='A' else 'en_core_web_trf',disable=['ner']);modelmeta=dict(spacy=spacy.__version__,spacy_model_version=nlp.meta['version'])
 elif a.parser=='C':
  from gliner2 import GLiNER2
  model=GLiNER2.from_pretrained(a.model).cuda().eval();modelmeta=dict(model_path=a.model)
 else:
  from transformers import AutoModelForCausalLM,AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained(a.model,padding_side='left');model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval();modelmeta=dict(model_path=a.model,dtype='BF16',thinking=False,do_sample=False,max_new_tokens=1600)
 load_seconds=time.perf_counter()-load
 if gpu:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 times=[];count=0;started=time.perf_counter();step=a.batch if a.parser=='D' else 1
 for pos in range(0,len(records),step):
  rr=records[pos:pos+step];raws=[];errors=[]
  if gpu:torch.cuda.synchronize()
  tick=time.perf_counter()
  try:
   if a.parser=='D':
    prompts=[tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(dict(instruction=r['text'],public_landmark_names=r['landmark_catalog']))}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in rr];x=tokenizer(prompts,padding=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():y=model.generate(**x,max_new_tokens=1600,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    raws=tokenizer.batch_decode(y[:,x['input_ids'].shape[1]:],skip_special_tokens=True)
   elif a.parser in ['A','B']:
    r=rr[0];matcher=PublicNames({'public':{str(i):dict(name=n) for i,n in enumerate(r['landmark_catalog'])}});mentions=matcher.match('public',r['text']);parsed=parse_doc(nlp(r['text']),r['text'],'public',matcher);raws=[dict(parsed=parsed,graph=convert_spacy(parsed,mentions))]
   else:
    r=rr[0];mentioned=[n for n in r['landmark_catalog'] if n.casefold() in r['text'].casefold()];schema=model.create_schema().entities({'navigation target':'The single object the navigator must reach, with its descriptive adjectives.','named landmark':'A named reference landmark. Public names appearing literally: '+', '.join(mentioned),'intermediate reference object':'An unnamed parking lot, road, building, car or other reference object linking target and landmark.'}).relations(RELATIONS)
    with torch.inference_mode():raws=[model.extract(r['text'],schema,threshold=.5,include_confidence=True,include_spans=True)]
  except Exception as e:raws=[dict(runtime_error=repr(e)) for _ in rr]
  if gpu:torch.cuda.synchronize()
  elapsed=time.perf_counter()-tick;times.append(elapsed/len(rr))
  for r,raw in zip(rr,raws):
   alignment=[]
   try:
    if isinstance(raw,dict) and 'runtime_error' in raw:raise ValueError(raw['runtime_error'])
    if a.parser=='D':
     clean=raw.strip();clean=re.sub(r'^```(?:json)?\s*','',clean);clean=re.sub(r'\s*```$','',clean);obj=json.loads(clean);es=[]
     for j,e in enumerate(obj.get('entities',[])):
      entity,warn=aligned_entity(r['text'],e,j)
      if entity:es.append(entity)
      if warn:alignment.append(warn)
     obj['entities']=es
    elif a.parser=='C':obj=gliner_graph(raw,r['text'],r['landmark_catalog'])
    else:obj=raw['graph']
    graph=validate_graph(r['text'],obj)
    for entity in graph['entities']:
     if entity['type']=='unknown':entity['type']=infer_type(entity['text']);entity['type_source']='shared_lexical_span_heuristic'
    graph['chains']=relation_paths(graph);error=None
   except Exception as e:graph=validate_graph(r['text'],{});graph['chains']=[];error=repr(e)
   result=dict(index=r['index'],parser=a.parser,graph=graph,raw=raw,error=error,alignment_warnings=alignment,amortized_latency_s=elapsed/len(rr),batch_wall_s=elapsed,batch_size=len(rr))
   with path.open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
   count+=1
  print(json.dumps(dict(parser=a.parser,finished=count,remaining=len(records)-count,batch_seconds=elapsed)),flush=True)
 summary=dict(**config,**modelmeta,model_load_s=load_seconds,resumed_existing=len(done),finished_this_run=count,wall_s=time.perf_counter()-started,mean_amortized_s=sum(times)/len(times) if times else None,mean_after_warmup_s=sum(times[2:])/len(times[2:]) if len(times)>2 else None,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else 0)
 (a.output/'runtime.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
