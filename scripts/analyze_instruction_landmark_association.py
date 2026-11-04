"""TRAIN-only language-to-public-landmark dependency graph audit, no navigation."""
import argparse,collections,csv,hashlib,json,re,unicodedata
from pathlib import Path
SOURCE=Path('/home/20261024_1/Workspace/DATA/sbfnav-data/original_citynav/citynav_train_seen.json')
CATALOG=Path('/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json')
OUT=Path('reports/instruction_landmark_association')
HELPERS={'front','back','rear','left','right','side','north','south','east','west','northeast','northwest','southeast','southwest','corner','end','edge','middle','center','centre','top','bottom','direction','distance'}
GENERIC_NAMES={'none','unknown','no name','unnamed','building','car','ground','parking','road','street','footpath','house','grass','trafficroad','water'}
TYPES={'Car':{'car','van','vehicle','truck','bus','automobile','jeep','suv','sedan','lorry','taxi'},'Building':{'building','buildling','build','house','home','block','structure','tower','church','chapel','hall','garage','school','complex','unit','warehouse','shed','midsection','section'},'Parking':{'parking','lot','area','park','garage'},'Ground':{'ground','grass','lawn','field','garden','area','section','patch','land','strip','piece'}}
PREPS={'in','on','at','near','beside','behind','between','under','above','below','over','around','across','opposite','outside','inside','within','along','off','by','from','to','of','with','toward','towards','beyond','past','through','adjacent','next','for','against','into','onto','alongside'}

def normalise(text):
 chars=[];mapping=[]
 for i,c in enumerate(text):
  translit=(' and ' if c=='&' else unicodedata.normalize('NFKD',c).encode('ascii','ignore').decode().lower()) or ' '
  for x in translit:
   chars.append(x if x.isalnum() else ' ');mapping.append(i)
 # Keep one source index per normalized character; collapse whitespace carefully.
 out=[];positions=[]
 for c,i in zip(chars,mapping):
  if c==' ' and (not out or out[-1]==' '):continue
  out.append(c);positions.append(i)
 if out and out[-1]==' ':out.pop();positions.pop()
 return ''.join(out),positions

def iter_train(path):
 decoder=json.JSONDecoder();idx=0
 with path.open() as stream:
  buf=stream.read(1<<20).lstrip();assert buf[0]=='[';buf=buf[1:]
  while True:
   buf=buf.lstrip(' \r\n\t,')
   if buf.startswith(']'):return
   try:row,end=decoder.raw_decode(buf)
   except json.JSONDecodeError:
    more=stream.read(1<<20)
    if not more:raise
    buf+=more;continue
   yield idx,row
   idx+=1;buf=buf[end:]

class PublicNames:
 def __init__(self,catalog):
  self.maps={};self.patterns={}
  for name,objects in catalog.items():
   table=collections.defaultdict(list)
   for oid,o in objects.items():
    variants=[o.get('name','')]+(o.get('aliases',[]) if isinstance(o.get('aliases',[]),list) else [])
    for variant in variants:
     if not isinstance(variant,str):continue
     key,_=normalise(variant)
     if key in GENERIC_NAMES or len(key)<3:continue
     table[key].append({'id':str(oid),'name':o.get('name',''),'type':o.get('object_type','')})
     if key.startswith('the ') and len(key)>7:table[key[4:]].append({'id':str(oid),'name':o.get('name',''),'type':o.get('object_type','')})
   self.maps[name]=dict(table);self.patterns[name]=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(k) for k in sorted(table,key=lambda k:(-len(k),k)))+r')(?!\w)') if table else None
 def match(self,map_name,text):
  norm,positions=normalise(text);pattern=self.patterns.get(map_name);out=[]
  if pattern:
   for m in pattern.finditer(norm):
    refs=self.maps[map_name][m.group()];out.append({'start':positions[m.start()],'end':positions[m.end()-1]+1,'text':text[positions[m.start()]:positions[m.end()-1]+1],'public_names':sorted({r['name'] for r in refs}),'public_ids':sorted({r['id'] for r in refs}),'types':sorted({r['type'] for r in refs})})
  return out

def classify(doc,mentions,category,identity_ids=()):
 # Merge complete mapped names so grammatical scope is not inferred from a name fragment.
 merged=[];occupied=set()
 for mention in mentions:
  span=doc.char_span(mention['start'],mention['end'],alignment_mode='expand')
  if span and not any(i in occupied for i in range(span.start,span.end)):
   merged.append((span,mention));occupied.update(range(span.start,span.end))
 with doc.retokenize() as retokenizer:
  for span,m in merged:retokenizer.merge(span)
 public={}
 for m in mentions:
  ts=[t for t in doc if t.idx<=m['start']<t.idx+len(t.text)]
  if ts:public[ts[0].i]=m
 nodes={t.i:t for t in doc if t.i in public or (t.pos_ in ('NOUN','PROPN') and t.lemma_.lower() not in HELPERS)}
 allowed=TYPES.get(category,set().union(*TYPES.values()));wrapped={t.head.i for t in doc if t.i in public and t.dep_ in ('compound','appos') and t.head.i in nodes};candidates=[t for t in nodes.values() if t.i not in public and t.i not in wrapped and t.lemma_.lower() in allowed]
 # Prefer main grammatical referent over a same-category object inside a prepositional prefix.
 def rank(t):
  sentindex=sum(s.start<t.sent.start for s in doc.sents)
  return (sentindex,0 if t.dep_ in ('ROOT','nsubj','nsubjpass','attr','dobj','obj') else 1,t.i)
 target=min(candidates,key=rank) if candidates else None
 if target is None:
  named_roots=[t for t in nodes.values() if t.i in public and set(public[t.i]['public_ids']) & set(identity_ids) and category in public[t.i]['types'] and (t.dep_ in ('ROOT','nsubj','attr','appos') or t.head.lemma_ in ('find','locate','reach'))]
  if len(named_roots)==1:return {'category':'direct_identity','target':named_roots[0].text,'paths':[],'edges':[],'reason':'named target identity, no explicit target-to-other-landmark geometric relation','scope_uncertain':False}
  return {'category':'uncertain' if mentions else 'no_confirmed_anchor','target':None,'paths':[],'edges':[],'reason':'target role unresolved','scope_uncertain':True}
 target_id=target.i;coref=set();aliases={};pronoun_ambiguous=False
 for t in doc:
  if t.i in nodes and t.i!=target_id and t.lemma_.lower()==target.lemma_.lower() and any(c.lower_ in ('this','that','these','those') and c.dep_=='det' for c in t.children):coref.add(t.i)
 for t in doc:
  if t.i in nodes and t.i not in public and any(c.lower_ in ('this','that','these','those') and c.dep_=='det' for c in t.children):
   previous=[n for n in nodes.values() if n.i<t.i and n.i not in public and n.lemma_.lower()==t.lemma_.lower()]
   if previous:aliases[t.i]=previous[-1].i
 for t in doc:
  if t.i in public and t.dep_ in ('compound','appos') and t.head.i in nodes and t.head.lemma_.lower() in {'building','house','church','chapel','hall','hotel','centre','center','school','depot'} and 'Building' in public[t.i]['types']:
   aliases[t.head.i]=t.i
  if t.i in nodes and t.i not in public and t.lemma_.lower() in {'parking','grass','housing'} and t.dep_=='compound' and t.head.i in nodes:aliases[t.i]=t.head.i
  if t.i in nodes and t.i not in public and t.lemma_.lower() in {'roof','door','wall','window','facade','storey','story','floor'} and t.dep_=='pobj' and t.head.lower_=='with' and t.head.head.i in nodes:
   aliases[t.i]=t.head.head.i
 def canonical(i):
  if i in coref:return target_id
  while i in aliases:i=aliases[i]
  return i
 def resolve(token,seen=None):
  nonlocal pronoun_ambiguous
  seen=set() if seen is None else set(seen)
  if token.i in seen:return None
  seen.add(token.i)
  if token.i in nodes:return canonical(token.i)
  if token.lower_ in ('it','this','that','which','they','these'):
   if token.head.dep_ in ('relcl','acl') and token.head.head.i in nodes:return canonical(token.head.head.i)
   # Explicit subject continuity only; competing noun subjects remain uncertain.
   earlier=[t for t in nodes.values() if t.i<token.i and t.dep_ in ('nsubj','nsubjpass','ROOT')]
   if earlier:return canonical(earlier[-1].i)
   return target_id if token.i>target_id else None
  if token.pos_ in ('VERB','AUX','ADJ','ADV') or token.lemma_.lower() in HELPERS:
   subj=[c for c in token.children if c.dep_ in ('nsubj','nsubjpass','csubj')]
   for child in subj:
    value=resolve(child,seen)
    if value is not None:return value
   if token.dep_ in ('relcl','acl') and token.head.i in nodes:return canonical(token.head.i)
   if token.dep_ in ('ROOT',) and token.pos_ in ('VERB','AUX'):
    obj=[c for c in token.children if c.dep_ in ('dobj','obj','attr') and c.i in nodes]
    if len(obj)==1:return canonical(obj[0].i)
  if token.head.i!=token.i:return resolve(token.head,seen)
  return None
 target_id=canonical(target_id)
 if target_id in public and set(public[target_id]['public_ids']) & set(identity_ids):
  m=public[canonical(target_id)]
  return {'category':'direct_identity','target':target.text,'paths':[],'edges':[],'reason':'explicit target name/type binding','scope_uncertain':False,'public_identity':m['public_names']}
 if target_id in public:
  return {'category':'uncertain','target':target.text,'paths':[],'edges':[],'reason':'target contracted to a reference landmark; target role unresolved','scope_uncertain':True}
 edges=[];seen_edges=set();scope_repairs=[];frame_only=[]
 def add(a,b,relation,evidence):
  if a is None or b is None or a==b:return
  a,b=canonical(a),canonical(b)
  if a==b:return
  key=(min(a,b),max(a,b),relation)
  if key not in seen_edges:edges.append({'a':a,'b':b,'relation':relation,'evidence':evidence});seen_edges.add(key)
 def coordination(t):return [t]+[c for c in t.conjuncts if c.i in nodes]
 for token in doc:
  if token.dep_ in ('pobj','pcomp') and token.head.lower_ in PREPS:
   prep=token.head
   if prep.lower_=='with' and re.search(r'(?:when viewed|viewed aerially|aerial view|orientated|oriented)',doc[max(0,prep.i-8):prep.i].text.lower()):
    frame_only.append(token.text);continue
   owner=resolve(prep.head);obj=resolve(token)
   # Geometric helper nouns collapse to one relation, not an intermediate object.
   if token.lemma_.lower() in HELPERS:continue
   rel=prep.lower_;chain=[prep.lower_];cur=prep.head
   for _ in range(4):
    if cur.lemma_.lower() in HELPERS:chain.insert(0,cur.lemma_.lower())
    if cur.head.i==cur.i or cur.i in nodes:break
    cur=cur.head
   rel=' '.join(chain)
   # Resolve a contiguous nested region phrase even when the small parser
   # attaches its final modifier to the sentence verb: car in [lot in front of X].
   if owner==target_id:
    intermediates=[n for n in nodes.values() if target_id<n.i<prep.i and n.i not in public and any({e['a'],e['b']}=={target_id,canonical(n.i)} and e['relation'] in ('in','inside','within','at','on') for e in edges)]
    if intermediates:
     region=max(intermediates,key=lambda n:n.i);tail=doc[region.i+1:prep.i].text.lower()
     if not re.search(r'[.,;]|\b(?:and|but|it|this|car|vehicle|target)\b',tail):
      owner=canonical(region.i);scope_repairs.append({'region':region.text,'relation':rel,'anchor':token.text})
   for sibling in coordination(token):add(owner,resolve(sibling),rel,f'{prep.head.text} -> {prep.text} -> {sibling.text}')
  if token.dep_=='poss' and token.i in public:add(resolve(token.head),token.i,'possessive region association',f'{token.text} owns {token.head.text}')
  if token.i in public and token.dep_=='compound' and token.head.i in nodes and canonical(token.head.i)!=token.i:
   add(resolve(token.head),token.i,'associated region',f'{token.text} -> {token.head.text}')
  if token.pos_=='VERB' and token.lemma_.lower() in ('surround','enclose','border','face','overlook','contain','include','occupy'):
   ss=[resolve(c) for c in token.children if c.dep_ in ('nsubj','nsubjpass')];oo=[resolve(c) for c in token.children if c.dep_ in ('dobj','obj')]
   for x in ss:
    for y in oo:add(x,y,token.lemma_,token.text)
 graph=collections.defaultdict(list)
 for e in edges:graph[e['a']].append((e['b'],e));graph[e['b']].append((e['a'],e))
 paths=[];queue=collections.deque([(target_id,[],[])])
 while queue:
  node,used,ee=queue.popleft()
  if node in public and node!=target_id:
   paths.append({'nodes':[target_id]+used,'labels':[target.text]+[nodes[i].text for i in used],'relations':[e['relation'] for e in ee],'hops':len(ee),'public_names':public[node]['public_names']});continue
  if len(ee)>=4:continue
  for nxt,e in graph[node]:
   if nxt!=target_id and nxt not in used:queue.append((nxt,used+[nxt],ee+[e]))
 paths.sort(key=lambda p:(p['hops'],p['labels']))
 if paths:
  direct=paths[0]['hops']==1;identity=direct and all(p['relations']==['identity/apposition'] for p in paths if p['hops']==1)
  cls='direct_identity' if identity else 'direct_geometry' if direct else 'indirect_geometry';reason='dependency-linked explicit public landmark path'
 else:cls='uncertain' if mentions else 'no_confirmed_anchor';reason='public names present but target-to-anchor scope unresolved' if mentions else 'no explicit binding to this map public catalog'
 ordinary=[nodes[e['b'] if e['a']==target_id else e['a']].text for e in edges if target_id in (e['a'],e['b']) and (e['b'] if e['a']==target_id else e['a']) not in public]
 return {'category':cls,'target':target.text,'target_token':target_id,'paths':paths[:12],'edges':edges,'generic_target_references':ordinary,'reason':reason,'scope_uncertain':cls=='uncertain','frame_only_mentions':frame_only,'scope_repairs':scope_repairs,'has_direct':any(p['hops']==1 for p in paths),'has_indirect':any(p['hops']>=2 for p in paths)}

def main():
 import spacy
 parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 catalog=json.loads(CATALOG.read_text());names=PublicNames(catalog);records=[]
 for ep,r in iter_train(SOURCE):
  map_name=f"{r['area']}_block_{r['block']}";text=r['descriptions'][0];target=catalog[map_name][str(r['object_ids'][0])]
  records.append({'episode':ep,'map_name':map_name,'instruction':text,'target_category':target['object_type'],'mentions':names.match(map_name,text),'multi_target_count':len(r['object_ids']),'analysis_target_ids':[str(x) for x in r['object_ids']]})
  if args.limit and len(records)>=args.limit:break
 nlp=spacy.load('en_core_web_sm',disable=['ner']);out=[]
 for i,(r,doc) in enumerate(zip(records,nlp.pipe((r['instruction'] for r in records),batch_size=128))):
  parsed=classify(doc,r['mentions'],r['target_category'],r['analysis_target_ids'])
  ignored=GENERIC_NAMES|HELPERS|set().union(*TYPES.values())|{'black','white','gray','grey','blue','red','green','brown','beige','yellow','orange','silver','tan'}
  cues=[chunk.text for chunk in doc.noun_chunks if any(t.pos_=='PROPN' and t.text.lower() not in ignored and len(t.text)>1 for t in chunk)] if not r['mentions'] else []
  parsed['unbound_name_cues']=cues
  if parsed['category']=='no_confirmed_anchor' and cues:
   parsed['category']='uncertain';parsed['reason']='unbound proper-name/alias hypothesis; not evidence of no landmark association';parsed['scope_uncertain']=True
  out.append({**r,**parsed})
  if (i+1)%1000==0:print('parsed',i+1,flush=True)
 (OUT/'records.json').write_text(json.dumps(out,ensure_ascii=False,separators=(',',':'))+'\n');counts=collections.Counter(r['category'] for r in out);summary={'n':len(out),'unique_map_instruction':len({(r['map_name'],r['instruction']) for r in out}),'maps':len({r['map_name'] for r in out}),'mutually_exclusive':dict(counts),'overlap':{'explicit_public_name':sum(bool(r['mentions']) for r in out),'direct_geometry_path':sum(r.get('has_direct',False) for r in out),'indirect_geometry_path':sum(r.get('has_indirect',False) for r in out)},'by_target_category':{c:dict(collections.Counter(r['category'] for r in out if r['target_category']==c)) for c in sorted({r['target_category'] for r in out})},'provenance':{'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'catalog':str(CATALOG),'catalog_sha256':hashlib.sha256(CATALOG.read_bytes()).hexdigest(),'spacy':spacy.__version__,'parser':'en_core_web_sm','parser_version':nlp.meta['version'],'GT_usage':'TRAIN target category guides noun-role selection; TRAIN instance IDs check explicit self-name identity; no target coordinates/contours used to construct links','limitations':'automated dependency interpretation; manual sample audit required; no name binding does not prove impossible association'}}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
 # Deterministic stratified audit sample, independent of manual judgment.
 import random
 rng=random.Random(20261105);sample=[]
 for c,n in [('direct_geometry',30),('indirect_geometry',30),('direct_identity',10),('no_confirmed_anchor',20),('uncertain',20)]:
  candidates=[r for r in out if r['category']==c];sample.extend(rng.sample(candidates,min(n,len(candidates))))
 (OUT/'manual_review_sample.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
