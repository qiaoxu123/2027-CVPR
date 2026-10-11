"""Nominal-span language graph, frozen dependency support and explicit scopes.

No target annotations, object IDs from labels, or GT coordinates are accepted.
Map IDs only identify PUBLIC name matches. Ambiguity is recorded, not hidden.
"""
import re,hashlib
from dataclasses import asdict
import numpy as np
from experiments.language_object_graph.language import TYPE_PATTERNS,COLORS,entity

REL=re.compile(r'\b(in\s+between|between|in front of|at the back(?: side)? of|behind|next to|near|beside|adjacent to|close to|along(?:side)?|connected to|attached to|'
 r'(?:to the |on the )?(?:east|west|north|south)(?:ern)?(?: side)? of|(?:to the |on the )?(?:left|right)(?: side)? of|'
 r'on|in|at|off(?: of)?|by)\b',re.I)
MODIFIERS=set(('a an the this that these those small large big tiny huge massive short tall light dark bright multi colored rectangular square round circular triangular long narrow curved shaped roofed mostly empty only grayish tan reddish silver beige white gray grey red blue brown black green yellow orange purple teal').split())
TYPES={**TYPE_PATTERNS,'Building':TYPE_PATTERNS['Building']+'|home|bungalow|apartment|structure', 'Ground':TYPE_PATTERNS['Ground']+'|land|yard', 'TrafficRoad':TYPE_PATTERNS['TrafficRoad']+'|lane|drive','Footpath':TYPE_PATTERNS['Footpath']+'|pavement|alleyway'}
TOKEN=re.compile(r"[A-Za-z0-9]+")

def words(text):return [(m.group().lower(),m.start(),m.end()) for m in TOKEN.finditer(text)]
def kind(trigger):
 x=trigger.lower()
 if 'between' in x:return 'between'
 if x in ('in front of',):return 'front'
 if x.startswith('at the back') or x=='behind':return 'behind'
 if x in ('along','alongside','on','in'):return 'along' if x!='in' else 'near'
 if x in ('near','next to','beside','adjacent to','close to','attached to','connected to','at','off','off of','by'):return 'near'
 return next((v for v in ('east','west','north','south','left','right') if v in x),None)

def mentions(text,graph):
 tokens=words(text);hits=[];names={}
 for i,name in enumerate(graph.names):
  ws=tuple(w[0] for w in words(name))
  if ws:names.setdefault(ws,[]).append(i)
 for sequence,ids in names.items():
  for j in range(len(tokens)-len(sequence)+1):
   if tuple(t[0] for t in tokens[j:j+len(sequence)])==sequence:
    start,end=tokens[j][1],tokens[j+len(sequence)-1][2];hits.append(dict(start=start,end=end,head_start=start,head_end=end,type=graph.types[ids[0]],ids=ids,name=True))
 for typ,pattern in TYPES.items():
  for m in re.finditer(r'\b(?:'+pattern+r')\b',text,re.I):hits.append(dict(start=m.start(),end=m.end(),head_start=m.start(),head_end=m.end(),type=typ,ids=[],name=False))
 # Long names and multiword typed nouns consume nested generic noun matches.
 hits.sort(key=lambda h:(not h['name'],-(h['end']-h['start']),h['start']));selected=[]
 for h in hits:
  if any(h['start']<x['end'] and h['end']>x['start'] for x in selected):continue
  selected.append(h)
 selected.sort(key=lambda h:h['start'])
 for h in selected:
  if not h['name']:
   before=[t for t in tokens if t[2]<=h['start']];n=0
   for word,start,end in reversed(before):
    gap=text[end:h['start']]
    if word not in MODIFIERS or re.search(r'[.;!?]',gap) or n>=8:break
    h['start']=start;n+=1
  e=asdict(entity(text[h['start']:h['end']],'reference',graph));e.update(object_type=h['type'],named_ids=tuple(h['ids']),named_names=tuple(dict.fromkeys(graph.names[i] for i in h['ids'])),span=[h['start'],h['end']],head_span=[h['head_start'],h['head_end']],head_text=text[h['head_start']:h['head_end']]);h['entity']=e
 return selected

def parse(text,graph,*,backend='rule',dependency=None):
 if not isinstance(text,str) or not text.strip():raise ValueError('raw instruction required')
 if backend not in ('rule','dependency'):raise ValueError('parser backend')
 if backend=='dependency' and (dependency is None or dependency.get('instruction')!=text):raise ValueError('same raw-text dependency parse required')
 hits=mentions(text,graph);tokens=dependency.get('tokens',[]) if dependency else [];unknown=[]
 if not hits:return dict(instruction=text,entities=[],factors=[],hypotheses=[],unknown=['no identifiable nominal entity'],confidence=0,backend=backend)
 scores=[]
 for j,h in enumerate(hits):
  before=text[:h['start']];local=before[-35:].lower();score=.55 if j==0 else .15
  if re.search(r'\bthis\s+(?:is\s+)?(?:a\s+|the\s+)?$',local):score=.95
  if re.search(r'\b(?:it is|it s|find|locate|look for|go to|navigate to)\s+(?:a\s+|the\s+)?$',local):score=.9
  if h['entity']['phrase'].lower().startswith('this '):score+=.65
  if h['name']:score-=.18
  if re.search(r'\b(?:on|in|at|of|off|from|by|near|behind|between|next to|with)\s+(?:a\s+|an\s+|the\s+)?$',local):score-=.3
  if re.search(r'\bif\b',before.lower()) and j==0:score-=.2
  if backend=='dependency':
   covered=[t for t in tokens if t['start']>=h['head_start'] and t['end']<=h['head_end']]
   if any(t['dep'] in ('nsubj','dobj','attr','ROOT') for t in covered):score+=.3
   if covered and all(t['dep'] in ('pobj','compound','amod') for t in covered):score-=.15
  scores.append(score)
 target=int(np.argmax(scores));best=scores[target]
 # Group referents and region-only directives are explicitly not a unique object.
 if re.match(r'\s*multiple buildings\b',text,re.I):unknown.append('group target: several instances')
 if hits[0]['name'] and target==0 and not re.search(r'\b(?:find|locate|go to|navigate to)\b',text[:hits[0]['start']],re.I):
  unknown.append('leading name may be reference; role unresolved')
 entities=[h['entity'].copy() for h in hits];original_to_new={target:0};ordering=[target]+[i for i in range(len(hits)) if i!=target]
 entities=[entities[i] for i in ordering]
 for i,j in enumerate(ordering):original_to_new[j]=i;entities[i]['role']='target' if i==0 else 'reference'
 # Return all names outside the nominal target as reference nodes, including
 # names in unknown relations; none get promoted into target unary.
 factors=[]
 for m in REL.finditer(text):
  if any(h['head_start']<=m.start()<h['head_end'] for h in hits if h['name']):continue
  after=[i for i,h in enumerate(hits) if h['head_start']>=m.end()]
  before=[i for i,h in enumerate(hits) if h['head_end']<=m.start()]
  if not after:continue
  anchor=after[0];subject=before[-1] if before else target;k=kind(m.group())
  if k is None or anchor==subject:continue
  # Separate discourse sentences usually attach a fresh subject; otherwise
  # nearest nominal and original target are retained as scope alternatives.
  subject_options=[original_to_new[subject]]
  if backend=='dependency':
   trigger_tokens=[i for i,t in enumerate(tokens) if t['start']>=m.start() and t['start']<m.end()]
   if trigger_tokens:
    node=tokens[trigger_tokens[0]]['head'];visited=set()
    for _ in range(5):
     if node in visited:break
     visited.add(node);t=tokens[node]
     matched=[i for i,h in enumerate(hits) if h['head_start']<=t['start']<h['head_end'] and i!=anchor]
     if matched:
      primary=original_to_new[matched[0]];subject_options=[primary]+[v for v in subject_options if v!=primary];break
     node=t['head']
  if subject!=target and before:
   textgap=text[hits[subject]['head_end']:m.start()]
   if not re.search(r'[.!?]',textgap):subject_options.append(0)
  anchors=[original_to_new[anchor]]
  if k=='between':
   a=hits[anchor];rest=[i for i in after[1:] if re.search(r'\band\b',text[a['end']:hits[i]['start']],re.I)]
   if rest:anchors.append(original_to_new[rest[0]])
   elif re.search(r'\b(?:two|2)\b',text[m.end():a['head_start']]) and text[a['head_start']:a['head_end']].lower().endswith('s'):
    clone=a['entity'].copy();clone['role']='reference';clone['unknown']=(*clone['unknown'],'plural instance expansion, identity unknown');entities.append(clone);anchors.append(len(entities)-1)
   else:unknown.append('Between has no second separately identified reference');continue
  subject_options=[s for s in dict.fromkeys(subject_options) if s not in anchors]
  if not subject_options:continue
  frame=['world'] if k in ('near','along','between','east','west','north','south') else ['world-north-up','current-uav','opposite-current-uav']
  factors.append(dict(kind=k,subject=subject_options[0],subject_hypotheses=[dict(subject=s,probability=(.65 if n==0 else .35/(len(subject_options)-1)) if len(subject_options)>1 else 1.) for n,s in enumerate(subject_options)],anchors=anchors,trigger=m.group(),trigger_span=[m.start(),m.end()],frame_hypotheses=frame,confidence=.8 if len(subject_options)==1 else .55,status='text-only scope, not reference identity truth'))
 hypotheses=[dict(target=0,probability=1.,status='nominal head interpretation')]
 if len(scores)>1:
  second=int(np.argsort(scores)[-2]);margin=best-scores[second]
  if margin<.2 and original_to_new[second]!=0:
   hypotheses=[dict(target=0,probability=.7,status='ambiguous target role'),dict(target=original_to_new[second],probability=.3,status='alternative target role')];unknown.append('target role has multiple plausible nominal heads')
 confidence=0. if any('group target' in u or 'leading name' in u for u in unknown) else .8 if len(hypotheses)==1 else .5
 return dict(instruction=text,entities=entities,factors=factors,hypotheses=hypotheses,unknown=unknown,confidence=confidence,backend=backend,source='raw nominal spans and public names; no labels',reference_instance_truth=None)
