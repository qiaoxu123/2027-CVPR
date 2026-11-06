"""Text-only entity and relation binding. No geometry/GT/trajectory inputs.
Confidence numbers are rule scores, not calibrated probabilities.
"""
from __future__ import annotations
import re,collections
from dataclasses import dataclass
WORDS='left right front behind back rear above below top bottom north south east west'.split()
TOKEN=re.compile(r'\b('+'|'.join(WORDS)+r')\b',re.I)
PATTERNS={w:re.compile(p,re.I) for w,p in {
 'left':r'left\s+(?:hand\s+)?(?:side\s+)?of\s+', 'right':r'right\s+(?:hand\s+)?(?:side\s+)?of\s+',
 'front':r'front\s+of\s+', 'back':r'back\s+of\s+', 'rear':r'rear\s+of\s+',
 'behind':r'behind\s+', 'above':r'above\s+', 'below':r'below\s+',
 'top':r'top\s+of\s+', 'bottom':r'bottom\s+of\s+',
 'north':r'north\s+(?:side\s+)?of\s+', 'south':r'south\s+(?:side\s+)?of\s+',
 'east':r'east\s+(?:side\s+)?of\s+', 'west':r'west\s+(?:side\s+)?of\s+'}.items()}
HEADS={'Building':{'building','house','home','shed','warehouse','structure','church','townhouse'},'Car':{'car','vehicle','van','truck','bus','minivan'},'Parking':{'parking','lot','park','area','space'},'Ground':{'ground','area','land','field','lawn','garden','patch','court','grass'}}
POSITION_HEADS={'side','front','back','rear','left','right','north','south','east','west','top','bottom'}

@dataclass(frozen=True)
class Entity:
 id:str
 name:str
 category:str

def tokens(s):
 # Explicit linguistic equivalents only; no edit distance or GT proximity.
 ts=list(re.finditer(r'[a-z0-9]+|&',s.lower().replace('’',"'")))
 vals=[('st' if t[0]=='saint' else 'and' if t[0]=='&' else t[0]) for t in ts]
 return ts,vals

def name_key(s):
 _,v=tokens(s)
 return tuple(v[1:] if v and v[0]=='the' else v)

class TextCatalog:
 def __init__(self,entities):
  self.entities={e.id:e for e in entities};self.names=collections.defaultdict(list)
  for e in entities:
   k=name_key(e.name)
   if k:self.names[k].append(e.id)
  self.first=collections.defaultdict(list)
  for k,ids in self.names.items():self.first[k[0]].append((k,ids))
 def spans(self,text):
  ts,vs=tokens(text);hits=[]
  for j,val in enumerate(vs):
   for k,ids in self.first.get(val,[]):
    if tuple(vs[j:j+len(k)])==k:hits.append({'start':ts[j].start(),'end':ts[j+len(k)-1].end(),'ids':ids,'name':' '.join(k)})
  return [h for h in hits if not any(g['start']<=h['start'] and g['end']>=h['end'] and g['end']-g['start']>h['end']-h['start'] for g in hits)]
 def resolve(self,hit):
  ids=hit['ids'];cats={self.entities[i].category for i in ids}
  if len(ids)==1:return ids,'unique_named_object',.99
  # Preserve ALL same-named road segments as one semantic road entity.
  # No segment is selected based on the target position or score.
  if cats=={'TrafficRoad'}:return ids,'all_named_road_segments',.80
  return None,'duplicate_nonroad_landmark_name',0.

def token_at(doc,pos):return next((t for t in doc if t.idx<=pos<t.idx+len(t.text)),None)

def root_subjects(verb):
 return [c for c in verb.children if c.dep_ in {'nsubj','nsubjpass','attr'}]

def initial_target_heads(doc,category):
 first=next(iter(doc.sents));root=first.root;heads=HEADS.get(category,set())
 candidates=[root]+root_subjects(root)
 result=[]
 for t in candidates:
  if t.lemma_.lower() in heads:
   # A noun attribute of a copular root is a target; not unrelated nested nouns.
   if t==root or t.dep_ in {'nsubj','nsubjpass','attr'}:result.append(t.i)
 # Unpunctuated comma-spliced annotations may make the SECOND clause the
 # dependency root. Recover only the first clause's explicit target subject.
 if not result:
  boundary=next((t.idx for t in first if t.text==','),first.end_char)
  candidates=[t for t in first if t.idx<boundary and t.lemma_.lower() in heads and t.dep_ in {'nsubj','nsubjpass','attr'} and t.head.idx<boundary]
  if candidates:result.append(candidates[0].i)
 return set(result)

def resolve_owner(tok,target_heads,category,doc):
 """Trace grammatical owner without traversing across arbitrary nominal NPs."""
 path=[];cur=tok
 for _ in range(14):
  if cur.i in [i for i,_,_ in path]:break
  path.append((cur.i,cur.text,cur.dep_))
  if cur.i in target_heads:return True,'initial_target_head',path
  if cur!=tok and cur.pos_ in {'NOUN','PROPN','PRON'} and cur.lemma_.lower() not in POSITION_HEADS:
   # Later explicit this-category subjects may refer to target; preserve confidence tier.
   if cur.lemma_.lower() in HEADS.get(category,set()) and any(c.lower_=='this' for c in cur.children):
    return True,'explicit_this_target_later_clause',path
   if cur.lower_=='it' and cur.dep_ in {'nsubj','nsubjpass'}:
    prev=[t for t in doc[:cur.i] if t.lemma_.lower() in HEADS.get(category,set()) and t.dep_ not in {'compound','amod'}]
    if len(prev)==1 and prev[0].i in target_heads:return True,'single_nominal_antecedent_it',path
   return False,'different_nominal_owner',path
  if cur.pos_ in {'VERB','AUX'} or (cur==tok and cur.dep_=='ROOT'):
   subjects=root_subjects(cur)
   for sub in subjects:
    if sub.i in target_heads:return True,'verb_target_subject',path+[(sub.i,sub.text,sub.dep_)]
    if sub.lower_ in {'that','which'} and cur.dep_ in {'relcl','acl'} and cur.head.i in target_heads:return True,'target_relative_clause',path+[(cur.head.i,cur.head.text,cur.head.dep_)]
    if sub.lemma_.lower() in HEADS.get(category,set()) and any(c.lower_=='this' for c in sub.children):return True,'explicit_this_target_later_clause',path+[(sub.i,sub.text,sub.dep_)]
    if sub.lower_=='it':
     prev=[t for t in doc[:sub.i] if t.lemma_.lower() in HEADS.get(category,set()) and t.dep_ not in {'compound','amod'}]
     if len(prev)==1 and prev[0].i in target_heads:return True,'single_nominal_antecedent_it',path+[(sub.i,sub.text,sub.dep_)]
   if cur.dep_ in {'acl','relcl'} and cur.head.i in target_heads:return True,'target_participial_clause',path+[(cur.head.i,cur.head.text,cur.head.dep_)]
   # Different explicit grammatical subject is a scope boundary.
   if subjects and not all(s.lower_ in {'that','which','it'} for s in subjects):return False,'other_clause_subject',path
  if cur.head==cur:break
  cur=cur.head
 return False,'owner_unresolved',path

def frame_cues(text):
 return {'explicit_view':bool(re.search(r'\b(?:viewed|viewing|view|aerially|aerial|looking|stand|standing|facing|vertically|horizontally)\b',text,re.I)),
 'local_layout':bool(re.search(r'\b(?:row|lane|column|parking|parked|driveway|houses|buildings|road|street)\b',text,re.I)),
 'intrinsic_front_candidate':bool(re.search(r'\b(?:front|back|rear)\b',text,re.I)),
 'motion_context':bool(re.search(r'\b(?:turn|turning|driving|drive|travel|travelling|traveling|coming|heading)\b',text,re.I))}

def parse_relations(doc,category,catalog):
 text=doc.text;hits=catalog.spans(text);target_heads=initial_target_heads(doc,category);out=[];cues=frame_cues(text)
 for m in TOKEN.finditer(text):
  w=m[0].lower();s,e=m.span();tok=token_at(doc,s)
  r={'word':w,'semantic_word':w,'span_start':s,'span_end':e,'status':'excluded','scope':'unknown','reason':'no_explicit_relation','reference_ids':[], 'entity_confidence':0.,'scope_confidence':0.,'parse_confidence':0.,'confidence_tier':'excluded','target_head_tokens':sorted(target_heads),'dependency_path':[], 'inverse':False, **cues}
  def reject(reason,scope='unknown'):
   r.update(reason=reason,scope=scope);out.append(r)
  if any(h['start']<=s and h['end']>=e for h in hits):reject('direction_in_proper_name','nonrelational');continue
  if w=='right' and re.match(r'\s+(?:next|beside|near|across|along|on|at|by|behind|in\s+front)\b',text[e:],re.I):reject('right_emphasis','nonrelational');continue
  if re.search(r'\b(?:turn|turning|take|taking|traveling|travelling|heading)\s+(?:a\s+|to\s+the\s+)?$',text[:s],re.I):reject('motion_direction_not_static_relation','nonrelational');continue
  if w in {'front','back','rear','top','bottom'} and re.match(r'\s+(?:yard|bumper|door|license|plate|seats?|roof|wheel|window|section)\b',text[e:],re.I):reject('object_part_attribute','nonrelational');continue
  rel=PATTERNS[w].match(text[s:])
  if not rel:
   # A fully explicit observer frame, e.g. stand at NAME facing the stream ... your right.
   anchor=re.search(r'\b(?:stand|standing)\s+at\s+(?:the\s+)?',text[:s],re.I)
   if anchor and re.search(r'\b(?:your|you)\s*$',text[max(0,s-12):s],re.I) and w in {'left','right','front','behind'}:
    hs=[h for h in hits if h['start']>=anchor.end() and not text[anchor.end():h['start']].strip()]
    facing=re.search(r'\bfacing\s+(?:the\s+)?(stream|river|road)\b',text,re.I)
    if len(hs)==1 and facing and target_heads:
     ids,res,ec=catalog.resolve(hs[0])
     if ids:
      r.update(status='bound',scope='observer_relative_direct',reason='',reference_ids=ids,reference_text=text[hs[0]['start']:hs[0]['end']],reference_span=[hs[0]['start'],hs[0]['end']],entity_resolution=res,entity_confidence=ec,scope_confidence=.97,parse_confidence=min(ec,.97),confidence_tier='high' if ec>=.95 else 'medium',binding_rule='explicit_stand_at_facing_frame',declared_facing_category=facing[1].lower());out.append(r);continue
   reject('ordinal_part_or_relation_without_bindable_marker','nonrelational' if w in {'top','bottom'} else 'unknown');continue
  if re.search(r'\b(?:upper|lower|top|bottom|north|south|diagonally|above\s+the|below\s+the)\s+(?:and\s+)?$',text[:s],re.I) or re.match(r'\s+(?:east|west)\b',text[e:],re.I):reject('compound_direction_not_single_axis');continue
  if w in {'top','bottom'}:reject('part_or_region_not_external_pair','nonrelational');continue
  if w=='front' and not re.search(r'\bin\s+(?:the\s+)?$',text[:s],re.I):reject('front_part_not_in_front','nonrelational');continue
  if w in {'back','rear'} and not re.search(r'\b(?:at|to|in)\s+(?:the\s+)?$',text[:s],re.I):reject('back_rear_part_or_unknown');continue
  after=s+rel.end();hs=[h for h in hits if h['start']>=after and re.fullmatch(r'\s*(?:the\s+)?',text[after:h['start']],re.I)]
  if len(hs)!=1:
   # Do not jump over a reference head to find a later landmark.
   later=[h for h in hits if after<h['start']<after+100]
   lead=text[after:min(after+90,later[0]['start'])] if later else text[after:after+90]
   reject('indirect_reference_np' if later and re.search(r'\b(?:of|in|on|behind|near|beside|next)\b',lead,re.I) else 'reference_not_independently_resolved','indirect' if later else 'unknown');continue
  hit=hs[0];r.update(reference_text=text[hit['start']:hit['end']],reference_span=[hit['start'],hit['end']],reference_candidate_ids=hit['ids'])
  ids,res,ec=catalog.resolve(hit);r.update(entity_resolution=res,entity_confidence=ec)
  if ids is None:reject(res);continue
  if re.match(r'\s+(?:road\s+car\s+park|car\s+park|parking\s+lot|yard|entrance|wall|side|corner|driveway)\b',text[hit['end']:],re.I):reject('reference_is_named_landmark_subregion','indirect');continue
  # Unbound coordinated reference can change number and relational interpretation.
  tail=re.sub(r'^\s+(?:building|road|church|hall)\b','',text[hit['end']:],flags=re.I)
  if re.match(r'\s+(?:and|or)\s+(?!(?:to|on|in|behind|above|below|left|right|north|south|east|west|is|has|the\s+building\s+is)\b)',tail,re.I):reject('coordinated_reference_uncertain');continue
  # Independent surface guard: statistical dependency attachment alone is not proof.
  # Containers and anonymous reference NPs before the direction are scope boundaries.
  start=tok.sent.start_char;prefix=text[start:s]
  prefix=re.sub(r'(?:\b(?:to|on|at|in)\s+(?:the\s+)?|\b(?:directly|just|immediately|far)\s+)+$','',prefix,flags=re.I)
  container=re.search(r'\b(?:in|inside|within|at|of)\s+(?:(?:the|a|an)\s+)?(?:\w+\s+){0,5}(?:parking\s+(?:lot|area|spot)|car\s+park|yard|backyard|row|lane|column|cluster)\b',prefix,re.I)
  embedded=re.search(r'\b(?:behind|beside|near|next\s+to|in\s+front\s+of|above|below|on|with)\s+(?:the|a|an|another)\s+(?:(?:gray|brown|white|black|blue|green|large|small|tall|long|red|light|dark|grey|roofed|and|colored|brick|silver|square|tiny|huge|first|second|two|three|four)\s+)*(?:building|house|car|vehicle|tree)\b',prefix,re.I)
  nearest=re.search(r'\b(?:nearest|closest)\b.+\b(?:trees?|buildings?|cars?|houses?)\b',prefix,re.I)
  preceding_names=[h for h in hits if start<=h['start']<h['end']<s]
  if any(re.match(r'\s+(?:building\s+)?(?:that|which)\b',text[h['end']:s],re.I) for h in preceding_names):
   reject('relative_clause_named_context_attachment_ambiguous','indirect');continue
  # Cross-check ALL other generic entity heads after the established target NP.
  # A parser can incorrectly attach a relation across 'between ... cars' or
  # 'in the parking space'; neither becomes direct merely by model attachment.
  masked=list(prefix)
  for named in hits:
   lo=max(named['start'],start);hi=min(named['end'],s)
   for k in range(lo,hi):masked[k-start]=' '
  surface=''.join(masked)
  first_heads=[doc[i] for i in target_heads if start<=doc[i].idx<s]
  if first_heads:
   cut=min(t.idx+len(t.text) for t in first_heads)-start;surface=surface[cut:]
  # Distance units in a target relative clause are not another object.
  surface=re.sub(r'\b(?:is|are)\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)?|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:houses?|buildings?|cars?|vehicles?)\b',' ',surface,flags=re.I)
  other_nominal=re.search(r'\b(?:cars?|vehicles?|vans?|trucks?|buildings?|houses?|lots?|yards?|backyards?|rows?|lanes?|columns?|clusters?|fields?|grounds?|areas?|trees?|spaces?|aisles?|alleys|alley|playgrounds?|courtyards?|enclosures?|spots?|corners?|crevices?)\b',surface,re.I)
  r['surface_other_nominal']=other_nominal[0] if other_nominal else None
  if container or embedded or nearest or other_nominal:
   reject('surface_dependency_scope_disagreement_or_embedded_np','indirect');continue
  if re.search(r'\bfrom\s+(?:the\s+)?$',text[:s],re.I):reject('ordinal_count_from_side_not_external_direction','nonrelational');continue
  ok,rule,path=resolve_owner(tok,target_heads,category,doc);r.update(binding_rule=rule,dependency_path=path)
  if not ok:reject('non_target_relation_owner' if rule in {'different_nominal_owner','other_clause_subject'} else 'relation_owner_unresolved','indirect' if rule in {'different_nominal_owner','other_clause_subject'} else 'unknown');continue
  sc=.97 if rule in {'initial_target_head','verb_target_subject','target_participial_clause','target_relative_clause'} else .82
  # A named road locator before a relation is not erased; retain lower-confidence
  # attachment rather than treating the nearest landmark as the reference.
  if re.search(r'\b(?:on|along|off|beside|near)\s+',prefix,re.I):sc=min(sc,.85)
  if re.match(r'\s*(?:are\s+you|is\s+this|what\s+is)\b',text,re.I):sc=min(sc,.8)
  # Coordinates, attributes and target noun plural do not establish a directed frame.
  plural=any(doc[i].tag_=='NNS' for i in target_heads)
  if plural:sc=min(sc,.8)
  score=min(ec,sc);r.update(status='bound',scope='direct',reason='',reference_ids=ids,scope_confidence=sc,parse_confidence=score,confidence_tier='high' if score>=.95 else 'medium',target_plural=plural)
  out.append(r)
 return out

def landmark_anchor(doc,row,catalog):
 """Population gate: a named landmark must be the actual reference NP.
 Directions between target and anonymous neighbors are outside this experiment.
 """
 if row.get('binding_rule')=='explicit_stand_at_facing_frame':return True
 s=row['span_start'];w=row['word'];hits=catalog.spans(doc.text)
 if any(h['start']<=s<h['end'] for h in hits):return False
 rel=PATTERNS[w].match(doc.text[s:])
 if not rel:return False
 after=s+rel.end()
 return any(h['start']>=after and re.fullmatch(r'\s*(?:the\s+)?',doc.text[after:h['start']],re.I) for h in hits)

def direct_named_subject(doc,row,catalog):
 """Resolve an explicit NAMED subject independently, never using target position."""
 s=row['span_start'];t=token_at(doc,s);hits=catalog.spans(doc.text);cur=t;seen=set();owner=None
 for _ in range(12):
  if cur.i in seen:break
  seen.add(cur.i)
  if cur!=t and cur.pos_ in {'NOUN','PROPN'} and cur.lemma_.lower() not in POSITION_HEADS:owner=cur;break
  if cur.pos_ in {'VERB','AUX'}:
   subs=[x for x in root_subjects(cur) if x.pos_ in {'NOUN','PROPN'}]
   if len(subs)==1:owner=subs[0];break
  if cur.head==cur:break
  cur=cur.head
 if owner is None:return None
 matches=[h for h in hits if h['start']<=owner.idx<h['end'] and h['end']<=s]
 if len(matches)!=1:return None
 h=matches[0];gap=doc.text[h['end']:s]
 if not re.fullmatch(r'\s*(?:(?:building|road|is|lies|stands|sits|located|situated|directly|to|the|on|at|in|just)\s*)*',gap,re.I):return None
 ids,res,ec=catalog.resolve(h)
 if ids is None:return None
 # A nested 'NAME behind LANDMARK' after another object's description may attach
 # to either NP. Require a finite explicit named-subject clause or initial NP.
 clause_prefix=doc.text[owner.sent.start_char:h['start']].strip().lower()
 explicit_copula=bool(re.search(r'\b(?:is|are|lies|stands|sits)\b',gap,re.I))
 if clause_prefix not in {'','the','a','an'} and not explicit_copula:return None
 return {'ids':ids,'text':doc.text[h['start']:h['end']],'span':[h['start'],h['end']],'resolution':res,'confidence':min(ec,.97)}

def parse_landmark_relations(doc,category,catalog):
 allrows=parse_relations(doc,category,catalog);kept=[]
 for r in allrows:
  if not landmark_anchor(doc,r,catalog):continue
  r['landmark_related']=True;r['subject_role']='annotated_target' if r['status']=='bound' else 'unresolved_context_object'
  r['subject_binding_rule']='annotation_target_subject' if r['status']=='bound' else None
  r['subject_text']=None;r['subject_ids']=[]
  # A direction may describe another NAMED object, not the annotation target.
  if r['status']!='bound' and r['reason'] not in {'direction_in_proper_name','compound_direction_not_single_axis','right_emphasis','coordinated_reference_uncertain','reference_is_named_landmark_subregion','front_part_not_in_front','back_rear_part_or_unknown','part_or_region_not_external_pair','ordinal_count_from_side_not_external_direction'}:
   named=direct_named_subject(doc,r,catalog)
   if named:
    hs=[h for h in catalog.spans(doc.text) if h['start']>=r['span_start']+PATTERNS[r['word']].match(doc.text[r['span_start']:]).end() and re.fullmatch(r'\s*(?:the\s+)?',doc.text[r['span_start']+PATTERNS[r['word']].match(doc.text[r['span_start']:]).end():h['start']],re.I)]
    if len(hs)==1:
     ids,res,ec=catalog.resolve(hs[0])
     if ids:
      score=min(ec,named['confidence']);r.update(status='bound',scope='named_context_subject_direct',reason='',reference_ids=ids,reference_text=doc.text[hs[0]['start']:hs[0]['end']],reference_span=[hs[0]['start'],hs[0]['end']],entity_resolution=res,entity_confidence=ec,scope_confidence=named['confidence'],parse_confidence=score,confidence_tier='high' if score>=.95 else 'medium',subject_role='named_context_object',subject_ids=named['ids'],subject_text=named['text'],subject_binding_rule='explicit_named_subject_dependency_and_surface_agree')
  if r['subject_role']=='unresolved_context_object':r['subject_unresolved_reason']=r['reason']
  kept.append(r)
 return kept,len(allrows)-len(kept)
