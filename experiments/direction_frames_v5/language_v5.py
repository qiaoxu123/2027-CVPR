"""Text/type-only semantic roles and conservative target-side scope extensions."""
from v5core import *
from road_language import extract as frozen_extract
import difflib
SEARCH=re.compile(r'\b(?:coming\s+from|turn(?:ing)?|enter(?:ing)?|approach(?:ing)?|along|facing|proceeds?|travell?ing|via)\b',re.I)

def role_pairs(text,catalog):
 hits=catalog.spans(text);roads=[]
 for h in hits:
  ids,res,c=catalog.resolve(h)
  if ids and all(catalog.entities[i].category=='TrafficRoad' for i in ids):roads.append({**h,'resolved_ids':ids,'confidence':c})
 candidates=[]
 for a in roads:
  for b in roads:
   if set(a['resolved_ids'])&set(b['resolved_ids']):continue
   preA=text[max(0,a['start']-65):a['start']];preB=text[max(0,b['start']-65):b['start']];kind=None
   if a['end']<=b['start'] and b['start']-a['end']<210:
    middle=text[a['end']:b['start']]
    if re.search(r'\b(?:coming\s+from)\s+(?:the\s+)?$',preB,re.I):
     if re.search(r'\b(?:onto|down|into)\s+(?:the\s+)?$',preA,re.I):kind='coming_from_deictic'
     elif re.search(r'\b(?:on|of|along)\s+(?:the\s+)?$',preA,re.I):kind='coming_from_road_side'
    elif re.search(r'\bfrom\s+(?:the\s+)?$',preB,re.I):
     if re.search(r'\b(?:turn(?:\s+(?:left|right))?\s+(?:onto|into)|take\s+a\s+(?:left|right)\s+onto)\s+(?:the\s+)?$',preA,re.I):kind='turn_onto_from'
     elif re.search(r'\bapproaching\s+(?:the\s+)?$',preA,re.I):kind='approaching_from_UNDERSPECIFIED'
     elif re.search(r'\b(?:along|down)\s+(?:the\s+)?$',preA,re.I):
      kind='along_from' if re.search(r'\b(?:proceeds?|proceeding|travel|travelling|traveling|drive|driving)\b',preA+middle,re.I) or re.fullmatch(r'\s*from\s+(?:the\s+)?',middle,re.I) else 'implicit_order_origin_UNDERSPECIFIED'
     elif re.search(r'\bfacing\s+(?:the\s+)?$',preA,re.I):kind='facing_from_OBSERVER'
    elif re.search(r'\bvia\s+(?:the\s+)?$',preB,re.I) and re.search(r'\benter(?:ing)?\s+(?:the\s+)?$',preA,re.I):kind='enter_via'
   elif b['end']<=a['start'] and a['start']-b['end']<180:
    middle=text[b['end']:a['start']]
    if re.search(r'\b(?:coming\s+from|turn(?:\s+(?:left|right))?\s+from)\s+(?:the\s+)?$',preB,re.I) and re.search(r'\b(?:onto|into|to)\s+(?:the\s+)?$',middle,re.I):kind='coming_from_to_onto'
    elif re.search(r'\bcoming\s+from\s+(?:the\s+)?$',preB,re.I) and re.search(r'\b(?:left|right)\s+(?:hand\s+)?(?:side\s+)?of\s+(?:the\s+)?$',middle,re.I):kind='prefix_coming_from_road_side'
   if kind:candidates.append({'A':a,'B':b,'template':kind,'role_span':[min(a['start'],b['start']),max(a['end'],b['end'])]})
 # Repeated names can describe identical roles; conflicting pairs are NOT resolved
 # by target proximity. Keep every role pair until scope can license one.
 dedup={}
 for r in candidates:dedup.setdefault((tuple(r['A']['resolved_ids']),tuple(r['B']['resolved_ids']),r['template']),r)
 return list(dedup.values()),hits

def side_scope(text,start,role,hits):
 # Reject movement commands, anonymous object comparisons and ordinal-only parts.
 word=re.match(r'\w+',text[start:])[0].lower();before=text[:start];after=text[start+len(word):]
 if re.search(r'\b(?:turn|take|turning)\s+(?:a\s+|to\s+the\s+)?$',before,re.I):return False,'turn_direction_not_target_relation',False
 if re.match(r'\s+(?:(?:hand\s+)?side\s+)?(?:of|from)\s+(?:(?:the|a|another|first)\s+)*(?:car|vehicle|tree|house|building|alley)\b',after,re.I):return False,'anonymous_reference_not_in_population',False
 if re.match(r'\s+(?:side\s+)?(?:lane|of\s+the\s+corner\s+turn)\b',after,re.I):return False,'lane_or_turn_region_not_direct_road_pair',True
 rel=PATTERNS[word].match(text[start:])
 if rel:
  pos=start+rel.end();named=[h for h in hits if h['start']>=pos and re.fullmatch(r'\s*(?:the\s+)?',text[pos:h['start']],re.I)]
  if len(named)==1 and not set(role['A']['resolved_ids'])<=set(named[0]['ids']):return False,'named_relation_reference_differs_from_travel_road',False
 context_start=max(role['A']['end'],role['B']['end'])
 if start>context_start and re.search(r'\b(?:alley|driveway|parking\s+lot|car\s+park)\b',text[context_start:start],re.I):return False,'intervening_local_frame_or_container_after_road_entry',True
 if start>context_start and re.search(r'\b(?:facing|looking|viewed|standing|stand|turn(?:ing)?)\b',text[context_start:start],re.I):return False,'new_observer_or_motion_frame_after_road_entry',True
 # Names are masked without deleting scope boundaries.
 masked=list(text[:start])
 for h in hits:
  for j in range(h['start'],min(h['end'],start)):masked[j]=' '
 prefix=''.join(masked);clause=max(prefix.rfind('.'),prefix.rfind(';'),prefix.rfind('!'))+1;local=prefix[clause:]
 markers=list(re.finditer(r"\b(?:this\s+is|it\s+is|it\s+will\s+be|it\s+would\s+be|it\s+would|it's|the\s+car\s+is|the\s+house\s+is)\b",local,re.I))
 if markers:
  owned=local[markers[-1].end():];mode='deictic_target_clause'
  if role['A']['end']<clause+markers[-1].start() and role['template']=='coming_from_road_side' and re.search(r'\b(?:lot|parking|corner\s+building)\b',prefix[:clause+markers[-1].start()],re.I):return False,'road_A_embedded_in_context_not_travel_anchor',True
 else:
  owned=local;mode='initial_target_NP'
  if not re.match(r'\s*(?:the|a|an|red|white|black|grey|gray)\b',owned,re.I):return False,'no_independent_target_subject',True
  if not re.search(r'\b(?:building|house|car|vehicle|van)\b',owned,re.I):return False,'target_head_missing_or_part_only',True
 if re.search(r'\b(?:driveway|alley|parking\s+(?:lot|area|space)|car\s+park|row|lane|portion|part|middle|section|between|next\s+to|behind|in\s+front\s+of)\b',owned,re.I):return False,'container_or_embedded_scope_not_simplified',True
 owned=re.sub(r'\b(building|car|house)\s+\1\b',r'\1',owned,flags=re.I)
 heads=re.findall(r'\b(?:building|house|car|vehicle|van|tree|cars|houses|buildings)\b',owned,re.I)
 if len(heads)>1 or any(h.lower() in {'tree','cars','houses','buildings'} for h in heads):return False,'multiple_or_contextual_entity_heads',True
 # Static target side must explicitly be your/the/on/to LEFT, or LEFT OF A.
 if not re.search(r'\b(?:on|to|at|your|the)\s*(?:the\s+)?$',before,re.I):return False,'not_a_target_side_marker',False
 return True,mode,True

def template_text(text,hits):
 for h in sorted(hits,key=lambda z:z['start'],reverse=True):text=text[:h['start']]+' ROAD '+text[h['end']:]
 text=re.sub(r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|twelfth|last|\d+)\b','#',text.lower());return ' '.join(re.findall(r'[a-z#]+',text))
