"""Additional target-to-NAMED-road travel-side relations absent from old OF syntax.
No target coordinates, trajectories, or polygon geometry are accepted.
"""
from shared import *

def immediate(hits,text,pos):
 a=[h for h in hits if h['start']>=pos and re.fullmatch(r'\s*(?:the\s+)?',text[pos:h['start']],re.I)]
 return a[0] if len(a)==1 else None

def extract(text,catalog):
 hits=catalog.spans(text);out=[]
 for cm in re.finditer(r'\bcoming\s+from\s+(?:the\s+)?',text,re.I):
  source=immediate(hits,text,cm.end())
  if source is None:continue
  sid,srule,sc=catalog.resolve(source)
  if not sid:continue
  onto=list(re.finditer(r'\b(?:onto|down)\s+(?:the\s+)?',text[:cm.start()],re.I))
  dest=immediate(hits,text,onto[-1].end()) if onto else None
  if dest is None:continue
  did,rule,dc=catalog.resolve(dest)
  if not did or set(did)&set(sid) or not all(catalog.entities[i].category=='TrafficRoad' for i in did+sid):continue
  # Coming-from source is immediately followed by a target-deictic clause.
  after=text[source['end']:]
  clause=re.match(r'\s*[, ]*(?:(?:then\s+)?(?:this\s+is|it\s+is|it\s+will\s+be|it\s+would\s+be|it\s+would|it\s+will|it\'s|the\s+car\s+is|the\s+house\s+is))\s+',after,re.I)
  if not clause:continue
  start=source['end']+clause.end();end=re.search(r'[.!;]',text[start:]);end=start+end.start() if end else len(text)
  for w in re.finditer(r'\b(left|right)\b',text[start:end],re.I):
   s=start+w.start();prefix=text[start:s]
   # Direct side of the travel frame, not turning commands or anonymous nesting.
   if not re.search(r'\b(?:your|the|on|to)\s*$',prefix,re.I):continue
   if re.search(r'\b(?:turn|take|alley|driveway|lot|parking\s+area|next|between|behind|front|houses|buildings)\b',prefix,re.I):continue
   # At most one explicit target head. More entities create a scope boundary.
   if len(re.findall(r'\b(?:car|vehicle|house|building|tree|van)\b',prefix,re.I))>1:continue
   out.append({'word':w[1].lower(),'span_start':s,'span_end':s+len(w[1]),'reference_ids':did,'reference_text':text[dest['start']:dest['end']],'reference_span':[dest['start'],dest['end']],'source_road_ids':sid,'travel_road_span':[dest['start'],dest['end']],'coming_from_span':[cm.start(),source['end']],'parse_confidence':min(dc,sc,.95),'binding_rule':'explicit_named_road_entry_target_deictic_side','confidence_tier':'high' if min(dc,sc,.95)>=.95 else 'medium'})
 return out
