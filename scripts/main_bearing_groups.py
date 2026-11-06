"""Post-hoc PUBLIC TEXT strata, not inference rules or human semantic gold."""
import re
from collections import Counter
from scripts.analyze_instruction_landmark_association import PublicNames,normalise


def strata(record,eligible,matcher):
 refs=record['landmarks'];counts=Counter(normalise(x['name'])[0] for x in refs)
 text=record['instruction'];indirect=False;attached=False
 for m in matcher.match(record['map_name'],text):
  prefix=normalise(text[:m['start']])[0]
  if re.search(r'\b(?:north|south|east|west)(?: side)? of(?: the)?$',prefix):
   attached=True
   if re.search(r'\b(?:in|inside|within|at)\b[^.!?;]{0,100}\b(?:parking lot|car park|row|lane)\b[^.!?;]{0,70}\b(?:north|south|east|west)(?: side)? of(?: the)?$',prefix):indirect=True
 return dict(public_direct_gate=bool(eligible),same_name_multi_polygon=any(n>1 for n in counts.values()),multiple_public_names=len(counts)>1,no_public_anchor=not refs,heuristic_indirect_cardinal_chain=indirect,attached_cardinal_declined=attached and not eligible)
