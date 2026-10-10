"""Conservative target-phrase binding. Labels are language weak supervision only."""
import re
from collections import defaultdict
COLORS=r'white|gray|grey|red|blue|brown|black|green|yellow|orange|beige'

def bind_phrase(phrase, object_type):
    p=str(phrase).lower().replace('grey','gray')
    result={}
    # Only explicitly adjacent color / object nouns, or roof color clauses.
    for color, noun in re.findall(r'\b('+COLORS+r')\s+(?:(?:colored|coloured)\s+)?(roof(?:ed)?|building|house|car|vehicle|automobile)\b',p):
        kind={'roof':'roof_color','roofed':'roof_color','building':'building_color','house':'building_color','car':'vehicle_color','vehicle':'vehicle_color','automobile':'vehicle_color'}[noun]
        if (object_type=='Building' and kind in ('roof_color','building_color')) or (object_type=='Car' and kind=='vehicle_color'):
            result.setdefault(kind,set()).add(color)
    for color in re.findall(r'\broof(?:s)?\s+(?:is|are|of|in)\s+('+COLORS+r')\b',p):
        if object_type=='Building':result.setdefault('roof_color',set()).add(color)
    for shape in re.findall(r'\b(rectangular|square|round|circular|l-shaped|u-shaped)\s+(?:roof(?:ed)?\s+)?(?:building|house)\b',p):
        if object_type=='Building':result.setdefault('building_shape',set()).add(shape)
    return {k:next(iter(v)) for k,v in result.items() if len(v)==1}

def object_attributes(descriptions, object_type):
    votes=defaultdict(set); examples=defaultdict(list)
    for d in descriptions:
        phrase=d.get('target','')
        for kind,label in bind_phrase(phrase,object_type).items():
            votes[kind].add(label);examples[kind].append(phrase)
    # Conflicting colors across descriptions are left unknown.
    return {k:{'label':next(iter(v)),'confidence':'consistent_language_weak_label_not_visual_verified','support_phrases':examples[k]} for k,v in votes.items() if len(v)==1}
