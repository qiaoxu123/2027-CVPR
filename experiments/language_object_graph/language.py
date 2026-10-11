"""Raw-instruction language graph. No processed/target annotation lookup."""
from dataclasses import dataclass,asdict
import hashlib
import re
import numpy as np
from graphnav.landmark_alias import canonical_name

TYPE_PATTERNS={
 'Building':r'building|buildings|house|houses|church|warehouse|hall|school|store|shop',
 'Car':r'car|cars|vehicle|vehicles|automobile|van|truck',
 'Parking':r'parking lot|parking area|parking space|car park|parking',
 'TrafficRoad':r'road|street|avenue', 'Footpath':r'footpath|walkway|path',
 'Ground':r'lawn|grass|field|ground|garden|empty land',
 'HighVegetation':r'tree|trees|vegetation', 'Water':r'river|water|brook',
 'Bridge':r'bridge','Bike':r'bicycle|bike','Wall':r'wall',
 'StreetFurniture':r'pole|lamp|bench|sign','Rail':r'railway|rail',
}
RELATIONS=('between','near','along','east','west','north','south','left','right','front','behind')
COLORS='white|gray|grey|red|blue|brown|black|green|yellow|orange|beige|silver'
TRIGGER=re.compile(r'\b(between|next to|near|beside|adjacent to|close to|alongside|along|'
 r'(?:to the |on the )?(?:east|west|north|south)(?:ern)?(?: side)?(?: of)?|'
 r'(?:to the |on the )?(?:left|right)(?: side)?(?: of)?|in front of|behind)\b',re.I)


def canon(text):return canonical_name(text).replace(' s ','s ')


@dataclass
class Entity:
    phrase:str
    role:str
    object_type:str|None
    colors:tuple[str,...]=()
    shape:str|None=None
    size:str|None=None
    named_ids:tuple[int,...]=()
    named_names:tuple[str,...]=()
    unknown:tuple[str,...]=()


def entity(phrase,role,graph):
    text=canon(phrase);mentions=[]
    for i,name in enumerate(graph.names):
        key=canon(name)
        if key and re.search(r'\b'+re.escape(key)+r'\b',text):mentions.append((len(key),key,i,name))
    # Preserve homonymous fragments, avoid a shorter nested name swallowing a full name.
    if mentions:
        longest=max(x[0] for x in mentions);mentions=[x for x in mentions if x[0]==longest]
    type_hits=[]
    for typ,p in TYPE_PATTERNS.items():
        m=re.search(r'\b(?:'+p+r')\b',text)
        if m:type_hits.append((m.start(),-len(m.group()),typ))
    typ=min(type_hits)[2] if type_hits else None
    color=tuple(dict.fromkeys(c.replace('grey','gray') for c in re.findall(r'\b('+COLORS+r')\b',text)))
    # A full entity phrase with several nouns/colors is not a reliable attribute binding.
    nouns=sum(bool(re.search(r'\b(?:'+p+r')\b',text)) for p in TYPE_PATTERNS.values())
    unknown=[]
    if len(color)>1 or nouns>1:
        color=();unknown.append('color scope ambiguous')
    shape=next((x for x in ['l shaped','u shaped','rectangular','square','round','circular','long','narrow'] if re.search(r'\b'+x+r'\b',text)),None)
    size=next((x for x in ['small','large','big','huge','tiny'] if re.search(r'\b'+x+r'\b',text)),None)
    return Entity(phrase,role,typ,color,shape,size,tuple(x[2] for x in mentions),tuple(dict.fromkeys(x[3] for x in mentions)),tuple(unknown))


def parse(instruction,graph):
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('raw instruction required')
    matches=list(TRIGGER.finditer(instruction));prefix=instruction[:matches[0].start()] if matches else instruction
    # Attribute/context clauses do not silently turn a car beside a building into the target.
    target_phrase=re.split(r'\b(?:with|that has|which has)\b',prefix,flags=re.I)[0].strip(' ,.')
    target=entity(target_phrase,'target',graph);entities=[target];factors=[]
    for j,m in enumerate(matches):
        clause=instruction[m.end():matches[j+1].start() if j+1<len(matches) else len(instruction)].strip(' ,.')
        trigger=m.group().lower();kind='near' if trigger in ['next to','near','beside','adjacent to','close to'] else 'along' if trigger in ['along','alongside'] else 'front' if trigger=='in front of' else next((k for k in RELATIONS if k in trigger),None)
        if kind is None or not clause:continue
        if kind=='between':
            groups=re.split(r'\band\b',clause,flags=re.I)
            if len(groups)<2:
                factors.append(dict(kind=kind,anchors=[],status='Unknown: two anchors not parsed',frame_hypotheses=[]));continue
            anchors=[]
            for phrase in groups[:3]:
                entities.append(entity(phrase,'reference',graph));anchors.append(len(entities)-1)
            factors.append(dict(kind=kind,anchors=anchors,status='weak parsed joint factor',frame_hypotheses=['world']))
        else:
            entities.append(entity(clause,'reference',graph));anchors=[len(entities)-1]
            frames=['world'] if kind in ['near','along','east','west','north','south'] else ['world-north-up','current-uav','opposite-current-uav']
            factors.append(dict(kind=kind,anchors=anchors,status='weak parsed factor; no identity truth',frame_hypotheses=frames))
    return dict(instruction=instruction,entities=[asdict(e) for e in entities],factors=factors,
                source='raw instruction and public names only',reference_instance_truth=None)


def vector(language,dimension=96):
    """Fixed language-graph descriptor + signed lexical hash, no trained encoder."""
    out=np.zeros(dimension,np.float32)
    for word in re.findall(r'[a-z]+',language['instruction'].lower()):
        h=hashlib.sha256(word.encode()).digest();j=int.from_bytes(h[:2],'little')%64;out[j]+=1 if h[2]%2 else -1
    out[:64]/=max(np.linalg.norm(out[:64]),1)
    for f in language['factors']:
        if f['kind'] in RELATIONS:out[64+RELATIONS.index(f['kind'])]+=1
    for i,typ in enumerate(TYPE_PATTERNS):out[76+i]=float(language['entities'][0]['object_type']==typ)
    out[-3:]=[float(language['entities'][0]['size'] is not None),float(language['entities'][0]['shape'] is not None),float(bool(language['entities'][0]['colors']))]
    return out
