"""Public-text-only named landmark retrieval, including conservative name variants.
No target, pose, distance-to-target or annotation reference IDs are accepted.
Actual mention spans survive as aliases for existing relation parsing.
"""
import re
from collections import defaultdict
from rapidfuzz.fuzz import partial_ratio_alignment
from scripts.analyze_instruction_landmark_association import normalise,GENERIC_NAMES

ABBREVIATIONS={'rd':'road','ave':'avenue','ln':'lane','dr':'drive','pl':'place','saint':'st'}
ROAD_TYPES={'TrafficRoad','Road'}
ROAD_SUFFIXES={'road','street','avenue','lane','drive','way','close','gardens','grove','crescent','terrace','place'}
BARE_GENERIC={'red','white','black','silver','grey','gray','brown','yellow','green','blue','orange','purple','pink','short','long','tall','small','large','big','little','middle','top','bottom','front','rear','back','left','right','old','new'}
OBJECT_NOUNS={'building','buildings','house','houses','church','college','school','library','hall','centre','center','car','cars','van','truck','vehicle'}
ROAD_CONTEXT=re.compile(r'\b(?:on|along|between|road|street|avenue|lane|drive|junction|intersection|corner|end)\b')


def normalized_span(text):
    raw,pos=normalise(text);parts=[];starts=[];ends=[]
    for m in re.finditer(r'\S+',raw):
        word=ABBREVIATIONS.get(m.group(),m.group())
        if parts:parts.append(' ');starts.append(pos[m.start()]);ends.append(pos[m.start()])
        parts.append(word);starts.extend([pos[m.start()]]*len(word));ends.extend([pos[m.end()-1]+1]*len(word))
    return ''.join(parts),starts,ends


class PublicLandmarkMentions:
    def __init__(self,catalog,fuzzy_threshold=.92,fuzzy_margin=.05):
        self.maps={};self.threshold=fuzzy_threshold;self.margin=fuzzy_margin
        for map_name,objects in catalog.items():
            full=defaultdict(dict);roads=defaultdict(set)
            for oid,obj in objects.items():
                name=obj.get('name','');key=normalized_span(name)[0]
                if not key or key in GENERIC_NAMES or len(key)<3 or len(obj.get('contour',[]))<3:continue
                item=dict(id=str(oid),name=name,type=obj.get('object_type','Unknown'))
                variants=[name]+(obj.get('aliases',[]) if isinstance(obj.get('aliases',[]),list) else [])
                for variant in variants:
                    if not isinstance(variant,str):continue
                    v=normalized_span(variant)[0]
                    if len(v)<3 or v in GENERIC_NAMES:continue
                    full[v][str(oid)]=item
                    if v.startswith('the ') and len(v)>7:full[v[4:]][str(oid)]=item
                if obj.get('object_type') in ROAD_TYPES:
                    words=key.split()
                    if len(words)>1 and words[-1] in ROAD_SUFFIXES:
                        base=' '.join(words[:-1])
                        if len(base)>=5 and base not in BARE_GENERIC and base not in GENERIC_NAMES and base not in {'north','south','east','west','upper','lower','church','school','college','garden','parking','avenue','street'}:roads[base].add(name)
            bare={k:{oid:v for entries in full.values() for oid,v in entries.items() if v['name'] in names} for k,names in roads.items() if len(names)==1}
            self.maps[map_name]=(dict(full),bare)

    def match(self,map_name,text):
        norm,starts,ends=normalized_span(text);full,bare=self.maps.get(map_name,({},{}));found=[];exact_names=set()
        def entry(start,end,key,items,kind,score):
            lo,hi=starts[start],ends[end-1]
            return dict(start=lo,end=hi,text=text[lo:hi],public_names=sorted({x['name'] for x in items.values()}),public_ids=sorted(items),types=sorted({x['type'] for x in items.values()}),kind=kind,score=float(score))
        # Scan each public alias independently: a compound road name must not
        # hide its explicit component road names, unlike longest-only regexes.
        for table,kind in [(full,'normalized_exact'),(bare,'unique_road_base')]:
            for key,items in table.items():
                for m in re.finditer(r'(?<!\w)'+re.escape(key)+r'(?!\w)',norm):
                    if kind=='unique_road_base':
                        if not ROAD_CONTEXT.search(norm[max(0,m.start()-60):m.end()+60]):continue
                        following=norm[m.end():].strip().split()
                        if following and following[0] in OBJECT_NOUNS:continue
                    found.append(entry(m.start(),m.end(),key,items,kind,1.));exact_names.update(x['name'] for x in items.values())
        fuzzy=[]
        for key,items in full.items():
            if len(key)<10 or all(x['name'] in exact_names for x in items.values()):continue
            a=partial_ratio_alignment(key,norm,score_cutoff=100*self.threshold)
            if a is None or a.dest_end-a.dest_start<.8*len(key):continue
            s,e=a.dest_start,a.dest_end
            if s and norm[s-1].isalnum() or e<len(norm) and norm[e].isalnum():continue
            fuzzy.append(entry(s,e,key,items,'fuzzy_public_name',a.score/100))
        def overlap(a,b):return max(0,min(a['end'],b['end'])-max(a['start'],b['start']))/max(1,min(a['end']-a['start'],b['end']-b['start']))
        for x in fuzzy:
            other=max((y['score'] for y in found+fuzzy if y['public_names']!=x['public_names'] and overlap(x,y)>.5),default=0.)
            if x['score']-other>=self.margin:found.append(x)
        # Entity scope: a street name containing a church name does not also
        # mention that church, and a church name containing a street name does
        # not by itself reference the street. Keep explicit component roads
        # only for a public compound road name (e.g. 'A Rd / B Rd').
        def shadowed(x,y):
            def core_start(row):
                article=re.match(r'the\s+',row['text'],re.I)
                return row['start']+(article.end() if article else 0)
            xs,ys=core_start(x),core_start(y)
            contained=ys<=xs and y['end']>=x['end'] and y['end']-ys>x['end']-xs
            if not contained:return False
            compound=all(t in ROAD_TYPES for t in y['types']) and any('/' in n for n in y['public_names'])
            if compound and all(t in ROAD_TYPES for t in x['types']) and x['public_names']!=y['public_names']:return False
            return True
        found=[x for x in found if not any(shadowed(x,y) for y in found)]
        dedup={}
        for x in found:
            key=(x['start'],x['end'],tuple(x['public_names']))
            if key not in dedup or x['score']>dedup[key]['score']:dedup[key]=x
        return sorted(dedup.values(),key=lambda x:(x['start'],-x['score'],x['public_names']))


def add_mention_aliases(record,mentions):
    """Keep the original words that existing geometric relation rules must match."""
    for landmark in record['landmarks']:
        aliases=list(landmark.get('aliases',[]))
        for m in mentions:
            if landmark['name'] in m['public_names'] and m['text']!=landmark['name'] and m['text'] not in aliases:aliases.append(m['text'])
        landmark['aliases']=aliases
    return record
