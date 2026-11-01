"""Conservative public-language role parsing; no target labels or future observations."""
import re
import torch

DIRECTIONS={'north':(0.,1.),'south':(0.,-1.),'east':(1.,0.),'west':(-1.,0.)}
OPPOSITE={'north':'south','south':'north','east':'west','west':'east','left':'right','right':'left','behind':'in front of','in front of':'behind'}


def parse_roles(text,names):
    matches=[]
    for index,name in enumerate(names):
        if not name.strip():continue
        name=re.sub(r'^(Building|TrafficRoad|Footpath|Bridge|Parking|Ground|road)\s+','',name)
        # Ignore punctuation differences, but do not fuzzy-match unrelated landmarks.
        parts=re.findall(r'\w+',name)
        if not parts:continue
        pattern=r'\b(?:(north|south|east|west|left|right)(?:ern)?\s+(?:side\s+)?of|(behind|near|beside|opposite|in front of|next to))\s+(?:the\s+)?'+r'\W+'.join(map(re.escape,parts))+r'\b'
        for match in re.finditer(pattern,text,re.I):matches.append((index,name,match))
    if len(matches)!=1:
        return dict(valid=False,target=text,reference='',relation='',direction=(0.,0.),reference_index=-1,opposite_text=text)
    index,name,match=matches[0];group=1 if match.group(1) else 2;word=match.group(group).lower()
    opposite=OPPOSITE.get(word,word)
    # Remove the bound relation clause; preserve remaining target appearance/context.
    target=(text[:match.start()]+' '+text[match.end():]).strip(' ,.;') or text
    return dict(valid=True,target=target,reference=name,relation=word,direction=DIRECTIONS.get(word,(0.,0.)),reference_index=index,
                opposite_text=text[:match.start(group)]+opposite+text[match.end(group):])


def role_batch(texts,names,encoder,device):
    rows=[parse_roles(t,n) for t,n in zip(texts,names)]
    # These encode the target and the referent independently in the same frozen BERT space.
    def encode(text):
        tokens,mask=encoder.encode_tokens(text)
        return ((tokens.float()*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).to(device).clone()
    target=encode([r['target'] for r in rows])
    reference=encode([r['reference'] for r in rows])
    relation=encode([r['relation'] for r in rows])
    return dict(role_target=target,role_reference=reference,role_relation=relation,
                role_direction=torch.tensor([r['direction'] for r in rows],device=device),
                role_valid=torch.tensor([r['valid'] for r in rows],device=device),
                role_reference_index=torch.tensor([r['reference_index'] for r in rows],device=device)),rows


class BalancedRoleSampler:
    """Half ordinary shard samples, half role-bound Train states; exclusions inherited."""
    def __init__(self,original,episodes,role_episodes):
        self.original=original;self.base=original.base;self.episodes=episodes;self.role_episodes=role_episodes

    def __len__(self):return len(self.original)

    def __iter__(self):
        indices=list(self.original)
        roles=[i for i in indices if self.episodes[i//4] in self.role_episodes]
        if not roles:raise ValueError('No eligible role-bound Train states')
        generator=torch.Generator().manual_seed(1000+self.base.epoch)
        choices=torch.randint(len(roles),(len(indices)//2,),generator=generator).tolist()
        for j,index in enumerate(indices):
            yield index if j%2==0 else roles[choices[j//2]]
