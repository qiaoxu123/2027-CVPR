"""Roles/scopes from raw text, without any GT annotation lookup."""
import copy,re
from experiments.role_aware_object_graph.parser import parse as role_parse


def parse(text,graph):
    language=role_parse(text,graph)
    language['variables']=[]
    for i,e in enumerate(language['entities']):
        e=copy.deepcopy(e);e['variable']='Z' if i==0 else f'X{i}';e['coreference']=None
        # Keep observable roof/part colours separate; never bind another noun's
        # colour to this variable merely because it appears in full instruction.
        e['observable_phrase']='a '+(' '.join(e['colors'])+' ' if e['colors'] else '')+(e.get('head_text') or e['object_type'] or 'object').lower()
        e['observable_supported']=bool(e['colors'] or e['object_type'])
        language['variables'].append(e)
    for f in language['factors']:
        f['symmetric_anchors']=f['kind']=='between'
        f['frame_unknown']=f['kind'] in ('left','right','front','behind')
    language['coreference_policy']='explicit reuse of an entity index allowed; unresolved pronouns remain Unknown'
    return language


def active_variables(language):
    """Only relation variables which can affect the target's marginal."""
    connected={0};factors=[f for f in language.get('factors',[]) if not f.get('frame_unknown') and f['kind'] not in ('left','right','front','behind')]
    for _ in range(len(language.get('variables',[]))+1):
        old=set(connected)
        for f in factors:
            endpoints={f['subject'],*f['anchors']}
            if endpoints&connected:connected|=endpoints
        if old==connected:break
    return connected,[f for f in factors if {f['subject'],*f['anchors']}<=connected]
