"""Public association/ranking and explicitly isolated offline statistics."""
import re
import math
import numpy as np
from shapely.geometry import Polygon
from graphnav.semantic_geometry import canonical,parse_spatial_relations,DIRECTIONS
from experiments.instance_grounding.evaluate import wilson


def percentile(values):
    x=np.asarray(values,float)
    if not np.isfinite(x).all():raise ValueError('finite observed scores required')
    if len(x)<=1:return np.zeros(len(x))
    order=np.argsort(x,kind='stable');out=np.zeros(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        out[order[i:j]]=((i+j-1)/2)/(len(x)-1);i=j
    return out


def public_relations(instruction,regions):
    """No target ID/coordinate in this signature; named observed anchors only."""
    text=canonical(instruction);refs=[r for r in regions if canonical(r.get('name','')) and re.search(r'\b'+re.escape(canonical(r['name']))+r'\b',text)]
    result=np.zeros(len(regions));judged=np.zeros(len(regions),bool);used=[]
    if not refs:return result,judged,used
    relations=parse_spatial_relations(instruction,[dict(name=r['name']) for r in refs])
    for rel in relations:
        if rel.kind not in (*DIRECTIONS,'near'):continue
        anchors=[refs[k] for k in rel.anchors[0]]
        if not anchors:continue
        used.append(dict(kind=rel.kind,anchors=[a['name'] for a in anchors]))
        for i,r in enumerate(regions):
            states=[]
            for a in anchors:
                if r is a:states.append(0);continue
                if rel.kind in DIRECTIONS:
                    d=np.asarray(DIRECTIONS[rel.kind],float);d/=np.linalg.norm(d)
                    s=float((np.asarray(r['center_world'])-np.asarray(a['center_world']))@d);states.append(1 if s>=10 else (-1 if s<=-10 else 0))
                else:
                    d=Polygon(r['contour']).distance(Polygon(a['contour']));states.append(1 if d<=25 else (-1 if d>=50 else 0))
            status=1 if 1 in states else (-1 if states and all(s==-1 for s in states) else 0)
            result[i]+=status;judged[i]|=status!=0
    if used:result/=len(used)
    return result,judged,used


def compose_scores(rgb,geometry,relation,language,modality,relation_available):
    """Fixed diagnostic ranking; no logits are written back to the model."""
    v=percentile(rgb)
    if language=='attribute_spatial' and relation_available:v=(v+(np.asarray(relation)+1)/2)/2
    if modality=='rgb':return v
    if modality=='geometry':return percentile(geometry)
    if modality=='combined':return (v+percentile(geometry))/2
    raise ValueError('unknown modality')


def offline_residual(logits,weights,selected,amplitude,*,offline_only=False):
    if not offline_only:raise ValueError('theoretical gain never a deployable inference path')
    logits=np.asarray(logits,float);w=np.asarray(weights,float);out=logits.copy();res=np.zeros_like(logits)
    if selected is not None:
        if selected<0 or selected>=len(w):raise ValueError('only an observed candidate permitted')
        a=float(logits.max()-logits.min()+1) if amplitude=='strong' else float(amplitude)
        if a<0 or not np.isfinite(a):raise ValueError('finite nonnegative theory amplitude required')
        res=w[selected]*a;out+=res
    assert np.array_equal(out[res==0],logits[res==0])
    return out,res


def auc(labels,scores):
    y=np.asarray(labels,bool);s=np.asarray(scores,float);n=int(y.sum());m=len(y)-n
    if not n or not m:return None
    # Tie-aware Mann–Whitney identity, no numerical threshold fitting.
    ranks=percentile(s)*(len(y)-1)+1
    return float((ranks[y].sum()-n*(n+1)/2)/(n*m))


def reliability(labels,prob,bins=10):
    y=np.asarray(labels,float);p=np.asarray(prob,float)
    if len(y)!=len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('aligned probabilities required')
    rows=[];ece=0
    for i in range(bins):
        mask=(p>=i/bins)&((p<(i+1)/bins) if i<bins-1 else (p<=1))
        if not mask.any():continue
        accuracy=float(y[mask].mean());confidence=float(p[mask].mean());count=int(mask.sum());ece+=count/max(1,len(y))*abs(accuracy-confidence)
        rows.append(dict(bin=i,count=count,accuracy=accuracy,confidence=confidence))
    return dict(n=len(y),ece=float(ece),brier=float(np.mean((p-y)**2)) if len(y) else None,auc=auc(y,p),bins=rows)


def risk_curve(records,confidence_key,thresholds):
    result=[]
    for t in thresholds:
        a=[r for r in records if r['eligible'] and r[confidence_key]>=t];groups={}
        for r in a:groups.setdefault(r['family_id'],[]).append(r['correct'])
        good=sum(all(v) for v in groups.values());lower=wilson(good,len(groups));risk=1-float(np.mean([r['correct'] for r in a])) if a else None
        result.append(dict(threshold=t,n=len(a),families=len(groups),coverage=len(a)/max(1,len(records)),risk=risk,family_any_error_risk=1-good/len(groups) if groups else None,family_correct_wilson95_lower=lower,supported_safe_interval=len(groups)>=10 and lower>=.9))
    return result
