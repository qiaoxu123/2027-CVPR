"""Offline counterfactual construction/scoring; never used by navigation."""
import re
import numpy as np
import shapely

PAIRS = {'NS': ('north','south'), 'EW': ('east','west'), 'IO': ('inside','outside')}
PHRASES = {'north':'north of','south':'south of','east':'east of','west':'west of',
           'inside':'inside','outside':'outside'}
AXES = {'north': np.array([0.,1.]),'south':np.array([0.,-1.]),
        'east':np.array([1.,0.]),'west':np.array([-1.,0.])}


def replacement_span(task, mentions):
    """Use the already scope-checked relation, never rewrite a landmark name."""
    anchor = task['anchors'][0]
    phrase = anchor['original_predicate']
    pattern = r'\b'+r'\s+'.join(re.escape(w) for w in phrase.split())+r'\b'
    start,end = anchor['source_span']
    found = [m for m in re.finditer(pattern,task['instruction'],re.I)
             if start<=m.start()<m.end()<=end
             and not any(m.start()<q['end'] and m.end()>q['start'] for q in mentions)]
    if not found:raise ValueError('No source-bound predicate outside public names')
    m=found[-1]
    return m.start(),m.end()


def replace_relation(text,span,word):
    start,end=span;phrase=PHRASES[word]
    return text[:start]+phrase+text[end:],(start,start+len(phrase))


def relation_regions(world,polygon):
    points=shapely.points(world);inside=shapely.covers(polygon,points)
    regions={'inside':inside,'outside':~inside}
    vertices=shapely.get_coordinates(polygon)
    for word,axis in AXES.items():regions[word]=(world@axis)>(vertices@axis).max()+1e-8
    return regions


def probabilities(logits):
    x=np.asarray(logits,dtype=np.float64);p=np.exp(x-x.max());return p/p.sum()


def score_pair(logits_a,logits_b,world,regions,pair):
    a,b=PAIRS[pair];pa,pb=probabilities(logits_a),probabilities(logits_b)
    ra,rb=regions[a],regions[b];ma=pa@world;mb=pb@world
    shift=ma-mb
    valid=bool(ra.any() and rb.any())
    change_a=float(pa[ra].sum()-pb[ra].sum())
    change_b=float(pb[rb].sum()-pa[rb].sum())
    out=dict(valid_regions=valid,region_cells=[int(ra.sum()),int(rb.sum())],
        centroid_a=ma.tolist(),centroid_b=mb.tolist(),centroid_delta=shift.tolist(),
        centroid_distance_m=float(np.linalg.norm(shift)),
        mass_a_in_a=float(pa[ra].sum()),mass_b_in_a=float(pb[ra].sum()),
        mass_a_in_b=float(pa[rb].sum()),mass_b_in_b=float(pb[rb].sum()),
        mass_change_a=change_a,mass_change_b=change_b,
        joint_mass_correct=bool(change_a>1e-8 and change_b>1e-8) if valid else None,
        top1_a_region=bool(ra[np.argmax(logits_a)]) if valid else None,
        top1_b_region=bool(rb[np.argmax(logits_b)]) if valid else None,
        total_variation=float(np.abs(pa-pb).sum()/2),
        area_fraction_a=float(ra.mean()),area_fraction_b=float(rb.mean()))
    if pair!='IO':
        signed=float(shift@AXES[a]);out.update(signed_centroid_shift_m=signed,
            centroid_direction_correct=bool(signed>1e-5) if valid else None)
    else:out.update(signed_centroid_shift_m=None,centroid_direction_correct=None)
    return out


def effective_logits(geometry,relation,word):
    """PR16 with no RGB: cardinal abstention; topology keeps Relation output."""
    return geometry if word in AXES else relation
