import numpy as np
import shapely
from scripts.relation_counterfactual_metrics import replacement_span,replace_relation,relation_regions,score_pair,effective_logits


def test_rewrite_preserves_names_attributes_and_other_clauses():
    text='A white car near North Road, beside a red van.'
    task=dict(instruction=text,anchors=[dict(original_predicate='near',source_span=[8,26])])
    span=replacement_span(task,[dict(start=17,end=27)])
    changed,_=replace_relation(text,span,'south')
    assert changed=='A white car south of North Road, beside a red van.'
    assert changed[:span[0]]==text[:span[0]]
    assert changed.endswith(text[span[1]:])


def test_known_positive_geometry_and_negative_reversal():
    yy,xx=np.meshgrid(np.arange(-2,3),np.arange(-2,3),indexing='ij')
    world=np.stack((xx.ravel(),yy.ravel()),-1).astype(float)
    regions=relation_regions(world,shapely.box(-.6,-.6,.6,.6))
    right=score_pair(world[:,1],-world[:,1],world,regions,'NS')
    wrong=score_pair(-world[:,1],world[:,1],world,regions,'NS')
    assert right['centroid_direction_correct'] and right['joint_mass_correct']
    assert not wrong['centroid_direction_correct'] and not wrong['joint_mass_correct']
    io=score_pair(regions['inside']*8,regions['outside']*8,world,regions,'IO')
    assert io['joint_mass_correct']


def test_empty_inside_is_unknown_not_model_failure():
    world=np.array([[0.,0.],[1.,1.]])
    regions=relation_regions(world,shapely.box(.1,.1,.2,.2))
    result=score_pair(np.array([0.,1.]),np.array([1.,0.]),world,regions,'IO')
    assert not result['valid_regions'] and result['joint_mass_correct'] is None


def test_cardinal_runtime_fallback_does_not_mean_geometry_ignores_text():
    g=np.array([1.,2.]);r=np.array([3.,4.])
    assert np.array_equal(effective_logits(g,r,'north'),g)
    assert np.array_equal(effective_logits(g,r,'inside'),r)
