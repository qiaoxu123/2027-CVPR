"""Regression protection for target binding, uncertain labels and clustered statistics."""
from experiments.visual_grounding.data.audit_attributes import bind_phrase,object_attributes
from experiments.visual_grounding.statistics import intervals,macro_f1


def test_surrounding_car_cannot_color_building():
    assert bind_phrase('gray building with a white car beside it','Building')=={'building_color':'gray'}
    assert bind_phrase('gray building with a white car beside it','Car')=={'vehicle_color':'white'}


def test_conflicting_object_descriptions_are_unknown():
    ds=[{'target':'white building'},{'target':'gray building'}]
    assert object_attributes(ds,'Building')=={}


def test_roof_is_distinct_from_wall_color():
    assert bind_phrase('red building with white roof','Building')=={'building_color':'red','roof_color':'white'}


def test_map_clusters_and_paired_difference():
    x=[1,1,0,0];m=['a','a','b','b'];r=intervals(x,m,draws=100)
    assert r['mean']==.5 and r['maps']==2
    diff=intervals([a-b for a,b in zip(x,x)],m,draws=100)
    assert diff['iid_95ci']==[0.,0.] and diff['map_cluster_95ci']==[0.,0.]


def test_macro_not_majority_accuracy():
    assert macro_f1(['white']*9+['red'],['white']*10)<.5
