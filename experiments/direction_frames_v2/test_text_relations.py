import pytest
from text_relations import Entity,TextCatalog,parse_relations
@pytest.fixture(scope='module')
def nlp():
 import spacy
 return spacy.load('en_core_web_sm',disable=['ner'])
@pytest.fixture
def cat():return TextCatalog([Entity('1','Church','Building'),Entity('2','Panton Hall','Building'),Entity('3','Wellington Road','TrafficRoad')])
def get(nlp,cat,text,word='behind',category='Building'):
 return next(r for r in parse_relations(nlp(text),category,cat) if r['word']==word)

def test_roof_attribute_not_indirect(nlp,cat):
 r=get(nlp,cat,'A big building with grey roof situated behind the Church.');assert r['status']=='bound' and r['reference_ids']==['1']
def test_indirect_parking_not_target(nlp,cat):
 r=get(nlp,cat,'The white car in the parking lot behind the Church.',category='Car');assert r['status']=='excluded' and r['scope']=='indirect'
def test_indirect_reference_not_flattened(nlp,cat):
 r=get(nlp,cat,'The white car behind the parking lot of the Church.',category='Car');assert r['status']=='excluded'
def test_road_context_does_not_change_reference(nlp,cat):
 r=get(nlp,cat,'The gray house on Wellington Road to the right of the Church.','right');assert r['status']=='bound' and r['reference_ids']==['1']
def test_named_road_relative_clause_is_ambiguous(nlp,cat):
 r=get(nlp,cat,'The gray house on Wellington Road that is 12 houses to the right of the Church.','right');assert r['status']=='excluded' and r['reason']=='relative_clause_named_context_attachment_ambiguous'
def test_multireference_words_bind_separately(nlp,cat):
 rs=parse_relations(nlp('The building is right of Panton Hall and left of the Church.'),'Building',cat)
 assert [(r['word'],r['reference_ids']) for r in rs]==[('right',['2']),('left',['1'])]
def test_nested_other_car_not_target(nlp,cat):
 r=get(nlp,cat,'The white car with a blue car behind the Church.',category='Car');assert r['status']=='excluded'
def test_duplicate_building_never_chosen(nlp):
 c=TextCatalog([Entity('1','Church','Building'),Entity('2','Church','Building')]);r=get(nlp,c,'The building behind the Church.');assert r['status']=='excluded' and r['reason']=='duplicate_nonroad_landmark_name'
def test_repeated_road_all_segments_retained(nlp):
 c=TextCatalog([Entity('1','Wellington Road','TrafficRoad'),Entity('2','Wellington Road','TrafficRoad')]);r=get(nlp,c,'The white car to the left of Wellington Road.','left',category='Car');assert r['reference_ids']==['1','2'] and r['confidence_tier']=='medium'
def test_no_gt_available_to_catalog():
 from dataclasses import fields
 assert {f.name for f in fields(Entity)}=={'id','name','category'}
def test_named_word_not_direction(nlp):
 c=TextCatalog([Entity('1','West Midlands Police Custody Suite','Building')]);r=get(nlp,c,'The car behind West Midlands Police Custody Suite.','west',category='Car');assert r['reason']=='direction_in_proper_name'
def test_front_is_target_relation(nlp,cat):
 r=get(nlp,cat,'The white car parked in front of the Church.','front',category='Car');assert r['status']=='bound'
def test_stand_facing_explicit(nlp):
 c=TextCatalog([Entity('9','Pembroke College Boat Club','Building')]);r=get(nlp,c,'This is a building located on Cutter Ferry Lane. If you stand at Pembroke College Boat Club facing the stream, this building would be on your right.','right');assert r['status']=='bound' and r['scope']=='observer_relative_direct' and r['reference_ids']==['9']
def test_subregion_of_landmark_abstains(nlp,cat):
 r=get(nlp,cat,'The white car behind the Church car park.',category='Car');assert r['status']=='excluded'
def test_spacy_attachment_must_not_flatten_parking_scope(nlp,cat):
 r=get(nlp,cat,'The black and white car in the middle of the parking lot to the left of the Church.','left',category='Car');assert r['status']=='excluded' and r['scope']=='indirect'
def test_relative_clause_on_embedded_house_not_car(nlp,cat):
 r=get(nlp,cat,'The blue car parked behind the brown house on Wellington Road that is 3 houses to the right of the Church.','right',category='Car');assert r['status']=='excluded'
def test_coordinated_generic_suffix_reference_abstains(nlp,cat):
 r=get(nlp,cat,'This is a tall building with a gray roof, to the right of the Church building and the railroad.','right');assert r['status']=='excluded'
def test_between_other_cars_does_not_bind_target(nlp,cat):
 r=get(nlp,cat,'A white car parked between a white and a black car behind the Church.','behind',category='Car');assert r['status']=='excluded'
def test_parking_space_reference_not_flattened(nlp,cat):
 r=get(nlp,cat,'This is a white car parked in the parking space behind the Church.','behind',category='Car');assert r['status']=='excluded'
def test_embedded_roofed_building_reference_not_target(nlp,cat):
 r=get(nlp,cat,'The gray car between white car and gray car at the white roofed building behind the Church.','behind',category='Car');assert r['status']=='excluded'
@pytest.mark.parametrize('case_index',[0,1,2,3])
def test_original_four_pairs_preserve_text_binding(nlp,case_index):
 import json
 from pathlib import Path
 cases=json.loads((Path(__file__).resolve().parents[1]/'direction_frames/results/same_pair_opposite_descriptions.json').read_text());x=cases[case_index]
 cats=['Building','Car','Building','Car'];c=TextCatalog([Entity(x['ref_id'],x['ref_name'],'Building')])
 for text,w in [(x['instruction1'],x['word1']),(x['instruction2'],x['word2'])]:
  r=get(nlp,c,text,w,category=cats[case_index]);assert r['status']=='bound' and r['reference_ids']==[x['ref_id']],r
def test_population_only_named_landmark_relations(nlp,cat):
 from text_relations import parse_landmark_relations
 r,skip=parse_landmark_relations(nlp('The car to the left of a red car and to the right of the Church.'),'Car',cat)
 assert len(r)==1 and r[0]['word']=='right' and skip==1

def test_context_parking_relation_cannot_use_target_car(nlp,cat):
 from text_relations import parse_landmark_relations
 r,_=parse_landmark_relations(nlp('The white car in the parking lot behind the Church.'),'Car',cat)
 assert len(r)==1 and r[0]['status']=='excluded' and r[0]['subject_role']=='unresolved_context_object'

def test_named_context_subject_independent_of_navigation_target(nlp,cat):
 from text_relations import parse_landmark_relations
 r,_=parse_landmark_relations(nlp('The Church is to the left of Panton Hall.'),'Car',cat)
 assert len(r)==1 and r[0]['status']=='bound' and r[0]['subject_role']=='named_context_object' and r[0]['subject_ids']==['1'] and r[0]['reference_ids']==['2']
def test_landmark_direction_modifies_enclosure_not_internal_lawn(nlp,cat):
 from text_relations import parse_landmark_relations
 r,_=parse_landmark_relations(nlp('The lawn in the lower left quadrant of the square enclosure to the left of the Church.'),'Ground',cat)
 assert r[-1]['status']=='excluded'
def test_right_side_is_reference_subregion_when_above_it(nlp,cat):
 r=get(nlp,cat,'The small building found above the right side of the Church.','right');assert r['status']=='excluded'
def test_ambiguous_relative_after_named_context_abstains(nlp,cat):
 r=get(nlp,cat,'The building next to Panton Hall that is in front of the Church.','front');assert r['status']=='excluded'
