import spacy
from scripts.analyze_instruction_landmark_association import PublicNames,classify,normalise

NLP=spacy.load('en_core_web_sm',disable=['ner'])
NAMES=PublicNames({'m':{'0':{'name':'Foo Building','object_type':'Building'},'1':{'name':'Bar Road','object_type':'TrafficRoad'}}})
def classify_text(text):return classify(NLP(text),NAMES.match('m',text),'Car')
def test_direct():
 r=classify_text('The red car is next to Foo Building.');assert r['category']=='direct_geometry';assert r['paths'][0]['hops']==1

def test_nested_region_indirect():
 r=classify_text('The red car is in the parking lot in front of Foo Building.');assert r['category']=='indirect_geometry';assert r['paths'][0]['hops']==2

def test_explicit_region_coreference():
 r=classify_text('The red car is in a parking lot. That parking lot is in front of Foo Building.');assert r['category']=='indirect_geometry'

def test_ordinary_car_not_public_landmark():
 r=classify_text('The white car beside another black car.');assert r['category']=='no_confirmed_anchor'

def test_comma_preserves_direct_target_scope():
 r=classify_text('The car is in a parking lot, next to Foo Building.');assert r['category']=='direct_geometry'

def test_apostrophe_and_ampersand_normalization():
 assert normalise("St Clement’s & Hall")[0]==normalise("St Clement's and Hall")[0]

def test_mentioned_but_disconnected_not_assumed_relation():
 r=classify_text('The car is red. Foo Building has a black roof.');assert r['category']=='uncertain'

def test_car_cannot_be_named_road_identity():
 text='A gray car and another car, on Bar Road.'
 r=classify_text(text);assert r['category']!='direct_identity'

def test_named_target_identity_requires_true_instance_binding():
 text='Foo Building.'
 r=classify(NLP(text),NAMES.match('m',text),'Building',identity_ids=['0']);assert r['category']=='direct_identity'
 r=classify(NLP(text),NAMES.match('m',text),'Building',identity_ids=['1']);assert r['category']!='direct_identity'
