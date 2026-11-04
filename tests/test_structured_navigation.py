import inspect
import pytest
from graphnav.structured_navigation import parse_instruction,relation_frame


def test_ambiguous_view_directions_never_become_world_axes():
    for word in ('behind','front of','left of','right of'):
        assert relation_frame(word)['frame']=='reference_frame_unknown'
        assert relation_frame(word)['requires_review']
    assert relation_frame('north of')['frame']=='world_cardinal'
    assert relation_frame('inside')['frame']=='topological'


def test_parser_cannot_take_private_target_information():
    assert set(inspect.signature(parse_instruction).parameters)=={'text','map_name','nlp','public_names'}


def test_spans_and_entities_keep_original_text():
    spacy=pytest.importorskip('spacy')
    from scripts.analyze_instruction_landmark_association import PublicNames
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    names=PublicNames({'toy':{'1':dict(name='Alpha Hall',object_type='Building')}})
    text='The red car in the parking lot in front of Alpha Hall.'
    result=parse_instruction(text,'toy',nlp,names)
    assert result['target']['category']=='Car'
    assert result['target']['name'] is None
    assert result['landmarks'][0]['text']=='Alpha Hall'
    for entity in result['entities']:
        assert text[entity['span'][0]:entity['span'][1]]==entity['text']
    assert any(x['value']=='red' for x in result['target']['attributes'])
    assert result['indirect_relations']
    assert not result['direct_relations']
    assert any(e['reference_frame']['frame']=='reference_frame_unknown' for e in result['relations'])


def test_generic_road_suffix_is_not_a_fake_intermediate_object():
    spacy=pytest.importorskip('spacy');from scripts.analyze_instruction_landmark_association import PublicNames
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    names=PublicNames({'toy':{'1':dict(name='Maids Causeway',object_type='TrafficRoad')}})
    r=parse_instruction('The black car behind the grey building on Maids Causeway Road.','toy',nlp,names)
    assert r['indirect_relations'][0]['hops']==2
    assert not any(e['text']=='Road' for e in r['entities'])


def test_target_attributes_exclude_determiners_and_other_object_wall():
    spacy=pytest.importorskip('spacy');from scripts.analyze_instruction_landmark_association import PublicNames
    nlp=spacy.load('en_core_web_sm',disable=['ner']);names=PublicNames({'toy':{'1':dict(name='Alpha Hall',object_type='Building')}})
    r=parse_instruction('The black car parked at the back wall of Alpha Hall.','toy',nlp,names)
    assert {x['value'].lower() for x in r['target']['attributes']}=={'black'}


def test_side_of_retains_direction_and_next_to_is_not_just_to():
    spacy=pytest.importorskip('spacy');from scripts.analyze_instruction_landmark_association import PublicNames
    nlp=spacy.load('en_core_web_sm',disable=['ner']);names=PublicNames({'toy':{'1':dict(name='Alpha Hall',object_type='Building')}})
    r=parse_instruction('The big building at the right side of Alpha Hall.','toy',nlp,names)
    edge=r['relations'][0]
    assert 'right' in edge['predicate'] and edge['reference_frame']['requires_review']
    r=parse_instruction('The car next to Alpha Hall.','toy',nlp,names)
    assert r['relations'][0]['predicate']=='next to'


def test_unbound_proper_name_is_retained_without_inventing_a_map_binding():
    spacy=pytest.importorskip('spacy');from scripts.analyze_instruction_landmark_association import PublicNames
    nlp=spacy.load('en_core_web_sm',disable=['ner']);names=PublicNames({'toy':{}})
    r=parse_instruction('The white building next to Unknown Museum.','toy',nlp,names)
    assert not r['landmarks']
    assert any('Unknown Museum' in c['text'] for c in r['unbound_landmark_candidates'])
