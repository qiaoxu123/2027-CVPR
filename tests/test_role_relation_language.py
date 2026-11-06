import pytest
from scripts.analyze_instruction_landmark_association import PublicNames


def test_gt_free_parser_direct_and_indirect():
    spacy=pytest.importorskip('spacy')
    from graphnav.role_relation_language import parse_instruction
    nlp=spacy.load('en_core_web_sm',disable=['ner'])
    matcher=PublicNames({'m':{'1':{'name':'One Stop','object_type':'Building'}}})
    direct=parse_instruction(nlp,'The white building north of One Stop.','m',matcher)
    indirect=parse_instruction(nlp,'The blue car in the parking lot north of One Stop.','m',matcher)
    assert direct['category']=='direct_geometry'
    assert indirect['category']=='indirect_geometry'
    assert any(p['hops']>=2 for p in indirect['paths'])
    assert direct['target_span']['text'].lower().startswith('the white building')
    assert any(p['public_ids']==['1'] for p in indirect['paths'])
    assert direct['GT_used'] is False


def test_paths_keep_relation_orientation_and_unknown_frame():
    spacy=pytest.importorskip('spacy')
    from graphnav.role_relation_language import parse_instruction
    nlp=spacy.load('en_core_web_sm',disable=['ner']);matcher=PublicNames({'m':{'1':{'name':'One Stop','object_type':'Building'}}})
    q=parse_instruction(nlp,'The car behind One Stop.','m',matcher)
    assert q['paths'] and q['paths'][0]['steps'][0]['orientation'] in [-1,1]
    assert q['paths'][0]['steps'][0]['reference_frame']=='unresolved_relative'
    assert all('GT' not in e for e in q['entities'])


def test_initial_target_and_compound_relation_are_retained():
    spacy=pytest.importorskip('spacy')
    from graphnav.role_relation_language import parse_instruction
    nlp=spacy.load('en_core_web_sm',disable=['ner']);matcher=PublicNames({'m':{'1':{'name':'One Stop','object_type':'Building'}}})
    q=parse_instruction(nlp,'This is a white car in the parking lot across from One Stop. There is another white car behind it.','m',matcher)
    assert q['target_span'] and 'white car' in q['target_span']['text']
    assert any('across from'==s['relation'] for p in q['paths'] for s in p['steps'])
