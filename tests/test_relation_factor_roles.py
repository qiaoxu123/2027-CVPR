import spacy

from scripts.parse_local_chain_roles import role
from scripts.analyze_instruction_landmark_association import PublicNames
from graphnav.relation_factor_roles import build_relation_roles

NLP = spacy.load('en_core_web_sm')
NAMES = PublicNames({'m': {'1': {'name': 'Foo Building', 'object_type': 'Building'}}})


def parse(text):
    doc = NLP(text)
    mentions = NAMES.match('m', text)
    parsed = role(doc, mentions)
    return build_relation_roles(doc, parsed, mentions)


def test_two_ordinary_references_remain_separate():
    result = parse('The white car beside a grey car and a black car.')
    assert len(result['references']) == 2
    assert {r['phrase'] for r in result['references']} == {'a grey car', 'a black car'}
    assert all(r['kind'] == 'near' for r in result['references'])


def test_nested_map_scope_attaches_to_parking_not_target():
    result = parse('The red car is in the parking lot in front of Foo Building.')
    refs = result['references']
    assert len(refs) == 1 and refs[0]['kind'] == 'inside'
    assert refs[0]['phrase'] == 'the parking lot'
    assert refs[0]['anchors'][0]['kind'] == 'front'
    assert refs[0]['anchors'][0]['public_ids'] == ['1']
    assert result['global_scope_uncertain'] is False


def test_direction_stays_categorical_and_unknown_frame():
    result = parse('The white car to the right of the grey car.')
    assert result['references'][0]['kind'] == 'right'
    assert result['references'][0]['frame'] == 'unspecified'


def test_between_abstains_without_fake_pairwise_labels():
    result = parse('The white car between a grey car and a black car.')
    assert not result['references']
    assert any(r['raw_relation'] == 'between' for r in result['unsupported_edges'])


def test_names_stay_public_and_no_private_fields():
    result = parse('The red car near Foo Building.')
    assert result['references'][0]['public_ids'] == ['1']
    assert result['references'][0]['anchors'][0]['kind'] == 'identity'
    assert not {'target_world', 'target_id', 'contour'} & result.keys()


def test_uncertain_scope_is_not_silently_enabled():
    doc = NLP('The white car beside the grey car.')
    parsed = role(doc, [])
    parsed['parse']['scope_uncertain'] = True
    result = build_relation_roles(doc, parsed, [])
    assert not result['references']
    assert result['global_scope_uncertain']
