import pytest
from scripts.diagnose_text_gap_restorations import build_variants
from scripts.analyze_instruction_landmark_association import PublicNames

@pytest.fixture
def parser():
    spacy=pytest.importorskip('spacy')
    return spacy.load('en_core_web_sm')

def test_exact_source_words_and_disjoint_blocks(parser):
    names=PublicNames({'test_map':{'1':{'name':'Landmark Hall','object_type':'Building'}}})
    text='The second white building in the parking lot north of Landmark Hall.'
    variants,tags=build_variants(text,'test_map',parser,names,'building; Building; north of Landmark Hall')
    a={tuple(x['span']) for x in tags['attributes']};q={tuple(x['span']) for x in tags['quantities']};c={tuple(x['span']) for x in tags['chain']}
    assert not a&q and not a&c and not q&c
    assert any(x['text']=='white' for x in tags['attributes'])
    assert any(x['text']=='second' for x in tags['quantities'])
    for entries in [tags['attributes'],tags['quantities'],tags['chain']]:
        assert all(text[x['span'][0]:x['span'][1]]==x['text'] for x in entries)
    assert 'white' not in variants['full_drop_attributes']
    assert 'second' not in variants['full_drop_quantities']
    assert 'white' in variants['restore_attributes']
    assert 'second' in variants['restore_quantities']

def test_no_landmark_no_forced_chain(parser):
    names=PublicNames({'test_map':{}})
    variants,tags=build_variants('Find a blue car.','test_map',parser,names,'car; Car')
    assert tags['chain']==[]
    assert variants['restore_chain']=='car; Car'

def test_empty_deletion_keeps_exact_source(parser):
    names=PublicNames({'test_map':{}})
    text='Find a blue car.'
    variants,tags=build_variants(text,'test_map',parser,names,'car; Car')
    assert tags['quantities']==[]
    assert variants['full_drop_quantities']==text
    assert variants['full_drop_chain']==text
