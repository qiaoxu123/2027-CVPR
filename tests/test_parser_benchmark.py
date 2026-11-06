import pytest
from graphnav.parser_benchmark import validate_graph,relation_paths,score_reviewed,public_record

def sample():
 return dict(text='The red car in the lot next to The Library.',landmark_catalog=['The Library'])
def graph():
 return dict(entities=[dict(id='t',text='The red car',start=0,end=11,role='target',type='car'),dict(id='p',text='the lot',start=15,end=22,role='intermediate',type='parking'),dict(id='l',text='The Library',start=31,end=42,role='landmark',type='building')],relations=[dict(subject='t',relation='inside',object='p'),dict(subject='p',relation='next to',object='l')])
def test_exact_spans_and_direction_preserved():
 g=validate_graph(sample()['text'],graph());assert not g['validation_errors'];path=relation_paths(g)[0];assert path['nodes']==['t','p','l'];assert path['hops']==2;assert path['steps'][1]['orientation']==1
 g['relations'][1]={'subject':'l','relation':'beside','object':'p'};assert relation_paths(g)[0]['steps'][1]['orientation']==-1

def test_invalid_spans_or_dangling_relations_are_not_silently_gold():
 g=graph();g['entities'][0]['end']=7;g['relations'].append(dict(subject='missing',relation='near',object='l'));v=validate_graph(sample()['text'],g);assert len(v['validation_errors'])==3

def test_public_record_drops_all_outcomes_and_GT():
 x=dict(index=1,instruction='hello',landmark_catalog=['Library'],target_xy=[1,2],target_id=99,group='large_error',map_name='secret',prediction=1)
 assert public_record(x)==dict(index=1,text='hello',landmark_catalog=['Library'])

def test_metrics_require_independent_review_and_directed_triples():
 gold=dict(index=1,text=sample()['text'],graph=graph(),reviewed=False,reviewer='human',source='human_editor');pred=dict(index=1,graph=graph());assert score_reviewed([pred],[gold])['reviewed_count']==0
 gold['reviewed']=True;gold['independent']=True;r=score_reviewed([pred],[gold]);assert r['entities']['F1']==1;assert r['triples']['F1']==1
 pred['graph']['relations'][0]['subject']='p';r=score_reviewed([pred],[gold]);assert r['triples']['F1']<1

def test_model_output_cannot_be_promoted_to_gold_and_empty_chain_is_not_success():
 review=dict(index=1,text=sample()['text'],graph=graph(),reviewed=True,reviewer='model',independent=True,source='model_output')
 assert score_reviewed([dict(index=1,graph=graph())],[review])['reviewed_count']==0
 review.update(source='human_editor',reviewer='independent_human',graph=dict(entities=[],relations=[]))
 result=score_reviewed([dict(index=1,graph=dict(entities=[],relations=[]))],[review]);assert result['complete_chain_accuracy'] is None;assert result['gold_chain_tasks']==0

def test_correct_direction_control_requires_all_anchors_correct():
 from scripts.prepare_parser_benchmark import direction_pools
 rows=[dict(index=1,anchor=0,top1=1,error=5,target_index_offline=1),dict(index=1,anchor=1,top1=0,error=150,target_index_offline=1),dict(index=2,anchor=0,top1=1,error=10,target_index_offline=1)]
 pools=direction_pools(rows);assert {x['index'] for x in pools['correct_direction']}=={2};assert {x['index'] for x in pools['large_error_context_present']}=={1}

def test_gliner_relation_endpoints_use_model_offsets_for_repeated_words():
 from scripts.run_parser_benchmark import gliner_graph
 text='car near the lot; car near Library.';second=text.index('car',1);library=text.index('Library')
 raw={'entities':{'navigation target':[dict(text='car',start=0,end=3)],'intermediate reference object':[dict(text='car',start=second,end=second+3)],'named landmark':[dict(text='Library',start=library,end=library+7)]},'relation_extraction':{'near':[dict(head=dict(text='car',start=second,end=second+3),tail=dict(text='Library',start=library,end=library+7))]}}
 graph=gliner_graph(raw,text,['Library']);entities={e['id']:e for e in graph['entities']};assert entities[graph['relations'][0]['subject']]['start']==second

def test_case_only_alignment_is_unique_and_does_not_choose_ambiguous_mentions():
 from scripts.run_parser_benchmark import aligned_entity
 entity,note=aligned_entity('The white car near Library.',dict(id='t',text='the white car',role='target'),0)
 assert entity and entity['text']=='The white car' and entity['start']==0;assert 'case' in note
 entity,note=aligned_entity('The car beside the car.',dict(id='t',text='THE CAR',role='target'),0);assert entity is None
