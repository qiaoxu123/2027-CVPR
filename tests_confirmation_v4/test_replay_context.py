"""Public reference lookup is query-bound and never uses destination fields."""
import copy
import importlib
import pytest
from experiments.a2_waldo_landmark_belief.paths import save,sha
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold


def module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_replay_context')


class Tokenizer:
    def __call__(self,text,**kwargs):
        assert text=='Main Road'
        return dict(offset_mapping=[[0,4],[5,9]])


def fixture(tmp_path,monkeypatch):
    m=module();base=tmp_path/'runs';data=tmp_path/'data'
    target=next(i for i in range(100) if calibration_fold('m_block_0',i)=='verify')
    annotation=tmp_path/'train.json'
    save([dict(area='m',block=0,object_ids=[target],ann_ids=[0],descriptions=['Main Road'],
               target_position=[999.,-999.,0.])],annotation)
    query=dict(index=0,map_name='m_block_0',object_id=target,description_id=0,
               reference_queries=1,split='calibration',fold='verify')
    entry={k:v for k,v in query.items() if k!='fold'}
    save(dict(annotation_path=str(annotation),annotation_sha256=sha(annotation),entries=[entry]),base/'source_manifest.json')
    contour=[[0,0],[1,0],[1,1],[0,1]]
    objects={str(i):dict(id=i,name='Main Road',object_type=kind,
        contour=[[x+i-1,y] for x,y in contour]) for i,kind in [(1,'TrafficRoad'),(2,'Footpath')]}
    save({'m_block_0':objects},data/'cityrefer/objects.json')
    save({'m_block_0':{str(target):[dict(landmarks=['Main Road'])]}},data/'cityrefer/processed_descriptions.json')
    monkeypatch.setattr(m,'BASE',base);monkeypatch.setattr(m,'DATA',data)
    return m,query,annotation


def test_fixed_public_query_resolves_complete_road_without_destination_use(tmp_path,monkeypatch):
    m,query,_=fixture(tmp_path,monkeypatch);context=m.ConfirmationReferenceContext(Tokenizer(),[query])
    text,offsets,refs=context.resolve(query)
    assert text=='Main Road' and offsets==[[0,4],[5,9]]
    assert len(refs)==1 and 'region' in refs[0]
    assert refs[0]['name']=='Main Road' and refs[0]['linked']
    for key,value in [('object_id',999),('description_id',9),('fold','fit'),('index',8)]:
        changed=copy.deepcopy(query);changed[key]=value
        with pytest.raises(ValueError):context.resolve(changed)


def test_context_rejects_duplicate_or_non_verify_queries_before_public_lookup(tmp_path,monkeypatch):
    m,query,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):m.ConfirmationReferenceContext(Tokenizer(),[query,query])
    bad=dict(query,fold='fit')
    with pytest.raises(ValueError):m.ConfirmationReferenceContext(Tokenizer(),[bad])


def test_changed_annotation_is_rejected_before_resolving_cached_reference(tmp_path,monkeypatch):
    m,query,annotation=fixture(tmp_path,monkeypatch);annotation.write_text('changed')
    with pytest.raises(ValueError):m.ConfirmationReferenceContext(Tokenizer(),[query])
