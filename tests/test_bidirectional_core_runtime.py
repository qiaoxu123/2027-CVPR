"""CPU-only public/private and runtime contracts; no pretrained network is loaded."""
import contextlib
import inspect
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from experiments.bidirectional_core import runtime
from experiments.bidirectional_core.model import CoreModel


def install_ground(monkeypatch):
    package=ModuleType('sbfnav');package.__path__=[]
    mapdata=ModuleType('sbfnav.mapdata');mapdata.GROUND_LEVEL={'m':10.}
    monkeypatch.setitem(sys.modules,'sbfnav',package)
    monkeypatch.setitem(sys.modules,'sbfnav.mapdata',mapdata)


class PrivateBomb:
    def __getitem__(self,key): raise AssertionError('private target accessed during public episode construction')


class StartOnlyTrajectory:
    def __getitem__(self,index):
        assert index==0,'future trajectory accessed before rollout'
        return [5.,7.,30.,.4]


def test_reverse_public_episode_uses_actual_start_and_anchor_only():
    record=dict(direction='reverse',map_name='m',instruction='Return north of Church.',
                source_episode_index=0,split='val_seen',trajectory_xyzyaw=StartOnlyTrajectory(),private_labels=PrivateBomb(),
                reverse_geometry_description={'public_anchor':{'name':'Church'},'label_positive_object_ids':['secret-target']})
    source=[SimpleNamespace(map_name='m',split='val_seen',target_position=PrivateBomb())]
    public=runtime.public_episode(record,source,PrivateBomb())
    np.testing.assert_array_equal(public.start_pose,[5,7,30,.4])
    assert public.reference_names==('Church',)
    assert not hasattr(public,'target_world') and not hasattr(public,'target_object_id')
    assert public.references==()


def test_forward_reference_lookup_uses_declared_raw_index_without_labels():
    item=SimpleNamespace(map_name='m',split='val_seen',instruction='Find a building.')
    source=[None,None,item]
    catalog=SimpleNamespace(referenced_landmark_names=lambda actual: ['Church'] if actual is item else pytest.fail('wrong source index'))
    record=dict(direction='forward',map_name='m',split='val_seen',instruction=item.instruction,source_episode_index=2,
                trajectory_xyzyaw=StartOnlyTrajectory(),private_labels=PrivateBomb())
    public=runtime.public_episode(record,source,catalog)
    assert public.reference_names==('Church',)


@pytest.mark.parametrize('altitude',[9.,10.])
def test_nonpositive_agl_is_unknown_and_never_reads_rgb(monkeypatch,altitude):
    install_ground(monkeypatch)
    builder=runtime.CurrentBuilder.__new__(runtime.CurrentBuilder)
    builder.renderer=SimpleNamespace(current_rgbd=lambda *args:pytest.fail('invalid AGL renderer called'))
    episode=SimpleNamespace(map_name='m')
    state=builder.build_from_poses(episode,[[0,0,altitude,0]])
    assert state.current_observation is None
    assert state.pose[2]==altitude  # no invented altitude correction


def test_valid_current_observation_hides_source_surface_raster(monkeypatch):
    install_ground(monkeypatch);calls=[]
    builder=runtime.CurrentBuilder.__new__(runtime.CurrentBuilder)
    def render(map_name,pose,ground):
        calls.append((map_name,pose.copy(),ground))
        return {'bgr':np.zeros((384,384,3),np.uint8),'surface_height_m':np.ones((384,384))}
    builder.renderer=SimpleNamespace(current_rgbd=render)
    state=builder.build_from_poses(SimpleNamespace(map_name='m'),[[0,0,20,0],[5,6,30,.7]])
    assert len(calls)==1 and 'surface_height_m' not in state.current_observation
    np.testing.assert_array_equal(calls[0][1],state.pose)
    assert state.step==1


def policy_stub(monkeypatch,pose,with_roi=False):
    install_ground(monkeypatch)
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:contextlib.nullcontext())
    policy=runtime.CorePolicy.__new__(runtime.CorePolicy)
    policy.device=torch.device('cpu');policy.record={'world_bounds':[0.,0.,100.,100.],'map_name':'m','target_xy':PrivateBomb()}
    policy.protocol={'stop_positive_weight':20.};policy.native_text=torch.zeros(1,1024);policy.landmarks=object()
    context=torch.full((1,3,8),.1234567)
    policy.frozen_field=lambda:(SimpleNamespace(belief_logits=torch.tensor([[0.,2.,1.]]),dense_context=context),{'grid_xy':torch.tensor([[[.1,.1],[.5,.5],[.9,.9]]])})
    captured=[]
    def head(public):
        captured.append(public)
        trajectory=SimpleNamespace(select=lambda:(torch.tensor([0]),torch.tensor([0]),torch.tensor([[50.,50.,30.]]),torch.tensor([math.log(20)+math.log(9)])))
        return SimpleNamespace(trajectory=trajectory,candidate_ids=torch.tensor([[1]]),belief_logits=public['static_logits'])
    policy.head=head
    if with_roi:
        policy.visual=SimpleNamespace(encode=lambda *args:[([{'center_world':[50.,50.],'quality':[1.,1.,1.],'object_id':'private-audit-id'}],torch.ones(1,1024))])
    else:policy.visual=SimpleNamespace(encode=lambda *args:pytest.fail('Unknown observation encoded'))
    state=SimpleNamespace(pose=np.asarray(pose),current_observation={'actual_observed_source':True} if with_roi else None)
    return policy,state,captured,context


def test_predict_unknown_visual_evidence_keeps_exact_public_contract_and_stop_guard(monkeypatch):
    policy,state,captured,context=policy_stub(monkeypatch,[0.,0.,30.,0.])
    prediction=policy.predict(state,'Find a building.')
    public=captured[0]
    assert set(public)==CoreModel.PUBLIC_KEYS
    assert not public['roi_valid'].any() and not public['roi_features'].any()
    torch.testing.assert_close(public['static_context'],context.half().float(),rtol=0,atol=0)
    assert not prediction.should_stop
    assert prediction.diagnostics['stop_probability_pass']
    assert not prediction.diagnostics['stop_predicted_goal_guard_pass']


def test_predict_associates_only_visible_roi_geometry_and_never_passes_identity(monkeypatch):
    policy,state,captured,_=policy_stub(monkeypatch,[50.,50.,30.,0.],with_roi=True)
    prediction=policy.predict(state,'Find a building.')
    public=captured[0]
    assert public['roi_grid_ids'].tolist()==[[1]] and public['roi_valid'].tolist()==[[True]]
    assert set(public)==CoreModel.PUBLIC_KEYS and prediction.should_stop
    assert prediction.diagnostics['predicted_goal_xy']==[50.,50.]


def test_scoring_labels_are_read_after_rollout_in_run():
    # Complement behavioral PublicBomb tests with the entrypoint's score boundary.
    source=inspect.getsource(runtime.run)
    call=source.index('poses, termination, timing, diagnostics = rollout')
    label_access=source.index("record['private_labels']['target_world']")
    assert call<label_access


def test_public_landmark_catalog_includes_unreferenced_geometry_without_hidden_selection():
    objects={'secret-target':{'name':'','object_type':'Building','contour':[[0,0],[10,0],[10,10],[0,10]]},
             'anchor':{'name':'Church','object_type':'Building','contour':[[20,0],[30,0],[30,10],[20,10]]}}
    index=runtime.public_landmarks(objects)
    candidates=index.candidates(np.array([-1,-1]),np.array([31,11]))
    assert {r['id'] for r in candidates}=={'secret-target','anchor'}


def stub_run_imports(monkeypatch):
    install_ground(monkeypatch)
    dataset=ModuleType('sbfnav.dataset')
    dataset.CityNavDataset=lambda *a,**k:pytest.fail('dataset constructed before unsafe resume was rejected')
    dataset.CityReferCatalog=lambda *a,**k:pytest.fail('catalog constructed before unsafe resume was rejected')
    navigation=ModuleType('sbfnav.navigation');navigation.BoundedWaypointController=object
    monkeypatch.setitem(sys.modules,'sbfnav.dataset',dataset)
    monkeypatch.setitem(sys.modules,'sbfnav.navigation',navigation)
    monkeypatch.setattr(runtime,'FrozenROIEncoder',lambda *a,**k:pytest.fail('GPU encoder loaded before unsafe resume rejection'))


def minimal_args(tmp_path):
    episodes=tmp_path/'episodes.jsonl';episodes.write_text('[]')
    (tmp_path/'cityrefer').mkdir(exist_ok=True)
    (tmp_path/'processed_citynav').mkdir(exist_ok=True)
    for name in ['objects.json','processed_descriptions.json']:(tmp_path/'cityrefer'/name).write_text('{}')
    (tmp_path/'processed_citynav'/'citynav_val_seen.json').write_text('[]')
    checkpoint=tmp_path/'head.pt';checkpoint.write_bytes(b'new head')
    a3=tmp_path/'a3.pt';a3.write_bytes(b'frozen a3')
    return SimpleNamespace(output=str(tmp_path/'out'),episodes=str(episodes),checkpoint=str(checkpoint),a3=str(a3),
                           split='val_seen',limit=0,sbf_source=str(tmp_path),data_root=str(tmp_path),device='cpu')


def test_resume_changed_checkpoint_is_rejected_before_loading_models(monkeypatch,tmp_path):
    import json
    stub_run_imports(monkeypatch);args=minimal_args(tmp_path);out=Path(args.output);out.mkdir()
    (out/'protocol.json').write_text(json.dumps({'checkpoint_sha256':'previous-different-checkpoint'}))
    (out/'predictions.jsonl').write_text('')
    before=(out/'protocol.json').read_bytes()
    with pytest.raises(ValueError,match='changed evaluation inputs'):runtime.run(args)
    assert (out/'protocol.json').read_bytes()==before


def test_resume_predictions_without_protocol_is_rejected_before_loading_models(monkeypatch,tmp_path):
    stub_run_imports(monkeypatch);args=minimal_args(tmp_path);out=Path(args.output);out.mkdir()
    (out/'predictions.jsonl').write_text('')
    with pytest.raises(ValueError,match='immutable evaluation protocol'):runtime.run(args)
    assert not (out/'protocol.json').exists()


@pytest.mark.parametrize('changed', ['map_name','split','instruction'])
def test_forward_wrong_source_binding_rejected_without_private_targets(changed):
    original=SimpleNamespace(map_name='m',split='val_seen',instruction='Find a building.')
    record=dict(direction='forward',map_name='m',split='val_seen',instruction='Find a building.',source_episode_index=0,
                trajectory_xyzyaw=StartOnlyTrajectory(),private_labels=PrivateBomb())
    record[changed]='wrong'
    with pytest.raises(ValueError,match='mismatch'):runtime.public_episode(record,[original],PrivateBomb())


def test_run_reads_private_score_labels_only_after_actual_rollout_boundary(monkeypatch,tmp_path):
    import json
    stub_run_imports(monkeypatch);args=minimal_args(tmp_path);completed=set();scored=[]
    records=[]
    for role in ('forward','reverse'):
        records.append(dict(episode_id='family:'+role,family_id='family',map_name='m',split='val_seen',direction=role,
            source_episode_index=0,instruction=role,trajectory_xyzyaw=[[0,0,30,0],[1,0,30,0]],static_language_eligible=True,
            reverse_geometry_description={'public_anchor':{'name':'Church'}},private_labels={'target_world':[1,0,30]}))
    Path(args.episodes).write_text(''.join(json.dumps(r)+'\n' for r in records))
    original_loads=json.loads
    class ScoreLabels(dict):
        def __init__(self,data,role):super().__init__(data);self.role=role
        def __getitem__(self,key):
            assert self.role in completed,'private score target accessed before rollout completion'
            return super().__getitem__(key)
    def guarded_loads(value,*a,**k):
        r=original_loads(value,*a,**k)
        if isinstance(r,dict) and 'private_labels' in r:r['private_labels']=ScoreLabels(r['private_labels'],r['direction'])
        return r
    monkeypatch.setattr(runtime.json,'loads',guarded_loads)
    original=SimpleNamespace(map_name='m',split='val_seen',instruction='forward')
    sys.modules['sbfnav.dataset'].CityNavDataset=lambda *a,**k:[original]
    sys.modules['sbfnav.dataset'].CityReferCatalog=lambda *a,**k:SimpleNamespace(referenced_landmark_names=lambda x:('Church',))
    sys.modules['sbfnav.navigation'].BoundedWaypointController=lambda **k:SimpleNamespace(**k)
    monkeypatch.setattr(runtime,'CurrentBuilder',lambda *a,**k:SimpleNamespace(transform=lambda name:object()))
    monkeypatch.setattr(runtime,'FrozenROIEncoder',lambda *a,**k:object())
    class FakePolicy:
        def __init__(self,*a,**k):self.protocol={'condition':'E0'}
        def set_episode(self,public,*a):assert not hasattr(public,'target_world')
        def predict(self,*a):raise AssertionError('not called by stub rollout')
    monkeypatch.setattr(runtime,'CorePolicy',FakePolicy)
    def fake_rollout(public,builder,controller,predict,ground,protocol,**kwargs):
        assert public.instruction not in completed
        assert controller.horizontal_step_m==protocol.horizontal_step_m==50
        assert controller.vertical_step_m==protocol.vertical_step_m==10
        completed.add(public.instruction)
        return np.asarray([[0,0,30,0],[1,0,30,0]]),'horizon',[],[]
    monkeypatch.setattr(runtime,'rollout',fake_rollout)
    score=runtime.score_trace
    def observed_score(*a,**k):
        assert len(completed)>len(scored)
        scored.append(True)
        return score(*a,**k)
    monkeypatch.setattr(runtime,'score_trace',observed_score)
    runtime.run(args)
    assert len(scored)==2 and completed=={'forward','reverse'}
    output=original_loads((Path(args.output)/'metrics.json').read_text())
    assert output['completed_episodes']==2 and output['forward']['SR']==1

@pytest.mark.parametrize('filename',['objects.json','processed_descriptions.json'])
def test_resume_changed_public_source_hash_rejected_before_models(monkeypatch,tmp_path,filename):
    import json
    stub_run_imports(monkeypatch);args=minimal_args(tmp_path);out=Path(args.output);out.mkdir()
    current=dict(split=args.split,episodes_sha256=runtime.file_sha256(args.episodes),
        checkpoint_sha256=runtime.file_sha256(args.checkpoint),a3_sha256=runtime.file_sha256(args.a3),limit=args.limit,
        horizon=20,horizontal_step_m=50.,vertical_step_m=10.,stop_threshold=.5,stop_guard_m=20.,
        vision_revision='1b426889ea62b5a72bf9839009a1b184bfc9c178',max_rois=32,
        runtime_sha256=runtime.file_sha256(runtime.__file__),vision_code_sha256=runtime.file_sha256(Path(runtime.__file__).with_name('vision.py')),
        data_root=str(Path(args.data_root).resolve()),device=args.device,
        public_objects_sha256=runtime.file_sha256(Path(args.data_root)/'cityrefer/objects.json'),
        public_language_bindings_sha256=runtime.file_sha256(Path(args.data_root)/'cityrefer/processed_descriptions.json'))
    (out/'protocol.json').write_text(json.dumps(current));(out/'predictions.jsonl').write_text('')
    (Path(args.data_root)/'cityrefer'/filename).write_text('{"changed":true}')
    before=(out/'protocol.json').read_bytes()
    with pytest.raises(ValueError,match='changed evaluation inputs'):runtime.run(args)
    assert (out/'protocol.json').read_bytes()==before


def test_unknown_direction_cannot_silently_become_reverse():
    record=dict(direction='typo',map_name='m',split='val_seen',instruction='instruction',source_episode_index=0,
                trajectory_xyzyaw=StartOnlyTrajectory(),private_labels=PrivateBomb())
    with pytest.raises(ValueError,match='direction'):
        runtime.public_episode(record,[SimpleNamespace(map_name='m',split='val_seen')],PrivateBomb())


def test_exact_static_cache_missing_public_query_is_rejected_without_encoding(monkeypatch):
    policy=runtime.CorePolicy.__new__(runtime.CorePolicy);policy.device=torch.device('cpu')
    policy.exact_static={'different-public-query':{}};policy.require_exact_static=True;policy.landmark_cache={}
    policy.visual=SimpleNamespace(texts=lambda *a:pytest.fail('text encoder used for missing exact cache query'))
    def set_public(self,episode,*args):self.record={'map_name':'m','instruction':episode.instruction,'world_bounds':[0,0,100,100],'target_xy':[0.,0.]}
    monkeypatch.setattr(runtime.A3WaypointPolicy,'set_episode',set_public)
    with pytest.raises(ValueError,match='public query missing'):
        policy.set_episode(SimpleNamespace(map_name='m',instruction='new language'),object(),object())


def test_exact_static_cache_exposes_only_frozen_public_fields_not_labels(monkeypatch):
    import json
    policy=runtime.CorePolicy.__new__(runtime.CorePolicy);policy.device=torch.device('cpu');policy.landmark_cache={}
    record={'map_name':'m','instruction':'Find a building.','world_bounds':[0,0,100,100],'target_xy':[0.,0.]}
    cached={'static_logits':torch.tensor([0.,1.]),'static_context':torch.ones(2,8),'grid_xy':torch.tensor([[.25,.25],[.75,.75]]),
            'native_text':torch.ones(1024),'target_world':PrivateBomb(),'target_object_id':PrivateBomb()}
    policy.exact_static={json.dumps(record,sort_keys=True):cached};policy.require_exact_static=True
    policy.visual=SimpleNamespace(texts=lambda *a:pytest.fail('cached exact text was recomputed'))
    def set_public(self,*args):self.record=record
    monkeypatch.setattr(runtime.A3WaypointPolicy,'set_episode',set_public)
    monkeypatch.setattr(runtime,'public_landmarks',lambda objects:'public-catalog')
    policy.set_episode(SimpleNamespace(map_name='m',instruction=record['instruction']),object(),SimpleNamespace(objects={'m':{}}))
    field,inputs=policy.frozen_field()
    assert set(vars(field))=={'belief_logits','dense_context'} and set(inputs)=={'grid_xy'}
    torch.testing.assert_close(field.belief_logits[0],cached['static_logits'])
    torch.testing.assert_close(field.dense_context[0],cached['static_context'])
    torch.testing.assert_close(inputs['grid_xy'][0],cached['grid_xy'])
    torch.testing.assert_close(policy.native_text[0],cached['native_text'])
