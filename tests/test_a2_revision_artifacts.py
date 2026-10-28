import pytest
from experiments.a2_waldo_landmark_belief.revision_artifacts import verify_revision_history


def history():
    metrics=dict(views=2,optimizer_updates=1,loss=1.,map_correct=1.,a0_correct=1.,corrected=0.,broken=0.,proposal_correct=1.,proposal_available_views=2.,map_r1=.5)
    return [dict(epoch=e,stage='ranking' if e<=20 else 'refinement',stage_epoch=e if e<=20 else e-20,train=dict(metrics),dev=dict(metrics,optimizer_updates=0)) for e in range(1,26)]


def test_revision_history_requires_all25_full_population_and_consistent_corrections():
    h=history();verify_revision_history(h,2,2)
    with pytest.raises(ValueError):verify_revision_history(h[:-1],2,2)
    h=history();h[23]['dev']['views']=1
    with pytest.raises(ValueError):verify_revision_history(h,2,2)
    h=history();h[24]['dev']['corrected']=1
    with pytest.raises(ValueError):verify_revision_history(h,2,2)


def test_completed_epoch_cannot_hide_missing_optimizer_updates():
    h=history();h[0]['train']['optimizer_updates']=0
    with pytest.raises(ValueError):verify_revision_history(h,2,2)
    h=history();h[0]['dev']['optimizer_updates']=1
    with pytest.raises(ValueError):verify_revision_history(h,2,2)
