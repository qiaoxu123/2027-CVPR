import pytest
from experiments.a2_waldo_landmark_belief.artifact_protocol import validate_training_history


def history():
    return [dict(epoch=i,stage='ranking' if i<=20 else 'refinement',stage_epoch=i if i<=20 else i-20,
                 train=dict(views=100,optimizer_updates=13),dev=dict(views=20)) for i in range(1,26)]


def test_all25_epochs_and_full_train_dev_views_are_required():
    rows=history();validate_training_history(rows,100,20)
    with pytest.raises(ValueError,match='25'):validate_training_history(rows[:-1],100,20)
    rows=history();rows[13]['train']['views']=99
    with pytest.raises(ValueError,match='views'):validate_training_history(rows,100,20)
    rows=history();rows[20]['stage']='ranking'
    with pytest.raises(ValueError,match='stage'):validate_training_history(rows,100,20)
