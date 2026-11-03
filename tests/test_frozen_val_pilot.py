import pytest
from scripts.evaluate_chain_val_pilot import validate_val_pilot


def manifest():
    return dict(split='val_seen',purpose='fixed_val_seen_128',no_parameter_selection=True,indices=list(range(128)))


def test_only_exact_frozen_val_seen128_is_allowed():
    validate_val_pilot(manifest(),2470)
    for split in ['train_seen','val_unseen','test_unseen']:
        f=manifest();f['split']=split
        with pytest.raises(ValueError):validate_val_pilot(f,2470)
    for indices in [list(range(127)),[0]*128,list(range(127))+[2470]]:
        f=manifest();f['indices']=indices
        with pytest.raises(ValueError):validate_val_pilot(f,2470)
    f=manifest();f['no_parameter_selection']=False
    with pytest.raises(ValueError):validate_val_pilot(f,2470)


def test_full_gate_is_strictly_over65percent_and_improves_original():
    from scripts.report_chain_val_pilot import full_evaluation_gate
    assert not full_evaluation_gate(0,83)['passed']
    assert full_evaluation_gate(83,84)['passed']
    assert not full_evaluation_gate(84,84)['passed']
    assert not full_evaluation_gate(91,90)['passed']


def test_grouped_delta_interval_preserves_constant_paired_difference():
    from scripts.report_chain_val_pilot import grouped_interval
    assert grouped_interval([0,0,0],['a','a','b'])==[0.,0.]
    assert grouped_interval([-1,-1,-1],['a','a','b'])==[-1.,-1.]
