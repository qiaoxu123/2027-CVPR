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
