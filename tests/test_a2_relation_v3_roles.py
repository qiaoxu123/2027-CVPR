import pytest
from scripts.a2_prepare_relation_v3_roles import verify_target_roles


def test_all_supervision_roles_must_be_target_disjoint():
    roles=dict(train={('m',1)},dev={('m',2)},calibration_fit={('m',3)},calibration_verify={('m',4)})
    verify_target_roles(roles)
    roles['train'].add(('m',4))
    with pytest.raises(ValueError):verify_target_roles(roles)
