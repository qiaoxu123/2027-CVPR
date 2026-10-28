import pytest
from experiments.a2_waldo_landmark_belief.revision_preflight import check_preflight_records


def rows(shift=0.):
    return [dict(index=1,step=5,xyz=[1.+shift,2.,3.],should_stop=False,belief_top16_world_xy=[[1.+shift,2.]],a2_updated=bool(shift))]


def test_real_preflight_requires_off_parity_and_nonzero_controller_coupling():
    result=check_preflight_records(rows(),rows(),rows(1.))
    assert result['off_parity_passed'] and result['controller_coupling_passed']
    with pytest.raises(ValueError):check_preflight_records(rows(),rows(1.),rows(1.))
    with pytest.raises(ValueError):check_preflight_records(rows(),rows(),rows())
