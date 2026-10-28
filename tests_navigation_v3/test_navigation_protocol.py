import copy
import pytest
from experiments.a2_waldo_landmark_belief.relation_v3_navigation_protocol import build_pilot_population,require_candidate
from experiments.a2_waldo_landmark_belief.calibration_protocol import verify_gate


def population():
    rows=[dict(index=i,map_name='m',object_id=f'o{i}',description_id=0,episode_id=[i],split='dev') for i in range(293)]
    roles=dict(target_roles=dict(train=[dict(map_name='m',object_id='train')],dev=[dict(map_name='m',object_id='dev')],
        calibration_fit=[dict(map_name='m',object_id=e['object_id']) for e in rows[:84]],
        calibration_verify=[dict(map_name='m',object_id=e['object_id']) for e in rows[84:]]))
    confirmation=dict(fresh_entries=[dict(e,fold='fit' if e['index']<84 else 'verify') for e in rows[:163]],
                      sealed_entries=[dict(e,fold='verify') for e in rows[163:]])
    return roles,confirmation,dict(entries=rows)


def candidate():
    selected=dict(minimum_gain=.8,maximum_harm=.1,temperature=1.)
    stats=dict(groups=209,supported_groups=40,a0_correct_groups=100,gain_groups=10,gain_ci95=[.01,.1],
               broken_group_wilson_upper95=.02,mean_ne_difference_m=-.2)
    result=dict(status='v3_candidate_requires_real_closed_loop_pilot',enabled=True,gate_final=False,fixed_action=.25,temperature=1.,
        temperature_fitting=False,benchmark_used=False,no_verify_reselection=True,fit_groups=84,verify_groups=209,
        fit_selected_parameters=selected,verification=stats,verification_performance_evaluated=True,verify_gate=verify_gate(stats),
        minimum_gain=.8,maximum_harm=.1)
    return result,dict(selected_parameters=selected)


def test_pilot_is_fixed_unique_subset_of_verify_and_order_independent():
    roles,confirmation,source=population()
    pilot=build_pilot_population(roles,confirmation,source)
    assert len(pilot)==64 and len({e['index'] for e in pilot})==64
    assert all(e['index']>=84 and e['fold']=='verify' for e in pilot)
    confirmation['fresh_entries'].reverse();confirmation['sealed_entries'].reverse();source['entries'].reverse()
    assert build_pilot_population(roles,confirmation,source)==pilot


@pytest.mark.parametrize('kind',['duplicate','overlap','missing','swapped_query','wrong_fold','extra_target'])
def test_pilot_rejects_changed_inventory_before_inference(kind):
    roles,confirmation,source=population()
    if kind=='duplicate':confirmation['sealed_entries'][1]=copy.deepcopy(confirmation['sealed_entries'][0])
    elif kind=='overlap':roles['target_roles']['train'].append(copy.deepcopy(roles['target_roles']['calibration_verify'][0]))
    elif kind=='missing':confirmation['sealed_entries'].pop()
    elif kind=='swapped_query':confirmation['sealed_entries'][0]['description_id']=99
    elif kind=='wrong_fold':confirmation['sealed_entries'][0]['fold']='fit'
    else:roles['target_roles']['calibration_verify'].append(dict(map_name='m',object_id='extra'))
    with pytest.raises(ValueError):build_pilot_population(roles,confirmation,source)


def test_only_original_verified_nonzero_candidate_can_enter_pilot():
    result,decision=candidate();require_candidate(result,decision)
    for field,value in [('enabled',False),('gate_final',True),('fixed_action',.5),('temperature',2.),
                        ('benchmark_used',True),('minimum_gain',.7),('verify_groups',208)]:
        bad=copy.deepcopy(result);bad[field]=value
        with pytest.raises(ValueError):require_candidate(bad,decision)
    bad=copy.deepcopy(result);bad['verification']['gain_ci95'][0]=0.
    with pytest.raises(ValueError):require_candidate(bad,decision)
