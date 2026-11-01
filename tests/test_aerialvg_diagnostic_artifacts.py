import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]/'reports/aerialvg_diagnostics_20261101'


def test_complete_diagnostics_preserve_denominators_and_belief_accounting():
    summary=json.loads((ROOT/'candidate_summary.json').read_text())
    assert summary['checkpoint_unchanged']
    assert summary['selection_reproduction']==dict(states=1910,deployed_selection_matches_prepared=1910)
    for partition,n in [('train',1275),('dev',635)]:
        data=summary['splits'][partition]
        for cohort in data.values():
            count=cohort['states'];a0=cohort['a0_correct']
            for values in cohort['fusion'].values():
                assert values['updated_correct']==a0+values['corrected']-values['damaged']
                assert values['covered_rank_wrong']+values['rank_correct']==values['covered']
            for losses in cohort['first_loss'].values():assert sum(losses.values())==count
            for strategy in ['native','a0_guided','native_spatial','a0_guided_spatial']:
                counts=[cohort['stages'][k]['covered'] for k in ['raw900',strategy+'/valid900',strategy+'/nms',strategy+'/top60',strategy+'/top30',strategy+'/top15']]
                assert counts==sorted(counts,reverse=True)
        assert data['all']['states']==n


def test_navigation_is_only_existing128_and_does_not_claim_accumulation_causality():
    nav=json.loads((ROOT/'navigation_summary.json').read_text())
    for model in ['baseline','current','history8']:
        assert len(nav['rows'][model])==128
        for r in nav['rows'][model]:
            low,high=r['terminal_goal_target_distance_bounds'];assert low<=high+1e-4
            if r['cause']=='定位偏差证据':assert low>20
    assert nav['history8_vs_current']['helped']==10 and nav['history8_vs_current']['harmed']==7
    assert 'prevents identifying' in nav['history8_vs_current']['mechanism']
