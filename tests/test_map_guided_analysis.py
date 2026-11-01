from scripts.analyze_map_guided_diagnostic import group_interval


def test_bootstrap_preserves_groups_and_reports_fixed_prediction_difference():
    rows=[]
    for group,left,right in [('a',5,30),('a',5,30),('b',30,5)]:
        rows.append(dict(group=group,methods=dict(map_rgb=dict(error=left),map_only=dict(error=right))))
    result=group_interval(rows,'map_rgb','map_only')
    assert result['groups']==2 and result['states']==3
    assert abs(result['delta']-1/3)<1e-9
    assert result['CI95'][0]<=result['delta']<=result['CI95'][1]
