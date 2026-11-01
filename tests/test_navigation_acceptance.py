from graphnav.navigation_acceptance import assess_seen


def test_positive_pilot_or_zero_crossing_interval_cannot_pass_full_seen_gate():
    criteria=dict(seen_episodes=2470,minimum_SR_gain_pp=2.,maximum_SPL_drop_pp=1.,maximum_NE_increase_m=1.)
    summary=dict(episodes=128,deltas=dict(SR=dict(difference=2.34,group_bootstrap_95ci=[-4.69,9.38]),SPL=dict(difference=.25),NE=dict(difference=.61)))
    verification=dict(passed=True,full_split=False,split='val_seen',episodes=128)
    assert not assess_seen(summary,verification,criteria)['accepted_for_heldout']
    summary['episodes']=2470;verification.update(full_split=True,episodes=2470)
    assert not assess_seen(summary,verification,criteria)['accepted_for_heldout']
    summary['deltas']['SR']['group_bootstrap_95ci']=[.5,4.2]
    assert assess_seen(summary,verification,criteria)['accepted_for_heldout']
    summary['deltas']['SPL']['difference']=-2
    assert not assess_seen(summary,verification,criteria)['accepted_for_heldout']
