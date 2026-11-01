"""Explicit pre-confirmation navigation criteria; pilot improvements are insufficient."""


def assess_seen(summary,verification,criteria):
    reasons=[]
    if not verification.get('passed') or not verification.get('full_split') or verification.get('split')!='val_seen' or verification.get('episodes')!=criteria['seen_episodes'] or summary.get('episodes')!=criteria['seen_episodes']:
        reasons.append('Incomplete or unverified full Seen result')
    delta=summary['deltas']
    if delta['SR']['difference']<criteria['minimum_SR_gain_pp']:reasons.append('SR gain below preset minimum')
    if delta['SR']['group_bootstrap_95ci'][0]<=0:reasons.append('SR interval does not exclude zero')
    if delta['SPL']['difference'] < -criteria['maximum_SPL_drop_pp']:reasons.append('SPL degradation exceeds preset limit')
    if delta['NE']['difference']>criteria['maximum_NE_increase_m']:reasons.append('NE degradation exceeds preset limit')
    return dict(accepted_for_heldout=not reasons,reasons=reasons,note='Passing Seen is a selection gate, not goal completion. Independent full held-out confirmation remains required.')
