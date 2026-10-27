from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold,group_summary,verify_gate


def row(group,before,after,changed=True):
    return dict(group=['m',group],a0_correct=before,a2_correct=after,a0_ne=30. if not before else 10.,
                a2_ne=30. if not after else 10.,changed=changed)


def test_repeated_observations_do_not_inflate_independent_calibration_groups():
    rows=[row(0,False,True),row(0,True,False),row(1,False,True)]
    stats=group_summary(rows)
    assert stats['groups']==2 and stats['mean_r1_difference']==.5
    assert calibration_fold('m',4)==calibration_fold('m',4)


def test_no_support_or_too_few_independent_groups_cannot_enable_gate():
    rows=[row(0,False,True)]*100
    assert not verify_gate(group_summary(rows))['passed']


def test_safe_gain_with_sufficient_independent_evidence_can_enable_gate():
    rows=[]
    for i in range(100):
        rows.extend([row(i,True,True,False),row(i,False,i<40,i<40)])
    stats=group_summary(rows)
    assert verify_gate(stats)['passed']
    assert stats['broken_groups']==0 and stats['gain_ci95'][0]>0


def test_temperature_labels_exclude_instances_in_other_calibration_fold():
    import torch
    from experiments.a2_waldo_landmark_belief.calibrate import calibration_known_mask
    fit=next(i for i in range(100) if calibration_fold('m',i)=='fit')
    verify=next(i for i in range(100) if calibration_fold('m',i)=='verify')
    packet=dict(target_instance=['m',fit],labels=dict(rank_label_mask=torch.tensor([True,True])),
                private=dict(association=[dict(object_id=fit),dict(object_id=verify)]))
    assert calibration_known_mask(packet,torch.tensor([True,True])).tolist()==[True,False]
