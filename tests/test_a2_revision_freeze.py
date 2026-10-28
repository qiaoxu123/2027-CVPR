import pytest
from experiments.a2_waldo_landmark_belief.revision_freeze import require_candidate


def candidate():
    return dict(status='revision_candidate_requires_real_closed_loop_pilot',enabled=True,scale=1.,threshold=.5,temperature=1.,no_verify_reselection=True,benchmark_used=False,
      verify_gate=dict(passed=True),verification=dict(supported_groups=40,a0_correct_groups=100,gain_groups=8,gain_ci95=[.01,.1],broken_group_wilson_upper95=.03,mean_ne_difference_m=-1.))


def test_final_evaluation_cannot_accept_off_or_failed_calibration_candidate():
    require_candidate(candidate())
    c=candidate();c['enabled']=False
    with pytest.raises(ValueError):require_candidate(c)
    c=candidate();c['verification']['gain_ci95']=[-.1,.1]
    with pytest.raises(ValueError):require_candidate(c)
    c=candidate();c['benchmark_used']=True
    with pytest.raises(ValueError):require_candidate(c)


def test_freeze_rejects_changed_calibration_sources_instead_of_rebinding_them(tmp_path):
    from experiments.a2_waldo_landmark_belief.revision_freeze import check_bound_digests
    from experiments.a2_waldo_landmark_belief.paths import sha
    p=tmp_path/'packets.json';p.write_text('{"v":1}')
    c=dict(packet_manifest_sha256=sha(p));check_bound_digests(c,{'packet_manifest_sha256':p})
    p.write_text('{"v":2}')
    with pytest.raises(ValueError):check_bound_digests(c,{'packet_manifest_sha256':p})


def test_final_freeze_rejects_changed_external_runtime_even_with_same_revision_code(tmp_path,monkeypatch):
    import json
    from experiments.a2_waldo_landmark_belief import revision_freeze as f
    from experiments.a2_waldo_landmark_belief.paths import sha
    base=tmp_path/'run';root=base/'revision_v1';(root/'training').mkdir(parents=True);(root/'calibration').mkdir()
    original=tmp_path/'a0.pt';original.write_bytes(b'a0');checkpoint=root/'training/best.pt';checkpoint.write_bytes(b'a2')
    c=candidate();c['checkpoint_sha256']=sha(checkpoint)
    (root/'calibration/candidate_calibration.json').write_text(json.dumps(c))
    (root/'final_calibration.json').write_text(json.dumps(dict(c,gate_final=True)))
    frozen=dict(status='revision_frozen_after_independent_calibration_and_real_pilot',enabled=True,a0_sha256=sha(original),checkpoint_sha256=sha(checkpoint),
                code_sha256={},external_dependencies={'vendor':'old'},bound_files={},data_signature={})
    path=root/'frozen.json';path.write_text(json.dumps(frozen))
    monkeypatch.setattr(f,'BASE',base);monkeypatch.setattr(f,'CHECKPOINT',original);monkeypatch.setattr(f,'A0_SHA',sha(original))
    monkeypatch.setattr(f,'evaluation_code_hashes',lambda:{});monkeypatch.setattr(f,'evaluation_data_signature',lambda:{})
    monkeypatch.setattr(f,'validate_calibration_sources',lambda *args:[])
    monkeypatch.setattr(f,'verify_frozen_assets',lambda:{'vendor':'changed'})
    with pytest.raises(ValueError,match='External'):f.validate_frozen(path)
