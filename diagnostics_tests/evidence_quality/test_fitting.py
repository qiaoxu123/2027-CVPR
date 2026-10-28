"""Freeze both20-round probes and conditions without reading DEV packets."""
import importlib

import numpy as np
import pytest

from diagnostics_tests.evidence_quality.test_packets import population


def api():
    return importlib.import_module('diagnostics.evidence_quality.fitting')


def exported_packets(tmp_path):
    from diagnostics.evidence_quality.packets import write_packets
    entries, arrays=population();binding=dict(checkpoint_sha256='c'*64)
    write_packets(tmp_path/'packets',role='train',identities=entries,arrays=arrays,binding=binding)
    # Deliberately invalid DEV bytes: choosing rules must never open this file.
    (tmp_path/'packets/dev').mkdir();(tmp_path/'packets/dev/packets.npz').write_bytes(b'not opened')
    return dict(binding=binding,entries=dict(train=entries,dev=[]))


def test_full20_round_fitting_freezes_weights_choices_and_curves_without_dev(tmp_path):
    exported=exported_packets(tmp_path)
    result=api().fit_and_freeze(tmp_path,exported)
    assert result['decision']['choices']==dict(baseline=None,extended=None)
    for arm in ('baseline','extended'):
        assert len(result['models'][arm]['history'])==20
    assert (tmp_path/'fit/fit_curves.png').is_file()
    assert (tmp_path/'fit/fit_curves.svg').is_file()
    assert list((tmp_path/'fit/tensorboard').glob('events.*'))
    cached=api().fit_and_freeze(tmp_path,exported)
    assert cached['decision']==result['decision']
    assert cached['manifest_sha256']==result['manifest_sha256']


def test_changed_frozen_choice_refused_instead_of_refitted(tmp_path):
    exported=exported_packets(tmp_path);api().fit_and_freeze(tmp_path,exported)
    path=tmp_path/'fit/fixed_choice.json';path.write_text('{}')
    with pytest.raises(ValueError):api().fit_and_freeze(tmp_path,exported)
