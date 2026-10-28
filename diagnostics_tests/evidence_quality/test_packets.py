"""Persist exact-role CPU packets without mixing sources or accepting corruption."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module('diagnostics.evidence_quality.packets')


def population(role='train'):
    n = 2025 if role=='train' else 1029
    entries = [dict(role=role, source_kind='revision_v1_A0_closedloop', directory='/original',
        filename=f'view_{i}.pt', episode=i, step=0, map_name='map', object_id=i,
        sha256='a'*64, overlay_sha256='b'*64) for i in range(n)]
    arrays = dict(features=np.zeros((n,22),np.float32), available=np.zeros(n,dtype=bool),
                  before=np.full(n,10.,np.float64), after=np.full(n,10.,np.float64),
                  outcomes=np.zeros(n,np.int64))
    return entries, arrays


def test_exact_population_roundtrip_and_existing_output_not_overwritten(tmp_path):
    entries, arrays = population(); binding = dict(checkpoint_sha256='c'*64)
    api().write_packets(tmp_path,role='train',identities=entries,arrays=arrays,binding=binding)
    result = api().load_packets(tmp_path,role='train',expected_entries=entries,binding=binding)
    assert result['identities'] == entries
    for key in arrays: np.testing.assert_array_equal(result['arrays'][key],arrays[key])
    with pytest.raises(ValueError):
        api().write_packets(tmp_path,role='train',identities=entries,arrays=arrays,binding=binding)


@pytest.mark.parametrize('change', ['wrong_role','missing_row','duplicate','wrong_source'])
def test_incomplete_or_foreign_population_rejected_before_write(tmp_path, change):
    entries, arrays = population()
    if change=='wrong_role': entries[0]['role']='calibration'
    if change=='missing_row': entries.pop()
    if change=='duplicate': entries[1]=dict(entries[0])
    if change=='wrong_source': entries[0]['source_kind']='teacher_snapshot'
    with pytest.raises(ValueError):
        api().write_packets(tmp_path,role='train',identities=entries,arrays=arrays,binding={})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', ['nan','wrong_outcome','unavailable_changed','false_availability'])
def test_invalid_or_inconsistent_packet_rejected(tmp_path, change):
    entries, arrays = population()
    if change=='nan': arrays['features'][0,0]=float('nan')
    if change=='wrong_outcome': arrays['outcomes'][0]=1
    if change=='unavailable_changed': arrays['after'][0]=30.;arrays['outcomes'][0]=2
    if change=='false_availability': arrays['features'][0,14]=1.
    with pytest.raises(ValueError):
        api().write_packets(tmp_path,role='train',identities=entries,arrays=arrays,binding={})


def test_changed_bytes_or_binding_rejected_on_read(tmp_path):
    entries, arrays = population('dev'); binding=dict(checkpoint_sha256='c'*64)
    api().write_packets(tmp_path,role='dev',identities=entries,arrays=arrays,binding=binding)
    with pytest.raises(ValueError):
        api().load_packets(tmp_path,role='dev',expected_entries=entries,binding={})
    path=tmp_path/'dev/packets.npz';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError):
        api().load_packets(tmp_path,role='dev',expected_entries=entries,binding=binding)


def test_changed_expected_observation_order_rejected(tmp_path):
    entries, arrays=population('dev');api().write_packets(tmp_path,role='dev',identities=entries,arrays=arrays,binding={})
    with pytest.raises(ValueError):
        api().load_packets(tmp_path,role='dev',expected_entries=entries[::-1],binding={})


def test_sealed_role_refused_before_file_access(tmp_path):
    with pytest.raises(ValueError):
        api().load_packets(tmp_path,role='calibration',expected_entries=[],binding={})
