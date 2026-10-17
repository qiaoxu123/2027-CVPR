"""Metric boundary cases and real exported-table integrity checks."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from scripts.build_paper_experiments import OUT, METHODS, SPLITS, path_metrics, aggregate, eid


def read(name):
    return list(csv.DictReader((OUT / name).open()))


def test_xy_inclusive_success_and_oracle_path():
    # Z is deliberately large: benchmark is horizontal; final radius boundary is inclusive.
    r = path_metrics([[100, 0, 900], [0, 0, 900], [20, 0, 900]], [0, 0, 0])
    assert r['NE'] == 20 and r['SR'] == r['OSR'] == 1
    assert r['SPL'] == pytest.approx(100 / 120)
    r = path_metrics([[100, 0], [0, 0], [21, 0]], [0, 0])
    assert r['SR'] == r['SPL'] == 0 and r['OSR'] == 1


def test_invalid_geometry_does_not_silently_drop_episode():
    with pytest.raises(AssertionError):
        path_metrics([[np.nan, 0]], [0, 0])


def test_episode_ids_are_not_row_indices_or_instruction_text():
    assert eid("('cambridge_block_2', 17, 3)") == ('cambridge_block_2', 17, 3)
    assert eid(['cambridge_block_2', '17', '3']) == ('cambridge_block_2', 17, 3)


def test_all_table1_targets_and_splits_present():
    rows = read('table1_ours_unified.csv')
    for method in METHODS:
        assert {r['split'] for r in rows if r['method'] == method} == set(SPLITS)
    for r in rows:
        if r['status'] == 'Done':
            assert Path(r['result_path']).is_file()
            assert int(r['episodes']) > 0
        else:
            assert r['missing'] and all(r[k] == '' for k in ('NE', 'SR', 'OSR', 'SPL'))


def test_difficulty_membership_and_weighted_recombination():
    manifest = json.loads((OUT / 'difficulty_episode_manifest.json').read_text())
    membership = manifest['versions']['revised']
    assert membership['counts'] == {'easy': 2081, 'medium': 1678, 'hard': 1522}
    ids = [tuple(r['episode_id']) for r in membership['membership']]
    assert len(ids) == len(set(ids)) == 5281
    full = next(r for r in read('table1_ours_unified.csv') if r['method'] == 'PolyBelief' and r['split'] == 'test_unseen')
    groups = [r for r in read('table2_difficulty_comparison.csv') if r['method'] == 'PolyBelief']
    assert len(groups) == 3 and sum(int(r['episodes']) for r in groups) == 5281
    for key in ('NE', 'SR', 'OSR', 'SPL'):
        combined = sum(float(r[key]) * int(r['episodes']) for r in groups) / 5281
        assert combined == pytest.approx(float(full[key]), abs=1e-10)


def test_paper_and_local_result_identities_do_not_mix():
    paper = read('table1_paper_reported.csv')
    assert {r['dataset'] for r in paper if r['method'] == 'SBFNav'} == {
        'original_CityNav', 'revised_paper_unspecified_file_hash'}
    assert all(not r.get('checkpoint_sha256') for r in paper)
    difficulty_paper = read('table2_paper_reported.csv')
    assert 'Seq2Seq' in {r['method'] for r in difficulty_paper}
    assert 'Seq2Seq+GSM' not in {r['method'] for r in difficulty_paper}


def test_calibrated_residual_has_exact_checkpoint_and_full_counts():
    rows = [r for r in read('ablation_results.csv') if r['variant'].startswith('Calibrated Visual Residual')]
    assert len(rows) == 8
    for r in rows:
        assert r['residual_checkpoint_sha256'] == '5fc1a3e115b71d14fe1e8fbaf3ed29c1863d8b2bc080455d89f7a4e503d25f11'
        assert int(r['grid_size']) == 28
        assert int(r['episodes']) == {'val_seen': 2470, 'val_unseen': 2697}[r['split']]


def test_sbf_epoch12_is_consistent_across_all_three_splits():
    rows = [r for r in read('table1_ours_unified.csv') if r.get('variant') == 'paper-based reimplementation epoch12']
    assert {r['split'] for r in rows} == set(SPLITS)
    assert {r['checkpoint_sha256'] for r in rows} == {'92db9ed470563d2fd7c57ef35eb8627f2a7552ef0d4f4dca46de8ddba0035a31'}
    assert all(r['comparable'] == 'False' and 'Val-Unseen' in r['missing'] for r in rows)


def test_exported_common_episode_metrics_match_each_completed_table_row():
    from collections import defaultdict
    grouped = defaultdict(list)
    seen = set()
    for r in read('common_episode_metrics.csv'):
        key = (r['experiment_id'], r['map_name'], r['object_id'], r['ann_id'])
        assert key not in seen
        seen.add(key)
        grouped[r['experiment_id']].append(r)
    for r in read('table1_ours_unified.csv'):
        if r['status'] != 'Done':
            continue
        episodes = grouped[r['experiment_id']]
        assert len(episodes) == int(r['episodes'])
        for metric in ('NE', 'SR', 'OSR', 'SPL'):
            assert np.mean([float(x[metric]) for x in episodes]) == pytest.approx(float(r[metric]), abs=1e-10)
