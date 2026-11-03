from scripts.report_region_semantic_probe import merge_metrics


def test_group_summary_merges_confusions_instead_of_averaging_frame_rates():
    rows = [{'methods': {'x': {'confusion': [[100, 0, 0], [0, 0, 0], [0, 0, 0]]}}},
            {'methods': {'x': {'confusion': [[0, 0, 0], [1, 0, 0], [0, 0, 1]]}}}]
    result = merge_metrics(rows, 'x')
    assert result['patches'] == 102
    assert result['balanced_accuracy'] == 2/3
    assert result['class_support'] == [100, 1, 1]
