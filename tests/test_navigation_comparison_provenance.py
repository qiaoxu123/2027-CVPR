import pytest
from scripts.summarize_native_navigation import canonical_data_hashes


def test_migrated_dataset_roots_preserve_relative_paths_and_content_hashes():
    old={'/old/sbfnav-data/cityrefer/objects.json':'abc'}
    new={'/new/sbfnav-data/cityrefer/objects.json':'abc'}
    assert canonical_data_hashes(old)==canonical_data_hashes(new)
    assert canonical_data_hashes(old)!=canonical_data_hashes({'/new/sbfnav-data/other/objects.json':'abc'})
    assert canonical_data_hashes(old)!=canonical_data_hashes({'/new/sbfnav-data/cityrefer/objects.json':'changed'})
    with pytest.raises(ValueError,match='Ambiguous'):
        canonical_data_hashes(dict(old,**new))
