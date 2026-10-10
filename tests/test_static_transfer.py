import pytest
from graphnav.static_transfer_runtime import validate_static_transfer


def test_controller_transfer_rejects_incompatible_hidden_or_grid():
    for key in ('hidden','grid_size','relative_position'):
        with pytest.raises(ValueError,match=key):validate_static_transfer({key:128},{key:256})


def test_original_legacy_default_relative_coordinates_are_compatible():
    validate_static_transfer({'hidden':128},{'hidden':128,'relative_position':False})
