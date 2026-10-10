from graphnav.checkpointing import resolve_checkpoint_resource


def test_preserved_relative_resource_resolves_original_repository(tmp_path, monkeypatch):
    original=tmp_path/'original';checkpoint=original/'runs'/'model'/'best.pt'
    checkpoint.parent.mkdir(parents=True);checkpoint.touch()
    expected=original/'runs'/'source_cache';expected.mkdir()
    unrelated=tmp_path/'current';unrelated.mkdir();(unrelated/'runs'/'source_cache').mkdir(parents=True)
    monkeypatch.chdir(unrelated)
    assert resolve_checkpoint_resource('runs/source_cache',checkpoint)==expected


def test_absolute_resource_is_not_silently_relocated(tmp_path):
    missing=tmp_path/'missing'
    assert resolve_checkpoint_resource(missing,tmp_path/'best.pt')==missing
