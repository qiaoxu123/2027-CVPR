import json
import pytest
from scripts.run_experiment_queue import validate_required_checks


def test_existing_failed_report_does_not_authorize_dependent_experiment(tmp_path):
    path=tmp_path/'check.json';path.write_text(json.dumps(dict(all_exact=False)))
    manifest=dict(required_checks=[dict(path=str(path),equals={'all_exact':True})])
    with pytest.raises(ValueError,match='stage check failed'):validate_required_checks(manifest)
    path.write_text(json.dumps(dict(all_exact=True)))
    validate_required_checks(manifest)
