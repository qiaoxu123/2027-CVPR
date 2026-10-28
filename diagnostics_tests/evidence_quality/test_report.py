"""Exercise positive and rejected diagnostic outcomes, never policy enablement."""
import importlib
import math

import numpy as np
import pytest

from diagnostics_tests.evidence_quality.test_packets import population
from diagnostics_tests.evidence_quality.test_fitting import exported_packets


def api():
    return importlib.import_module('diagnostics.evidence_quality.report')


def signal_packets(tmp_path,binding=None):
    from diagnostics.evidence_quality.packets import write_packets
    binding=binding if binding is not None else dict(checkpoint_sha256='c'*64);entries={}
    for role in ('train','dev'):
        ids,a=population(role)
        if role=='dev':
            for row in ids:row['map_name']='heldout'
        i=np.arange(len(ids));gain=i%5==0;harm=(i%5==1)|(i%5==2)
        a['available'][:]=True;a['features'][:,14]=1.
        a['features'][:,15:17]=math.log(2)
        a['features'][gain,15]=math.log(8);a['features'][harm,16]=math.log(8)
        a['before'][gain]=30.;a['after'][harm]=30.
        a['outcomes'][gain]=1;a['outcomes'][harm]=2
        write_packets(tmp_path/'packets',role=role,identities=ids,arrays=a,binding=binding)
        entries[role]=ids
    return dict(binding=binding,entries=entries)


def test_no_train_condition_produces_honest_rejection_without_dev_read(tmp_path):
    exported=exported_packets(tmp_path)
    result=api().assess_and_report(tmp_path,exported)
    assert result['status']=='no_safe_train_selection_conditions'
    assert result['internal_dev_evaluated'] is False
    assert result['internal_results'] is None
    assert result['promising_for_formal_iteration'] is False
    assert result['objective_achieved'] is False


def test_fixed_new_information_signal_can_pass_without_enabling_runtime(tmp_path):
    exported=signal_packets(tmp_path)
    result=api().assess_and_report(tmp_path,exported)
    assert result['internal_dev_evaluated'] is True
    assert result['internal_results']['baseline']['summary']['corrected']==0
    advanced=result['internal_results']['extended']
    assert advanced['summary']['corrected']>30 and advanced['summary']['broken']==0
    assert advanced['passes_internal'] is True
    assert result['promising_for_formal_iteration'] is True
    assert result['no_runtime_enablement'] is True
    assert result['objective_achieved'] is False
    assert advanced['by_map']['heldout']['views']==1029
    assert (tmp_path/'report/report_ZH.md').is_file()
    assert api().assess_and_report(tmp_path,exported)==result


def test_modified_published_report_refused_without_overwrite(tmp_path):
    exported=exported_packets(tmp_path);api().assess_and_report(tmp_path,exported)
    path=tmp_path/'report/report_ZH.md';path.write_text('changed')
    with pytest.raises(ValueError):api().assess_and_report(tmp_path,exported)
    assert path.read_text()=='changed'
