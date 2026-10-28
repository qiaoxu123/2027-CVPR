"""Exercise the real final-report path using invented tiny benchmark records."""
import csv,hashlib,json,shutil
from pathlib import Path
from types import SimpleNamespace
import pytest
import importlib

def _module():
    return importlib.import_module('experiments.a2_waldo_landmark_belief.reference_v4_results')
from experiments.a2_waldo_landmark_belief.relation_v3_reporting import compare_pairs
from tests_reporting_v3.test_reporting_statistics import row


def fixture_runtime(tmp_path,monkeypatch,*,incomplete=False,changed_freeze=False):
    reports=_module()
    import torch
    from experiments.a2_waldo_landmark_belief import paths,collected_snapshot,reference_v4_navigation as navigation
    from experiments.a2_waldo_landmark_belief import reference_v4_navigation_freeze as freeze,reference_v4_navigation_sources as sources,revision_eval_protocol as protocol
    import sbfnav.dataset as datasets
    import sbfnav.planning_state as planning
    def forbidden(*a,**k):raise AssertionError('Report attempted GPU or real tensor/data access')
    monkeypatch.setattr(torch,'load',forbidden);monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(paths,'ROOT',tmp_path);monkeypatch.setattr(paths,'DATA',tmp_path/'invented_data')
    monkeypatch.setattr(collected_snapshot,'BASE',tmp_path/'run')
    root=tmp_path/'run/revision_v4';(root/'training').mkdir(parents=True);(root/'frozen.json').write_text('frozen')
    for ext in ('png','svg'):(root/f'training/training_curves.{ext}').write_bytes(b'preserved synthetic training curve')
    (root/'training/curve_manifest.json').write_text(json.dumps({ext+'_sha256':paths.sha(root/f'training/training_curves.{ext}') for ext in ('png','svg')}))
    for name in ('reference_v4_results.py','relation_v3_results.py','relation_v3_reporting.py'):
        target=tmp_path/'experiments/a2_waldo_landmark_belief'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(reports.__file__).parent/name,target)
    tests=tmp_path/'tests_reporting_v4';tests.mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copyfile(source,tests/source.name)
    counts={'val_seen':3,'val_unseen':2};monkeypatch.setattr(protocol,'FULL_COUNTS',counts)
    seen=[]
    class Dataset(list):
        def __init__(self,data,scope):
            self.scope=scope
            super().__init__([SimpleNamespace(index=i,object_id=f't{i}',instruction=f'query{i}',target_position=[1.,2.,0.]) for i in range(counts[scope])])
    monkeypatch.setattr(datasets,'CityNavDataset',Dataset);monkeypatch.setattr(datasets,'CityReferCatalog',lambda data:object())
    monkeypatch.setattr(planning,'PlanningStateBuilder',lambda *args,**kwargs:object())
    validations=[]
    def validate(root):
        validations.append(root)
        if len(validations)==2 and changed_freeze:(root/'frozen.json').write_text('changed')
    monkeypatch.setattr(freeze,'validate_frozen',validate);monkeypatch.setattr(sources,'code_hashes',lambda:{'frozen_runtime':'digest'})
    def arm(scope,name,dataset,catalog,builder):
        seen.append((scope,name))
        def rows(success):
            result=[]
            for i in range(counts[scope]):
                value=row(i,target=f't{i}',sr=int(success and i==0),stop=1,false=int(not(success and i==0)),recall=int(success and i==0),updated=success)
                value['poses']=[[0,0,30,0],[1 if success else 0,1,30,0]];result.append(value)
            return result
        a0,a2=rows(False),rows(True);comparison=compare_pairs(a0,a2,draws=100)
        selected=a0 if name=='A0' else a2
        if incomplete and scope=='val_unseen' and name=='A2':selected=selected[:-1]
        from graphnav.closedloop import aggregate
        return selected,dict(metrics=aggregate([r['metrics'] for r in selected]),candidate_coverage=None if name=='A0' else comparison['candidate_coverage'],
            goal_retrieval=dict(unique_queries=len(dataset),**{k:v[name] for k,v in comparison['recall'].items()})),dict(complete_sha256='synthetic',signature_sha256='synthetic')
    monkeypatch.setattr(navigation,'verified_arm',arm)
    return root,seen,validations,tmp_path/'reports/a2_execution_20261027/revision_v4_final'


def test_real_report_runner_verifies_all_four_arms_and_exports_tables_curves_and_plots(tmp_path,monkeypatch):
    reports=_module()
    root,seen,validations,output=fixture_runtime(tmp_path,monkeypatch)
    output.mkdir(parents=True);(output/'complete.json').write_text('previous complete');(output/'results.json').write_text('previous results')
    reports.run()
    archived=list((output.parent/'revision_v4_final_archive').iterdir())
    assert len(archived)==1 and (archived[0]/'results.json').read_text()=='previous results'
    assert seen==[(s,a) for s in ('val_seen','val_unseen') for a in ('A0','A2')]
    assert len(validations)==2
    result=json.loads((output/'results.json').read_text());complete=json.loads((output/'complete.json').read_text())
    assert result['positive_SR_on_both_splits'] is True
    assert result['splits']['val_seen']['episodes']==3
    for name,sha in complete['files_sha256'].items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==sha
    assert (output/'paired_differences.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (output/'paired_differences.svg').read_text()
    assert (output/'training_curves.png').read_bytes()==(root/'training/training_curves.png').read_bytes()
    rows=list(csv.DictReader((output/'metrics.csv').open()))
    rate=next(r for r in rows if r['split']=='val_seen' and r['metric']=='False_Stop_Rate')
    assert float(rate['A0'])==100 and float(rate['A2'])==pytest.approx(200/3)
    assert (output/'by_map.csv').exists() and (output/'ANALYSIS_ZH.md').exists()
    assert len(json.loads((output/'examples.json').read_text()))==2


@pytest.mark.parametrize('failure',['incomplete','changed_freeze'])
def test_partial_benchmark_or_changed_freeze_cannot_publish_a_final_report(tmp_path,monkeypatch,failure):
    reports=_module()
    root,seen,validations,output=fixture_runtime(tmp_path,monkeypatch,**{failure:True})
    with pytest.raises(ValueError):reports.run()
    assert not (output/'complete.json').exists() and not (output/'results.json').exists()


def test_report_source_changed_during_plotting_cannot_be_published(tmp_path,monkeypatch):
    reports=_module()
    root,seen,validations,output=fixture_runtime(tmp_path,monkeypatch)
    original=reports.plot_differences
    def changed(result,directory):
        original(result,directory)
        (tmp_path/'experiments/a2_waldo_landmark_belief/relation_v3_reporting.py').write_text('changed during reporting')
    monkeypatch.setattr(reports,'plot_differences',changed)
    with pytest.raises(ValueError,match='source'):reports.run()
    assert not (output/'complete.json').exists()


def test_plotting_failure_preserves_existing_complete_report_byte_for_byte(tmp_path,monkeypatch):
    reports=_module()
    root,seen,validations,output=fixture_runtime(tmp_path,monkeypatch)
    output.mkdir(parents=True);(output/'complete.json').write_text('old complete report');(output/'results.json').write_text('old results')
    def failed(*a):raise RuntimeError('invented plot failure')
    monkeypatch.setattr(reports,'plot_differences',failed)
    with pytest.raises(RuntimeError,match='plot failure'):reports.run()
    assert (output/'results.json').read_text()=='old results'
    assert (output/'complete.json').read_text()=='old complete report'
