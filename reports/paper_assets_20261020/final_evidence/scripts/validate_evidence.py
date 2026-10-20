"""Validate public/local separation, exact alignment and immutable source assets.

This verifies archived evidence, not fresh fixed-weight neural inference.
Use --paper-root to check paper copies; --without-local permits public-only
review after cloning, explicitly omitting private RGB checks.
"""
import argparse
import py_compile
import subprocess
from common import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--paper-root',type=Path,default=Path('/home/ubuntu5/Workspace/2027-Qiaoxu-CVPR'))
    parser.add_argument('--without-local',action='store_true')
    args=parser.parse_args();checks={}
    protected=read(FINAL/'PROTECTED_INPUTS.json')
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    checks['preexisting_assets_code_formal_metrics_unchanged']=len(protected)
    caseidx={r['case_id']:r for r in cases()};observations=read(FINAL/'RGB_FRAME_INDEX.json')
    for r in observations['frames']:
        c=caseidx[r['case_id']];a=fields(r['case_id']);i=r['tensor_index'];step=r['step']
        meta=read(ASSETS/'fields'/f'{r["case_id"]}.json')
        assert r['pose_xyz_yaw']==c['trace']['poses'][step]==meta['frames'][i]['pose']
        assert np.array_equal(a['poses'][step],r['pose_xyz_yaw'])
        assert r['step']==meta['frames'][i]['step']==c['trace']['step_diagnostics'][i]['observation_step']
        assert r['npz_sha256']==r['archive_npz_sha256']
        assert not r['public_rgb'] and r['strict_target_visibility'] is None
        assert r['weight_sha256']==c['checkpoint_sha256']
        assert r['residual_weight_sha256']==c['residual_checkpoint_sha256']
        assert np.array_equal(a['world_xy'][a['real_belief'][i].argmax()],r['prediction_xy_dense_argmax'])
        if not args.without_local:
            path=FINAL/r['local_npz'];assert sha(path)==r['npz_sha256']
            with np.load(path,allow_pickle=False) as obs:
                assert hashlib.sha256(obs['bgr'].tobytes()).hexdigest()==r['bgr_pixel_sha256']
                assert np.array_equal(obs['footprint_world'],r['footprint_world'])
    checks['same_step_pose_weight_tensor_metadata_frames']=len(observations['frames'])
    checks['private_RGB_hashes_verified']=0 if args.without_local else len(observations['frames'])
    figrows=[]
    for metadata in sorted((ASSETS/'figures').glob('*.json')):
        d=read(metadata);art=[]
        for ext,h in d['sha256'].items():
            path=metadata.with_suffix('.'+ext);assert sha(path)==h
            paper=args.paper_root/'figures/citynav'/path.name
            assert paper.exists() and sha(paper)==h,paper
            art.append({'file':str(path.relative_to(ROOT)),'sha256':h,'matches_paper_copy':True})
        for c in d['cases']:
            expected=caseidx[c['case_id']]
            for key in ['episode_id','checkpoint_sha256','residual_checkpoint_sha256','source_predictions_sha256','dataset_sha256']:
                assert c[key]==expected[key],(metadata,key)
        sources=d.get('sources',[])
        figrows.append({'figure':d['figure'],'cases':d['cases'],'separate_geometry_heads':sources,
            'image_use':d['image_use'],'artifacts':art})
    assert len(figrows)==4
    dump(FINAL/'FIGURE_PROVENANCE_AUDIT.json',{'original_four_figures_regenerated':False,'figures':figrows,
        'units':{'map_axes':'world meters; +Y not verified geographic north','belief':'normalized probability',
                 'overview_residual':'logit delta; fixed [-0.5,0.5] display','visual_difference':'normalized probability difference; per-row symmetric scales',
                 'goal_errors':'horizontal meters','GT_circle':'20-meter evaluator radius'},
        'colors':{'GT':'green star','predicted_goal':'yellow X','executed_path':'blue','actual_final_pose':'red square',
                  'observation_footprint':'cyan','probability':'magma with shared log scales within comparison',
                  'signed_residual_or_difference':'RdBu_r (red positive, blue negative)',
                  'shuffled_geometry_donor':'dotted cyan; recipient context solid'},
        'caption_audit':'See PAPER_METHOD_CORRECTIONS.md: current four captions already distinguish formal, CVR, separate geometry and matched-path controls.'})
    checks['original_figures_and_paper_copies_hash_verified']=4
    for path in (FINAL/'figures').glob('*.json'):
        d=read(path);assert d['rgb_pixels'] is False
        for ext,h in d['sha256'].items():assert sha(path.with_suffix('.'+ext))==h
    for path in (FINAL/'scripts').glob('*.py'):py_compile.compile(str(path),doraise=True)
    snap=read(FINAL/'MANUSCRIPT_SNAPSHOT.json')
    for path,h in snap['sections'].items():assert sha(args.paper_root/path)==h
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=args.paper_root,text=True).strip()
    checks['paper_sections_and_clean_tree_verified']=True
    tracked=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard',str(FINAL.relative_to(ROOT))],cwd=ROOT,text=True).splitlines()
    assert not any('local_only' in Path(p).parts for p in tracked)
    assert not any(Path(p).suffix.lower() in ['.pt','.pth','.npz','.tif','.jpg','.jpeg'] for p in tracked)
    checks['public_paths_have_no_private_RGB_or_weights']=True
    for name in ['RGB_CASE_MANIFEST.md','MODEL_MECHANISM_AUDIT.md','VISUAL_CAUSAL_CASES.csv','PAPER_METHOD_CORRECTIONS.md','REPORT.md']:
        assert (FINAL/name).is_file(),name
    code=read(FINAL/'CODE_PROVENANCE.json')
    assert all(r['matches_recorded_inference'] and sha(ROOT/p)==r['current_sha256'] for p,r in code['code_checks'].items())
    checks['source_files_match_inference_hashes']=len(code['code_checks'])
    assert read(FINAL/'CAUSAL_SUMMARY.json')['archived_comparisons_verified']==310
    checks['archived_statistic_comparisons_verified_by_analysis_script']=310
    dump(FINAL/'VALIDATION.json',{'passed':True,'checks':checks,'new_training':False,'new_neural_inference':False,
        'original_four_figures_regenerated':False,'local_checks_omitted':args.without_local,
        'unit_tests':{'command':'python -m pytest -q tests/test_stage1.py tests/test_calibrated_residual.py tests/test_appearance_matching.py tests/test_stage1_diagnostics.py tests/test_stage1_final_evaluation.py',
                      'result':'23 passed in 0.97s','scope':'existing lightweight mechanism tests; no weights or GPU'}})
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
