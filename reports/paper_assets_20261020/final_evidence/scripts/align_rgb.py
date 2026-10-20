"""Recover exact archived model-input crops, not new FPV or model inference.

Every NPZ must equal its archived hash before being paired with belief tensors.
Decoded observations and RGB-containing figures are always local_only.
"""
import argparse
import sys
import cv2
import numpy as np
import rasterio
from common import *
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import world_to_frame


def main():
    p=argparse.ArgumentParser();p.add_argument('--rgbd',type=Path,default=Path('/home/ubuntu5/Workspace/HETT/data/rgbd'))
    p.add_argument('--ground-table',type=Path,default=Path('/home/ubuntu5/Workspace/hett-experiments/02-static-belief-memory/multiagent/mapdata.py'))
    a=p.parse_args()
    # Read only published map constants, no episode target/trajectory inference.
    import ast
    tree=ast.parse(a.ground_table.read_text());ground=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GROUND_LEVEL' for t in node.targets):ground=ast.literal_eval(node.value)
    assert ground is not None
    index=read(ASSETS/'observation_index.json');old={(r['case_id'],r['step']):r for r in index['frames']}
    originals={Path(r['path']).name:r for r in index['source_assets']}
    selected={r['case_id']:r for r in cases() if r['case_id'] in SELECTED}
    render=CityNavObservedViewRenderer(a.rgbd,FINAL/'local_only/scratch',image_size=384)
    output=[];source=[];selection={}
    for name in sorted({r['map_name'] for r in selected.values()}):
        for ext in ['.png','.tif']:
            path=a.rgbd/(name+ext);orig=originals[path.name]
            row={'map_name':name,'local_path':str(path.resolve()),'sha256':sha(path),
                 'archive_source_sha256':orig['sha256'],'file_bytes_match_archive':sha(path)==orig['sha256']}
            if ext=='.tif':
                with rasterio.open(path) as f:row.update(shape=[f.height,f.width],affine=list(f.transform),nodata=f.nodata,crs=str(f.crs))
            source.append(row)
    for cid in SELECTED:
        case=selected[cid];data=fields(cid);meta=read(ASSETS/'fields'/f'{cid}.json');frames=meta['frames']
        assert sha(ASSETS/meta['arrays'])==meta['arrays_sha256']
        real=data['real_belief'];static=data['real_static'];changed=real.argmax(1)!=static.argmax(1)
        interior=[j for j in np.flatnonzero(changed) if 0<j<len(frames)-1]
        key=int(interior[0]) if interior else len(frames)//2
        picks=[0,key,len(frames)-1];selection[cid]={'tensor_indices':picks,'steps':[frames[i]['step'] for i in picks],
            'key_selection':'first interior static/observed Top1 change; otherwise temporal midpoint',
            'model_identity':case['model_identity'],'checkpoint_sha256':case['checkpoint_sha256'],
            'residual_checkpoint_sha256':case['residual_checkpoint_sha256']}
        for i,frame in enumerate(frames):
            step=frame['step'];original=old[(cid,step)];pose=np.asarray(frame['pose'],np.float64)
            assert np.array_equal(pose,np.asarray(case['trace']['poses'][step]))
            assert np.array_equal(pose,np.asarray(data['poses'][step]))
            assert np.array_equal(pose,np.asarray(original['pose']))
            assert case['trace']['step_diagnostics'][i]['observation_step']==step
            obs=render.current_rgbd(case['map_name'],pose,ground[case['map_name']])
            assert np.array_equal(obs['footprint_world'],np.asarray(frame['footprint_world'],np.float32))
            path=FINAL/'local_only/observations'/f'{cid}_step{step}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,bgr=obs['bgr'],footprint_world=obs['footprint_world'],depth_valid=obs['depth_valid'])
            actual=sha(path)
            if actual!=original['crop_sha256']:
                raise RuntimeError(f'{cid}:{step} reconstructed pixels/support fail archived hash; do NOT pair this crop with old beliefs')
            image=path.with_suffix('.png');assert cv2.imwrite(str(image),obs['bgr'])
            target=np.asarray(case['target_xyz'][:2]);pixel=world_to_frame(target[None],obs['footprint_world'],obs['bgr'].shape)[0]
            inside=bool(np.isfinite(pixel).all() and (pixel>=0).all() and (pixel<=383).all())
            support=False
            if inside:
                u,v=np.rint(pixel).astype(int);support=bool(obs['depth_valid'][v,u] and obs['bgr'][v,u].max()>0)
            output.append({'case_id':cid,'episode_id':case['episode_id'],'map_name':case['map_name'],'split':case['split'],
                'step':step,'tensor_index':i,'pose_xyz_yaw':pose.tolist(),
                'world_coordinate_units':'meters','yaw_units':'radians',
                'camera_pitch':None,'camera_roll':None,'camera_model':'square nadir orthophoto crop; not measured perspective camera',
                'footprint_world':obs['footprint_world'].tolist(),'ground_level_m':ground[case['map_name']],
                'image_size':[384,384,3],'channel_order':'saved PNG is RGB-compatible OpenCV BGR encoding; arrays explicitly bgr',
                'interpolation':'cv2.INTER_LINEAR','border':'OpenCV constant zero; no inpainting',
                'local_npz':str(path.relative_to(FINAL)),'local_png':str(image.relative_to(FINAL)),
                'npz_sha256':actual,'archive_npz_sha256':original['crop_sha256'],'exact_archive_crop_match':True,
                'bgr_pixel_sha256':hashlib.sha256(obs['bgr'].tobytes()).hexdigest(),
                'source_rgb':str((a.rgbd/(case['map_name']+'.png')).resolve()),
                'target_xy_projected_pixel':pixel.astype(float).tolist(),'target_xy_inside_footprint':inside,
                'target_projection_supported_proxy':support,'strict_target_visibility':None,
                'belief_file':str((ASSETS/'fields'/f'{cid}.npz').relative_to(ROOT)),
                'belief_tensor':'real_belief','residual_tensor':'real_residual','weight_sha256':case['checkpoint_sha256'],
                'residual_weight_sha256':case['residual_checkpoint_sha256'],'is_selected_panel':i in picks,
                'instruction':case['instruction'],'GT_target_xyz':case['target_xyz'],
                'prediction_xy_dense_argmax':data['world_xy'][real[i].argmax()].tolist(),
                'actual_controller_selected_goal_xy':case['trace']['step_diagnostics'][i]['selected_goal_xy'],
                'trajectory_source':'same-case executed poses; not an expert route',
                'fields_sha256':meta['arrays_sha256'],'renderer_sha256':sha(ROOT/'graphnav/renderer.py'),
                'public_rgb':False})
        print(cid,'exact crops',len(frames),'selected steps',selection[cid]['steps'],flush=True)
    dump(FINAL/'RGB_FRAME_INDEX.json',{'frames':output,'sources':source,'selection':selection,
        'all_frames_match_archived_NPZ_SHA256':True,'new_neural_inference':False,
        'true_perspective_FPV_available':False,'file_hash_mismatch_resolution':'Do not assert whole-raster byte identity. Every complete reconstructed model-input NPZ independently equals its archived SHA256, including RGB, footprint and valid-depth support.'})


if __name__=='__main__':main()
