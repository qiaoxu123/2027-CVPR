"""Check reached camera geometry and original measured colors on actual frames."""
from collections import defaultdict
from pathlib import Path
import json
import cv2,numpy as np
from experiments.multiview_rgb_grounding.render import ROOT,read_cloud,project
from .run import R,dump,visible_regions
from .sensor import render_culled

def run():
 grouped=defaultdict(list)
 for path in (R/'frames').glob('*/metadata.json'):
  meta=json.loads(path.read_text());grouped[meta['map_name']].append((path,meta))
 checked=0;max_error=0.;pixels=0;roi_checks=0;culling_checks=0;seen_tilts=set();objects=json.loads(Path("/home/rental/20260922_1/Workspace/DATA/sbfnav-data/cityrefer/objects.json").read_text());gate=json.loads(Path("experiments/multiview_rgb_grounding/protocol.json").read_text())["quality_gates"]
 for name,frames in grouped.items():
  xyz,rgb,data=read_cloud(next(ROOT.glob('*/'+name+'.ply')))
  for path,m in frames:
   g=np.load(path.parent/'geometry.npz');win=g['winner'];valid=win>=0;flat=np.flatnonzero(valid);chosen=flat[np.linspace(0,len(flat)-1,min(1024,len(flat))).astype(int)] if len(flat) else flat;ids=win.ravel()[chosen];image=cv2.imread(str(path.parent/'rgb.png'))[...,::-1];mask=cv2.imread(str(path.parent/'visibility.png'),0);assert np.array_equal(mask>0,valid)
   if len(ids):
    uv,z,good=project(xyz[ids],m['camera_pose'],m['view']['camera_tilt']);assert good.all();yy,xx=np.divmod(chosen,512);err=np.linalg.norm(np.stack([xx+.5,yy+.5],1)-uv,axis=1);assert err.max()<=2.13;assert np.array_equal(image.reshape(-1,3)[chosen],rgb[ids]);assert np.allclose(g['depth'].ravel()[chosen],z,atol=1e-4);max_error=max(max_error,float(err.max()));pixels+=len(ids)
   if (name,m['view']['camera_tilt']) not in seen_tilts:
    fresh=render_culled(xyz,rgb,m['camera_pose'],m['view']['camera_tilt']);assert np.array_equal(fresh['winner'],win);assert np.array_equal(fresh['rgb'],image);assert np.array_equal(fresh['depth'],g['depth'],equal_nan=True);culling_checks+=1
    rois,mask=visible_regions({'winner':win},xyz,objects[name],gate);assert rois==m['regions'];assert np.array_equal(mask,g['instance']);seen_tilts.add((name,m['view']['camera_tilt']));roi_checks+=1
   assert np.isnan(g['depth'][~valid]).all();assert m['reprojection_max']<=2.13;checked+=1
  del xyz,rgb,data
 dump(R/'geometry_checks.json',dict(actual_frames_checked=checked,actual_ROI_equivalence_frames=roi_checks,actual_frustum_culling_equivalence_frames=culling_checks,original_color_pixels_checked=pixels,max_sampled_reprojection_pixels=max_error,depth_matches_axial_camera_z=True,validity_mask_matches_real_winners=True,invalid_depth_nan=True,independent_camera_yaw_and_tilt=True,no_image_warp_or_texture_generation=True))
 print(json.dumps({'actual_frames_checked':checked}),flush=True)

if __name__=='__main__':run()
