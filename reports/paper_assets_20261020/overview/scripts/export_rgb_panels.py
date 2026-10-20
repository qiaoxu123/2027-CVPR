"""Export existing real-data artists as independent figures, never a panel montage."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,out,name,bbox=None):
 for ext in ['pdf','png','svg']:
  p=out/f'{name}.{ext}';fig.savefig(p,dpi=450,bbox_inches=bbox,pad_inches=.035,facecolor='white')
  if ext=='svg':p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
 return {ext:{'file':f'{name}.{ext}','sha256':sha(out/f'{name}.{ext}')} for ext in ['pdf','png','svg']}

def export_panels(fig,axes,im,E,D,out,root):
 entries=[]
 for text in fig.texts:text.set_visible(False)
 # The input-image inset is a separate deliverable, not embedded in navigation.
 for ax in axes:
  for child in ax.child_axes:child.set_visible(False)
 titles=['(a) Language-guided target search','(b) Map-geometric goal grounding','(c) Belief-guided navigation']
 names=['panel_a_language_search_rgb','panel_b_geometric_belief_rgb','panel_c_navigation_rgb']
 for i,(ax,title,name) in enumerate(zip(axes,titles,names)):
  for other in axes:other.set_visible(other is ax)
  ax.set_title(title,fontsize=8,fontweight='bold',pad=8)
  fig.canvas.draw();bbox=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(.04)
  entries.append({'name':name,'step':0 if i<2 else 7,'instruction':E['instruction_verbatim'],'dataset_version':E['dataset_version'],'checkpoint_sha256':E['checkpoint_sha256'],'source_fields':E['source_array_sha256'],'source_rgb_sha256':E['raster']['sha256'],'display_world_bounds':E['display_world_bounds'],'separate_observation_inset':True,'files':save(fig,out,name,bbox)})
 row=E['observation_crops'][-1];p=Path(row['local_path']);assert sha(p)==row['sha256'];obs=np.load(p);rgb=obs['bgr'][...,::-1]
 f,ax=plt.subplots(figsize=(2.6,2.6));ax.imshow(rgb,interpolation='nearest');ax.set_axis_off();ax.set_title('Pose-conditioned Orthophoto Observation\nStep 7',fontsize=8,pad=6)
 entries.append({'name':'observation_step7','step':7,'pose':row['pose'],'footprint_world':row['footprint_world'],'source_npz_sha256':row['sha256'],'bgr_pixel_sha256':hashlib.sha256(obs['bgr'].tobytes()).hexdigest(),'files':save(f,out,'observation_step7','tight')});plt.close(f)
 raw=out/'observation_step7_native_384.png';Image.fromarray(rgb).save(raw);assert np.array_equal(np.array(Image.open(raw)),rgb)
 f=plt.figure(figsize=(3,.60));ax=f.add_axes([.10,.45,.80,.25]);cb=f.colorbar(ScalarMappable(norm=im.norm,cmap=im.cmap),alpha=im.get_alpha(),cax=ax,orientation='horizontal',ticks=[1e-5,1e-3,1e-2]);cb.set_label('Goal probability (same scale for B0 and B7)',fontsize=7,labelpad=2);cb.ax.tick_params(labelsize=7);cb.solids.set_rasterized(False)
 entries.append({'name':'shared_belief_colorbar','files':save(f,out,'shared_belief_colorbar','tight')});plt.close(f)
 manifest={'episode_id':E['episode_id'],'model_identity':E['model_identity'],'checkpoint_sha256':E['checkpoint_sha256'],'dataset_sha256':E['dataset_sha256'],'source_predictions_sha256':E['source_predictions_sha256'],'source_evidence_sha256':sha(root/'OVERVIEW_EVIDENCE.json'),'instruction_verbatim':E['instruction_verbatim'],'outputs_are_independent_panels':True,'combined_overview_generated':False,'navigation_panel_contains_observation_inset':False,'shared_color_norm':{'vmin':im.norm.vmin,'vmax':im.norm.vmax,'alpha':im.get_alpha(),'type':'log'},'metrics':E['metrics'],'entries':entries,'native_observation':{'file':raw.name,'sha256':sha(raw),'size':[384,384],'pixel_exact':True},'upload_scope':'User-requested private paper repository research assets only; public image redistribution rights are not asserted.','sources':'OVERVIEW_EVIDENCE.json + source/overview_fields.npz + source/map.json + SHA-matched original raster and observation NPZ','command':'python scripts/draw_overview.py --include-rgb --separate-panels --output-dir ../local_only/overview_panels_rgb','scripts':{n:sha(Path(__file__).parent/n) for n in ['draw_overview.py','export_rgb_panels.py']}}
 (out/'RGB_PANELS_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (out/'README.md').write_text('''# Separate real RGB panels — private research assets

Episode: `birmingham_block_5:8:4`, Revised CityNav, formal frozen C epoch1. No independent CVR. Original instruction: “This is a large building just besides a large dirt track on Aldridge Road.”

Each panel is a separate PDF/PNG/editable SVG. No combined overview is generated here. Navigation excludes the former observation inset: the actual input is supplied separately as `observation_step7.*`. `observation_step7_native_384.png` retains the exact384×384 RGB pixels (no resizing or annotations). The shared probability colorbar is also separate.

- `panel_a_language_search_rgb.*`: actual scene, start, referenced road, candidates and analysis-only GT.
- `panel_b_geometric_belief_rgb.*`: same scene, B0 and measured road relationship.
- `panel_c_navigation_rgb.*`: same scene, B7, executed path, current support, goal and final pose.
- `observation_step7.*`: Pose-conditioned Orthophoto Observation, not perspective FPV.
- `shared_belief_colorbar.*`: common probability normalization for B0/B7.

Coordinates, arrays and source images are unchanged. The display crop is x560–800m,y830–1070m. Final NE11.9533m, belief L1 change0.026204, unchanged Top-1. ROI support is not semantic visibility. The model uses fixed s0 plus a residual, not recursive probability accumulation.

Source: CityNav / SensatUrban orthophoto and CityNav / OpenStreetMap geographic geometry. Map attribution: https://www.openstreetmap.org/copyright . Imagery is cropped and annotated; all modifications are display-only. Underlying RGB public redistribution and publication clearance remain unconfirmed. These derived panels are uploaded only to the private paper repository under the user's explicit instruction; this does not establish or grant public image rights. No full source raster, model checkpoint or large raw dataset is included.

See `RGB_PANELS_MANIFEST.json` for per-file hashes, exact episode/step and source provenance; `../OVERVIEW_EVIDENCE.json` holds source paths and transforms. Rebuild in the lab from the experiment overview directory with the command in the manifest. Public code is retained without RGB assets in the experiment repository.
''')
 print(out)
