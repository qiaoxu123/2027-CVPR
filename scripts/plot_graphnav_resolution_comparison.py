"""Replay initial public observation at both real resolutions and align to RGB."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from graphnav.workspace_paths import current_paths,PROJECT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(PROJECT/'Ablations/GraphNav/diagnostics_20261013'))
    parser.add_argument('--output')
    args=parser.parse_args();root=Path(args.root);out=Path(args.output) if args.output else root/'resolution_visualization'
    if out.exists():raise ValueError('Refusing to overwrite replay')
    out.mkdir(parents=True)
    settings=current_paths(json.loads((root/'heatmap_28/metrics.json').read_text())['args'])
    sys.path.insert(0,settings['sbf_source'])
    from sbfnav.config import load_config
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from graphnav.closedloop import PublicEpisode,PublicCatalog
    from graphnav.dynamic_runtime import NativeObservationBuilder
    from graphnav.stage1_runtime import GraphNavStage1Policy
    from graphnav.diagnostic_fields import install_field_capture
    from graphnav.landmark_projection import frame_to_world
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=Path(settings['data_root']);config=load_config(settings['config'])
    catalog=CityReferCatalog(data/'cityrefer');dataset=CityNavDataset(data,'val_seen',version=config['data']['version'])
    row=json.loads((root/'heatmap_28/predictions.jsonl').read_text().splitlines()[0]);source=dataset[row['index']]
    public=PublicEpisode(source.map_name,source.instruction,source.start_pose.copy(),tuple(catalog.referenced_landmarks(source)),tuple(catalog.referenced_landmark_names(source)))
    builder=PlanningStateBuilder(data,PublicCatalog(catalog),canvas_size=config['model']['canvas_size'],field_size=config['model']['field_size'],max_height_above_ground_m=config['state']['max_height_above_ground_m'],trajectory_radius_m=config['state']['trajectory_radius_m'])
    builder=NativeObservationBuilder(builder,data/'rgbd',out/'observations')
    torch.manual_seed(0);np.random.seed(0)
    policy=GraphNavStage1Policy(settings['checkpoint'],out/'public_inputs')
    arrays=[]
    for grid_size in (28,384):
        policy.args.grid_size=grid_size
        policy.set_episode(public,builder.transform(public.map_name),catalog)
        remove=install_field_capture(policy,out/str(grid_size))
        state=builder.build_from_poses(public,np.asarray([public.start_pose]))
        prediction=policy.predict(state,public.instruction)
        remove()
        array=dict(np.load(next((out/str(grid_size)).glob('*.npz'))));arrays.append(array)
        # Verify that the replay follows the measured resolution intervention.
        measured=json.loads((root/f'heatmap_{grid_size}/predictions.jsonl').read_text().splitlines()[0])
        np.testing.assert_allclose(prediction.diagnostics['belief_top16_world_xy'],measured['step_diagnostics'][0]['belief_top16_world_xy'],atol=1e-4,rtol=0)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    summaries=[]
    for axes_row,array in zip(axes,arrays):
        g=int(array['grid_size']);rgb=array['bgr'][...,::-1];grid=array['world_xy'].reshape(g,g,2)
        logits=array['belief_logits'].reshape(g,g)
        axes_row[0].imshow(rgb);axes_row[0].set_title('Actual observed RGB 384×384')
        heat=axes_row[1].pcolormesh(grid[...,0],grid[...,1],logits-logits.max(),shading='nearest',cmap='magma',vmin=-15,vmax=0)
        quad=array['footprint_world'];closed=np.concatenate([quad,quad[:1]])
        axes_row[1].plot(closed[:,0],closed[:,1],color='cyan',label='Observed footprint')
        axes_row[1].set_aspect('equal');axes_row[1].set_title(f'Actual global model grid {g}×{g}')
        fig.colorbar(heat,ax=axes_row[1],label='Belief logit relative to global max')
        x,y=np.meshgrid(np.arange(384,dtype=np.float32),np.arange(384,dtype=np.float32))
        world=frame_to_world(np.stack([x.ravel(),y.ravel()],-1),quad,rgb.shape)
        sx=float(grid[0,1,0]-grid[0,0,0]);sy=float(grid[1,0,1]-grid[0,0,1])
        mx=((world[:,0]-grid[0,0,0])/sx).reshape(384,384).astype(np.float32)
        my=((world[:,1]-grid[0,0,1])/sy).reshape(384,384).astype(np.float32)
        local=cv2.remap(logits,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float(logits.min()))
        axes_row[2].imshow(rgb)
        axes_row[2].imshow(local-local.max(),cmap='magma',alpha=.55,vmin=-15,vmax=0)
        axes_row[2].set_title('Photo-aligned display (bilinear resampling)')
        for ax in (axes_row[0],axes_row[2]):ax.axis('off')
        summaries.append(dict(grid_size=g,cells=g*g,grid_spacing_world_m=[sx,sy],photo_shape=list(rgb.shape),actual_field_shape=list(logits.shape),matches_measured_top16=True))
    fig.suptitle(public.map_name+' / initial observed fields; labels unused')
    fig.savefig(out/'resolution_comparison.png',dpi=160);plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(cases=summaries,index=row['index'],checkpoint=settings['checkpoint'],note='Model fields are genuinely evaluated on 28/384 grids. RGB overlays are only geometry-aligned bilinear displays; no extra image detail or semantic resolution is invented.'),indent=2))
    print(out/'resolution_comparison.png')


if __name__=='__main__':main()
