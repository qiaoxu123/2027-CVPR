"""Post-rollout source DEM clearance proxy; does not change navigation scoring."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import rasterio
from scripts.summarize_closedloop import read_run
from graphnav.checkpointing import provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',required=True,help='name=completed run')
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing audit overwrite')
    runs={};result={}
    for spec in a.run:
        name,path=spec.split('=',1);report,rows,manifest=read_run(path)
        runs[name]=rows;result[name]=dict(episodes=len(rows),valid_points=0,unknown_points=0,source_initial_below_surface=0,later_below_surface_points=0,later_below_surface_episodes=0,clearance_values=[],manifest_sha256=manifest,run=path)
    for mapname in sorted({r['map_name'] for rows in runs.values() for r in rows.values()}):
        rgb=cv2.imread(str(Path(a.data_root)/'rgbd'/(mapname+'.png')))
        if rgb is None:raise IOError('missing source RGB: '+mapname)
        with rasterio.open(Path(a.data_root)/'rgbd'/(mapname+'.tif')) as raster:
            for name,rows in runs.items():
                for row in rows.values():
                    if row['map_name']!=mapname:continue
                    poses=np.asarray(row['poses']);height=np.array([float(v[0]) for v in raster.sample(poses[:,:2])])
                    ii,jj=rasterio.transform.rowcol(raster.transform,poses[:,0],poses[:,1]);ii=np.asarray(ii);jj=np.asarray(jj)
                    valid=np.isfinite(height)&(height>=0)&(ii>=0)&(ii<rgb.shape[0])&(jj>=0)&(jj<rgb.shape[1])
                    support=np.zeros(len(height),bool);support[valid]=rgb[ii[valid],jj[valid]].max(-1)>0;valid&=support
                    clearance=poses[:,2]-height;bad=valid&(clearance<0);d=result[name]
                    d['valid_points']+=int(valid.sum());d['unknown_points']+=int((~valid).sum());d['source_initial_below_surface']+=int(bad[0]);d['later_below_surface_points']+=int(bad[1:].sum());d['later_below_surface_episodes']+=int(bad[1:].any());d['clearance_values'].extend(clearance[valid].tolist())
    for d in result.values():
        values=d.pop('clearance_values');d['clearance_min_m']=min(values) if values else None;d['clearance_p05_m']=float(np.quantile(values,.05)) if values else None
    report=dict(models=result,provenance=provenance(),note='Private post-rollout nearest source DEM sample with nonblack RGB support. Pointwise clearance proxy; source grid/annotation errors and unknown pixels are explicit. Not swept-path collision, semantic ground truth or proof of physical flight safety. Does not change policy, inputs, success, SR/SPL or source data.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
