"""TRAIN-fit-only attribution of missing near-field ROI support. No rule changes."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from graphnav.checkpointing import atomic_json
from graphnav.landmark_projection import world_to_frame
from graphnav.observed_candidate_rois import CurrentCandidateROIEnvironment, camera_support_mask
from scripts.audit_reference_roi_supervision import reference_indices, reference_observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'data', 'cache', 'data-root', 'sbf-source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    labels=json.loads((a.data/'labels_private.json').read_text())
    manifest=json.loads((a.audit/'manifest.json').read_text())
    fit=[i for i in manifest['selected'] if labels[i]['split']=='fit']
    ids=sorted(np.random.default_rng(42).choice(fit,min(128,len(fit)),replace=False).tolist())
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    ds=CityNavDataset(a.data_root,'train_seen',version='processed_citynav')
    env=CurrentCandidateROIEnvironment(a.data_root/'rgbd',a.output/'observations')
    rows=[]
    for i in ids:
        episode=ds[i]
        with np.load(a.cache/'public_features'/f'{i:05d}.npz') as z:
            centers=z['candidate_world'].copy()
        for phase,step in reference_indices(episode.trajectory):
            pose=episode.trajectory[step]
            obs,rois,reason=reference_observation(env,episode.map_name,pose,GROUND_LEVEL[episode.map_name],centers)
            row=dict(index=i,phase=phase,unknown=reason,gt_in_frame=False,gt_sensor_supported=False,
                     correct_center_supported=False,correct_roi90=False,correct_visible_portion90=False)
            if reason is not None:
                rows.append(row);continue
            # Target labels only classify existing public ROI evidence, never render/select it.
            target=np.asarray(labels[i]['target_world'])
            pixel=world_to_frame(target[None],obs['footprint_world'],obs['bgr'].shape)[0]
            row['gt_in_frame']=bool(np.isfinite(pixel).all() and ((pixel>=0)&(pixel<=383)).all())
            if row['gt_in_frame']:
                x,y=np.rint(pixel).astype(int)
                row['gt_sensor_supported']=bool(obs['depth_valid'][y,x] and obs['bgr'][y,x].max()>0)
            distance=np.linalg.norm(centers-target,axis=1)
            for roi in rois:
                if distance[roi['candidate']]>20:
                    continue
                row['correct_center_supported']=True
                row['correct_roi90'] |= bool(roi['valid'])
                # Diagnostic denominator separates out-of-frame from invalid
                # pixels INSIDE the currently observable intersection.
                support=[]
                for key,size in [('native',roi['native_size']),('low',roi['low_size'])]:
                    geometric=float(camera_support_mask(roi['quad'],obs['footprint_world'],size).mean())
                    observed=roi[key+'_support']
                    support.append(observed/max(geometric,1e-9))
                row['correct_visible_portion90'] |= bool(min(support)>=.9)
            rows.append(row)
    summary={}
    keys=('gt_in_frame','gt_sensor_supported','correct_center_supported','correct_roi90','correct_visible_portion90')
    for phase in ('start','middle','end'):
        rr=[r for r in rows if r['phase']==phase]
        summary[phase]=dict(n=len(rr),unknown=sum(r['unknown'] is not None for r in rr),
                           **{key:sum(r[key] for r in rr) for key in keys},
                           gt_sensor_and_correct_center=sum(r['gt_sensor_supported'] and r['correct_center_supported'] for r in rr),
                           gt_sensor_but_rejected_roi=sum(r['gt_sensor_supported'] and r['correct_center_supported'] and not r['correct_roi90'] for r in rr))
    atomic_json(dict(selected=ids,scope='TRAIN-fit reference observations, posthoc geometry proxies only',summary=summary),a.output/'summary.json')
    atomic_json(rows,a.output/'rows.json');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
