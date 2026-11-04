"""Post-rollout public-Polygon/coordinate audit; never imported by navigation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely
from graphnav.landmark_projection import world_to_frame, frame_to_world


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(s) for s in (a.run/'predictions.jsonl').read_text().splitlines()]
    tasks={t['index']:t for t in json.loads((a.selection/'tasks_public.json').read_text())}
    details=[]; manifests=[]
    for r in rows:
        region=unary_union([Polygon(c) for c in tasks[r['index']]['public_polygons']])
        for t,d in enumerate(r['step_diagnostics']):
            directory=a.run/'decisions'/f'{r["index"]:02d}_{r["method"]}'
            path=directory/f'step_{t:02d}.npz'
            data=np.load(path)
            bounds=data['bounds']; fp=data['footprint_world']; pose=data['pose']
            v=(np.arange(28)+.5)/28; yy,xx=np.meshgrid(v,v,indexing='ij')
            xy=np.stack([xx.ravel(),yy.ravel()],axis=-1)*(bounds[2:]-bounds[:2])+bounds[:2]
            mask=shapely.distance(shapely.points(xy),region)<=20
            def prob(x):
                out=np.exp(x.astype(float)-x.max()); return out/out.sum()
            before=prob(data['original_logits']); after=prob(data['final_logits'])
            delta=data['final_logits'].astype(float)-data['original_logits'].astype(float)
            pts=np.vstack([fp,fp.mean(0)])
            pixels=world_to_frame(pts,fp,(384,384,3))
            reconstructed=frame_to_world(pixels,fp,(384,384,3))
            before_correct=bool(mask[int(data['original_candidate_ids'][0])])
            after_correct=bool(mask[int(data['final_candidate_ids'][0])])
            record=dict(visual_corrected_goal=int(not before_correct and after_correct),visual_damaged_goal=int(before_correct and not after_correct),index=r['index'],method=r['method'],step=t,
                public20_mass_before=float(before[mask].sum()),public20_mass_after=float(after[mask].sum()),
                public20_mass_delta=float((after-before)[mask].sum()),
                max_abs_delta=float(abs(delta).max()),geometry_logit_range=float(np.ptp(data['original_logits'])),
                world_pixel_roundtrip_max_m=float(np.linalg.norm(pts-reconstructed,axis=1).max()),
                footprint_pose_center_distance_m=float(np.linalg.norm(fp.mean(0)-pose[:2])),
                recorded_pose_matches=bool(np.allclose(pose,r['poses'][t])),
                zero_delta=bool(np.all(delta==0)),
                evidence_cells=int(data['pgvb_covered'].sum()) if 'pgvb_covered' in data else 0,
                **{k:d.get(k) for k in ('landmark_tracking_state','landmark_visual_gain','stop_probability','predicted_goal_distance')})
            details.append(record)
            for artifact in (path,path.with_suffix('.png')):
                manifests.append(dict(path=str(artifact.relative_to(a.run)),bytes=artifact.stat().st_size,
                    sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
    out=dict(frames=len(details),all_recorded_poses_match=all(d['recorded_pose_matches'] for d in details),
        max_world_pixel_roundtrip_m=max(d['world_pixel_roundtrip_max_m'] for d in details),
        max_footprint_pose_center_distance_m=max(d['footprint_pose_center_distance_m'] for d in details),
        methods={m:dict(frames=sum(d['method']==m for d in details),
            corrected_goal_decisions=sum(d['visual_corrected_goal'] for d in details if d['method']==m),
            damaged_goal_decisions=sum(d['visual_damaged_goal'] for d in details if d['method']==m),
            mass_increase=sum(d['method']==m and d['public20_mass_delta']>1e-8 for d in details),
            mass_decrease=sum(d['method']==m and d['public20_mass_delta']< -1e-8 for d in details),
            max_abs_delta=max(d['max_abs_delta'] for d in details if d['method']==m),
            geometry_logit_range_median=float(np.median([d['geometry_logit_range'] for d in details if d['method']==m])))
            for m in sorted({d['method'] for d in details})})
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'coordinate_evidence_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    (a.output/'step_evidence.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in details))
    (a.output/'raw_artifacts_manifest.json').write_text(json.dumps(dict(server_root=str(a.run),artifacts=manifests),indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
