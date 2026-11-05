"""Recompute every swapped ROI/token/candidate mask on cached real RGB, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.polygon_rgb_matching import candidate_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','nearfield','depth-root','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();rows=list(map(json.loads,(a.run/'results.jsonl').open()));count=0;real=0;same_sparse=0
    for row in rows:
        if not row['candidates']:continue
        image=cv2.imread(str(a.nearfield/'frames'/f'{row["key"]}.png'))
        depth=np.load(a.depth_root/'frames'/f'{row["key"]}.npz')['depth_valid']
        old=json.loads((a.nearfield/'frames'/f'{row["key"]}.json').read_text())['anchor_audit']
        valid=np.array(old['valid']);uv=np.array(old['local_uv'])
        obs=dict(bgr=image,depth_valid=depth,footprint_world=np.array(row['footprint_world']))
        A=row['candidates'][0]
        for c in row['candidates']:
            recomputed=candidate_inputs(c['contours'],obs)
            assert hashlib.sha256(recomputed['anchor']['geometry_mask'].tobytes()).hexdigest()==c['roi_mask_sha256']
            np.testing.assert_allclose(recomputed['tokens'],c['polygon_tokens'],atol=1e-7,rtol=0)
            expected=uv[(recomputed['anchor']['labels']==1)&valid]
            np.testing.assert_allclose(expected,c['positive_candidate_uv'],atol=1e-7,rtol=0)
            if c['family']!='correct':
                assert .5-1e-8<=c['local_area_m2']/A['local_area_m2']<=2+1e-8
                assert abs(c['visible_fraction']-A['visible_fraction'])<=.2+1e-8
                assert c['roi_mask_sha256']!=A['roi_mask_sha256']
                if c['positive_candidate_uv']==A['positive_candidate_uv']:same_sparse+=1
            if c['family']=='real_other_instance':
                assert not set(c['public_ids'])&set(A['public_ids']);real+=1
            count+=1
        if row['reference']:
            assert row['reference']['step']<row['step']
            assert row['reference']['rgb_sha256']!=row['rgb_sha256']
    result=dict(candidate_evaluations=count,real_negative_candidates=real,all_recomputed_ROI_SHA_match=True,
        all_shape_tokens_match=True,all_positive_candidate_regions_match=True,all_negative_area_visibility_matched=True,
        all_references_strict_past_and_distinct_RGB=True,
        changed_pixel_ROI_but_same_sparse_positive_centers=same_sparse,
        note='Sparse centers can remain unchanged for different pixel masks; regions are recomputed, not reused.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
