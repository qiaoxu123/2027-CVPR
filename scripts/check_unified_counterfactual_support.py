"""Public-grid feasibility audit; retain all64 raw direction interventions."""
import argparse,json
from pathlib import Path
import numpy as np
from graphnav.checkpointing import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--counterfactual',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cf=json.load(open(a.counterfactual));centers=np.array([np.array(s['landmarks'][0]['contour']).mean(0) for s in cf['source_samples']])
    mask={'NS':(centers[:,1]>1/56)&(centers[:,1]<55/56),'EW':(centers[:,0]>1/56)&(centers[:,0]<55/56)}
    result=dict(all_raw_samples_retained=64,GT_used=False,feasible_counts={k:int(v.sum()) for k,v in mask.items()},infeasible_sample_indices={k:np.flatnonzero(~v).tolist() for k,v in mask.items()},methods={})
    for method,heads in cf['methods'].items():
        result['methods'][method]={}
        for head,v in heads.items():
            rows=v['per_sample'];stats={}
            for axis,direction,mass in [('NS','north_south_centroid_shift_m','north_mass_response'),('EW','east_west_centroid_shift_m','east_mass_response')]:
                accepted=[row for row,ok in zip(rows,mask[axis]) if ok]
                stats[axis]=dict(n=len(accepted),centroid_response=float(np.mean([x[direction]>1e-5 for x in accepted])),mass_response=float(np.mean([x[mass]>1e-7 for x in accepted])))
            result['methods'][method][head]=stats
    atomic_json(result,a.output)

if __name__=='__main__':main()
