"""Warm cached inference costs, separately from rendering/encoding."""
import time, numpy as np, torch
from experiments.semantic_polygon_grounding.audit import save
from .resources import OUT
from .evaluate import materialize
from .inference import posterior

def main():
    torch.set_num_threads(1);maps=torch.load(OUT/'full_geometry/maps.pt',weights_only=True);data=torch.load(OUT/'observed/dev_public.pt',weights_only=True,mmap=True);result={}
    for method,version,hard in [('independent','geometry',False),('joint','geometry',False),('hard','geometry',True),('hsv_siglip','real',False)]:
        durations=[];sizes=[];assignments=[]
        with torch.no_grad():
            for d in data:
                p=materialize(d['versions'][version],maps[d['metadata']['map_name']],d['a0']);posterior(p,hard=hard,relations=method!='independent')
                t=time.perf_counter()
                for _ in range(5):posterior(p,hard=hard,relations=method!='independent')
                durations.append((time.perf_counter()-t)/5);sizes.append(p['target_features'].shape[0]);assignments.append(p['log_phi'].shape[0])
        result[method]=dict(families=len(durations),repetitions=5,cpu_threads=1,mean_ms=1000*float(np.mean(durations)),p50_ms=1000*float(np.median(durations)),p95_ms=1000*float(np.quantile(durations,.95)),objects_range=[min(sizes),max(sizes)],assignments_range=[min(assignments),max(assignments)])
    save(OUT/'runtime.json',dict(prepared_cpu_inference=result,excludes='graph construction, raster I/O, encoder inference and cold start; see separate stage manifests for these',gpu_training='separate completed history files'))
if __name__=='__main__':main()
