"""Gated A4/A5 runner; observed pixels only, same graph/grid/eligible memory.

This is intentionally not launched when A2's pre-registered Dev gate fails.
It offers fixed pixel-color and frozen same-checkpoint image/text scoring.
Cosine similarities remain Unknown until a threshold is supported on Dev.
No candidate lacks evidence merely because it has not been observed.
"""
import argparse,json,hashlib,time
from pathlib import Path
import cv2
import numpy as np
import torch
from .run import OUT,load,row,aggregate,dump_rows
from .objects import ObjectGraph
from .pipeline import forward,model_inputs,reconstruct
from .language import parse
from .rgb import CausalAttributes,frozen_phrase_evidence
from .reasoning import mix
from experiments.semantic_polygon_grounding.audit import DATA,save
from graphnav.landmark_projection import project_landmark
from graphnav.renderer import CityNavObservedViewRenderer

class FrozenAttributes:
    """Official frozen encoder supplied externally; no hidden model downloads.

    Image keys include pixels and model fingerprint. Raw and foreground modes
    therefore cannot accidentally reuse another crop's feature cache.
    """
    def __init__(self,encoder,fingerprint,*,mode='polygon',threshold=None):
        if mode not in ('polygon','foreground'):raise ValueError('ROI mode')
        self.encoder=encoder;self.fingerprint=fingerprint;self.mode=mode;self.threshold=threshold
        self.images={};self.texts={};self.nodes={};self.last_step=-1
    def observe(self,obs,graph,step,cutoff,*,control='real'):
        if step>cutoff or step<=self.last_step:raise ValueError('noncausal frozen RGB update')
        self.last_step=step;eligible=CausalAttributes();eligible.observe(obs,graph,step,cutoff,control='real')
        rows=[]
        for i,attr in eligible.nodes.items():
            roi=project_landmark(graph.objects[i],obs['footprint_world'],obs['bgr'].shape,min_visible=.9,min_pixels=256)
            image=obs['bgr'].copy()
            if self.mode=='foreground':
                mask=cv2.erode(roi.mask,np.ones((5,5),np.uint8));image[mask==0]=0
            a,b,c,d=roi.bbox_xyxy;rows.append((i,image[b:d,a:c].copy(),attr['quality']))
        if control=='black':rows=[(i,np.zeros_like(image),q) for i,image,q in rows]
        elif control=='shuffled':
            crops=[image for _,image,_ in rows];rows=[(i,cv2.resize(crops[(k-1)%len(crops)],(image.shape[1],image.shape[0])),q) for k,(i,image,q) in enumerate(rows)]
        elif control!='real':raise ValueError('control')
        missing=[];keys=[]
        for _,image,_ in rows:
            h=hashlib.sha256(self.fingerprint.encode()+self.mode.encode()+np.asarray(image.shape,np.int64).tobytes()+image.tobytes()).hexdigest();keys.append(h)
            if h not in self.images:missing.append((h,image))
        if missing:
            vectors=np.asarray(self.encoder.images([image for _,image in missing]),np.float32)
            if not np.isfinite(vectors).all():raise ValueError('nonfinite frozen vectors')
            for (h,_),v in zip(missing,vectors):self.images[h]=v
        for (i,image,q),key in zip(rows,keys):
            if i not in self.nodes or q>self.nodes[i]['quality']:self.nodes[i]=dict(vector=self.images[key],quality=q,step=step,control=control)
    def evidence(self,entity,count):
        result=np.zeros(count)
        # No asserted safe acceptance means all evidence must abstain.
        if self.threshold is None or not self.nodes or entity['unknown']:return result
        words=[*entity['colors'],entity['shape'] or '',entity['object_type'] or '']
        phrase='a '+ ' '.join(x for x in words if x)
        if not entity['colors'] and not entity['shape']:return result
        if phrase not in self.texts:self.texts[phrase]=np.asarray(self.encoder.texts([phrase]),np.float32)[0]
        ids=list(self.nodes);scores,use=frozen_phrase_evidence(np.stack([self.nodes[i]['vector'] for i in ids]),self.texts[phrase],np.ones(len(ids),bool),[self.nodes[i]['quality'] for i in ids])
        for i,score,valid in zip(ids,scores,use):
            if valid and score>=self.threshold:result[i]=float(score)
        return result

def evaluate(states,labels,graphs,*,weight,control,model=None,device='cpu',encoder=None,fingerprint=None,threshold=None,mode='polygon'):
    renderer=CityNavObservedViewRenderer(DATA/'rgbd',OUT/'rgb_eval',image_size=384);rows=[];cost=[]
    for state in states:
        m=state['metadata'];graph=graphs[m['map_name']];start=time.perf_counter()
        memory=FrozenAttributes(encoder,fingerprint,threshold=threshold,mode=mode) if encoder is not None else CausalAttributes()
        by_step={}
        for r in sorted(state['regions'],key=lambda r:(r['source_step'],r['view_key'])):
            if r['source_step']>m['cutoff_step']:raise ValueError('future source region')
            by_step.setdefault(r['source_step'],r)
        for step,r in sorted(by_step.items()):
            obs=renderer.current_rgbd(m['map_name'],r['actual_camera_pose'],r['ground_level']);memory.observe(obs,graph,step,m['cutoff_step'],control=control)
        lang=parse(m['instruction'],graph);evidence={i:memory.evidence(entity,len(graph.ids)) for i,entity in enumerate(lang['entities'])}
        result=forward(graph,m['instruction'],state['public'],method='A4',weight=weight,visual_evidence=evidence[0],reference_visual={i:v for i,v in evidence.items() if i})
        logits=result['logits']
        if model is not None:
            with torch.no_grad():
                p,selected=model_inputs(graph,result,device);out=model(p);full=reconstruct(result,selected,out['delta'])
            result['object_belief']=full['object_belief'].cpu().numpy();logits=mix(state['public']['static_logits'],full['heatmap'].cpu().numpy(),weight)
        output=row(state,labels[m['sample_id']],graph,result,logits);output.update(observed_nodes=len(memory.nodes),eligible_source_steps=list(by_step),control=control,seconds=time.perf_counter()-start);rows.append(output);cost.append(output['seconds'])
    return rows,dict(seconds=sum(cost),mean_ms=float(1000*np.mean(cost)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','dev'],default='dev');args=p.parse_args()
    result=json.loads((OUT/'metrics.json').read_text())
    if not result['stages']['geometry_dev_gate']['passed']:raise SystemExit('STOP: pre-registered geometry Dev gate failed. Do not run A4/A5.')
    graphs={name:ObjectGraph.restore(torch.load(path,weights_only=True)) for path in (OUT/'graphs').glob('*.pt') for name in [path.stem]}
    states,labels,_=load(args.split);weight=result['selected']['A2']
    for control in ('real','black','shuffled'):
        rows,cost=evaluate(states,labels,graphs,weight=weight,control=control);dump_rows(OUT/f'{args.split}_pixel_{control}.jsonl.gz',rows);save(OUT/f'{args.split}_pixel_{control}_metrics.json',dict(metrics=aggregate(rows),timing=cost,supervision='weak measured color, not human attributes'))

if __name__=='__main__':main()
