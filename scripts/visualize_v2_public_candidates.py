"""Post-hoc TRAIN/Val report visualization only; no candidate generation or model calls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
Image.MAX_IMAGE_PIXELS=None

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--sbf-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
 ds=CityNavDataset(a.data,'val_seen',version='processed_citynav');rows=json.load(open(a.results/'original_v2_rows.json'));z=np.load(a.results/'original_v2_public_fields.npz');chosen=[]
 for predicate,n in [(lambda r:not r['raw_hits']['5'] and r['nms_hits']['5'],3),(lambda r:not r['nms_hits']['20'],2),(lambda r:r['raw_hits']['1'],1),(lambda r:not r['public_anchor_n'],1)]:chosen.extend([r for r in rows if predicate(r)][:n])
 manifest=[];a.output.mkdir(exist_ok=True,parents=True)
 for r in {r['index']:r for r in chosen}.values():
  ep=ds[r['index']];xy=z['xy'][r['index']];logits=z['logits'][r['index']];prob=np.exp(logits-logits.max());prob/=prob.sum();m=ep.map_name
  with rasterio.open(a.data/'rgbd'/(m+'.tif')) as f:b=f.bounds
  im=Image.open(a.data/'rgbd'/(m+'.png'));im.thumbnail((1000,1000));fig,ax=plt.subplots(figsize=(8,7));ax.imshow(im,extent=[b.left,b.right,b.bottom,b.top]);heat=ax.scatter(xy[:,0],xy[:,1],c=prob,cmap='magma',s=12,alpha=.6);raw=r['raw_top20'][:5];nms=r['nms_top20'][:5];ax.scatter(xy[raw,0],xy[raw,1],s=120,facecolors='none',edgecolors='orange',label='Raw top5');ax.scatter(xy[nms,0],xy[nms,1],s=100,marker='s',facecolors='none',edgecolors='cyan',label='20m NMS top5');gt=ep.target_position[:2];ax.scatter(*gt,s=130,marker='*',color='red',label='GT scoring only');ax.add_patch(plt.Circle(gt,20,color='red',fill=False));ax.set_title(f"{m} | task {r['index']} | NE={r['NE']:.1f}m\nRaw5={r['raw_hits']['5']}; NMS5={r['nms_hits']['5']}; first hit rank={r['first_correct_rank']}");ax.legend();fig.colorbar(heat,ax=ax,label='Frozen V2 Geometry probability');fig.tight_layout();fn='v2_'+str(r['index'])+'.png';fig.savefig(a.output/fn,dpi=120);plt.close(fig);manifest.append(dict(index=r['index'],episode=r['episode'],map_name=m,instruction=ep.instruction,raw5=r['raw_hits']['5'],nms5=r['nms_hits']['5'],NE=r['NE'],image=fn,usage='Orthophoto and target for post-hoc visualization only, never inference input'))
 (a.output/'v2_cases.json').write_text(json.dumps(manifest,indent=2));print('cases',len(manifest))
if __name__=='__main__':main()
