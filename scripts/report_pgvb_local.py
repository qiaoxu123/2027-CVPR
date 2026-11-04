"""Package sealed local-Anchor experiments; does not train or select models."""
import argparse,json,math,shutil,html
from pathlib import Path
import numpy as np
import torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.landmark_projection import world_to_frame


def finite(x):
    if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
    if isinstance(x,list):return [finite(v) for v in x]
    return None if isinstance(x,float) and not math.isfinite(x) else x

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run_root;o=a.output;o.mkdir(parents=True,exist_ok=True)
    for src,dst in [('data/manifest.json','data_manifest.json'),('pilot_s0/config.json','config.json'),('pilot_s0/freeze.json','freeze.json'),('pilot_s0/history.json','history.json'),('smoke_s0/raw_gpu_smoke.json','raw_gpu_smoke.json')]:shutil.copyfile(r/src,o/dst)
    for folder in ['validation48_fast','content_probe']:
        shutil.copytree(r/folder,o/folder,dirs_exist_ok=True)
    logs=o/'logs';logs.mkdir(exist_ok=True)
    for name in ['prepare','smoke','resume','train','evaluate','content_probe']:
        for ext in ['log','sh','exit']:
            f=r/f'{name}.{ext}'
            if f.exists():shutil.copyfile(f,logs/f.name)
    e=o/'evaluation';e.mkdir(exist_ok=True)
    for name in ['metrics.json','predictions.json','same_image_pairs.json','cases.json']:
        v=json.loads((r/'evaluation'/name).read_text())
        if name=='predictions.json':v=[x for x in v if x['subset']=='dev']
        (e/name).write_text(json.dumps(finite(v),ensure_ascii=False,indent=2,allow_nan=False))
    data=torch.load(r/'data/cache.pt',weights_only=True,map_location='cpu');maps=torch.load(r/'evaluation/visual_maps.pt',weights_only=True,map_location='cpu');rows=json.loads((e/'predictions.json').read_text());lookup={(x['index'],x['control']):x for x in rows}
    added=[dict(index=i,query_key=x['query_key'],subset=x['subset'],type=x['type'],local_anchor=x['local_anchor'],image_sha256=data['frames'][x['frame_id']]['frame']['image_sha256']) for i,x in enumerate(data['records']) if not x['legacy_common']]
    (o/'added_samples.json').write_text(json.dumps(added,indent=2))
    cases=json.loads((e/'cases.json').read_text())
    for n,c in enumerate(cases):
        i=c['index'];rec=data['records'][i];f=data['frames'][rec['frame_id']];q=data['queries'][rec['query_key']];im=cv2.cvtColor(cv2.imread(f['image_path']),cv2.COLOR_BGR2RGB);fig,axes=plt.subplots(1,4,figsize=(16,4));axes[0].imshow(im)
        for poly in q['contours']:
            xy=world_to_frame(np.asarray(poly),f['frame']['footprint_world'],im.shape);axes[0].plot(xy[:,0],xy[:,1],color='cyan',lw=1)
        axes[0].set_xlim(0,383);axes[0].set_ylim(383,0);axes[0].set_title('RGB + public Polygon (proxy)')
        controls=['old_legacy','real','zero_rgb'];vmax=max(float(maps[(i,k)]['probability'].max()) for k in controls);b=f['bounds']
        for ax,k in zip(axes[1:],controls):
            v=maps[(i,k)];z=np.where(v['covered'].numpy(),v['probability'].numpy(),np.nan).reshape(28,28);ax.imshow(z,origin='lower',extent=[b[0],b[2],b[1],b[3]],vmin=0,vmax=vmax,cmap='magma')
            for poly in q['contours']:
                xy=np.asarray(poly);ax.plot(xy[:,0],xy[:,1],'c-',lw=.6)
            rr=lookup[(i,k)];ax.scatter(*rr['world_xy'],c='red',s=15);ax.set_xlim(b[0],b[2]);ax.set_ylim(b[1],b[3]);ax.set_title(f'{k}: NE={rr["ne"]:.1f}m')
        fig.suptitle(q['name']+' | geometry proxy; visibility unknown');fig.tight_layout();fig.savefig(e/c['file'],dpi=130);plt.close(fig);cv2.imwrite(str(e/f'case_{n}_rgb.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    h=json.loads((o/'history.json').read_text());fig,ax=plt.subplots(1,3,figsize=(13,4));ax[0].bar(['Old fit','New fit','Old Dev','New Dev'],[6265,6275,699,701]);ax[0].set_title('Retained + added queries')
    ax[1].plot([x['epoch'] for x in h],[x['total'] for x in h]);ax[1].set_title('TRAIN loss (fixed 5112 updates)')
    xx=np.arange(2);ax[2].bar(xx-.18,[42.60,56.93],.36,label='Old');ax[2].bar(xx+.18,[92.83,90.97],.36,label='Local Anchor');ax[2].set_xticks(xx,['Building','Road']);ax[2].set_ylabel('Polygon R20 (%)');ax[2].legend();fig.tight_layout();fig.savefig(o/'summary.png',dpi=160);plt.close(fig)
    page='<meta charset="utf-8"><title>PGVB局部Anchor实验</title><style>body{max-width:1400px;margin:auto;font:16px sans-serif}img{max-width:100%}a{margin:8px}</style><h1>PGVB局部Anchor：几何代理诊断</h1><p>无人工像素Gold；真实可见性未知。未运行导航。</p><a href="REPORT_zh.md">中文报告</a><a href="evaluation/metrics.json">完整指标</a><img src="summary.png">'
    for n,c in enumerate(cases):page+=f'<h2>{html.escape(c["query_key"])}</h2><a href="evaluation/case_{n}_rgb.png">原生384 RGB</a><a href="evaluation/{c["file"]}"><img src="evaluation/{c["file"]}"></a>'
    page+='<h2>此前48例局部Anchor（青色/有效Mask见图例，非人工分割）</h2>'
    for n in range(48):page+=f'<a href="validation48_fast/case_{n:02d}/local_anchor.png"><img width="320" src="validation48_fast/case_{n:02d}/local_anchor.png"></a>'
    (o/'index.html').write_text(page)
if __name__=='__main__':main()
