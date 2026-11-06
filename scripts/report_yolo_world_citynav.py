"""Summarize real frozen detector results; no inference or tuning."""
import argparse,json,csv,time,html
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scripts.yolo_world_citynav import family

def aggregate(rows,bank,method,category,threshold='proxy_iou50'):
    d=[r['methods'][method][threshold][category] for r in rows if r['bank']==bank]
    n=sum(x['detections'] for x in d);g=sum(x['proxies'] for x in d);t=sum(x['tp'] for x in d)
    return dict(images=len(d),detections=n,proxies=g,tp=t,proxy_precision=t/n if n else None,proxy_recall=t/g if g else None,proxy_f1=2*t/(n+g) if n+g else None,unverified=n-t)

def main():
    import cv2,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    rows=[json.load(open(f)) for f in sorted((Path(a.run)/'results').glob('*.json'))];assert len(rows)==360,len(rows)
    summary={};tables=[]
    for b in ['base','color','parking']:
        for m in ['whole','tile']:
            for c in ['vehicle','parking','building','road']:
                z=aggregate(rows,b,m,c);summary[f'{b}/{m}/{c}']=z
                if z['detections'] or z['proxies']:tables.append(dict(bank=b,method=m,category=c,**z))
    (root/'metrics.json').write_text(json.dumps(summary,indent=2))
    with open(root/'metrics.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    per=[]
    for r in rows:
        for m,v in r['methods'].items():
            for c,z in v['proxy_iou50'].items():per.append(dict(sample_id=r['sample_id'],map=r['map_name'],bank=r['bank'],method=m,category=c,width=r['width'],native_samples=r['source_native_samples'],sampling_ratio=r['render_sampling_ratio'],tiles=len(v['tiles']),seconds=v['seconds'],peak_mib=v['peak_allocated_mib'],detections=z['detections'],proxies=z['proxies'],tp=z['tp']))
    with open(root/'per_image.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    base=[r for r in rows if r['bank']=='base'];maps=sorted(set(r['map_name'] for r in base));rng=np.random.default_rng(42)
    def recall(seq,m):
        g=sum(r['methods'][m]['proxy_iou50']['vehicle']['proxies'] for r in seq);t=sum(r['methods'][m]['proxy_iou50']['vehicle']['tp'] for r in seq);return t/g if g else np.nan
    boot=[]
    for _ in range(2000):
        seq=[r for mp in rng.choice(maps,len(maps),replace=True) for r in base if r['map_name']==mp];boot.append(recall(seq,'tile')-recall(seq,'whole'))
    ci=np.nanpercentile(boot,[2.5,97.5]).tolist()
    strata={}
    for name,fn in [('map',lambda r:r['map_name']),('resolution',lambda r:'800' if r['width']==800 else '2048' if r['width']==2048 else '801–2047'),('sampling',lambda r:'upsampled_native' if r['render_sampling_ratio']>1.01 else 'native_or_downsampled'),('density',lambda r:'dense>=20' if sum(g['category']=='vehicle' and g['eligible'] for g in r['proxies'])>=20 else 'sparse<20')]:
        strata[name]={}
        for k in sorted(set(fn(r) for r in base)):
            seq=[r for r in base if fn(r)==k];strata[name][k]={m:{c:aggregate(seq,'base',m,c) for c in ['vehicle','parking','building','road']} for m in ['whole','tile']}
    small={}
    for m in ['whole','tile']:
        total=hit=0
        for r in base:
            ids={i for i,g in enumerate(r['proxies']) if g['eligible'] and g['category']=='vehicle' and (g['box'][2]-g['box'][0])*(g['box'][3]-g['box'][1])*(640/max(r['width'],r['height']))**2<32**2}
            total+=len(ids);hit+=sum(x['proxy'] in ids for x in r['methods'][m]['proxy_iou50']['vehicle']['matches'])
        small[m]=dict(proxies=total,tp=hit,proxy_recall=hit/total if total else None)
    timing={b:{m:{'mean_ms':np.mean([r['methods'][m]['seconds']*1000 for r in rows if r['bank']==b]),'median_ms':np.median([r['methods'][m]['seconds']*1000 for r in rows if r['bank']==b]),'peak_mib':max(r['methods'][m]['peak_allocated_mib'] for r in rows if r['bank']==b)} for m in ['whole','tile']} for b in ['base','color','parking']}
    extra=dict(map_bootstrap_recall_delta_CI95=ci,small_vehicle=small,strata=strata,timing=timing,aux_iou30={f'{b}/{m}/{c}':aggregate(rows,b,m,c,'proxy_iou30') for b in ['base','color','parking'] for m in ['whole','tile'] for c in ['vehicle','parking','building','road']},counts=dict(frames=len(base),maps=len(maps),projected=sum(len(r['proxies']) for r in base),eligible=sum(g['eligible'] for r in base for g in r['proxies']),unknown=sum(not g['eligible'] for r in base for g in r['proxies']),oversampled_frames=sum(r['render_sampling_ratio']>1.01 for r in base)))
    (root/'analysis.json').write_text(json.dumps(extra,indent=2))
    # Publish full-resolution JPEG previews; all original lossless PNGs remain on server.
    (root/'images').mkdir(exist_ok=True);(root/'results').mkdir(exist_ok=True)
    atlas=[]
    for r in base:
        name=r['sample_id'].replace(':','_');img=cv2.imread(r['image_path']);h,w=img.shape[:2]
        cv2.imwrite(str(root/'images'/f'{name}.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY,92])
        # A browser canvas displays all boxes/polygons interactively without raster truncation.
        data={k:r[k] for k in ['sample_id','map_name','width','height','pose','footprint','image_sha','source_native_samples','render_sampling_ratio','proxies']}
        data['banks']={rr['bank']:rr['methods'] for rr in rows if rr['sample_id']==r['sample_id']}
        (root/'results'/f'{name}.json').write_text(json.dumps(data,separators=(',',':')))
        atlas.append(dict(id=name,map=r['map_name']))
    template=r'''<!doctype html><meta charset="utf-8"><title>CityNav YOLO-World TRAIN 120</title><style>body{font:16px sans-serif;margin:20px;background:#171b22;color:#eee}canvas{max-width:100%;height:auto}select,button{margin:5px;padding:8px}pre{white-space:pre-wrap}a{color:#8cd}</style><h1>YOLO-World-V2-S · TRAIN 120</h1><p>青色：公开Polygon（几何代理，非人工Mask）；绿色：IoU≥.5代理匹配；橙色：未核验候选；灰色：几何支持不足。JPEG为原尺寸有损预览，原生PNG保留服务器。可缩放浏览器查看。</p><select id="task"></select><select id="bank"><option>base</option><option>color</option><option>parking</option></select><select id="method"><option>whole</option><option>tile</option></select><label><input id="poly" type="checkbox" checked>Polygon</label><label><input id="det" type="checkbox" checked>检测</label><button id="prev">上一条</button><button id="next">下一条</button><a id="raw">图像</a> <a id="json">JSON</a><pre id="meta"></pre><canvas id="c"></canvas><script>const tasks=TASKS;const $=x=>document.getElementById(x);let data,img;tasks.forEach(x=>$('task').add(new Option(x.id+' '+x.map,x.id)));async function load(){let id=$('task').value;data=await(await fetch('results/'+id+'.json')).json();img=new Image;img.onload=draw;img.src='images/'+id+'.jpg';$('raw').href=img.src;$('json').href='results/'+id+'.json'}function draw(){let c=$('c');c.width=data.width;c.height=data.height;let ctx=c.getContext('2d');ctx.drawImage(img,0,0);ctx.lineWidth=Math.max(2,data.width/700);let v=data.banks[$('bank').value][$('method').value];if($('poly').checked)data.proxies.forEach(g=>{ctx.strokeStyle=g.eligible?'#00ffff':'#999999';g.contours.forEach(q=>{ctx.beginPath();q.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.stroke()})});let matched=new Set(Object.values(v.proxy_iou50).flatMap(z=>z.matches.map(x=>x.detection)));if($('det').checked)v.detections.forEach((d,i)=>{let [x,y,x2,y2]=d.box;ctx.strokeStyle=matched.has(i)?'#00ff65':'#ff9900';ctx.strokeRect(x,y,x2-x,y2-y);ctx.fillStyle=ctx.strokeStyle;ctx.font=Math.max(13,data.width/95)+'px sans-serif';ctx.fillText(d.label+' '+d.score.toFixed(2),x,Math.max(15,y-3))});$('meta').textContent=data.sample_id+' | '+data.map_name+' | '+data.width+'×'+data.height+' | '+v.detections.length+' detections | '+(v.seconds*1000).toFixed(1)+'ms | '+v.tiles.length+' tiles\n'+JSON.stringify(v.proxy_iou50,null,1)}$('task').onchange=load;['bank','method','poly','det'].forEach(x=>$(x).onchange=draw);$('prev').onclick=()=>{$('task').selectedIndex=Math.max(0,$('task').selectedIndex-1);load()};$('next').onclick=()=>{$('task').selectedIndex=Math.min(tasks.length-1,$('task').selectedIndex+1);load()};load();</script>'''
    (root/'index.html').write_text(template.replace('TASKS',json.dumps(atlas)))
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
    for i,m in enumerate(['whole','tile']):ax.bar(x+i*.35,[summary[f'base/{m}/{c}']['proxy_recall'] or 0 for c in ['vehicle','parking','building','road']],.35,label=m)
    ax.set_xticks(x+.175,['vehicle','parking','building','road']);ax.set_ylabel('Public registry BBox proxy recall @ IoU .5');ax.legend();fig.tight_layout();fig.savefig(root/'proxy_recall.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(base={k:v for k,v in summary.items() if k.startswith('base')},extra={k:v for k,v in extra.items() if k!='strata'}),indent=2))
if __name__=='__main__':main()
