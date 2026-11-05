"""Read-only nearfield scene proxies, graph evidence, and a local HTML atlas."""
import argparse
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import shutil
import numpy as np
from graphnav.landmark_projection import world_to_frame
from scripts.build_nearfield_scene_dataset import CLASSES
from scripts.nearfield_geometry import enclosure_road_names
from shapely.geometry import Point, Polygon

COLORS=dict(Building='#d55e00',Road='#f0df20',Parking='#ff00ee',Vegetation='#35cd45',Car='#ffffff',Ground='#b4a37f',Other='#999999',Water='#328dff')


def metric(objects):
    visible=[o for o in objects if 'predicted_class' in o]
    if not visible:return dict(n=0,top1_map_proxy=None,top3_map_proxy=None,shuffled_map_proxy=None,zero_rgb_map_proxy=None)
    return dict(n=len(visible),top1_map_proxy=float(np.mean([o['predicted_class']==o['expected_class'] for o in visible])),
        top3_map_proxy=float(np.mean([o['expected_class'] in o['top3_classes'] for o in visible])),
        shuffled_map_proxy=float(np.mean([o['shuffled_predicted_class']==o['expected_class'] for o in visible])),
        zero_rgb_map_proxy=float(np.mean([o['zero_rgb_predicted_class']==o['expected_class'] for o in visible])))


def band(v,limits):
    for lo,hi in limits:
        if lo<=v<hi:return f'{lo}-{hi}'
    return 'other'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-figures',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'frames').mkdir(exist_ok=True);(a.output/'maps').mkdir(exist_ok=True)
    frames=list(map(json.loads,(a.dataset/'dataset.jsonl').open()));frames.sort(key=lambda r:(r['index'],r['step']))
    maps={int(f.stem):json.loads(f.read_text()) for f in (a.dataset/'maps').glob('*.json')}
    for m in maps.values():
        lookup={o['public_id']:o for o in m['neighbors']}
        lookup.update({i:dict(name=m['landmark_name']) for i in m['anchor_ids']})
        for hole in m['road_enclosures']:
            hole.update(enclosure_road_names(hole['adjacent_road_ids'],lookup))
    near=[r for r in frames if r['near']];objects=[]
    for r in near:
        objects += [{**o,'sample_key':r['key'],'task_index':r['index'],'map_name':r['map_name'],'landmark_name':r['landmark_name'],
            'height_agl_m':r['height_agl_m'],
            'uav_polygon_distance_m':float(Polygon(next(n['contour'] for n in maps[r['index']]['neighbors'] if n['public_id']==o['public_id'])).distance(Point(r['pose'][:2])))} for o in r['objects'] if o['public_id']!='__anchor__']
    known=[r for r in near if r['anchor_audit'].get('real_auc_proxy') is not None]
    def auc_mean(key):
        values=[r['anchor_audit'].get(key) for r in known];values=[v for v in values if v is not None]
        return dict(n=len(values),mean=float(np.mean(values)) if values else None)
    chains=[];enclosures=[]
    for r in near:
        obs={o['public_id']:o for o in r['objects']}
        for chain in maps[r['index']]['car_inside_parking_chains']:
            car=obs.get(chain['car_id']);parking=obs.get(chain['parking_id'])
            both=car and parking and 'predicted_class' in car and 'predicted_class' in parking
            chains.append(dict(key=r['key'],**chain,both_RGB_roi_available=bool(both),
                RGB_nodes_match_map_types_proxy=bool(both and car['predicted_class']=='Car' and parking['predicted_class']=='Parking')))
        for hole in maps[r['index']]['road_enclosures']:
            if hole['distinct_named_road_count']<3 or hole['area_m2']<25:continue
            roads=[obs.get(i) for i in hole['adjacent_road_ids']]
            available=all(o and 'predicted_class'in o for o in roads)
            enclosures.append(dict(key=r['key'],adjacent_road_ids=hole['adjacent_road_ids'],area_m2=hole['area_m2'],
                all_road_RGB_roi_available=available,all_road_node_types_match_map_proxy=bool(available and all(o['predicted_class']=='Road' for o in roads))))
    unique_objects={(o['map_name'],o['public_id']) for o in objects}
    track=[r for r in near if r['tracking_state']=='tracking']
    summary=dict(physical_tasks=64,maps=len({r['map_name'] for r in frames}),frames=len(frames),near_frames=len(near),near_tasks=len({r['index'] for r in near}),
        independent_human_instance_labels=0,true_landmark_identity_accuracy=None,true_tracking_accuracy=None,true_scene_type_accuracy=None,
        pixel_mask_IoU=None,true_direct_relation_accuracy=None,true_indirect_relation_accuracy=None,
        status_proxy_frames=dict(Counter(r['landmark_status_proxy'] for r in near)),
        tracking_frames_in_sample=len(track),tracked_near_tasks=len({r['index'] for r in track}),
        top_scored_patch_label_proxy=dict(Counter(str(r['anchor_audit']['top_patch_label']) for r in near if 'top_patch_label' in r['anchor_audit'])),RGB_supported_tracking_frames=sum(r['landmark_status_proxy']=='RGB_supported_proxy' for r in track),
        anchor_controls={k:auc_mean(k) for k in ('real_auc_proxy','shuffled_auc_proxy','zero_features_auc_proxy','wrong_name_auc_proxy')},
        same_type_wrong_name_auc_not_worse=sum(r['anchor_audit'].get('wrong_name_auc_proxy') is not None and r['anchor_audit']['wrong_name_auc_proxy']>=r['anchor_audit']['real_auc_proxy'] for r in known),
        same_type_wrong_name_auc_within_005=sum(r['anchor_audit'].get('wrong_name_auc_proxy') is not None and abs(r['anchor_audit']['wrong_name_auc_proxy']-r['anchor_audit']['real_auc_proxy'])<=.05 for r in known),
        logged_score_mae_max=max(r['anchor_audit'].get('logged_score_mae',0) for r in frames),
        neighbor_object_views=len(objects),unique_neighbor_objects=len(unique_objects),
        out_of_view_views=sum(o['status']=='out_of_view' for o in objects),
        insufficient_patch_views=sum(o['status']=='insufficient_RGB_patch_support' for o in objects),
        neighborhood_visual_class_proxy=metric(objects),
        by_type={c:metric([o for o in objects if o['expected_class']==c]) for c in CLASSES},
        by_distance={b:metric([o for o in objects if band(o['anchor_distance_m'],[(0,20),(20,50),(50,100.00001)])==b]) for b in ('0-20','20-50','50-100.00001')},
        by_uav_distance={b:metric([o for o in objects if band(o['uav_polygon_distance_m'],[(0,20),(20,50),(50,float('inf'))])==b]) for b in ('0-20','20-50','50-inf')},
        by_height={b:metric([o for o in objects if band(o['height_agl_m'],[(0,20),(20,40),(40,float('inf'))])==b]) for b in ('0-20','20-40','40-inf')},
        geometric_mask_support=dict(roi_with_pixels=sum(o['polygon_pixels']>0 for o in objects),
            mean_supported_fraction_on_projected_ROIs=float(np.mean([o['rgb_support_fraction'] for o in objects if o['polygon_pixels']>0])),
            caveat='Coverage and valid depth are geometric/source support, not semantic mask IoU or real visibility'),
        relation_proxies=dict(map_car_parking_chains=sum(len(m['car_inside_parking_chains']) for m in maps.values()),
            car_parking_chain_views=len(chains),both_RGB_rois_available=sum(c['both_RGB_roi_available'] for c in chains),
            both_node_types_match_map=sum(c['RGB_nodes_match_map_types_proxy'] for c in chains),
            qualified_named_multi_road_holes=sum(h['distinct_named_road_count']>=3 and h['area_m2']>=25 for m in maps.values() for h in m['road_enclosures']),
            exact_road_union_holes=sum(len(m['road_enclosures']) for m in maps.values()),
            qualified_multi_road_hole_views=len(enclosures),all_road_ROIs_available=sum(c['all_road_RGB_roi_available'] for c in enclosures),
            all_road_types_match_map=sum(c['all_road_node_types_match_map_proxy'] for c in enclosures),
            map_building_nearest_road_queries=sum(len(m['building_nearest_road']) for m in maps.values())),
        source='Frozen64 rollout derived analysis; exploratory, not an independent navigation/generalization evaluation')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'relation_views.json').write_text(json.dumps(dict(car_parking=chains,road_enclosures=enclosures),indent=2)+'\n')
    # All categorical comparisons use public annotations as weak references only.
    with (a.output/'object_view_predictions.csv').open('w') as f:
        fields=['sample_key','task_index','map_name','public_id','name','map_type','expected_class','status','polygon_pixels','supported_pixels','native_patch_count','whole_polygon_coverage','anchor_distance_m','uav_polygon_distance_m','height_agl_m','predicted_class','shuffled_predicted_class','zero_rgb_predicted_class','cosine_margin']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(objects)
    (a.output/'human_review_template.json').write_text(json.dumps([dict(key=r['key'],index=r['index'],step=r['step'],verified_identity=None,
        status='unreviewed',visible_landmark_mask=None,scene_labels=[],relations=[],notes='',annotator=None,evidence='pending_independent_review') for r in frames],indent=2)+'\n')
    shutil.copy2(a.dataset/'dataset.jsonl',a.output/'dataset.jsonl')
    for name in ('manifest.json','runtime.json','selection.json'):shutil.copy2(a.dataset/name,a.output/name)
    atlas=[]
    for index,m in maps.items():(a.output/'maps'/f'{index:02d}.json').write_text(json.dumps(m,ensure_ascii=False)+'\n')
    for r in frames:
        key=r['key'];m=maps[r['index']];data=np.load(a.dataset/'frames'/f'{key}.npz');bounds=data['bounds'];fp=data['footprint_world']
        for suffix in ('json','png'):shutil.copy2(a.dataset/'frames'/f'{key}.{suffix}',a.output/'frames'/f'{key}.{suffix}')
        projections=[]
        for obj in [{'public_id':'__anchor__','type':r['landmark_type'],'contour':c} for c in m['anchor_contours']]+m['neighbors']:
            polygon=np.array(obj['contour']);px=world_to_frame(polygon,fp,(384,384,3))
            if px[:,0].max()<0 or px[:,0].min()>383 or px[:,1].max()<0 or px[:,1].min()>383:continue
            projections.append(dict(id=obj['public_id'],type=obj['type'],px=px.tolist()))
        scores=[];aa=r['anchor_audit']
        if 'actual_scores'in aa:
            uv=np.array(aa['local_uv']);world=uv*(bounds[2:]-bounds[:2])+bounds[:2];px=world_to_frame(world,fp,(384,384,3))
            scores=[[*xy.tolist(),float(score)] for xy,score,valid in zip(px,aa['actual_scores'],aa['valid']) if valid]
        atlas.append(dict(key=key,index=r['index'],step=r['step'],name=r['landmark_name'],type=r['landmark_type'],status=r['landmark_status_proxy'],
            pose=r['pose'],height=r['height_agl_m'],tracking=r['tracking_state'],objects=[o for o in r['objects'] if o['polygon_pixels']>0],projections=projections,scores=scores,
            anchor_audit={k:v for k,v in aa.items() if k not in ('actual_scores','valid','labels','local_uv')},
            map_index=r['index'],footprint=r['footprint_world']))
    (a.output/'atlas_data.js').write_text('const MAPS='+json.dumps(maps,ensure_ascii=False,separators=(',',':'))+';const ATLAS='+json.dumps(atlas,ensure_ascii=False,separators=(',',':'))+';\n')
    (a.output/'index.html').write_text(HTML)
    if not a.skip_figures:make_figures(frames,maps,a.dataset,a.output,summary)
    print(json.dumps(summary,indent=2))


def make_figures(frames,maps,dataset,output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    indices=(0,1,10,13,18,21,24,27,33,34,39,43,49,50,62,63)
    chosen=[]
    for index in indices:
        group=[r for r in frames if r['index']==index]
        r=max(group,key=lambda r:sum(o['map_type']=='Parking' and 'predicted_class'in o for o in r['objects']))
        chosen.append(r);key=r['key'];m=maps[index];im=plt.imread(dataset/'frames'/f'{key}.png');fp=r['footprint_world'];data=np.load(dataset/'frames'/f'{key}.npz')
        fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
        for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.imshow(im);ax.set_xlim(0,383);ax.set_ylim(383,0)
        axes[0,0].set_title('Actual visited RGB (native 384px)')
        for obj in m['neighbors']:
            px=world_to_frame(np.array(obj['contour']),fp,im.shape);color=COLORS.get({'TrafficRoad':'Road','Footpath':'Road','Bridge':'Road','HighVegetation':'Vegetation'}.get(obj['type'],obj['type']),'#999999')
            axes[0,1].plot(px[:,0],px[:,1],color=color,alpha=.75,lw=.7)
        for contour in m['anchor_contours']:
            px=world_to_frame(np.array(contour),fp,im.shape);axes[0,1].plot(px[:,0],px[:,1],color='cyan',lw=2)
        axes[0,1].set_title('Map projections: anchor cyan; others type colors')
        aa=r['anchor_audit']
        if 'actual_scores'in aa:
            bounds=data['bounds'];uv=np.array(aa['local_uv']);xy=uv*(bounds[2:]-bounds[:2])+bounds[:2];px=world_to_frame(xy,fp,im.shape);valid=np.array(aa['valid']);scores=np.array(aa['actual_scores'])
            axes[1,0].scatter(px[valid,0],px[valid,1],c=scores[valid],cmap='coolwarm',s=45,alpha=.7)
        axes[1,0].set_title('Frozen PGVB patch scores (not segmentation)')
        ax=axes[1,1];anchor=unary_union([Polygon(c) for c in m['anchor_contours']]);cx,cy=anchor.centroid.coords[0]
        for obj in m['neighbors']:
            c=np.array(obj['contour']);color=COLORS.get({'TrafficRoad':'Road','Footpath':'Road','Bridge':'Road','HighVegetation':'Vegetation'}.get(obj['type'],obj['type']),'#999999');ax.plot(c[:,0],c[:,1],color=color,lw=.7)
        for c in m['anchor_contours']:
            c=np.array(c);ax.plot(c[:,0],c[:,1],color='cyan',lw=2)
        pp=np.vstack([fp,fp[0]]);ax.plot(pp[:,0],pp[:,1],'k--');ax.scatter(r['pose'][0],r['pose'][1],color='red',s=20)
        ax.set_xlim(cx-150,cx+150);ax.set_ylim(cy-150,cy+150);ax.set_aspect('equal');ax.set_title('Public map World XY + actual Footprint')
        fig.suptitle(f'{index}: {r["landmark_name"]} | step {r["step"]} | {r["height_agl_m"]:.1f}m AGL\n{r["tracking_state"]} / {r["landmark_status_proxy"]}',fontsize=10)
        fig.savefig(output/f'case_{index:02d}.png',dpi=110);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(14,14),constrained_layout=True)
    for ax,r in zip(axes.flat,chosen):
        im=plt.imread(dataset/'frames'/f'{r["key"]}.png');ax.imshow(im)
        for c in maps[r['index']]['anchor_contours']:
            px=world_to_frame(np.array(c),r['footprint_world'],im.shape);ax.plot(px[:,0],px[:,1],color='cyan',lw=1)
        ax.set_xlim(0,383);ax.set_ylim(383,0);ax.set_title(f'{r["index"]}: {r["landmark_type"]} / {r["height_agl_m"]:.0f}m',fontsize=9);ax.axis('off')
    fig.savefig(output/'contact_sheet.png',dpi=130);plt.close(fig)
    conf=np.zeros((8,8),int)
    for r in frames:
        if not r['near']:continue
        for o in r['objects']:
            if o['public_id']!='__anchor__' and 'predicted_class'in o:conf[CLASSES.index(o['expected_class']),CLASSES.index(o['predicted_class'])]+=1
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True);ax.imshow(conf,cmap='Blues')
    ax.set_xticks(range(8),CLASSES,rotation=40,ha='right');ax.set_yticks(range(8),CLASSES)
    for i in range(8):
        for j in range(8):ax.text(j,i,str(conf[i,j]),ha='center',va='center',fontsize=8)
    ax.set_xlabel('Frozen RGB zero-shot prediction');ax.set_ylabel('Public map annotation (proxy only)');ax.set_title('Object-view confusion: NOT true recognition accuracy')
    fig.savefig(output/'scene_proxy_confusion.png',dpi=120);plt.close(fig)


HTML='''<!doctype html><meta charset="utf-8"><title>PGVB近场场景与关系审查</title>
<style>body{font:15px system-ui;margin:20px;background:#f5f6f7}button,select{font:inherit;margin:5px}canvas{width:31%;max-width:500px;background:#222;border:1px solid #bbb}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:4px}pre{white-space:pre-wrap;max-height:350px;overflow:auto}#map{width:48%}</style>
<h1>冻结轨迹近场场景审查</h1><p>地图标签、Polygon投影和弱RGB响应不能等同真实实例识别。未独立核验的准确率均为代理或未知。<a href="REPORT_zh.md">中文报告</a></p>
<button onclick="move(-1)">上一帧</button><select id="samples" onchange="show(this.selectedIndex)"></select><button onclick="move(1)">下一帧</button>
<label><input id="overlays" type="checkbox" checked onchange="show(current)">显示公开Polygon</label>
<h2 id="title"></h2><p id="info"></p>
<canvas id="raw" width="384" height="384"></canvas><canvas id="projected" width="384" height="384"></canvas><canvas id="scores" width="384" height="384"></canvas>
<p>左：原始384px RGB；中：公开Polygon投影（青色为指定地标）；右：PGVB原始Patch评分（独立色标，并非实例分割）。</p>
<canvas id="map" width="600" height="600"></canvas><div id="objectTable"></div>
<details><summary>公开地图直接与间接关系（不是视觉关系预测）</summary><pre id="relations"></pre></details>
<details><summary>视觉反事实及证据状态</summary><pre id="evidence"></pre></details>
<h3>独立核验（本地浏览器保存，不自动当作GT）</h3><select id="review" onchange="saveReview()"><option>unreviewed</option><option>visual_identity_supported</option><option>geometry_only</option><option>insufficient_evidence</option><option>wrong_identity</option><option>unknown</option></select><input id="note" style="width:60%" placeholder="依据、疑点及配准问题" oninput="saveReview()"><button onclick="exportReview()">导出核验JSON</button>
<script src="atlas_data.js"></script><script>
let current=0;const colors={Building:'#d55e00',Road:'#f0df20',TrafficRoad:'#f0df20',Footpath:'#f0df20',Bridge:'#f0df20',Parking:'#ff00ee',Car:'#ffffff',HighVegetation:'#35cd45',Ground:'#b4a37f'};
const sel=document.getElementById('samples');ATLAS.forEach(r=>{let o=document.createElement('option');o.textContent=r.key+' '+r.name;sel.append(o)});
function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}
function line(ctx,points,color,width=1){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function move(d){show(Math.max(0,Math.min(ATLAS.length-1,current+d)))}
function show(i){current=i;sel.selectedIndex=i;const r=ATLAS[i];r.map=MAPS[r.map_index];document.getElementById('title').textContent=r.key+' '+r.name;document.getElementById('info').textContent='World pose '+JSON.stringify(r.pose)+' | AGL '+r.height.toFixed(1)+'m | '+r.tracking+' | '+r.status;
let image=new Image();image.onload=()=>{if(i!==current)return;['raw','projected','scores'].forEach(id=>{let ctx=document.getElementById(id).getContext('2d');ctx.clearRect(0,0,384,384);ctx.drawImage(image,0,0,384,384)});let c=document.getElementById('projected').getContext('2d');if(document.getElementById('overlays').checked)r.projections.forEach(p=>line(c,p.px,p.id==='__anchor__'?'cyan':(colors[p.type]||'#999'),p.id==='__anchor__'?2:1));
let sc=document.getElementById('scores').getContext('2d');let values=r.scores.map(p=>p[2]);let lo=Math.min(...values),hi=Math.max(...values);r.scores.forEach(p=>{let q=(p[2]-lo)/(hi-lo||1);sc.fillStyle=`rgba(${Math.round(255*q)},0,${Math.round(255*(1-q))},.55)`;sc.fillRect(p[0]-4,p[1]-4,8,8)})};image.src='frames/'+r.key+'.png';
let map=document.getElementById('map').getContext('2d');map.clearRect(0,0,600,600);map.fillStyle='#222';map.fillRect(0,0,600,600);let all=r.map.anchor_contours.flat();let minx=Math.min(...all.map(p=>p[0])),maxx=Math.max(...all.map(p=>p[0])),miny=Math.min(...all.map(p=>p[1])),maxy=Math.max(...all.map(p=>p[1]));let span=Math.max(maxx-minx,maxy-miny)+200;let cx=(minx+maxx)/2,cy=(miny+maxy)/2;let tr=p=>[(p[0]-cx)/span*600+300,300-(p[1]-cy)/span*600];r.map.neighbors.forEach(o=>line(map,o.contour.map(tr),colors[o.type]||'#777'));r.map.anchor_contours.forEach(c=>line(map,c.map(tr),'cyan',3));line(map,r.footprint.map(tr),'white',2);let pos=tr(r.pose);map.fillStyle='red';map.beginPath();map.arc(...pos,4,0,2*Math.PI);map.fill();
let objects=r.objects.filter(o=>o.polygon_pixels>0);document.getElementById('objectTable').innerHTML='<h3>投影进入当前RGB的对象（'+objects.length+'）</h3><table><tr><th>ID/名称</th><th>地图类别</th><th>RGB零样本类别</th><th>有效像素/Patch</th><th>到Anchor距离(m)</th></tr>'+objects.map(o=>'<tr><td>'+esc(o.public_id+' '+o.name)+'</td><td>'+esc(o.map_type)+'</td><td>'+esc(o.predicted_class||'unknown: '+o.status)+'</td><td>'+o.supported_pixels+'/'+o.native_patch_count+'</td><td>'+o.anchor_distance_m.toFixed(2)+'</td></tr>').join('')+'</table>';
document.getElementById('relations').textContent=JSON.stringify({building_nearest_road:r.map.building_nearest_road,car_inside_parking:r.map.car_inside_parking_chains,road_enclosures:r.map.road_enclosures},null,2);document.getElementById('evidence').textContent=JSON.stringify(r.anchor_audit,null,2);let rev=JSON.parse(localStorage.getItem('nearfield:'+r.key)||'{}');document.getElementById('review').value=rev.status||'unreviewed';document.getElementById('note').value=rev.note||'';
}
function saveReview(){localStorage.setItem('nearfield:'+ATLAS[current].key,JSON.stringify({key:ATLAS[current].key,status:document.getElementById('review').value,note:document.getElementById('note').value,time:new Date().toISOString(),source:'browser_review_not_automatically_verified_GT'}))}
function exportReview(){let data=ATLAS.map(r=>JSON.parse(localStorage.getItem('nearfield:'+r.key)||JSON.stringify({key:r.key,status:'unreviewed'})));let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='nearfield_reviews.json';a.click()}
show(0);
</script>'''

if __name__=='__main__':main()
