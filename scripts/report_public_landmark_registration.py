"""Publish frozen matching results; evaluates heldout anchors, never refits using errors."""
import argparse,json,csv,hashlib,html,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import shape
import rasterio
from rasterio.warp import transform
from PIL import Image
Image.MAX_IMAGE_PIXELS=None


def writecsv(path, rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    root=a.run/'landmarks_grouped';summary=json.load(open(root/'summary.json'));rows=json.load(open(root/'matches.json'));batch=json.load(open(a.batch/'summary.json'));catalog=json.load(open(a.data/'cityrefer_original/objects.json'));old={r['map_name']:r for r in batch['rows']}
    # Immutable split audit, including adjacent parts; all touching OSM buildings must stay together.
    audits={}
    for city in summary['cities']:
        rr=[r for r in rows if r['city']==city and r['high_confidence']];fit=[r for r in rr if r['split']=='fit'];val=[r for r in rr if r['split']=='validation'];shared=set(t for r in fit for t in r['physical_tokens'])&set(t for r in val for t in r['physical_tokens']);touch=sum(shape(r['osm_polygon_en']).distance(shape(s['osm_polygon_en']))<=.5 for r in fit for s in val if r['kind']==s['kind']=='building');assert not shared and touch==0;audits[city]=dict(shared_physical_tokens=sorted(shared),cross_split_touching_buildings=touch)
    scene_rows=[]
    for m,b in sorted(old.items()):
        city=m.split('_')[0];rr=[r for r in rows if r['map_name']==m];stats=summary['cities'][city]['per_scene'][m]['auto_heldout_proxy'];scene_rows.append(dict(map_name=m,previous_grade=b['grade'],matches=len(rr),high_confidence=sum(r['high_confidence'] for r in rr),heldout_objects=stats['n'] if stats else 0,heldout_groups=stats['n_groups'] if stats else 0,proxy_rmse_m=stats['rmse_m'] if stats else None,proxy_p95_m=stats['p95_m'] if stats else None,proxy_max_m=stats['max_m'] if stats else None,new_geographic_verification=False,comment='自动身份未确认；误差仅代理' if stats else '无独立自动对应；继续待验证'))
    writecsv(out/'per_scene.csv',scene_rows)
    writecsv(out/'previous_12_unreliable.csv',[dict(map_name=r['map_name'],previous_local_max_corner_shift_m=old[r['map_name']].get('local_max_corner_shift_m'),previous_building_proxy_n=old[r['map_name']].get('building_proxy_n'),rejected_reasons=json.dumps(dict(Counter(reason for match in rows if match['map_name']==r['map_name'] for reason in match['reasons'])),ensure_ascii=False),public_matches=r['matches'],high_confidence=r['high_confidence'],proxy_rmse_m=r['proxy_rmse_m'],cause='结构匹配不可靠；具体树木/年代/重复建筑原因未独立确认，不能自动归因',action='拒绝低置信度；不批准地图输入') for r in scene_rows if r['previous_grade']=='unreliable'])
    unmatched=[];named_counts=Counter()
    for m,objects in catalog.items():
        if m not in old:continue
        for oid,obj in objects.items():
            if not obj.get('name'):continue
            typ=obj.get('object_type');named_counts[typ]+=1
            if typ=='Building' and not any(r['map_name']==m and r['kind']=='building' and r['source_id']==str(oid) for r in rows):unmatched.append(dict(map_name=m,source_id=oid,name=obj['name'],reason='no plausible name/geometry candidate within frozen public search window'))
    (out/'unmatched_buildings.json').write_text(json.dumps(unmatched,indent=2));(out/'matches.json').write_text(json.dumps(rows,separators=(',',':')));shutil.copy(root/'summary.json',out/'matching_summary.json')
    manual={}
    for city,v in summary['cities'].items():
        controls=json.load(open(a.batch/(city+'_controls_snapshot.json')))['points'];vv=[p for p in controls if p['split']=='validation'];M=np.array(v['matrix']);info=dict(n=len(vv),scope='previous manual validation points, not new independent scenes')
        if vv:
            e,n=transform('EPSG:4326','EPSG:27700',[p['lon'] for p in vv],[p['lat'] for p in vv]);xy=np.array([[p['x'],p['y']] for p in vv]);d=np.linalg.norm(xy@M[:,:2].T+M[:,2]-np.array([e,n]).T,axis=1);info.update(rmse_m=float(np.sqrt(np.mean(d*d))),p95_m=float(np.quantile(d,.95)),max_m=float(d.max()))
        manual[city]=info
    (out/'independence_audit.json').write_text(json.dumps(dict(audits=audits,existing_manual_validation_on_new_city_proposal=manual,named_public_types=dict(named_counts),unmatched_named_buildings=len(unmatched),new_independently_verified_scenes=0),indent=2))
    # Geometric paired overlays on true orthophoto. No target annotations in this tool.
    chosen=[]
    for city in ['birmingham','cambridge']:
        rcity=[r for r in rows if r['city']==city];good=sorted([r for r in rcity if r['high_confidence'] and r['kind']=='building'],key=lambda r:r['anchor_residual_m']);bad=sorted([r for r in rcity if r['high_confidence'] and r['split']=='validation' and r['representative']],key=lambda r:-r['anchor_residual_m']);rejected=sorted([r for r in rcity if not r['high_confidence']],key=lambda r:-r['confidence']);roads=[r for r in rcity if r['kind']=='road_junction' and r['high_confidence']]
        chosen.extend(good[:2]+bad[:2]+rejected[:2]+roads[:2])
    chosen=list({r['match_id']:r for r in chosen}.values());gallery=[]
    for r in chosen:
        m=r['map_name'];city=r['city'];M=np.array(summary['cities'][city]['matrix']);g=shape(r['source_polygon']);q=shape(r['osm_polygon_en']);points=np.array(g.exterior.coords) if g.geom_type=='Polygon' else np.array([r['source_anchor_xy']]);qp=np.array(q.exterior.coords) if q.geom_type=='Polygon' else np.array([r['osm_anchor_en']]);ql=(qp-M[:,2])@np.linalg.inv(M[:,:2]).T
        with rasterio.open(a.data/'rgbd'/(m+'.tif')) as f:bounds=f.bounds
        im=Image.open(a.data/'rgbd'/(m+'.png'));allp=np.concatenate([points,ql]);mn=allp.min(0)-25;mx=allp.max(0)+25;W,H=im.size;px=lambda x:(x-bounds.left)/(bounds.right-bounds.left)*W;py=lambda y:(bounds.top-y)/(bounds.top-bounds.bottom)*H;box=(int(max(0,px(mn[0]))),int(max(0,py(mx[1]))),int(min(W,px(mx[0]))),int(min(H,py(mn[1]))))
        if box[2]<=box[0] or box[3]<=box[1]:continue
        crop=im.crop(box);extent=[bounds.left+box[0]/W*(bounds.right-bounds.left),bounds.left+box[2]/W*(bounds.right-bounds.left),bounds.top-box[3]/H*(bounds.top-bounds.bottom),bounds.top-box[1]/H*(bounds.top-bounds.bottom)]
        fig,ax=plt.subplots(figsize=(7,6));ax.imshow(crop,extent=extent);ax.plot(points[:,0],points[:,1],color='#ff9c21',label='CityRefer public');ax.plot(ql[:,0],ql[:,1],color='#00deee',ls='--',label='OSM under city proposal');ax.set_title(f"{m} | {r['kind']} | {r['split']}\nscore={r['confidence']:.3f}; proxy residual={r['anchor_residual_m']:.2f}m");ax.legend();ax.set_xlabel('Local X (m)');ax.set_ylabel('Local Y (m)');fig.tight_layout();fn='cases/'+r['match_id']+'.png';fig.savefig(out/fn,dpi=120);plt.close(fig);gallery.append(dict(match_id=r['match_id'],name=r['source_name'],map=m,image=fn,split=r['split'],score=r['confidence'],residual=r['anchor_residual_m'],reasons=r['reasons']))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,(city,v) in zip(axs,summary['cities'].items()):
        z=[r for r in rows if r['city']==city and r['high_confidence'] and r['split']=='validation' and r['representative']];M0=np.array(v['initial_matrix']);before=[np.linalg.norm(np.array(r['source_anchor_xy'])@M0[:,:2].T+M0[:,2]-r['osm_anchor_en']) for r in z];after=[r['anchor_residual_m'] for r in z];ax.hist(before,bins=15,alpha=.5,label='City seed');ax.hist(after,bins=15,alpha=.5,label='Public object fit');ax.set_title(city+' automatic heldout PROXY');ax.set_xlabel('Anchor residual (m), unconfirmed identity');ax.legend()
    fig.tight_layout();fig.savefig(out/'heldout_proxy_residuals.png',dpi=130);plt.close(fig)
    v2=json.load(open(a.run/'v2/summary.json'));shutil.copy(a.run/'v2/summary.json',out/'v2_summary.json');table=[]
    for label,v in v2['metrics'].items():
        rr=json.load(open(a.run/'v2'/(label+'_rows.json')));writecsv(out/(label+'_per_task.csv'),[dict(index=r['index'],episode=json.dumps(r['episode']),map_name=r['map_name'],NE=r['NE'],first_correct_rank=r['first_correct_rank'],public_anchor_n=r['public_anchor_n'],**{'raw'+str(k):r['raw_hits'][str(k)] for k in [1,5,10,20]},**{'nms'+str(k):r['nms_hits'][str(k)] for k in [1,5,10,20]}) for r in rr]);table.append('|'+label+'|'+ '|'.join(f"{100*v[t][str(k)]:.2f}%" for t in ['raw','spatial_nms'] for k in [1,5,10,20])+f"|{v['NE']:.4f}|")
    fig,ax=plt.subplots(figsize=(7,4))
    for label,v in v2['metrics'].items():
        for key,ls in [('raw','--'),('spatial_nms','-')]:ax.plot([1,5,10,20],[100*v[key][str(k)] for k in [1,5,10,20]],marker='o',ls=ls,label=label+' '+key)
    ax.set(xlabel='Candidate budget K',ylabel='World <=20m recall (%)',title='2470 Val-Seen; frozen 20m spatial NMS');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'v2_recall.png',dpi=140);plt.close(fig)
    city_table='\n'.join('|'+city+f"|{v['matches']}|{v['high_confidence_rows']}|{v['fit_objects']} / {v['validation_objects']}|{v['fit_groups']} / {v['validation_groups']}|{v['initial_validation_proxy']['rmse_m']:.2f} → {v['optimized_validation_proxy']['rmse_m']:.2f}|{v['optimized_validation_proxy']['p95_m']:.2f}|{v['optimized_validation_proxy']['max_m']:.2f}|" for city,v in summary['cities'].items())
    report=f'''# 公开地标自动对应、城市级配准与V2候选召回

基于dce07766，seed=42；仅公开具名对象及OSM用于配准。未训练、未修改任何V2/V3权重，未运行导航或Test。完整逐场景与逐任务表见CSV；[交互图册](index.html)，公网审核 http://101.43.173.9:42024/control 。

## 1. 自动对应与拟合

34场景实际运行得到890条候选，424条通过固定启发式置信条件、仍待身份审核；466条自动拒绝。分数不是概率。建筑通过名称、平移对齐后的轮廓IoU、面积、邻域名称一致性共同筛选；OSM POI必须位于实际建筑Polygon内。道路采用同名道路的OSM共享节点与CityRefer公开道路交集，排除桥/隧道/异层交叉。建筑搜索窗100m，道路80m，不能恢复初始误差更大的对应，也不代表完整召回。

|城市|候选|高置信待审|拟合/验证对象|拟合/验证物理组|留出代理RMSE(m)|P95(m)|Max(m)|
|---|---:|---:|---:|---:|---:|---:|---:|
{city_table}

只用拟合对象RANSAC估计相似变换，阈值3m；不使用验证残差选择变换。Cambridge明显改善但仍有30m异常，Birmingham无改善，均不替换已验证变换。首次分组发现8组相邻/嵌套建筑存在跨组风险，已合并后以相同seed重新分组；没有修改匹配分数/阈值。最终共享物理ID、跨组接触建筑均为0。同一道路关联节点整组隔离。分组按物理对象而非随机角点；部分场景同时有拟合和验证对象，因此这是跨对象检验，并非整场景完全留出泛化。

**没有新确认的地理精度场景：新增0/34。** 原有Birmingham block3的5个验证点及已批准导出保留，原34场景质量为1已验证、21待验证、12不可靠。自动对应身份尚未经独立确认，其留出残差只能算匹配代理，不能充当地理真值；新候选不自动升级或导出。既有人控点在新城市提议上的事后误差见independence_audit.json，既有点不是新增独立证据，且潜在同建筑关联未经核验，不据此批准新方案。

旧12不可靠场景逐一见previous_12_unreliable.csv。可报告的是支持对象数、置信拒绝与代理残差；仅凭这些不能断言树木、年代差异或重复建筑为真实原因，具体原因标未知。道路Footprint裁剪、同名对象、POI与建筑功能不一致、数据年代是仍需审核的风险。未匹配具名建筑{len(unmatched)}个，保留排除原因；不把无支持场景强行配准。

## 2. 网页与数据安全

增加自动对应表、状态过滤、两侧Polygon、名称/形状/邻域/差距依据、物理分组及残差；身份确认、拒绝和恢复待审独立写入landmark_reviews.json，不修改人工控制点、物理分组、变换或现有导出。原Basic认证、写接口请求头、FRP均保留。公开对应JSON及自动拟合输出不含导航目标GT/指令答案。只有原已独立验证场景允许现有OSM导出；未启动OSM训练。

## 3. V2独立完整初始定位

全部2470任务计入，无公开文本匹配地标91条也保留。两个冻结Checkpoint分别报告，均用原Geometry权重、原规则场/解析、完整BERT指令、28×28原生候选；没有OSM/RGB/导航。空间NMS半径20m事先固定，按预测排序贪心去重，不用GT选点。

|Checkpoint|Raw@1|Raw@5|Raw@10|Raw@20|NMS@1|NMS@5|NMS@10|NMS@20|NE(m)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

**原始V2独立Top-5为77.37%，后续联合训练V2的Geometry为77.77%。** 原始Raw Top-20平均仅7.43个空间独立位置，热点重复确实影响Raw Top-K。这里的命中是世界位置20m覆盖，不是正确实例识别、真实可见性或导航SR。GT只在forward后评分；FullCandidateDataset的target_xy使用占位[0,0]，目标标签不进入forward。

重要协议差异：历史runtime使用按标注targetID关联的reference-name元数据；本次为确保公开盲输入，具名地标由原始指令与公开名录匹配，再调用原V2公开Polygon解析接口。不能将本次指标视为历史闭环SR的逐位复现，也不能与其他Polygon绑定不加说明混比。Checkpoint/数据SHA见v2_summary.json。

HETT来源核验状态：{v2['data_identity']}。数据来源结论以保存的provenance.json为准，不能只凭2470数量证明相同。V3既有Raw Top-1/5/20为45.59/48.79/53.20%，采用384×384插值密集候选，与V2原生28×28密度不同；本轮没有重跑V3统一NMS，不能据Raw Top-K宣布模型提升。

## 4. 决策

公开地标组合匹配可提供有价值的城市级初始化，Cambridge结果支持继续核验；尚不足以证明34场景都可靠。优先审核大残差与困难同名对象，随后引入未参与拟合的、身份已确认的跨场景物理参照；不要依靠结构代价或自动分数批准。现阶段新增长精度证明为0，保留原验证场景使用范围。

V2候选池经空间去重已有约77% Top-5世界覆盖，后续视觉验证应针对这些公开候选，在因果可观测范围内确认真实实例；不能读取整幅未来区域。OSM是否增加候选召回仍未评估，不能从配准改善推导定位收益。未启动A/B/C训练。
'''
    report += '''
## 5. 异常实例和测试

- Cambridge block10，1 Chesterton Lane（373016703fbd33023ff5）：城市提议下两侧轮廓位于不同建筑，残差30.38m；初始三点种子残差仅7.77m。名称/形状在有偏初值下可能共同支持错误实例，不能用低初始残差确认身份。原样计入47个留出对象，未根据该误差重新拟合或剔除。
- Birmingham block5，Regina Drive × Walsall Road（95e2e94d4644b428fa3c）：道路Polygon相交区与OSM共享节点相差32.55m，图上位置显著不同。道路面积重叠质心不必等于拓扑中心；仅同名拓扑不足以提供米制控制点。保留该异常，不放宽阈值。
- 旧5个人工验证点在新Birmingham城市提议上RMSE2.98m、最大4.61m，较原验证方案RMSE2.24m/最大3.74m更差，不替换原方案。

公网浏览器实际验证认证、对应表、两侧Polygon/RGB加载，JS错误0；原控制点文件SHA未变。无认证401、无写确认头403。确认/拒绝/重启恢复在隔离存储测试，不伪造真实用户审核。相关测试及最终执行记录见TESTS_zh.md。
'''
    (out/'REPORT_zh.md').write_text(report)
    cards=''.join(f'<article><h3>{html.escape(r["map"]+" / "+r["name"])}</h3><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}" width="480"></a><p>{html.escape(json.dumps(r,ensure_ascii=False))}</p></article>' for r in gallery)
    v2gallery=a.output/'cases/v2_cases.json'
    if v2gallery.exists():
        for r in json.load(open(v2gallery)):
            cards+=f'<article><h3>V2 #{r["index"]} / {r["map_name"]}</h3><a href="cases/{r["image"]}"><img loading="lazy" src="cases/{r["image"]}" width="480"></a><p>{html.escape(r["instruction"])}</p><p>事后GT仅评分；Raw5={r["raw5"]}; NMS5={r["nms5"]}; NE={r["NE"]:.2f}m</p></article>'
    payload=json.dumps([{k:r.get(k) for k in ['match_id','map_name','source_name','osm_name','kind','confidence','split','anchor_residual_m','high_confidence','reasons']} for r in rows],ensure_ascii=False).replace('</','<\\/')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>CityNav公开地标对应诊断</title><style>body{font:16px system-ui;margin:25px}article{display:inline-block;vertical-align:top;width:49%}img{max-width:100%}td{padding:6px;border:1px solid #aaa}p{overflow-wrap:anywhere}</style><h1>公开地标对应与V2候选</h1><p>自动匹配残差仅代理，不是确认地理精度。橙色CityRefer；青色OSM城市提议反投影。<a href="REPORT_zh.md">中文报告</a> · <a href="per_scene.csv">逐场景</a></p><img src="heldout_proxy_residuals.png"><img src="v2_recall.png"><p>名称/场景筛选 <input id="q"></p><table id="t"></table><script>const rows='''+payload+''';function render(){const q=document.querySelector('#q').value.toLowerCase();let t=document.querySelector('#t');t.replaceChildren();for(const r of rows.filter(r=>JSON.stringify(r).toLowerCase().includes(q))){let tr=document.createElement('tr');for(const k of ['map_name','source_name','osm_name','kind','confidence','split','anchor_residual_m','high_confidence','reasons']){let td=document.createElement('td');td.textContent=typeof r[k]==='object'?JSON.stringify(r[k]):r[k];tr.append(td)}t.append(tr)}}document.querySelector('#q').oninput=render;render()</script>'''+cards)
    (out/'provenance.json').write_text(json.dumps(dict(run=str(a.run),data=str(a.data),source_summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),checkpoint_shas={k:v for k,v in v2.items() if 'sha256' in k},hett_status=v2['data_identity'],no_target_gt_in_registration=True,no_training=True),indent=2))
    print(json.dumps(dict(cases=len(gallery),scenes=len(scene_rows),manual=manual,output=str(out)),indent=2))
if __name__=='__main__':main()
