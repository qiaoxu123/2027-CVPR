"""Package projection geometry, named multi-fragment context and qualitative AI review."""
import argparse,collections,html,json,shutil
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shapely
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_projection import world_to_frame,frame_to_world
from graphnav.polygon_projection_audit import _fill_geometry


NOTES=[
'局部建筑边缘及树木可见，轮廓仅部分入镜；不能确认屋顶像素边界。',
'道路走向与投影路域大致一致；所选源片段只部分入镜，不核验真实道路名称。',
'小型建筑/院落区域含明显源像素缺失，不能因几何完整就认为可可靠识别。',
'斜向道路走向与投影路域大致一致；黄色边界不是道路视觉分割。',
'标为Building的投影局部覆盖停车区和车辆，需核验地块/屋顶语义或源轮廓，不能直接判定程序平移错误。',
'该源片段投影为空；仅确认几何视野外，不通过图像证明命名身份不存在。',
'该源Building片段在Footprint外，画面内其它建筑不应冒认为该命名地标。',
'可见道路横向贯穿，裁剪轮廓保留大块内部区域，适合检查整物体覆盖率误筛。',
'该源片段几何上在视野外，当前RGB不足以核对具体建筑名称。',
'道路区域与投影方向大致相符，图像中汽车不构成该路域Polygon的像素真值。',
'道路仅在画面下缘出现，所选细长片段可作为几何支持但非整条道路可见。',
'高空的小源片段位于道路/街角，不能将单片段完整等同整个道路名称完整。',
'投影为路口附近楔形路域，外观与公共道路区域相容，精确边界仍未知。',
'画面被屋顶占满，Polygon边界在视野外但内部Mask应为整幅；此例能直接揭示整物体覆盖率过滤问题。',
'小建筑/院落轮廓大致位于建筑群中，细边界和命名身份无法用总览确认。',
'局部大建筑及边缘树木入镜，几何轮廓跨屋顶/边缘区域，不作为精确屋顶Mask。',
'只在右侧边缘有小块几何重叠，不足以核验实例身份或边界偏移。',
'投影跨下方屋顶和道路交界；需区分地面轮廓、屋顶范围及源标注语义，不可当精确建筑分割。',
'所选Road类源片段实际为Footpath，需查看同名并集，不能据此判断整条道路边界。',
'轮廓沿建筑/街角区域，部分RGB缺失；完整几何覆盖不能消除源数据不足。',
'画面左侧道路走向与细长投影大致一致；剪裁边缘几何支持不等于全路可见。',
'投影局部覆盖停车位与通道，不能直接充当教堂屋顶的视觉Mask；深度支持也较低。',
'仅画面右侧少量重叠，可确认裁剪行为但无法确认细粒度建筑对应。',
'大建筑局部屋顶位于下方，轮廓在视野边缘裁剪；应保留局部区域而非要求整物体入镜。',
'道路细长片段位于下缘，几何走向大致一致；边界需放大查看且没有人工Mask。',
'所选源Building片段几何视野外；没有为其制造可见Mask。',
'所选源Road片段几何视野外；画面内其它道路不是该片段的证据。',
'所选源Road片段不与Footprint相交；保持几何未知/无覆盖，不评价目标检测。',
'斜向路廊与投影大致相符；像素边界精度没有人类Gold。',
'小建筑区域过小，轮廓大致落在建筑密集区，无法确认实际命名或完整外观。',
'完整的源片段很小且大部分源像素不足，不应把几何完整等同有效观测。',
'所选Chesterton Road片段在视野外，但同名其他片段与画面相交；必须看具名地标并集。',
'所选Building片段在几何视野外，当前图像不能核对指定名称。',
'细小Footpath片段位于视野边缘且有源缺失，无法可靠确认精细视觉边界。',
'建筑轮廓大致落在画面下方建筑区，边缘源缺失使真实轮廓判断不可靠。',
'Road分组中的源类型为Footpath/铺装区；外观可作粗区域参考，不能代替道路语义分割。',
'Boat Club源轮廓在几何视野外，无伪造可见Mask。',
'部分路域出现在街角附近，轮廓形状与铺装路域相容，但命名及精细边界未知。',
'仅建筑边缘少量入镜，完整物体不在画面中；几何部分覆盖不等于不可用。',
'Castle Street所选片段几何视野外，不以画面中的其它道路作替代。',
'小型建筑投影很小且源支持有限，需要更可靠轮廓标注才能评价像素误差。',
'路域及路侧区域沿相同走向，可作为粗空间锚点，不能当精确沥青Mask。',
'所选Building源Polygon几何视野外，当前树木/道路画面不能验证该实例。',
'L形建筑轮廓与可见建筑群轮廓大致相容，是较直观的粗对应案例；仍非人工像素GT。',
'多叉路/院落源区域大致对应图像铺装区域；完整源片段不自动等于完整命名道路。',
'Quayside源片段几何视野外，没有用其它可见地物填补对应。',
'只有上缘少量路域片段可见，图像边界及小面积使精确对应难判断。',
'部分建筑轮廓在右下角裁剪，与屋顶区域大致相容；精细边界/命名尚未人工确认。']


def main():
    p=argparse.ArgumentParser()
    for k in ('audit','catalog','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((a.audit/'summary.json').read_text());samples=json.loads((a.audit/'samples.json').read_text());catalog=json.loads(a.catalog.read_text())
    assert len(samples)==48 and len(NOTES)==48
    for name in ('summary.json','samples.json','selection.json'):shutil.copyfile(a.audit/name,a.output/name)
    for page in range(4):shutil.copyfile(a.audit/f'contact_{page}.png',a.output/f'contact_{page}.png')
    named=[];reviews=[];human=[];world_error=[]
    for case in samples:
        i=case['index'];source=a.audit/f'case_{i:02d}';dest=a.output/f'case_{i:02d}';dest.mkdir(exist_ok=True)
        for name in ('sample.json','rgb.png','geometric_mask.png','source_valid_mask.png','depth_valid.png','overlay.png','comparison.png'):
            shutil.copyfile(source/name,dest/name)
        bgr=cv2.imread(str(dest/'rgb.png'));assert file_sha256(dest/'rgb.png')==case['rgb_sha256']
        xyz=frame_to_world(np.asarray(case['pixel_polygon']),case['footprint_world'],bgr.shape)
        world_error.append(float(np.linalg.norm(xyz-np.asarray(case['contour']),axis=-1).max()))
        objects=[dict(id=str(oid),name=obj['name'],type=obj['object_type'],contour=obj['contour']) for oid,obj in catalog[case['map_name']].items() if canonical_name(obj.get('name',''))==canonical_name(case['public_name'])]
        polygons=[shapely.Polygon(x['contour']) for x in objects];valid=[poly for poly in polygons if poly.is_valid and poly.area>0]
        union=shapely.union_all(valid);view=shapely.Polygon(case['footprint_world']);intersection=union.intersection(view)
        fraction=intersection.area/union.area;state='outside' if fraction==0 else 'full' if fraction>=.98 else 'partial'
        overlay=bgr.copy();pixel_polygons=[]
        for obj in objects:
            px=world_to_frame(np.asarray(obj['contour']),case['footprint_world'],bgr.shape)
            color=(0,255,255) if obj['id']==case['public_id'] else (255,255,0)
            cv2.polylines(overlay,[np.rint(px).astype(np.int32)],True,color,2)
            poly=shapely.Polygon(px)
            if poly.is_valid:pixel_polygons.append(poly)
        region=shapely.union_all(pixel_polygons).intersection(shapely.box(0,0,383,383));mask=np.zeros((384,384),np.uint8);_fill_geometry(mask,region)
        depth=cv2.imread(str(dest/'depth_valid.png'),cv2.IMREAD_GRAYSCALE)>0;support=mask.astype(bool)&(bgr.max(-1)>0)&depth
        colored=np.zeros_like(bgr);colored[mask.astype(bool)]=(130,130,130);colored[support]=(0,220,0)
        cv2.imwrite(str(dest/'named_overlay.png'),overlay);cv2.imwrite(str(dest/'named_geometric_mask.png'),mask*255);cv2.imwrite(str(dest/'named_source_valid_mask.png'),support.astype(np.uint8)*255)
        cv2.imwrite(str(dest/'named_comparison.png'),np.concatenate((bgr,overlay,colored),axis=1))
        info=dict(index=i,map_name=case['map_name'],public_name=case['public_name'],source_public_id=case['public_id'],fragments=objects,
            valid_geometry_fragments=len(valid),total_fragments=len(objects),geometry_state=state,polygon_coverage=fraction,view_occupancy=intersection.area/view.area,
            clipped_mask_pixels=int(mask.sum()),valid_observed_region_pixels=int(support.sum()),
            source_fragment_outside_but_name_observed=case['state']=='outside' and intersection.area>0,
            actual_visibility='unknown',human_pixel_GT=False)
        named.append(info);atomic_json(info,dest/'named_context.json')
        review=dict(index=i,reviewer='current_assistant_AI_not_human',method='all four contact sheets, plus native case images for selected ambiguous/oversized examples',
            source_fragment_state=case['state'],named_region_geometry_state=state,note=NOTES[i],pixel_GT=False,approved_as_visual_segmentation=False)
        reviews.append(review)
        human.append(dict(index=i,map_name=case['map_name'],public_name=case['public_name'],source_polygon_id=case['public_id'],
            object_to_review='all same-name public polygons (source fragment also separately available)',human_result=None,human_notes='',verified_mask_path=None,review_time=None))
    named_counts=dict(collections.Counter(x['geometry_state'] for x in named));valid_pool=[r for r in samples if r['valid_observed_region_pixels']>=256]
    potential={k:sum(not r['thresholds'][k] for r in valid_pool) for k in ('roi50_256','roi90_256','current_roi90_pipeline')}
    augmented=dict(named_geometry_counts=named_counts,named_fragment_outside_but_name_overlaps=[r['index'] for r in named if r['source_fragment_outside_but_name_observed']],
        native_source_types=dict(collections.Counter(x['source_type'] for x in samples)),
        world_coordinate_roundtrip_max_m=max(world_error),
        pixel_center_warp_error_by_height={str(h):dict(n=len(rr),max_px=max(r['raster_warp']['max_center_convention_px'] for r in rr),median_px=float(np.median([r['raster_warp']['max_center_convention_px'] for r in rr]))) for h in (15,35,60,100) if (rr:=[r for r in samples if r['height']==h])},
        supported_region_pool=len(valid_pool),rejected_in_supported_pool=potential,human_review_completed=False)
    atomic_json(named,a.output/'named_region_audit.json');atomic_json(reviews,a.output/'AI_visual_review48.json');atomic_json(human,a.output/'human_review_template.json');atomic_json(augmented,a.output/'additional_metrics.json')
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    states=('full','partial','oversized','outside');width=.35
    for j,typ in enumerate(('Building','Road')):
        axes[0].bar(np.arange(4)+j*width,[summary['counts'][typ+'/'+state]['threshold_pass']['roi90_256'] for state in states],width=width,label=typ)
    axes[0].set_xticks(np.arange(4)+width/2,states,rotation=20);axes[0].set_title('90% whole-source-Polygon pass / 6 each');axes[0].legend()
    heights=[15,35,60,100];axes[1].bar(range(4),[augmented['pixel_center_warp_error_by_height'][str(h)]['max_px'] for h in heights]);axes[1].set_xticks(range(4),heights);axes[1].set_xlabel('AGL height (m)');axes[1].set_ylabel('px');axes[1].set_title('Max renderer quantization discrepancy')
    axes[2].bar(['Geometry\nsource >=256','Pass 50%','Pass 90%','Full ROI\npipeline'],[len(valid_pool)]+[len(valid_pool)-potential[k] for k in ('roi50_256','roi90_256','current_roi90_pipeline')]);axes[2].set_title('Same 35 supported geometric regions')
    fig.tight_layout();fig.savefig(a.output/'summary.png',dpi=160);plt.close(fig)
    counts=summary['counts'];text=f'''# 具名地标Polygon到RGB投影：冻结诊断

**结论：当前世界坐标到图像的变换可作为粗几何锚点，尚不能当作真实视觉分割。没有发现大幅旋转/缩放程序错误；更明确的问题是整物体覆盖率筛选会丢掉有大量图像支持的部分地标。暂不据此新增视觉精配准网络，先核验源Polygon语义与筛选口径。**

本轮只做投影、可视化和数值审查：没有加载/训练模型、没有关系推理、Visual Belief训练或闭环导航；原渲染器、Yaw规则、阈值及Checkpoint不变。

## 1. 样本与Mask定义

来自既有TRAIN公开帧元数据及CityRefer公开目录，选取48个源Polygon—视图组合，22张地图，Building24、Road24（TrafficRoad15、Footpath9）。48个不同map+源ID，按城市同名约45组，不能当48个独立真实地标。为覆盖边界情况，在记录TRAIN XY处用未修改的原生渲染器重新采样Yaw=0/45/90/135/180/225/270/315度（各6例）及AGL15/35/60/100m（14/12/11/11例）。这是明确标注的**受控重渲染观测，不冒充原人类轨迹当时的帧**。RGB实际保存384×384 PNG，没有插值放大冒充高清。

源Polygon分层为完整/部分/大于视野/视野外，各类型各6例。大于视野指面积>视野1.25倍且占当前视野≥10%，不要求整幅视野都在物体内。初次过严采样未找到足够大型建筑，未产生评测结果；修订后按此定义固定选择，原失败日志保留服务器。这里是刻意平衡的压力样本，不代表全TRAIN过滤比例。

**黄色为所选源Polygon，青色为同名其他源片段；灰色Mask仅为几何投影裁剪，绿色进一步满足非黑RGB及有效垂直深度代理。均不是实际可见性或人工语义Mask。**完整原始轮廓、Pose、Footprint、RGB/Mask SHA逐例保存。正射RGB不是重新渲染的三维斜视图，深度为相机高度减表面高度，不是遮挡/射线深度。

道路名称常有多个片段，既检查实际ROI代码所用的单源轮廓，也补充完整同名Polygon并集。按同名并集计：几何完整{named_counts.get('full',0)}、部分{named_counts.get('partial',0)}、视野外{named_counts.get('outside',0)}。Case31所选Chesterton Road片段在视野外，但同名其他片段入镜；不能把源片段的无覆盖当整个地标不可见。

## 2. 坐标/缩放/旋转/裁剪

|数值检查|实测|解释|
|---|---:|---|
|像素往返最大差|{summary['analytic_roundtrip_max_px']:.6f}px|仅内部可逆性，不能证明图像语义配准|
|世界坐标往返最大差|{augmented['world_coordinate_roundtrip_max_m']:.8f}m|同样是数值检查|
|实际整数栅格warp与名义Footprint，像素中心约定|最大{summary['warp_center_max_px']:.3f}px；样本最大值中位{summary['warp_center_median_sample_max_px']:.3f}px|独立核查renderer取整，不只自我往返|
|改用栅格边缘约定的上界|{max(r['raster_warp']['max_edge_convention_px'] for r in samples):.3f}px|报告坐标约定差异，不冒充人工误差|
|有效角点与名义角点最大世界差|{summary['warp_max_world_corner_shift_m']:.4f}m|假设PNG与TIF共享像素中心配准|
|角点取整导致首边角差上界|{summary['max_edge_angle_difference_deg']:.4f}度|未发现90度轴向翻转|
|首边长度比例误差上界|{summary['max_edge_length_ratio_error']*100:.4f}%|未发现大幅缩放错误|

PNG/TIF源尺寸匹配。渲染器使用`raster.index`取整角点，投影器使用连续Footprint，二者并非完全相同的变换；低空差异更明显。本次最大像素中心偏差仍小于1px，不能据此解释数十米目标定位误差。没有修改渲染器或偷偷补偿偏差。

剪裁单元测试覆盖旋转视野、斜视野包围盒假阳性、画外物体、部分入镜，以及Polygon完全包住视野但所有原顶点在画外的情况。Case13中源Polygon覆盖整个图像，Mask应全满，**画面内没有黄色边界线是因为边界在视野外，不是投影消失**。当前实际填充能保留内部区域；问题出现在后续整物体覆盖率判定。

## 3. 50%/90%阈值

必须区分四个口径：源物体入镜比例、当前视野被物体覆盖比例、RGB/深度有效性、以及patch弱标签覆盖率。不能把它们合为一个“可见率”。

|源Polygon几何分层|样本|50%通过（同256像素下限）|90%通过（同256像素下限）|当前完整ROI流水线通过|
|---|---:|---:|---:|---:|
'''
    for state,label in [('full','完整'),('partial','部分'),('oversized','大于视野'),('outside','视野外')]:
        cc=[counts[t+'/'+state] for t in ('Building','Road')]
        text+=f'|{label}|{sum(c["n"] for c in cc)}|{sum(c["threshold_pass"]["roi50_256"] for c in cc)}|{sum(c["threshold_pass"]["roi90_256"] for c in cc)}|{sum(c["threshold_pass"]["current_roi90_pipeline"] for c in cc)}|\n'
    text+=f'''
在35个几何区域中有效源像素≥256的共同样本里，50%拒绝{potential['roi50_256']}，90%拒绝{potential['roi90_256']}；完整ROI流水线最后只留下{len(valid_pool)-potential['current_roi90_pipeline']}。这些是**潜在丢证据**计数，不是已核验实例的误过滤准确率。

12个大于视野的样本全部有足够有效源像素，却全部被50%/90%整物体覆盖率拒绝。尤其Case13：源物体入镜16.8%，但占视野100%，有效区域145,030像素；50%与90%均拒绝，这一口径不适合判定是否存在可用局部Anchor。

`project_landmark`默认50%且32像素；上表公平比较时固定像素下限256。`current_landmark_rois`显式调用90%且256像素，随后还有bbox最短边、RGB支持90%和腐蚀稳定性筛选。几何全入镜也不保证源数据完整，例如Case02/19/30的黑色缺失区会触发后续限制，不能一概认为所有拒绝都错误。深度在现有ROI中降低置信度而非保证语义可见性。

**本版PGVB缓存没有直接调用整物体90% ROI门槛**：它使用native patch有效RGB/深度90%、Polygon内缩后单patch覆盖90%，并要求≥2正/≥4负窗口。48例中仅20例满足这个训练筛选；Case13虽有135个正窗口，却因0个负窗口被排除。这是训练样本/负例构造限制，不是投影失败，不能用“把全部90%降低”混合解决。本轮所有规则保持原样，只记录。

## 4. 目视检查与证据边界

当前AI浏览全部4张总览（48例），并放大Case04/13/17/21等原生对照。逐例意见见`AI_visual_review48.json`，**不是人类标注或视觉配准准确率**。没有可靠人工Mask，因此不报告视觉IoU、真实实例像素误差或人为通过率。提供48条`human_review_template.json`，待真人核验时单独填写，不会自动进入训练。

- Road例03、09、28、41中，路域走向与投影大致相符；边界仍是公共地图几何，不等于真实道路像素分割。
- Building例43的L形轮廓可作为较直观的粗对应；例13显示大物体内部区域不能因边界不入镜而丢弃。
- 例04、17、21中，部分Building投影覆盖停车位、通道或屋顶/道路交界。原因可能是公共Polygon表示地块/地面范围、轮廓近似、影像时间差或局部配准偏差，**没有Gold无法唯一归因，更不能训练时当作精确屋顶Mask**。
- 例30的小Road源片段虽几何完整，有效源像素只有177；像素支持不足与整物体覆盖不足是不同问题。

## 5. 决策

**可以保留现有投影作为粗地标空间锚点；不能把它直接当作目标或地标的真实视觉分割监督。** 当前最明确、可复现的限制是“大地标/部分地标被整物体覆盖阈值拒绝”，以及源几何语义与RGB实例边界尚未核验。

暂不建议直接增加视觉精配准网络。下一步先：①统一Footprint与实际warp的亚像素约定；②在独立修改中，将是否有局部Anchor与整物体入镜比例分开，并保留源支持/边界不确定性；③少量可靠轮廓核验，区分地面范围与屋顶/道路像素。只有确认存在稳定、可学习的残差后，才评估轻量精配准是否有价值。本轮没有实现这些生产修改。

所有样本来自TRAIN；没有目标GT辅助筛选，没有正式Val/Test，没有模型前向/训练或导航。原Checkpoint及生产源文件SHA核验见`verification.json`。`index.html`支持逐例查看原RGB、源片段与完整同名地标投影、几何Mask及编辑/导出人工核验记录。CPU渲染/量化/作图循环约{summary['seconds']:.2f}s，不含候选枚举与本报告整理，不是推理时延。
'''
    (a.output/'REPORT_zh.md').write_text(text)
    page='''<!doctype html><meta charset="utf-8"><title>Polygon到RGB投影审查</title><style>body{max-width:1550px;margin:auto;font-family:sans-serif}img{max-width:100%}article{border-bottom:1px solid #bbb;padding:15px}textarea{width:90%}button,select{margin:8px;padding:6px}</style><h1>TRAIN Polygon到RGB：48例冻结审查</h1><p>未训练或运行模型。黄色为所选源片段，青色为同名其它片段；灰色为几何投影，绿色仅为有效RGB/深度支持。不是人工分割或真实可见性Gold。</p><a href="REPORT_zh.md">中文报告</a> | <a href="samples.json">源片段逐例数值</a> | <a href="named_region_audit.json">完整同名地标</a> | <a href="human_review_template.json">人工核验模板</a><br><button onclick="exportReview()">导出本浏览器人工核验记录</button><img src="summary.png">'''
    for case,named_case,review in zip(samples,named,reviews):
        i=case['index'];name=html.escape(case['public_name']);page+=f'''<article id="case{i}"><h2>{i:02d} {name} | {case['map_name']} | {case['type']}</h2><p>源片段：{case['state']}；完整同名地标几何：{named_case['geometry_state']}。Yaw {case['yaw_deg']}°，AGL {case['height']}m。</p><p>源片段入镜 {case['geometry']['polygon_coverage']:.3f} / 视野占比 {case['geometry']['view_occupancy']:.3f}；同名并集入镜 {named_case['polygon_coverage']:.3f}。</p><a href="case_{i:02d}/named_comparison.png"><img src="case_{i:02d}/named_comparison.png"></a><p><a href="case_{i:02d}/rgb.png">原始384 PNG</a> | <a href="case_{i:02d}/named_geometric_mask.png">同名几何Mask</a> | <a href="case_{i:02d}/named_source_valid_mask.png">同名有效源Mask</a> | <a href="case_{i:02d}/named_context.json">全部原始轮廓</a></p><details><summary>仅所选源片段对照</summary><img src="case_{i:02d}/comparison.png"></details><p>AI目视意见（非人工）：{html.escape(review['note'])}</p><label>人工意见<select id="r{i}" onchange="saveReview({i})"><option value="">未核验</option><option>粗对应合理</option><option>疑似偏移/尺度/旋转问题</option><option>轮廓语义不明确</option><option>信息不足</option><option>几何视野外</option></select></label><textarea id="n{i}" placeholder="人工备注：可分别说明源片段与完整具名地标" onchange="saveReview({i})"></textarea></article>'''
    page+='''<script>const key='polygon-rgb-audit48-v1';const review=JSON.parse(localStorage.getItem(key)||'{}');function saveReview(i){review[i]={index:i,result:document.getElementById('r'+i).value,notes:document.getElementById('n'+i).value,time:new Date().toISOString(),reviewer_type:'human_user_input'};localStorage.setItem(key,JSON.stringify(review))}function exportReview(){const blob=new Blob([JSON.stringify(review,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_projection_review.json';a.click();URL.revokeObjectURL(a.href)}for(const [i,r] of Object.entries(review)){if(document.getElementById('r'+i)){document.getElementById('r'+i).value=r.result;document.getElementById('n'+i).value=r.notes}}</script>'''
    (a.output/'index.html').write_text(page)
    print(json.dumps(augmented,indent=2))


if __name__=='__main__':main()
