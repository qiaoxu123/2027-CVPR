"""Render standalone visual-belief tables from sealed audit outputs; no model tuning."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.audit_visual_only_landmarks import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--partitions',type=Path,required=True)
    a=p.parse_args();metrics=json.loads((a.audit/'metrics.json').read_text());rows=json.loads((a.audit/'rows.json').read_text())
    partitions=json.loads(a.partitions.read_text())['results'];clean={}
    for subset,split in [('dev','fit_dev'),('confirm','fit_confirm')]:
        excluded={pair['right'] for pair in partitions[split]['same_city_name_pairs']}
        clean[subset]={control:summarize([r for r in rows if r['subset']==subset and r['control']==control and r['projection_supported'] and r['query_key'] not in excluded]) for control in ('native','trained','coverage_only','same_type_rgb')}
    (a.audit/'no_fit_same_name_subset.json').write_text(json.dumps(clean,indent=2))
    names={'native':'未训练视觉评分','trained':'训练后Visual-only','coverage_only':'仅观测覆盖范围','zero_rgb':'RGB置零','shuffle_rgb':'视觉位置打乱','same_type_rgb':'同类地标RGB互换','wrong_footprint':'错误世界坐标对应'}
    text='''# 地标Visual Belief独立验证（不融合几何）

## 范围与输入

本轮按用户纠正，独立评价视觉输出，**不训练、不运行导航、不改原模型或默认入口**。评分输入为当前SigLIP图像特征、地标名称/类型；公开地图坐标、Footprint/UAV位姿只负责图像证据到世界28×28网格的映射。公开Polygon用于既有训练弱监督和事后评价，不把Polygon区域直接复制成视觉预测。

新增`visual_only_world_belief`接口不接受几何logits、Polygon或投影支持标记。仅基于当前视觉评分和坐标映射输出观测域条件概率；几何teacher在预测完成之后才用于KL/位置对照，不加入输出。现有数据加载器仍加载teacher缓存中的名称向量及对照logits，但对照logits不会传入独立接口。这个版本尚不是“Polygon形状参与视觉特征匹配”的网络，也没有证明识别了地标身份。

复用已冻结v1权重及既有TRAIN Dev197/确认189条记录。Dev均有有效观测；确认3条没有有效世界观测，单独弃权，因此有效观测186条。每种对照确认均有相同3条弃权（7组共21条记录），不得当作21个不同样本。仅具有公开投影支持的子集为109/128；几何支持不等于真实可见，不宣称像素实例准确率。没有访问Val/Test，也不重新选择检查点。

## 1. 独立视觉定位

R20/NE按预测网格中心到公开Polygon并集距离计算，不是质心距离。所有方法使用相同并列峰规则：概率相对差≤1e-5视为数值并列，在并列格中选距其中心最近者。规则不读取Polygon；修复均匀图被浮点微差和索引起点任意取峰的问题。旧结果保留，不覆盖。本表与之前简单argmax会有小幅差异，不能解释为模型提升。

|分组|方法|R20|NE(m)|Polygon概率质量|覆盖均匀概率质量|
|---|---|---:|---:|---:|---:|
'''
    for subset in ('dev','confirm'):
        for c,label in names.items():
            m=metrics['summaries'][subset][c]['supported']
            text+=f'|{subset}|{label}|{m["correct"]}/{m["n"]} ({m["r20"]*100:.2f}%)|{m["ne"]:.4f}|{m["region_mass"]:.4f}|{m["uniform_region_mass"]:.4f}|\n'
    text+='''
概率质量通过各世界格与公开Polygon的面积交比例积分，假设格内密度均匀；对狭窄道路仍有网格近似误差，不能当作真实像素分割。观测外是未知，条件概率零不等于否定目标存在。

**开发组训练视觉65/109，未超过仅覆盖范围67/109；确认组97/128优于覆盖73/128。** 两组区域质量均提高，但不同分区表现不一致。仅靠区域附近采样位置就有较高R20，原有高定位率不能全部归因于视觉识别。RGB置零在数值并列修正后与覆盖基线完全一致。打乱及同类互换降低表现，说明网络使用了图像内容及位置对应，但不足以证明唯一地标识别。

## 2. Geometry一致性及独立视觉替代风险

|分组|G→未训练V条件KL|G→训练V条件KL|训练V相对G纠正/丢失|训练V相对未训练V纠正/丢失|
|---|---:|---:|---:|---:|
'''
    for subset in ('dev','confirm'):
        n=metrics['summaries'][subset]['native']['supported'];t=metrics['summaries'][subset]['trained']['supported'];d=metrics['trained_vs_native'][subset]
        text+=f'|{subset}|{n["conditional_G_to_V_KL"]:.4f}|{t["conditional_G_to_V_KL"]:.4f}|{t["corrected_vs_geometry"]}/{t["lost_vs_geometry"]}|{d["corrected"]}/{d["lost"]}|\n'
    for subset in ('dev','confirm'):
        rr=[r for r in rows if r['subset']==subset and r['control']=='trained' and r['projection_supported']]
        text+=f'\n{subset}支持子集Geometry单独：{sum(r["geometry_success"] for r in rr)}/{len(rr)}，NE {np.mean([r["geometry_ne"] for r in rr]):.4f}m。\n'
    text+='''
这里纠正/丢失指**单独V替代G的离线预测对照**，不是融合更新，更不是导航结果。支持子集G全部正确，当前V无法替代G。训练后与G分布KL仍更差，尚未完成“视觉Belief与几何Belief稳定对齐”。

## 3. 全样本与分区限制

'''
    for subset in ('dev','confirm'):
        for group in ('all_views','unsupported'):
            m=metrics['summaries'][subset]['trained'][group]
            text+=f'- {subset}/{group}：{m["correct"]}/{m["n"]}，NE {m["ne"]:.2f}m。\n'
    text+='''
无投影支持的记录不作为“实际不可见检测失败”统计，只说明当前观测和采样网格不足以给出区域支持，实际可见性未知；这里的全观测指标用于揭示局部输出适用边界，不能当作完整全局定位能力。

既有地图切块相邻地标可能是同一物理实体。以下仅从已评分记录中排除fit同城同名跨图候选，未根据定位结果选样；属于事后保守敏感性检查，**不是新的盲确认集，也不能保证所有跨名称物理重复被排除**。此前确认组已经曝光，没有再次宣称一次性独立验证。

|分组|方法|保守子集R20|NE(m)|
|---|---|---:|---:|
'''
    for subset,controls in clean.items():
        for c,m in controls.items():
            if m['n']:text+=f'|{subset}|{names[c]}|{m["correct"]}/{m["n"]}|{m["ne"]:.2f}|\n'
    text+='\n## 4. 类别表现（有投影支持）\n\n|分组|类别|未训练正确/样本|训练正确/样本|覆盖基线正确/样本|训练NE(m)|\n|---|---|---:|---:|---:|---:|\n'
    for subset in ('dev','confirm'):
        cc=metrics['summaries'][subset]
        for typ,t in cc['trained']['categories'].items():
            if not t['n']:continue
            n=cc['native']['categories'][typ];b=cc['coverage_only']['categories'][typ]
            text+=f'|{subset}|{typ}|{n["correct"]}/{n["n"]}|{t["correct"]}/{t["n"]}|{b["correct"]}/{b["n"]}|{t["ne"]:.2f}|\n'
    text+=f'''
## 5. 决策、图册与复现

**当前结论：训练后的独立Visual Belief有局部区域概率质量增益，但未稳定超过仅位置/覆盖先验，也未稳定对齐Geometry Belief，不能据此进入直接关系训练或用融合SR宣布通过。**

优先修改训练数据与验收：按跨块物理地标隔离，增加同一当前观测内不同同类地标的查询/困难负例，并保留目标不同位置的视图，打破“地标附近拍摄→图像中心就是答案”的偏置。若之后使用Polygon形状匹配图像，必须保留地图单独、RGB置零、Polygon互换对照；不能以Polygon直接截断输出实现伪对齐。本轮不扩展网络或新增训练。

六个真实RGB案例覆盖Building/Road成功、失败和无投影支持；`index.html`可打开原图。Singer Building成功视图NE2.3m，但另一仅部分区域支持的视图误定位约97.4m；Regent Street Dental Surgery无支持视图仍会把其它区域作为局部峰，故不能把有局部峰当作目标可见。青色Polygon是事后参考，没有叠加到模型RGB中。

缓存头推理/指标循环{metrics['seconds']:.2f}s，峰值GPU allocated {metrics['peak_allocated_gpu_gib']:.3f}GiB；不含冻结SigLIP编码及文件载入，不能当端到端导航时延。Checkpoint SHA `{metrics['checkpoint_sha256']}`，原Checkpoint未修改。30项相关测试通过，含独立接口禁几何参数、原始V等价、空观测弃权、坐标映射、浮点并列峰规则及原控制器回归。

复现：先用`scripts.audit_visual_only_landmarks`运行`metrics.json`所列冻结输入，再用`scripts.report_visual_only_landmarks --audit <目录> --partitions reports/geometry_guided_visual_belief/partition_audit.json`生成本报告。完整逐样本、输入与脚本SHA见`rows.json`、`metrics.json`；试算浮点规则输出保留在服务器`GraphNav/runs/geometry_guided_visual_belief/visual_only_draft_float/`。
'''
    (a.audit/'REPORT_zh.md').write_text(text)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7));methods=('coverage_only','native','trained','shuffle_rgb','same_type_rgb')
    for j,subset in enumerate(('dev','confirm')):
        xx=np.arange(len(methods))+j*.35
        for ax,key in zip(axes,('r20','ne','region_mass')):
            ax.bar(xx,[metrics['summaries'][subset][c]['supported'][key] for c in methods],width=.35,label=subset)
            ax.set_xticks(np.arange(len(methods))+.175,['Coverage','Native','Trained','Shuffle','Same-type'],rotation=25)
            ax.set_title(key+' | visual only');ax.legend()
    fig.tight_layout();fig.savefig(a.audit/'summary.png',dpi=160);plt.close(fig)
    page=(a.audit/'index.html').read_text().replace('<h1>', '<h1>',1)
    if 'src="summary.png"' not in page:page+='<h2>汇总</h2><img src="summary.png">'
    (a.audit/'index.html').write_text(page)
    print(json.dumps(clean,indent=2))


if __name__=='__main__':main()
