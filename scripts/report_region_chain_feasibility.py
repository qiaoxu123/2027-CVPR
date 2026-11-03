"""Summarize TRAIN-only semantic-region counterfactuals, without promotion."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','requirements','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if (a.output/'REPORT_zh.md').exists():raise FileExistsError('preserve prior report')
    a.output.mkdir(parents=True,exist_ok=True);r=a.root
    sparse=json.loads((r/'region_chain_feasibility_v2/results.json').read_text());full=json.loads((r/'region_chain_full_observed_oracle_v2/results.json').read_text())
    for label,path in [('sparse_v2',r/'region_chain_feasibility_v2/results.json'),('full_v2',r/'region_chain_full_observed_oracle_v2/results.json'),('sparse_v1',r/'region_chain_feasibility_v1/results.json'),('full_v1',r/'region_chain_full_observed_oracle_v1/results.json')]:shutil.copy2(path,a.output/(label+'.json'))
    methods=dict(sparse['summary']);f=full['summary'];methods['FullObservedRegionOracle']=dict(frames=900,correct=f['oracle_correct'],goal_r1=f['oracle_correct']/900,goal_ne_m=f['oracle_ne_m'])
    lines=['# 地标—中间区域—目标：区域约束可修复性诊断','','## 决策','','**不直接增加区域过滤模块或继续训练区域辅助头。** 本轮只有拟合组机制证据：扩大到整个实际观测视域后，区域Oracle净增3个位置成功状态；实际RGB类别过滤净损失2个。不能把粗区域分类准确视为目标实例定位能力。','',
           '仅原900个TRAIN拟合观察、351个目标、18张地图，冻结V3与已训练区域探针；没有新模型训练、闭环或Val/Test。未读取此前65确认组来设计本方案。','',
           '|方法|正确状态/900|Goal R1|NE(m)|纠正/破坏原V3|','|---|---:|---:|---:|---:|']
    for name,m in methods.items():
        pair=full['summary'] if name=='FullObservedRegionOracle' else sparse['paired'][name]
        lines.append(f"|{name}|{m['correct']}/900|{m['goal_r1']:.2%}|{m['goal_ne_m']:.4f}|{len(pair['corrected'])}/{len(pair['harmed'])}|")
    lines+=['','Sparse Oracle、RGB、Shuffle只覆盖144个1patch窗口的内接圆；Full Oracle改为完整实际RGB/depth有效像素覆盖下的世界网格中心，避免把稀疏覆盖误当语义能力上限。Oracle匿名Contour/类别均为私有TRAIN诊断输入，不进入正常推理。硬过滤仅作一次诊断，不是新的部署规则；不扫阈值。两种Oracle都不是最优上界，区域几何也不是像素金标准。','',
            '## 当前失败Goal究竟在哪里','','315个原V3失败状态，按公开区域要求及**预测Goal位置**分层：','',
            '|情况|状态|','|---|---:|','|没有可靠公开区域要求|110|','|预测Goal在实际观测支持之外|136|','|预测Goal已观测，但区域代理未知|34|','|预测Goal已观测，属于所要求的同类区域|26|','|预测Goal已观测，属于明确不符区域|9|','',
            '**136是错误预测Goal在视域外，不是136个真实目标不可见。** 对有区域要求且Goal已观测的69个失败，只有9个具有明确区域类型不符；其余类型正确或未知。','',
            '65/315个失败在实际视域内存在目标20m内、类别符合要求的区域世界格点。这仅是位置支持机会：不证明是同一实例、不证明能被模型识别，也不是65条可恢复导航。','',
            '## 语言与作用域限制','','公开语言提案应用于584/900状态，其余316弃权。车辆经中间区域的链只有35个状态（含解析目标角色错误的潜在情况），远少于真实语言中可能存在的关系，不能推广为完整关系链覆盖率。','',
            '固定seed直接AI复核8条提案：6条有明确包含关系，1条把目标停车场中的车辆误作主要目标但区域类型巧合同为Parking，1条“across ... vehicles ... in a parking lot”作用域不确定。不是人工金标准，未按其删样本或调过滤强度。','',
            '初版将“parking grounds”误判为Ground；先新增失败测试，再按文字修复为Parking。原版输出留存。这个错误曾产生一个偶然的位置纠正，修正后真实RGB纠正由1降至0；不能把错误解析的偶然收益算作成功。','',
            '## 为什么正确区域Oracle仍会破坏成功','','04154：Parking任务，原Goal误差12.78m，硬区域排除后27.05m。08481：Building任务，10.19m变24.14m。20m成功允许Goal落在邻近其他区域，语义类型更一致并不保证最终位置更准；还存在轮廓配准、屋顶与地面投影及边界未知问题。','',
            '完整Oracle纠正5个状态、破坏2个，涉及4个纠正目标与2个破坏目标；没有Car状态被完整Oracle纠正。不能据此给出SR提升数字。','',
            '## 下一步','','停止把通用Building/Parking/Ground类别直接当成目标身份约束。下一阶段优先验证原生图像细节和实例候选空间分辨率是否足以区分车辆/相似建筑：在相同视域、yaw和冻结SigLIP2下，比较真实高分辨率局部观测与低分辨率插值对照，再决定是否重训实例级视觉匹配。推理ROI必须来自公开地图或固定观测网格，不能用GT框/位置选ROI。','',
            '只有真实细节在独立TRAIN数据产生可靠位置/实例收益，才扩大训练并进入闭环；旧确认65和Val128不再用于选方案。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(10,4));names=list(methods);ax.bar(names,[methods[n]['goal_r1']*100 for n in names]);ax.set_ylim(0,100);ax.set_ylabel('TRAIN fit Goal R1 (%)');ax.tick_params(axis='x',labelrotation=15);ax.set_title('Mechanism diagnosis only; not SR or independent performance');fig.tight_layout();fig.savefig(a.output/'region_counterfactuals.png',dpi=150);plt.close(fig)
    print(json.dumps(methods))


if __name__=='__main__':main()
