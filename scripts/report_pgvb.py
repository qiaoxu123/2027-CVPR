"""Package frozen PGVB/structured-language evidence; never trains or rescores models."""
import argparse,gzip,hashlib,html,json,shutil
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_projection import world_to_frame


def write_review_page(output,sample):
    payload=json.dumps(sample,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>TRAIN指令解析核验</title>
<style>body{max-width:1100px;margin:auto;font:16px sans-serif}pre{white-space:pre-wrap;background:#eee;padding:12px}textarea{width:95%;height:80px}select,button{margin:8px;padding:8px}</style>
<h1>TRAIN解析人工核验（尚无人类标签）</h1><p>自动解析不是Gold。此页仅存储人工审核，不会自动修改训练数据。先阅读原文，再核对目标、地标、关系作用域及参考系。</p>
<button id="prev">上一条</button><button id="next">下一条</button><span id="progress"></span><h2 id="english"></h2><pre id="parsed"></pre>
<label>目标角色/类别<select id="target"><option value="">未核验</option><option>正确</option><option>错误</option><option>无法判断</option></select></label>
<label>地标绑定<select id="landmark"><option value="">未核验</option><option>正确</option><option>错误</option><option>无法判断</option></select></label>
<label>关系链/方向/作用域<select id="relation"><option value="">未核验</option><option>正确</option><option>错误或不完整</option><option>无法判断</option></select></label>
<textarea id="notes" placeholder="修正说明"></textarea><button id="save">保存到本浏览器</button><button id="export">导出人工审核JSON</button><span id="saved"></span>
<script>const tasks=__PAYLOAD__;let pos=0;const key='pgvb-human-review-v1-'+tasks[0].record.episode;let answers=JSON.parse(localStorage.getItem(key)||'{}');
const el=id=>document.getElementById(id);function save(){answers[tasks[pos].record.episode]={episode:tasks[pos].record.episode,target:el('target').value,landmark:el('landmark').value,relation:el('relation').value,notes:el('notes').value,review_time:new Date().toISOString(),reviewer_type:'human_user_input'};localStorage.setItem(key,JSON.stringify(answers));el('saved').textContent='已保存';}
function show(){const t=tasks[pos].record;el('english').textContent=t.original_text;el('parsed').textContent=JSON.stringify({target:t.target,landmarks:t.landmarks,entities:t.entities,relations:t.relations,direct:t.direct_relations,indirect:t.indirect_relations,ambiguity:t.ambiguity},null,2);const a=answers[t.episode]||{};for(const field of ['target','landmark','relation','notes'])el(field).value=a[field]||'';el('progress').textContent=(pos+1)+'/'+tasks.length;el('saved').textContent='';}
el('prev').onclick=()=>{save();pos=Math.max(0,pos-1);show()};el('next').onclick=()=>{save();pos=Math.min(tasks.length-1,pos+1);show()};el('save').onclick=save;el('export').onclick=()=>{save();const blob=new Blob([JSON.stringify({human_annotations:answers,source:'TRAIN structured parser review'},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_annotations.json';a.click();URL.revokeObjectURL(a.href)};show();</script>'''
    (output/'review.html').write_text(page.replace('__PAYLOAD__',payload))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);language=out/'language';language.mkdir(exist_ok=True);evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True);logs=out/'logs';logs.mkdir(exist_ok=True)
    for file in ('structured_train.json.gz','summary.json','review_sample.json'):shutil.copyfile(a.run_root/'language'/file,language/file)
    for file in ('metrics.json','predictions.json','same_image_pairs.json','cases.json'):shutil.copyfile(a.run_root/'evaluation'/file,evaluation/file)
    for file in ('config.json','history.json','freeze.json'):shutil.copyfile(a.run_root/'pilot_s0'/file,out/file)
    shutil.copyfile(a.run_root/'smoke_s0/raw_gpu_smoke.json',out/'raw_gpu_smoke.json');shutil.copyfile(a.run_root/'data/manifest.json',out/'data_manifest.json')
    for name in ('parse','prepare','smoke','resume','train','evaluate'):
        if (a.run_root/(name+'.log')).exists():shutil.copyfile(a.run_root/(name+'.log'),logs/(name+'.log'))
        if (a.run_root/(name+'.sh')).exists():shutil.copyfile(a.run_root/(name+'.sh'),logs/(name+'.sh'))
    summary=json.loads((language/'summary.json').read_text());metrics=json.loads((evaluation/'metrics.json').read_text());manifest=json.loads((out/'data_manifest.json').read_text());history=json.loads((out/'history.json').read_text());freeze=json.loads((out/'freeze.json').read_text())
    with gzip.open(language/'structured_train.json.gz','rt') as f:parsed=json.load(f)
    # Independent AI inspection of 24 fixed diagnostic texts. These are NOT human gold.
    # IDs were sampled before the final scope/direction corrections, not selected for success.
    checked=[(17403,'correct','correct','Primary building on Mud Lane; other-car attributes must stay references.'),
    (15368,'correct','incorrect','Essex House/Regent Street scope remains uncertain; white wall attribute incomplete.'),
    (6662,'correct','correct','Building on Aldridge Road; car colors belong to surrounding objects.'),
    (9383,'correct','correct','Ground/lawn surrounded by named building; passive relation retained.'),
    (10133,'correct','incorrect','Bridge Street reverse/coreference relation missing; square/roof-circle attributes incomplete.'),
    (12720,'correct','correct','Between Butterfield Cafe and Combined Facility must be interpreted jointly.'),
    (14472,'correct','correct','Building on Victoria Road and right of Stamps Direct; view reference frame unknown.'),
    (462,'correct','incorrect','Willmore direct anchor found; closest/intersection/Wellington chain incomplete.'),
    (16369,'correct','incorrect','Intersection has two road references but chain drops Clarendon and view framing.'),
    (13198,'correct','correct','Across-from vans then in-front-of Pathology is a plausible chain; frame remains unknown.'),
    (867,'correct','incorrect','Counting from far side is a view/counting reference, not direct target location; shared-lot names incomplete.'),
    (15952,'correct','correct','Building-middle-of-block linked to Mud Lane/Eden Street; roof/solar attributes incomplete.'),
    (4770,'correct','incorrect','Front-of-building may modify target rather than another car; attachment not reliable.'),
    (19966,'abstain','uncertain','Target is front section of multicolored building; category/Elizabeth Way binding cannot be confirmed.'),
    (16775,'correct','correct','Car next to tree on City Road; next-to surface phrase restored.'),
    (7290,'correct','correct','Car in parking lot for Police Custody Suite is an association chain; ordinal context must remain original.'),
    (18655,'correct','unresolved','Grafton/parking-lot chain incomplete; blue car black-top attribute also incomplete.'),
    (12345,'abstain','unresolved','Section of dirt area: do not force Building/Ground category; road/building scope unresolved.'),
    (7204,'incorrect','unresolved','Parser selects reference plural cars instead of first black car; earlier-noun ambiguity now flagged.'),
    (17968,'correct','unresolved','Connected-to Match Hairdressing not connected into named-anchor path.'),
    (7761,'correct','unresolved','12 Castle chart name/binding is not reliable; retain unresolved candidate.'),
    (6586,'correct','unresolved','Parking lot surrounding house at Ascott Gardens; missing/alias-sensitive public binding.'),
    (8182,'correct','unresolved','Cambridge graduate reference not resolved into a reliable public anchor path.'),
    (1666,'correct','unresolved','Named One Stop through parking area/building chain not resolved; ordinal and side constraints remain in original text.')]
    lookup={r['episode']:r for r in parsed}
    reviewed=[dict(episode=i,target_category_judgement=t,main_landmark_path_judgement=r,note=note,record=lookup[i],reviewer='current_assistant_AI_not_human') for i,t,r,note in checked]
    ai=dict(n=24,target_correct=21,target_incorrect=1,target_abstain=2,main_anchor_path_correct=9,main_anchor_path_incorrect=6,main_anchor_path_uncertain=1,unresolved_path=8,
        note='Stratified diagnostic development sample, not representative accuracy or human gold; no relation labels approved for training.')
    atomic_json(dict(summary=ai,rows=reviewed),language/'AI_review24.json');write_review_page(language,json.loads((language/'review_sample.json').read_text()))
    # Primary queries are linked to parsed TRAIN names; unmentioned public
    # regions remain explicitly identified augmentation, never target GT.
    from graphnav.landmark_alias import canonical_name
    bound={(r['map_name'],canonical_name(name)) for r in parsed for m in r['landmarks'] for name in m['public_names']}
    # Quantify physical-name-cluster uncertainty, without using it for model selection.
    rows=json.loads((evaluation/'predictions.json').read_text());actual={r['index']:r for r in rows if r['subset']=='dev' and r['control']=='real'};baseline={r['index']:r for r in rows if r['subset']=='dev' and r['control']=='coverage_only'}
    groups={}
    for i,r in actual.items():groups.setdefault(r['query_key'].split('::',1)[1],[]).append(i)
    rng=np.random.default_rng(0);names=sorted(groups);boot=[]
    for _ in range(2000):
        selected=[i for name in rng.choice(names,len(names),replace=True) for i in groups[name]]
        boot.append([np.mean([int(actual[i]['success'])-int(baseline[i]['success']) for i in selected]),np.mean([actual[i]['ne']-baseline[i]['ne'] for i in selected])])
    ci=dict(named_groups=len(names),r20_difference_ci95=np.quantile(np.array(boot)[:,0],[.025,.975]).tolist(),NE_difference_ci95=np.quantile(np.array(boot)[:,1],[.025,.975]).tolist(),
        caution='Exploratory city+canonical-name cluster bootstrap; aliases may share physical objects, only one held-out city and one seed; no significance/generalization claim.')
    atomic_json(ci,out/'uncertainty.json')
    # Replot sealed fields with probability zero anchored scale. This avoids
    # magnifying round-off in uniform zero-RGB maps into spurious visual texture.
    data=torch.load(a.run_root/'data/cache.pt',weights_only=True,map_location='cpu');maps=torch.load(a.run_root/'evaluation/visual_maps.pt',weights_only=True,map_location='cpu')
    binding={subset:dict(total=sum(r['subset']==subset for r in data['records']),
        parsed_name_queries=sum(r['subset']==subset and (data['frames'][r['frame_id']]['map_name'],canonical_name(data['queries'][r['query_key']]['name'])) in bound for r in data['records'])) for subset in ('fit','dev')}
    atomic_json(binding,out/'query_binding_audit.json')
    cases=json.loads((evaluation/'cases.json').read_text())
    for n,case in enumerate(cases):
        index=case['index'];r=data['records'][index];f=data['frames'][r['frame_id']];q=data['queries'][r['query_key']];bgr=cv2.imread(f['image_path'])
        assert file_sha256(f['image_path'])==f['frame']['image_sha256']
        fig,axes=plt.subplots(1,4,figsize=(16,4));axes[0].imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        for poly in q['contours']:
            pixels=world_to_frame(np.asarray(poly),f['frame']['footprint_world'],bgr.shape);axes[0].plot(pixels[:,0],pixels[:,1],'c-',lw=.8)
        axes[0].set_xlim(0,bgr.shape[1]-1);axes[0].set_ylim(bgr.shape[0]-1,0);axes[0].set_title('Real RGB + diagnostic Polygon');axes[0].axis('off')
        bound=f['bounds'];extent=[bound[0],bound[2],bound[1],bound[3]]
        vmax=max(float(maps[(index,c)]['probability'].max()) for c in ('coverage_only','zero_rgb','real'))
        for ax,c,title in zip(axes[1:],('coverage_only','zero_rgb','real'),('Coverage','Zero RGB','PGVB')):
            packed=maps[(index,c)];values=np.where(packed['covered'].numpy(),packed['probability'].numpy(),np.nan).reshape(28,28)
            im=ax.imshow(values,origin='lower',extent=extent,cmap='magma',vmin=0,vmax=vmax)
            for poly in q['contours']:
                xy=np.asarray(poly);ax.plot(xy[:,0],xy[:,1],'c-',lw=.8)
            rr=next(x for x in rows if x['index']==index and x['control']==c);ax.scatter(*rr['world_xy'],color='red',s=20);ax.set_xlim(bound[0],bound[2]);ax.set_ylim(bound[1],bound[3]);ax.set_title(f'{title}: NE {rr["ne"]:.1f}m, mass {rr["mass"]:.3f}',fontsize=9)
        fig.colorbar(im,ax=list(axes[1:]),shrink=.6,label='P(world cell | current observed domain)');fig.suptitle(q['name']+' | '+r['type']+' | geometry proxy, actual visibility unknown')
        fig.savefig(evaluation/case['file'],dpi=150,bbox_inches='tight');plt.close(fig);cv2.imwrite(str(evaluation/f'case_{n}_rgb.png'),bgr)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    exclusive=[summary['direct']-summary['both'],summary['indirect']-summary['both'],summary['both'],summary['neither']]
    axes[0].bar(['Direct only','Indirect only','Both','Unresolved'],exclusive);axes[0].set_title('Automatic path proposals, not gold labels')
    categories=summary['target_category'];axes[1].bar(list(categories),list(categories.values()));axes[1].set_title('Text-inferred target category');fig.tight_layout();fig.savefig(out/'language_distribution.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6));epochs=[r['epoch'] for r in history]
    for name in ('total','region','bce','pair'):axes[0].plot(epochs,[r[name] for r in history],label=name)
    axes[0].legend();axes[0].set_title('TRAIN losses, all fixed epochs')
    methods=('coverage_only','zero_rgb','real','polygon_zero')
    for j,typ in enumerate(('Building','Road')):
        xx=np.arange(len(methods))+j*.35
        axes[1].bar(xx,[metrics['summary']['dev'][c][typ]['R20'] for c in methods],width=.35,label=typ)
        axes[2].bar(xx,[metrics['summary']['dev'][c][typ]['NE'] for c in methods],width=.35,label=typ)
    for ax in axes[1:]:ax.set_xticks(np.arange(len(methods))+.175,['Coverage','Zero RGB','PGVB','No Polygon'],rotation=20);ax.legend()
    axes[1].set_title('Held-out-city R20');axes[2].set_title('Held-out-city NE (m)');fig.tight_layout();fig.savefig(out/'summary.png',dpi=160);plt.close(fig)
    total=summary['instructions'];pct=lambda n:f'{n/total*100:.2f}%'
    text=f'''# TRAIN语言结构化与PGVB地标视觉定位

**结论：完成全量22,002条语言候选解析及真实GPU PGVB训练，但本版尚未通过跨城市地标区分验收。不能进入直接/间接关系目标推理。原Checkpoint与正常推理入口未变。**

## 1. 全量语言解析（目标GT不参与）

只读取官方TRAIN英文、地图标识及公开地标名称/类别；不使用目标ID、目标类别或目标位置辅助选择目标。相较历史依赖GT类别的审计，本次是新的文本推断口径，数量不能直接混用。

|自动候选统计|数量|占全部TRAIN|
|---|---:|---:|
|绑定至少一个公开地标|{summary['named_landmark']}|{pct(summary['named_landmark'])}|
|多地标|{summary['multiple_landmarks']}|{pct(summary['multiple_landmarks'])}|
|有直接关系路径，允许与间接重叠|{summary['direct']}|{pct(summary['direct'])}|
|有间接关系路径，允许与直接重叠|{summary['indirect']}|{pct(summary['indirect'])}|
|两者都有|{summary['both']}|{pct(summary['both'])}|
|两者均未可靠连通|{summary['neither']}|{pct(summary['neither'])}|

直接单独{summary['direct']-summary['both']}、间接单独{summary['indirect']-summary['both']}。路径包含空间和所属/关联边；“未连通”包含解析失败、别名缺失及语言歧义，**不等于指令无法关联地标**。全部输出为未核验候选，`approved_for_training=false`；不会污染本轮视觉训练。

结构保留原文字符span、实体ID、目标名称（无则null）、文本推断类别、外观属性、每条有方向的边、链路正/反向遍历及between多参照组。取消旧4跳截断，保存完整最短关系链。left/right/front/behind不强制转换世界XY；明确为`reference_frame_unknown`。原文中的视角说明仍需后续人工确认，不据此推断相机角度。

抽样修复了Road后缀伪中间实体、把参考墙面混入目标属性、right side of丢方向词等问题。保留未绑定专名候选，不编造地图匹配。目标类别未知{summary['target_category'].get('unknown',0)}条；“抽取到属性”是字段覆盖，不代表完整保留所有外观/序数约束。

**人工解析准确率尚不可测：没有人类Gold。** 提供60条分层可编辑核验样本及`language/review.html`，尚无真人审核记录。另由当前AI检查固定24条诊断文本：目标类别21正确、1错误、2弃权；主地标链9可接受、6错误或不完整、1不确定、8未解析。该分层开发检查不是随机总体准确率，更不是人工核验。详细问题见`language/AI_review24.json`，其中第7204条把相似参考车辆选成目标，已显式标记多候选风险，但没有伪称完全修复。

完整结构化JSON为`language/structured_train.json.gz`，可用`gzip -dc`读取；22,002条原文、实体、边和路径引用一致性检查通过。初版及修订中间结果留在服务器，不覆盖原语言数据。

## 2. PGVB输入与物理隔离

复用当前真实384×384正射RGB和冻结SigLIP2三尺度原生特征，名称向量重新编码。真实RGB→SigLIP2→PGVB前向/反向Smoke通过，新提取与缓存特征MAE=0；SigLIP2无梯度。本轮不使用LoRA，也不训练原几何分支或视觉骨干。

输入公开名称/类别、Polygon形状、当前RGB；Footprint与UAV位姿提供合法观测和世界映射。新增Polygon token Cross-Attention及视觉Belief评分头。为防复制地图中心，Polygon在相机尺度下去中心，绝对位置只用于最终28×28网格投影，**没有输入原A3 logits或融合G**。这是“去中心形状条件PGVB”，不是完整绝对Polygon位置配准模型；负结果不能推广到所有Polygon方法。

Cambridge拟合：{manifest['counts']['fit']['frames']}帧/{manifest['counts']['fit']['queries']}查询；Birmingham开发：{manifest['counts']['dev']['frames']}帧/{manifest['counts']['dev']['queries']}查询。城市、地图和图像SHA无交叉；同一物理地标不可能跨两个城市分区。`data_manifest`历史键`physical_name_groups`实际是map+name组；按city+canonical-name合并为425/49，仍不是人工确认的真实实例数。只在一个留出城市/一个种子检查，不能宣称普遍泛化。

同图多地标训练查询减少“附近拍摄→中心就是答案”的偏置，但没有证明完全消除。公开地图同类区域提供困难负例；同一图的多个查询共享RGB且正窗口不重叠。未提及的公开同类名称可作为增强查询，不根据原任务真实目标选择。训练查询中{binding['fit']['parsed_name_queries']}/{binding['fit']['total']}的名称可在全量TRAIN解析中绑定；开发查询为{binding['dev']['parsed_name_queries']}/{binding['dev']['total']}，其余为公开地图增强查询，单独记录于`query_binding_audit.json`。所有监督为Polygon内缩/外扩的保守弱标签，边界忽略，不等于像素GT或真实可见性。无可靠正窗口的状态不强行监督；本轮是既有缓存Pilot，不是全量TRAIN视觉数据重建。

## 3. 固定训练与开发结果

seed0、12轮、AdamW lr0.0005，{freeze['steps']}次更新；区域概率损失+0.2平衡BCE+0.3同图困难负例排序。最佳权重只按训练轮平均Loss选择，冻结后一次性开发评分，不根据开发结果重训。全部Cross-Attention、Polygon、视觉及文本适配层有有限梯度。

|开发类别|方法|R20|NE(m)|Polygon概率质量|
|---|---|---:|---:|---:|
'''
    labels={'real':'PGVB','coverage_only':'仅观测覆盖','zero_rgb':'RGB置零'}
    for typ in ('Building','Road','all'):
        for control,label in labels.items():
            m=metrics['summary']['dev'][control][typ];text+=f'|{typ}|{label}|{m["correct"]}/{m["n"]} ({100*m["R20"]:.2f}%)|{m["NE"]:.3f}|{m["polygon_mass"]:.3f}|\n'
    text+='''
R20/NE为预测世界网格点到公开Polygon并集的距离，非质心、非真实实例准确率。概率质量用格子与Polygon相交面积近似积分。观测外未知；本表只含具有保守区域弱监督的样本，不能推广到不可见目标。RGB置零与覆盖基线相同，说明网络没有直接把绝对Polygon位置复制成输出。

|同图同类区分代理|TRAIN查询偏好准确率|开发查询偏好准确率|开发两查询都正确|
|---|---:|---:|---:|
'''
    for typ in ('Building','Road','all'):
        fit=metrics['pairs']['fit::real'][typ];dev=metrics['pairs']['dev::real'][typ]
        text+=f'|{typ}|{fit["query_preference_accuracy"]*100:.2f}%|{dev["query_preference_accuracy"]*100:.2f}%|{dev["strict_both_correct"]}/{dev["pairs"]}|\n'
    text+='''
区分代理比较同图两个互不重叠的公开区域：每个名称查询在自身正窗口的平均分是否超过另一地标，平局计0.5；“两查询都正确”要求两个方向都严格大于。零RGB所有开发查询平局、偏好50%，严格双正确0。352对来自176帧，不是352个独立实体。只是几何弱区域代理，不冒充真人像素实例标注。

## 4. 消融、纠正/破坏与根因

'''
    for key,label in [('coverage_only','覆盖基线'),('polygon_zero','关闭Polygon形状'),('wrong_polygon','同图同类Polygon形状互换'),('wrong_name','同图同类名称互换')]:
        d=metrics['dev_paired'][key];text+=f'- 相对{label}：共同{d["paired"]}条，PGVB纠正{d["real_corrected"]}、破坏{d["real_lost"]}；NE {d["real_NE"]:.3f} vs {d["control_NE"]:.3f}m。\n'
    text+=f'''
关闭Polygon后开发R20为{metrics['summary']['dev']['polygon_zero']['all']['correct']}/699，高于PGVB的366/699。Wrong-Polygon/Wrong-Name只有442条存在合法同图同类替换，必须使用共同子集，不能混用699分母。以上是独立V对照，不是对A0融合纠正，也不是SR。

PGVB相对覆盖只净增7条R20（142纠正/135破坏），NE反而恶化约9.29m。名称组bootstrap的R20差95%区间为[{ci['r20_difference_ci95'][0]*100:.2f}, {ci['r20_difference_ci95'][1]*100:.2f}]个百分点；仅是单城市单种子的探索性不确定性估计，不做显著性声明。

证据最支持的问题是**名称/场景记忆和Polygon匹配泛化不足**：拟合同图区分94.06%，跨城市50.28%；正确/错误名称的开发差异不稳定，Polygon形状消融没有独立收益。图像位置打乱使性能下降，说明确实使用RGB内容和对应，但这不等于可靠识别具名地标。去中心形状输入也舍弃了绝对地图配准约束；公开名称本身未必包含可见外观，这是本设计的实质限制。弱Polygon投影与实际屋顶/可见性的误差仍未有人工Gold核验，不能把所有失败归因于模型。

下一步只优先改**地图Polygon与当前图像的配准/区域验证监督**：明确已观测区域内哪些形状及类别证据支持指定Polygon，加入同图相似区域与错误配准反事实，并验证模型能拒绝不匹配图像。若恢复绝对Polygon位置输入，必须用地图单独/错配RGB对照证明新增的是视觉支持，不能靠复制位置通过验收。暂不扩大关系网络、LoRA或训练轮数；先取得跨物理地标的稳定区分收益。

## 5. 保存与工程验收

数据准备约{manifest['seconds']:.2f}s，缓存训练头{freeze['seconds']:.2f}s，峰值allocated {freeze['peak_gpu_gib']:.3f}GiB（不含冻结骨干编码）；头部耗时见`evaluation/metrics.json`，不冒充端到端时延。Smoke实际RGB及网络输入均384，不宣称高清细节增加。检测器/控制器/Stop/正式评测协议未改，未访问Val/Test、未开展闭环或下一阶段关系训练。资源扫描器未识别GPU，已用实际CUDA及真实前后向验证补证。

最佳Checkpoint SHA：`{freeze['best_sha256']}`；服务器`{a.run_root}/pilot_s0/best.pt`，last含优化器/RNG、支持轮末恢复。大缓存与权重留服务器，Git只保存脚本、结构化文本、配置、日志、逐样本指标和图册。原Checkpoint SHA `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`保持不变。参见`CHECKPOINTS_zh.md`。

所有本轮训练/缓存/评分已退出；原有无关tmux、服务及暂停任务未操作。人工语言准确率仍待真实核验，不把AI判断冒充人工数据；当前结果不足以进入直接/间接关系目标Belief训练。
'''
    if (out/'tests.log').exists():text+='\n相关单元/回归测试见`tests.log`；当前保存的验收为42项通过。静态核验页保存/恢复/导出仅作Node沙箱模拟，不冒充真人或服务器浏览器验收。\n'
    (out/'REPORT_zh.md').write_text(text)
    (out/'CHECKPOINTS_zh.md').write_text(f'# 权重与恢复\n\n原权重未覆盖。\n\n- 新增模型代码：`graphnav/polygon_guided_visual.py`，独立Visual Belief。\n- 服务器根目录：`{a.run_root}`。\n- `pilot_s0/best.pt` SHA256 `{freeze["best_sha256"]}`。\n- `pilot_s0/last.pt` SHA256 `{freeze["last_sha256"]}`，保存优化器、Python/Torch/CUDA RNG和轮末状态。\n- 新权重不加入Git；原A3/视觉分支Checkpoint保持原位。\n- 冻结SigLIP2版本和SHA见`data_manifest.json`；不使用LoRA，不宣称全参数训练。\n- 复现命令与真实日志见`logs/`；原输出保护，重新实验需新建路径，恢复用`--resume`。\n- 配置冻结后仅评分一次开发组，没有正式导航SR结果。\n')
    page='<!doctype html><meta charset="utf-8"><title>PGVB与TRAIN结构化解析</title><style>body{max-width:1600px;margin:auto;font-family:sans-serif}img{max-width:100%}</style><h1>PGVB：城市隔离冻结诊断</h1><p>没有通过跨城市同图地标区分。全部监督为公开Polygon几何代理，非人工像素GT；图中青线仅事后叠加，未输入模型RGB。</p><p><a href="REPORT_zh.md">中文报告</a> | <a href="language/structured_train.json.gz">全部22,002条结构化JSON</a> | <a href="language/review.html">60条人工核验工具</a> | <a href="language/AI_review24.json">24条AI检查（非人工）</a></p><img src="language_distribution.png"><img src="summary.png">'
    for ci,c in enumerate(cases):page+=f'<h2>{html.escape(c["query_key"])} | frame {c["frame_id"]}</h2><a href="evaluation/{c["file"]}"><img src="evaluation/{c["file"]}"></a><a href="evaluation/case_{ci}_rgb.png">原始实际RGB</a>'
    (out/'index.html').write_text(page)
    atomic_json(dict(language_all_train_complete=True,PGVB_training_complete=True,heldout_evaluation_complete=True,
        human_language_accuracy_verified=False,visual_alignment_acceptance_passed=False,direct_indirect_relation_model_started=False,
        normal_entry_changed=False,original_checkpoint_changed=False,formal_val_test_accessed=False),out/'STATUS.json')
    print(json.dumps(dict(language=summary['instructions'],PGVB_dev=metrics['summary']['dev']['real']['all'],AI_review=ai),indent=2))


if __name__=='__main__':main()
