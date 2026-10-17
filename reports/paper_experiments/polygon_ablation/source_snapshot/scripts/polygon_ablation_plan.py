"""Pre-result definitions, geometry audit and immutable experiment plan."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch

from graphnav.checkpointing import file_sha256, atomic_json
from graphnav.workspace_paths import PROJECT, WORKSPACE, current_paths

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'reports/paper_experiments/polygon_ablation'
BASE = PROJECT / 'GraphNav/runs/graphnav_v1_stage1/best.pt'
FROZEN = [('descriptor_only', x) for x in ('full_polygon', 'centroid', 'bounding_box', 'no_boundary', 'no_inside', 'no_extent', 'no_inside_information')]
FROZEN += [('end_to_end', x) for x in ('centroid', 'bounding_box')]
FROZEN += [('descriptor_only', x) for x in ('prior_to_centroid','rules_to_centroid','nodes_to_centroid')]
TRAINED = [(scope, rep, seed) for scope in ('descriptor_only','end_to_end')
           for rep in ('centroid','bounding_box','full_polygon') for seed in (0,1,2)]


def canonical_job(scope, representation, seed):
    if representation == 'full_polygon':
        scope = 'shared_full_polygon'
    return f'{scope}__{representation}__seed{seed}'


def query_key(record):
    return hashlib.sha256(json.dumps([record['map_name'],record['instruction']],ensure_ascii=False).encode()).hexdigest()


def init():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'frozen_plan.json').exists():
        plan = json.loads((OUT/'frozen_plan.json').read_text())
        for path, expected in plan['input_sha256'].items():
            if file_sha256(path) != expected:
                raise ValueError('Frozen input changed: '+path)
        return plan
    weights = torch.load(BASE,map_location='cpu',weights_only=True)
    sources = current_paths(json.loads((PROJECT/'GraphNav/runs/graphnav_v1_stage1/initial_sources.json').read_text()))
    args = current_paths(weights['a3_args'])
    static = PROJECT/'Ablations/GraphNav/2027-CVPR-full-token-rgb/data/revised_static'
    paths = {s:str(static/f'graphnav_{s}.jsonl') for s in ('train_seen','val_seen','val_unseen')}
    inputs = [str(BASE),*paths.values(),*(s['path'] for s in sources.values()),args['train_bert_cache'],args['seen_bert_cache']]
    for k in sources:
        if file_sha256(sources[k]['path']) != sources[k]['sha256']:
            raise ValueError('Pretrained source differs: '+k)
    records = {s:[json.loads(l) for l in Path(p).read_text().splitlines()] for s,p in paths.items()}
    assert len(records['val_seen'])==2426 and len(records['val_unseen'])==2679
    assert all(len({(r['map_name'],r['instruction']) for r in rs})==len(rs) for rs in records.values())
    train_keys={(r['map_name'],r['instruction']) for r in records['train_seen']}
    assert not any(train_keys & {(r['map_name'],r['instruction']) for r in records[s]} for s in ('val_seen','val_unseen'))
    membership=[]
    for i,r in enumerate(records['train_seen']):
        # Split by target object identity ONLY inside training metadata. All
        # instructions for an object remain together; not by target coordinates.
        group=':'.join(r['episode_id'].split(':')[:2])
        dev=int(hashlib.sha256(('train-dev-20261017:'+group).encode()).hexdigest()[:8],16)%10==0
        membership.append(dict(index=i,query_id=query_key(r),group=group,split='train_dev' if dev else 'train_fit'))
    atomic_json(membership,OUT/'train_dev_manifest.json')
    split_sets={k:{r['group'] for r in membership if r['split']==k} for k in ('train_fit','train_dev')}
    assert not split_sets['train_fit'] & split_sets['train_dev']
    for split,rs in records.items():
        atomic_json([dict(index=i,query_id=query_key(r),episode_id=r['episode_id'],map_name=r['map_name']) for i,r in enumerate(rs)],OUT/f'{split}_query_manifest.json')
    plan=dict(version='polygon-independent-contribution-v1',created=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              base_checkpoint=str(BASE),sources=sources,args=args,paths=paths,
              input_sha256={p:file_sha256(p) for p in inputs},
              seeds=[0,1,2],epochs=15,batch_size=16,lr=5e-5,optimizer='AdamW',weight_decay=.01,gradient_clip_norm=1.,
              selection='Train-Dev R@1/20m, then lowest NE, earliest tied epoch; no Val-Seen/Unseen/Test selection',
              training_order='same seed and epoch -> same torch.Generator permutation over Train-Fit; same per-epoch dropout RNG seed; matching random initialization across representations/scopes',
              loss='existing cross_entropy_goal Gaussian spatial target, unchanged default sigma_cells=0.9',
              frozen_interventions=FROZEN,training_conditions=TRAINED,
              unique_training_jobs=sorted({canonical_job(*r) for r in TRAINED}),
              shared_full_note='Full Polygon inputs are identical in both scopes with GAT disabled, hence one trained run per seed is reused by byte-identical checkpoint SHA; 18 logical conditions,15 unique runs.',
              feature_center='original arithmetic vertex mean, including source closing vertex; shared same point across all representations, even when it is not the AABB center',
              road_policy='verified open line: never close. Centroid/BBox unavailable line boundary/inside masked; near/along fall back to available point. TrafficRoad source areas remain polygons. Actual dataset max_roads=0; counts audited.',
              groups=dict(irregular='any valid polygon area/minimum_rotated_rectangle_area < .85; otherwise regular',
                          elongated='any valid polygon minimum_rotated_rectangle aspect >=3; otherwise compact',
                          large='max valid surface polygon area >1000m^2; otherwise small',
                          point_like='nonroad surface area <=25m^2',road_like='verified open polyline OR TrafficRoad surface, original type preserved',
                          area_like='nonroad surface >25m^2',landmark_count='distinct canonical referenced names; ambiguous same-name fragments not counted as distinct names',
                          relations='only explicitly parsed, uniquely bound named anchor groups; >=2 for multiple; not GT relational annotations',
                          overlapping='type flags may overlap in multi-landmark queries; report N separately; invalid shapes marked unknown, never dropped from overall metrics'),
              navigation=dict(splits={'val_seen':2470,'val_unseen':2697},grid=28,stop=.3,horizon=20,horizontal_step_m=50.,vertical_step_m=10.,arrival_guard_m=.001,
                              head_checkpoint=sources['head']['path'],head_sha256=sources['head']['sha256'],execution='guarded-blend',
                              seed_order=[0,1,2],smoke_indices='same existing evenly_spaced_indices(...,128)',
                              observations='geometry/pose-only policy; no RGB branch. Original rollout and controller reused. Pose-only rendering optimization requires exact Seen128 trajectory parity first.'),
              bootstrap=dict(draws=10000,seed=20261017,cluster='map_name; retain all queries/episodes per sampled map',seed_summary='mean and sample std(ddof=1); seed-averaged per-sample effects, fixed seed set; seeds are not extra independent maps'),
              no_test_access=True)
    atomic_json(plan,OUT/'frozen_plan.json')
    (OUT/'POLYGON_FEATURE_AUDIT.md').write_text('''# PolyBelief Polygon Geometry 信息通路审计

本审计和frozen_plan.json在任何新实验结果产生之前写入。原权重、共享数据和归档输出不修改。

|通路|使用位置|包含的完整几何信息|本实验处理|
|---|---|---|---|
|候选–地标10-D|candidate_geometry.relative_entity_features → geometry_proj / pair attention|相对XY、中心距离、方向、signed boundary、Inside、AABB宽高、开放线类型|所有组相同10维；不可用值乘显式mask置零；不新增可学习参数|
|节点10-D|semantic_graph._physical_node_geometry → gnn.node_geometry，即使variant=geometry仍使用|中心、宽高、周长、面积、方向、面标志、顶点数|Descriptor Only完全保留；End-to-End重建各表示可用统计，Centroid移除全部形状量；BBox面积/周长/顶点数只从矩形推导|
|prior/prior_buildings|StaticLocalizationDataset._prepare → baseline、token_head、geometry_head|候选到完整区域的距离，内部为0|Descriptor Only原样；End-to-End为点/真实AABB区域/完整区域距离；building-only旁路也同步处理|
|rules|prior_and_rule_fields → baseline及评分头|near/along使用完整区域距离；direction/between使用顶点均值|Descriptor Only原样；End-to-End按允许几何重算，相同规则公式和sigma|
|relation_fields|rule_map → query → token attention/geometry head|上述单条规则场，可能绕过pair_geometry|同rules同步重算，不能只清除pair特征|
|图节点/边/拓扑|prepare_semantic_graph / TypedGeometryGAT|边界距离、相交、邻近、道路拓扑|所有组统一variant=geometry，无GAT消息；传入的edges/adjacency同时清零；不变更网络参数结构|
|候选位置编码|FullTokenGeometryLocalizer._candidate_positional_features|公共地图网格；可选参照中心|统一原28×28与同一relative_position配置；中心约定完全相同|
|公开地标绑定/选择|public_static_record、数据转换、静态打包|名称/类型/实例/原顶点均值；可选道路选择|所有组相同名称类型/实例顺序/有效mask，禁止GT最近地标；当前正式数据max_roads=0|
|训练标签|cross_entropy_goal；评测层|目标坐标只用于loss与forward之后的指标|不进入geometry_view、_batch_inputs或模型forward；Train-Dev组划分仅使用Train对象ID|
|RGB/历史/完整观测|正式完整模型其他模块|可能额外包含形状或目标外观|本次静态几何模型及闭环均不启用RGB/历史；不能冒充完整RGB方法的同配置收益|
|固定Trajectory/Stop|A3TrajectoryPolicy.predict_field / 原Guarded-Blend|候选上下文、候选坐标、公开姿态和地图范围|全组使用相同预训练head SHA，不重新微调；重训上下文分布改变是明确的控制器迁移局限|

中心点遵循原代码的顶点算术均值，非面积质心，也未剔除重复闭合顶点。所有组共享这个原始点。BBox另保留真实轴对齐包围盒；此基线实际为“共享原始中心点 + AABB”，以避免中心定义差异混淆形状贡献。

开放Road/Polyline不闭合；Full使用真实开放线距离且Inside恒0。Centroid与BBox缺少真实线形时，boundary/inside通道mask为0，near/along仅用可用中心点，绝不伪造连线。TrafficRoad/Footpath等CityRefer面状轮廓继续按其源类型作为区域，不混作中心线。

## 两类实验的归因边界

冻结干预使用正式冻结C Epoch1中现有几何定位子模型，控制其权重不变。Descriptor Only只替换pair_geometry，其prior/rules/relation_fields及节点形状仍保留完整Polygon，不能称Centroid完整表示。End-to-End同时替换全部可学习模型实际消费的几何通路。

No Inside仅清除Inside通道，signed boundary的负号仍编码Inside；因此额外加入No Inside Information，把边界距离取绝对值并清除Inside。No Boundary/No Extent也只是冻结模型的通道依赖诊断，其他旁路仍在，不是公平表示比较。

独立重训保持相同原网络、15epoch、batch16、lr5e-5、AdamW、损失及随机种子0/1/2，匹配初始化与数据顺序；只用Train内部固定对象分组Dev选择checkpoint。Descriptor Only与End-to-End分别报告；Full输入相同的两scope按同一SHA复用，避免无意义重复训练。

几何分组阈值在frozen_plan.json中预先固定，不读取结果；按公开参照物形状统计，不使用GT目标形状或不可靠目标–参照物标签。多地标的类型分组允许重叠并分别报告N。4张Unseen地图不能当2679或2697个独立场景。
''')
    for name in ('polygon_ablation_results.csv','localization_results.csv','navigation_results.csv','geometry_subgroup_results.csv'):
        (OUT/name).write_text('status,phase,scope,representation,seed,split,samples\n')
    atomic_json(dict(status='Pending',note='No new results yet; no archived result substituted'),OUT/'paired_statistics.json')
    (OUT/'POLYGON_ABLATION_REPORT.md').write_text('# PolyBelief Polygon 独立贡献验证\n\n状态：初始化。冻结审计与预注册规则已保存，结果尚不可用于结论。\n\n1. Full Polygon是否更好：待实验。\n2. Seen/Unseen一致性与统计支持：待实验。\n3. 边界/Inside/Extent及旁路贡献：待冻结干预和独立重训，不先假设正收益。\n')
    return plan


if __name__=='__main__':
    print(json.dumps(init(),ensure_ascii=False))
