# CityNav named-landmark reference-frame audit v2

中文结果：[REPORT_ZH.md](REPORT_ZH.md)。基于v1分支，**主统计只研究以具名地标为参照的方向关系**，不研究目标与匿名邻车、树、车位的关系。真实关系主语可能不是导航目标；未知上下文主语不拿目标GT代替。

纯离线诊断：不训练、不修改Geometry/Relation/PGVB、不读Val/Test。未来运动方向仅作离线诊断，不得用于部署Belief。PCA正反符号从不按目标选取。

在现有conda Python基础上创建独立解析环境（不改现有环境的依赖）：

```bash
/home/20261024_1/miniconda3/envs/graphnav/bin/python -m venv --system-site-packages /tmp/citynav_direction_parser_env
/tmp/citynav_direction_parser_env/bin/python -m pip install 'spacy==3.7.5' 'numpy<2'
/tmp/citynav_direction_parser_env/bin/python -m pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl'
```

从仓库根目录运行：

```bash
PYTHON=/tmp/citynav_direction_parser_env/bin/python
DATA_ROOT=/home/20261024_1/Workspace/DATA/sbfnav-data
$PYTHON experiments/direction_frames_v2/extract.py --data-root "$DATA_ROOT"
$PYTHON experiments/direction_frames_v2/analyze_frames.py --data-root "$DATA_ROOT"
$PYTHON experiments/direction_frames_v2/case_audit.py --data-root "$DATA_ROOT"
$PYTHON experiments/direction_frames_v2/anchor_sensitivity.py
$PYTHON experiments/direction_frames_v2/render.py --data-root "$DATA_ROOT"
$PYTHON experiments/direction_frames_v2/report.py
$PYTHON -m pytest experiments/direction_frames_v2/test_text_relations.py experiments/direction_frames_v2/test_geometry_frames.py -q
$PYTHON experiments/direction_frames_v2/verify.py --data-root "$DATA_ROOT"
```

组件：

- `text_relations.py`：仅文本与Entity(id/name/category)，不接收GT/图像/轨迹。依存路径与表面作用域交叉检查，具名地标参照门控，真实主语单独记录。
- `extract.py`：TRAIN逐条指令解析，保存地标关系账本及哈希。冻结预训练spaCy只见文本，没有CityNav训练。
- `geometry_frames.py`：绑定后计算轮廓和参考方向；同名道路保留全部路段，未来/GT条件化标志显式保存。
- `analyze_frames.py`：联合D4词典、局部视线/运动/PCA候选、交叉拟合、同支持配对比较、cluster CI。
- `case_audit.py`：四组原始候选反例的地标、主语、图像、原始TRAIN annotation复核。
- `anchor_sensitivity.py`：保持绑定和词典不变，检验position、Polygon centroid和边界位移对结果的影响。
- `render.py`：真实地图和统计图，320dpi PNG及PDF。坐标重投影不是恢复原始描述相机图像。
- `verify.py`：原始记录逐条复算、范围门控、主语GT隔离、轮廓、参考方向和交叉拟合检查。

confidence是规则分数，不是正确概率；`high`与`all_effective`分开。匿名主语、作用域冲突、不可靠方向符号仍不可评估。TIF无CRS，真实地理北向仍未独立校准。

原始来源：[CityNav](https://github.com/water-cookie/citynav)、[CityRefer](https://github.com/ATR-DBI/CityRefer)。保留数据集原始授权和署名要求；本分支保存衍生统计、必要原文与地图裁剪，没有复制完整栅格或原始数据归档。
