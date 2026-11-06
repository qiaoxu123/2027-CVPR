# CityNav OSM 恢复：来源核验阶段

状态：未完成OSM恢复，未部署地图网页；没有声称地理对齐精度。

## 已验证证据

- 官方CityNav论文说明CityFlight使用Potree，展示与3D同步的OpenStreetMap：[论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.pdf)。论文说明地图用途，不能替代坐标转换参数。
- [官方仓库](https://github.com/water-cookie/citynav)的本地参考副本提供`gsamllavanav/mapdata.py`中的局部米制MAP_BOUNDS，以及`train.py`/`evaluate.py`中的局部XY到图像归一化转换。该转换不是WGS84配准。
- 本地TIF的CRS、仿射、边界、分辨率逐图见`raster_georeference.json`。例如Birmingham block1仿射为0.1m像素，原点(15.25,800)，CRS为空。不能直接将这些局部坐标当作经纬度或英国国家网格坐标。
- 旧公开地标地理编码尝试记录明确为未取得控制点，不能复用为已验证地理配准。
- 现有G5指令核验网页只绘制局部XY SVG轨迹与公开Polygon；尚无可验证OSM地理转换。

## 尚缺资源及后续验收

需要CityFlight官方local XY到地理坐标转换参数，或公开地标的可靠地理控制点。若采用控制点拟合，需按城市拟合并用未参与拟合的公开对象独立检查，报告米制误差、尺度、方向和每场景异常，不能用目标GT拟合。

当前不能提供精确对齐的OSM截图或可用地图地址。未取得配准依据前不静默近似叠加。官方标注时期OSM与当前OSM可能存在地图版本差异，恢复样式与恢复历史地图内容须分别记录。

复现：`python -m scripts.audit_citynav_osm_sources --rgbd <RGBD目录> --output <输出JSON>`。本阶段无训练、权重改动、候选生成、Val/Test推理或闭环。

## 2026-11-07 局部XY网页阶段

已实现并实际启动FastAPI只读网页：服务器本机 `http://127.0.0.1:8792/`。可用SSH本地转发 `ssh -L 8792:127.0.0.1:8792 <服务器>` 后浏览；没有新建公网frpc映射或改动其他服务。tmux会话`citynav_xy_map`。网页支持34图正射RGB、局部平面、Geometry热图切换，公开具名Polygon、官方TRAIN参考轨迹、候选点、可选事后目标叠加。点击候选显示位置/分数；RGB-D ROI点击查看尚未接入。无匹配数据时明确显示缓存缺失。

使用TIF bounds和仿射定义局部XY显示；图像顶行对应y_max。没有把局部XY直接换成经纬度。正射RGB按最长边最大2048直接从原图缩小，不插值宣称新增细节。23图已生成真实冻结V3 Geometry输出（28×28），空间NMS预设20m，不依赖GT。这里不是原始V2完整规则分支。GT在候选生成后才读取为可选诊断图层，不用于配准。

实际验收：34场景列表成功；Birmingham block1、Cambridge block3 metadata/RGB端点200，非法场景404，OSM disabled。编译通过。浏览器端截图测试尚未完成（当前Python环境缺少Playwright/浏览器）；`previews/`为实际数据Matplotlib对照预览，不冒称网页截图。

## CityRefer资源核查更新

[官方README](https://github.com/ATR-DBI/CityRefer/blob/main/README.md)列出[官方Google Drive数据包](https://drive.google.com/drive/folders/1FToSspQ1EzO7Pb7BwYYBSl5Zqf5c04fJ)，包括metadata及box3d，字段描述没有列出配准矩阵。不能由README没写推断数据包不存在控制点；官方Drive包内部尚未完整核查。本地CityNav下载包20条目名称中没有control/transform/osm字段，亦非官方CityRefer原包的完整替代。

新增`fit_citynav_osm_control_points.py`可执行仿射RANSAC，要求至少4拟合点和3独立验证点，输入为人工确认的公开建筑角点/道路交叉点及投影米制OSM坐标。预设最大验证误差5m才标记通过。尚无可靠控制点输入，因此没有实际RANSAC配准结果，34图地理配准均未验证，逐图误差为未知而不是0。未开放任何OSM伪精确叠加。

仍未完成：CityRefer官方数据包参数检索、真实控制点拟合与独立验证、已验证OSM切换、候选RGB-D ROI查看、真实浏览器截图和公网访问。本阶段不声称完成全部地图恢复目标。
