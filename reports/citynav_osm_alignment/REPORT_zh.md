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
