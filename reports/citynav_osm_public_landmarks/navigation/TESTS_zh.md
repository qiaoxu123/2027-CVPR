# 验收

13项相关测试通过（test_v2_observability_proxy、test_closedloop、test_osm_landmark_correspondence）；一个第三方Starlette弃用警告。py_compile及git diff --check通过。

GPU Smoke两组各2条通过，随后完成各2470条。逐任务Pose/Action/Goal/Stop均保存。没有删除失败、修改Stop/Goal、训练模型或接入OSM。

观测审计限定1911条固定NMS5命中集合，仅对实际已经观测的Pose做GT事后量测；目标中心与Contour冲突、无Contour、像素尺寸代理不等于实际识别均有测试。历史未观测的终止移动不计入。

模型、控制器和数据SHA以及源码指纹见run_manifest.json。原批量/单条初始排名差异保留并说明，未通过Val选择精度、阈值或训练配置。
