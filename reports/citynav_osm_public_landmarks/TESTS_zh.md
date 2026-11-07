# 实际验收记录

- `python -m pytest -q tests/test_osm_landmark_correspondence.py tests/test_osm_batch_quality.py tests/test_osm_public_vectors.py tests/test_citynav_auto_registration.py tests/test_citynav_control_registration.py`：19 passed（一个Starlette依赖弃用警告）。
- 物理分组、相邻建筑、同道路不跨组；改变验证anchor不影响拟合；预测顺序/20m空间NMS不接收GT。
- 隔离存储验证确认/拒绝接口、重建app后恢复、分组不变和禁止自动批准。
- 公网Chromium实际加载 `/control`，打开自动对应实例，两侧Polygon及真实RGB加载完成，页面JS错误0。无认证401；缺少写确认头403；原人工控制点SHA未变化。详情见web_test.json及截图。真实用户待审核记录未被测试改写。
- HETT官方Refined Val-Seen独立下载，与本地50453758字节文件SHA逐字节相同，共2470条；只下载Val-Seen，没有下载Test。
- GPU完整初始定位两组各2470，无导航、无OSM特征、无训练；模型输入target_xy为占位，真实目标在forward后评分。
- 当前Git改动不含graphnav模型代码或Checkpoint；未修改原控制器、阈值、原配准和已验证导出。
