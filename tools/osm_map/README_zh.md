# CityNav–OSM 人工控制点网页

入口 `/control`，原局部XY查看页仍为 `/`。服务加载原始RGB PNG（不降采样）与TIF bounds；OSM采用Leaflet1.9.4和OSM标准瓦片。Leaflet随代码保存BSD许可，OSM归属标注保留。

## 运行

使用现有GraphNav环境：FastAPI、uvicorn、requests、Rasterio/GDAL、OpenCV。浏览器验收另需Playwright及Chromium。

```bash
export MAP_REVIEW_USER='<用户名>'
export MAP_REVIEW_PASSWORD='<至少16字符密码>'
# 如服务器需要出口代理，可选：export OSM_HTTP_PROXY=http://127.0.0.1:7890
export CONTROL_PUBLIC_URL='http://<公网主机>:<端口>/'
python -m tools.osm_map.control_app --data <CityNav数据根目录> \
  --overlays <已有Geometry叠加目录> --store <私有控制点保存目录> --port 8793
```

服务只监听127.0.0.1，由FRPC映射。Basic认证保护页面、原图和全部API；写入另要求同源请求自定义头。控制点以临时文件+fsync+原子替换保存；版本冲突返回409，拒绝静默覆盖。历史快照存私有目录。FRPC token、网页密码均不应提交Git。

## 标注

1. 先选Cambridge及一个熟悉场景，左右缩放/拖动，右侧可搜索地名。场景尚未地理配准，两图不会自动同步。
2. 选择“拟合”或“独立验证”、建筑角点/道路交点等来源。左侧点局部XY，再在右侧点同一真实地物；也支持先右后左。
3. 两侧齐全后生成编号并在1秒后自动保存。两图显示编号连线。点击表格行可重选任一侧，或拖动标记；可切换用途、删除、撤销。
4. “取消选择/待配对”用于结束编辑并开始新点。只点了一侧时是待配对状态，尚不会作为完整控制点保存。
5. 建议至少10拟合点、5独立验证点，分布到地图四周；不能把同一建筑角点重复用作验证。保存后自动计算误差；点数不足仍保存但不批准配准。
6. 选择相似或仿射模型。右侧可显示透明正射图拟合预览；预测点到人工OSM点以青线显示误差。预览将局部仿射在Web Mercator屏幕作三角基近似，不是精密地理栅格导出。
7. “导出JSON”包含控制点、矩阵、EPSG:27700、拟合/独立验证的mean/RMSE/P95/max、逐点和分场景验证误差。独立验证max≤5m只作为本工具可视化验收，不足以自动批准模型的小目标监督。

同城市各场景共享一个控制点集/变换，点记录原场景；只有具有独立验证点的场景才有分场景误差，不能默认34图通过。控制点不能使用私有目标GT。目标、轨迹和Geometry不参与拟合。

## 浏览器Smoke

`tools/osm_map/smoke_browser.py`仅允许在空的Cambridge点集运行，模拟点击后保存/刷新/导出/删除/撤销，最终恢复空集。不要在正式标注开始后运行。使用`CONTROL_TEST_URL`和已有G5_REVIEW_USER/PASSWORD环境变量。合成点击不是地理控制点。
