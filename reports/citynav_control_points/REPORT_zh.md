# CityNav–OSM控制点系统：实际部署与验收

## 当前访问

实际通过公网验收：**http://101.43.173.9:42024/control**。用户名`reviewer`，密码沿用现有G5人工核验服务，私有环境文件`~/.local/share/citynav-g5/review.env`。原局部XY网页仍可从顶部链接访问，原8792服务未修改。

指定FRPS `101.200.133.203:7000`连通，但现有用户FRP token均遭拒绝；已申请正确token。现有SSH别名cvpr-source连接亦被对端关闭。因此目前使用已认证的现有101.43.173.9:8888 FRPS新增TCP42024映射，明确不是指定主机。没有修改、重启既有FRP/SSH/G5网页映射。拿到正确token后可将私有新客户端配置切回指定服务器。

## 实际实现

双Leaflet地图，左侧真实原始PNG+TIF局部XY+具名Polygon，右侧OSM道路/建筑/名称及显式地名搜索；支持34场景切换。默认Cambridge block3。人工点击成对保存CityNav XY与OSM lon/lat；编号、跨图连线、拖动/点击修改、删除、撤销、用途切换、1秒自动保存、重载和JSON导出。

控制点按城市保存，保留场景ID，优先检验统一变换。WGS84经Rasterio/GDAL转换EPSG:27700，seed42 RANSAC相似/仿射，阈值3m；至少10拟合点和5独立验证点；拒绝重复ID、重复XY/OSM点、共线退化。报告拟合残差与独立验证mean/RMSE/P95/max、逐点误差、分场景误差。保存前后不使用目标GT。青色矢量显示拟合预测与人工对应的差异。透明正射叠加是拟合预览，屏幕三角基近似不当作地理栅格金标准。OSTN15网格可用性尚未独立核查，真实控制点验证包含该转换误差。

## 服务器位置与恢复

- 私有数据：`/home/20261024_1/.local/share/citynav-osm-controls/data/cambridge.json`、`birmingham.json`，历史快照同目录。控制点和密码不提交Git。
- 应用tmux：`citynav_osm_controls`，私有启动脚本`~/.local/share/citynav-osm-controls/run.sh`。
- 新FRPC tmux：`citynav_osm_controls_frp`，私有配置`frpc-fallback.toml`，应用端口127.0.0.1:8793。
- 应用/FRP日志位于同一私有目录。服务由tmux保持，SSH断开不影响；未改systemd或其他用户服务。

## 真实验收

- 公网浏览器自动化：打开登录页→加载双图→点击成对点→自动保存→刷新恢复→导出→删除→撤销→再删除并确认空集。零JavaScript错误。`browser_public.json`记录公网URL；`control_page.png`为实际公网页面截图。
- 公网未认证401、认证200；相邻OSM瓦片SHA不同，确认没有重复占位图；城市地名Cambridge搜索返回真实结果。较细的St Johns College无结果时不编造位置，允许人工拖动。
- 曾发现上游将瓦片访问拒绝图以200返回，现使用带项目来源的User-Agent/Referer，并检测x-blocked拒绝占位图。重新测试及截图确认Cambridge城区显示正常。
- 9项CPU/本地接口测试通过：配准计算、独立验证失效、重复点拒绝、权限、持久化、409冲突、拟合报告和角点投影。合成测试只验证工程，不代表真实地理精度。
- 测试人工模拟点已删除，历史测试快照移入`data/_automated_smoke/`并注明synthetic。当前Cambridge有效点集为空，尚无真实配准误差，不声称任何场景通过。

## 限制

指定101.200.133.203公网地址尚待FRPS凭据；当前交付临时实际公网地址。真实控制点须由用户标注后才能产生地理误差与实际叠加，不以自动随机点击充当控制点。控制点图片只作人工地图配准，不进入正常导航推理。模型、权重和训练流程没有修改。
