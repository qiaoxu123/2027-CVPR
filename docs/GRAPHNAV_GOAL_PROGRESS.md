# GraphNav goal progress

Branch: `2027-CVPR/graphnav-closedloop-sbf-parity`, based on `11e0409b322f08db4459888c6d38ba08547da730`. Goal remains active: full closed-loop navigation and >=95% SR / >=90% SPL of the local SBFNav baseline on identical Val Unseen evaluation. No acceptance level is achieved yet.

## Unified baseline — in progress

Located genuine local SBFNav epoch-14 best checkpoint, SHA256 `e991b4e46ebc20b9cf7a1e350a0993c75028e863b10b55df166acd4e59510885`, and immutable source/config snapshot. Historical selection used Val Unseen proxy: disclosed, not replicated for new GraphNav tuning. Shared rollout now separates public language/pose/resolved polygons from GT/evaluator labels, uses XY metrics, 20m radius, 20 macro decisions, 50m horizontal and 10m vertical limits, map bounds and ground+10..100m altitude limits. Original SBF has horizon/stagnation termination and no trained Stop. False-Stop Rate is null when no stop decisions exist.

128 evenly spaced Val Seen episodes: SR 44.53%, SPL 36.89%, OSR 61.72%, NE 32.96m, nDTW 27.55%, sDTW 14.68%, mean 15.94 moves, mean policy latency 47.56ms. All 128 trajectories/termination lengths reproduce historical checkpoint outputs exactly. Full Seen/Unseen rerun follows. Closed-loop denominator is 2470/2697 raw episodes; static retrieval denominator 2426/2679 unique map+text queries is intentionally separate.

79 tests pass. SBF tokenizer was incompatible with the prior GraphNav environment (protobuf absent); baseline is run using existing SBFNav environment without modifying either environment. Failed startup output/log retained. Raw depth/collision checks and true-camera intrinsics fidelity remain explicit simulator limitations; no improved safety claims are made.

## P0/P1/P2 — completed first validation, further controlled tests pending

See `BIDIRECTIONAL_VISUAL_LOCAL_VALIDATION_20261009.md`. 54 projection/ROI examples reviewed. Reliable filtered RGB regions and cross-frame pairs exist; source holes/coarse footprints are filtered, not called segmentation GT.

P1 five epochs total train-only 70.02s. G0/G1 matched A3 train-only 226.30/229.32s; G1 total pretrain+fine-tune 299.35s. G0 Unseen R@8=47.14%, NE54.68m; G1 46.40%,56.31m: no demonstrated transfer gain. Historical A3 remains 50.88%,48.42m and is preserved.

P2 five formal epochs 65.62s, peak allocated GPU4.10GiB; Unseen visual retrieval R@1/R@5=69.57/93.91% on115 identities. Frozen Large mean-patch control=66.96/81.74%. Footpath R@1 declines vs frozen baseline (40% vs60%, only10 samples). This is visual matching, not language localization or navigation success. Base/Large control, pose-indexed online cache, distance/visibility-stratified evaluation and aligned native Depth remain pending. No LoRA or Unseen-driven architecture tuning.

## Next experiments

1. Finish unified full SBFNav baseline and source/data/protocol audit.
2. Evaluate preserved A3 with a basic bounded waypoint policy in the same simulator, including all episodes even when landmark binding is unavailable.
3. Select trajectory/Stop training and confidence-gated graph/visual history additions using train/internal dev and Val Seen only, at most3 major configurations per stage.
4. Freeze final candidates before full Val Unseen comparisons, then assess paired episode/map uncertainty and acceptance levels.

## Engineering candidates prepared during full baseline replay

Native RGB-D diagnostic added: 54 cases exactly reproduce the original RGB pixels and use source surface-height-derived vertical depth. Water/bridge source gaps are substantial; this is a benchmark depth proxy, not perspective depth or a 3D occlusion model. Category support/2px sensitivity and image pairs are in `reports/p0_depth_20261009/`.

A3 public-input bounded-waypoint adapter and zero-gated dynamic graph/visual residual candidates pass CPU regression tests. They are untrained integration candidates; no performance gain or navigation completion is claimed. The visual update covers all grid candidates, supports positive landmark history and treats unobserved nodes as unknown. Native RGB/Depth, node association and a learned trajectory policy still require integration/training.

P1/P2 now support atomic Last/Best and explicit resume with optimizer/RNG and input fingerprints. Source or configuration mismatch aborts; existing experiment outputs are protected. P2 epoch-indexed view sampling is worker-independent for fair Base224/Large384 controls. Old P2 results retain their original random-worker sampling and are not overwritten. The new controlled comparison is predeclared in `reports/sbf_protocol/phase1_plan.json` and queued only after full SBFNav replay, preceded by A3 Seen128 smoke. Model choice uses internal train dev; different native encoder resolutions are a disclosed confound. Current regression suite:90 passing tests (before additional queued experiment execution).

Candidate-aligned multimodal Waypoint/Stop head now has label-separated losses and genuine human XY-arclength waypoint interpolation. Tests cover executed-first-step constraints and consistent candidate/mode/Stop selection. It remains untrained and is not integrated into measured policy behavior.

Protocol input audit:1470 Seen and1208 Unseen episodes have different native SBF/GraphNav reference sets, usually multiple genuine same-name road fragments. Both get the same raw annotation assets/reference names; preserve native preprocessing and disclose the difference. The A3 adapter was corrected before its first measured evaluation to retain fragments, with a regression test.1020 Seen/1261 Unseen start poses exceed the old ground+100m altitude cap and remain unchanged. Full episode/schema/raster SHA256 and baseline checkpoint manifests are in `reports/sbf_protocol/`.

## Unified SBFNav full Seen complete

2470 episodes:SR43.52%,SPL36.12%,OSR60.40%,NE31.52m,nDTW28.28%,sDTW14.74%;15.77 mean executed movements,47.97ms mean policy latency. Evaluation took32.3min; every trajectory exactly matches the historical frozen checkpoint replay. Full Unseen now running. No GraphNav acceptance level achieved.

## Trajectory supervision and cache preparation — 2026-10-10

Training-only preparation completed with128509 samples from21864 approved Train Seen episodes,21864 explicit recovery samples,30796 positive Stop labels and32659 near-false-stop samples at20–40m.14 episodes with nontraining query text are excluded by the prior audited training whitelist; no validation episode is removed.50300 training poses are outside the legacy altitude range; source labels are preserved with explicit vertical-loss masks. The initial109258-sample preparation is retained, while v2 adds near-boundary Stop negatives.

The128509-sample dataset and21,169 distinct public priors reside in `data/trajectory_supervision_v2_seen/`; manifests record source/output SHA256. Distinct supplied priors occasionally share map+text; temporal feature caching must preserve these rows, while static recall keeps its original deduplication policy. Regression tests cover this index alignment and illegal teacher-z loss masking. No trajectory model has yet been trained or selected.

Real preserved A3 CPU cache smoke passed:1 public query,20 proposal contexts and784 dense contexts, all finite, no GT/future fields stored. Frozen A3 batch64 caching with workers4 is queued after the unified SBF replay and predeclared Base/Large controls. Chunk checkpoints validate source/model fingerprints and support interrupted cache resume. Dense context caching avoids repeated BERT/A3 evaluation for later geometry-residual and trajectory experiments. See `reports/sbf_protocol/phase2_plan.json`.

## Candidate trajectory training/runtime ready — 2026-10-10

Training and evaluation interfaces now support a frozen-A3 candidate-aligned trajectory/ranking/Stop head. Source labels remain separate from all forward inputs. Query-family-disjoint internal Train Seen development loss selects the checkpoint; deterministic epoch shuffling, atomic Last/Best, RNG/optimizer restore and source/cache SHA checks support interrupted training. This baseline explicitly supplies observed_fraction=0 and makes no visual/history claim.

Two implementation issues were fixed before training: raw per-mode score offsets could override candidate ranking, so mode scores are now conditional log-probabilities; identical zero-residual hypotheses caused winner-take-all mode collapse, so observable straight/left/right initial paths provide distinct hypotheses and converge to the candidate. Regression tests cover both.95 tests pass, including a real CPU optimizer step and proof that changing GT/future labels cannot change forward inputs.

The frozen phase3 plan queues full A3 Seen evaluation,5-epoch seed0 trajectory training and exactly three Seen128 Stop-threshold checks (0.5/0.3/0.7). Selection uses Seen only, ranked by SR then SPL then lower failed-stop episode fraction. No GraphNav Unseen run is queued. See `reports/sbf_protocol/phase3_plan.json`. No learned-head performance gain has been measured yet; dynamic observed RGB/Depth/graph/history integration is still incomplete.

## Unified SBFNav baseline verified in full — 2026-10-10

Full Unseen completed:2697 episodes,SR21.5054%,SPL15.8907%,OSR38.7468%,NE47.5653m,nDTW23.5093%,sDTW7.5936%,16.6507 mean movements,47.7632ms policy latency,36.96min evaluation. Every full Seen/Unseen pose array matches the historical trained-source/checkpoint replay exactly (0 coordinate discrepancy). No oracle target-setting call is made. CSV/JSON and acceptance thresholds are recorded in `reports/sbf_protocol/`.

Level2 minimums are now measured:Unseen SR20.4301%,SPL14.3016%, plus required pipeline and Seen/Stop/safety checks. SBF has no explicit learned policy Stop here; its false-stop rate is null. This limitation is disclosed, not converted into perfect Stop performance.

First A3 Seen128 bounded Top1 smoke:SR38.2813%,SPL37.6067%,OSR44.5313%,NE29.7557m,9.625 movements,3.2127ms policy latency;12.93s evaluation. On exactly the same128 episodes, SBF SR44.5313%,SPL36.8868%. A3 has−6.25pp SR and+0.7199pp SPL; direct paths are efficient but target selection misses some successes. This provides a real controlled Seen diagnostic, not an Unseen/closed-loop completion claim. Base/Large visual controls are now running; full A3 Seen and learned-head training remain queued after frozen features.

## Controlled vision comparison and first trajectory failure — 2026-10-10

P2 Base224/Large384 fair controls completed. Train dev R@1 selects Large86.17% vsBase82.98%; Unseen cross-frame retrieval R@1/R@5:Large73.04%/93.91%,Base72.17%/92.17%. Large's net R@1 gain is only1 of115 IDs and its paired95% CI spans0; Footpath/TrafficRoad degrade. No navigation gain is claimed. Training50.50s/56.28s,peak1.94GiB/4.10GiB, with unequal preexisting cache hits disclosed. See `docs/P2_CONTROLLED_BASE_LARGE_20261010.md`.

Preserved A3 full Seen bounded waypoint result:SR40.8907%,SPL39.7682%,OSR47.6923%,NE31.4657m. Relative to SBF full Seen:SR−2.6316pp,SPL+3.6498pp; efficient direct paths but insufficient target selection. No GraphNav Unseen evaluation has been performed.

Seed0 trajectory/rank/Stop head trained5 epochs on115559 Train/12950 internal-dev states,40.57s training+4.06s validation,peak54MiB allocated for the cached head. Dev Loss improves, but initial Seen128 SR only11.72%/10.94%/16.41% at raw thresholds0.5/0.3/0.7; failed policy stops83.59–89.06%. These failures are retained in `reports/trajectory_seed0_20261010/`. Weighted BCE positive-weight20 shifts model odds; interpreting raw sigmoid as Stop probability was incorrect. Analytical−log(20) calibration is now implemented and regression tested; it is not yet empirically validated. The same head/three thresholds will be checked on Seen only before any Unseen test.

Current-view landmark encoder implemented:actual384 RGB/vertical-depth proxy, all public named polygons rather than Top20 destination crops, source/boundary/depth uncertainty and preserved context. Pixel-keyed frozen features can be reused; evidence history resets each episode. No offline/future Unseen gallery is supplied. Interface remains unmeasured until the queued real-current smoke completes. Dynamic belief can consume frozen cached A3 fields without re-encoding BERT.100 tests pass. Subsequent queued jobs preserve an isolated Python source snapshot so ongoing development cannot change late imports mid-experiment.

## Stop correction and planning isolation — 2026-10-10

Analytically corrected weighted-Stop odds still fail Seen128:threshold0.3 SR18.75%/SPL15.95%,0.5 SR17.19%/SPL12.52%,0.7 SR18.75%/SPL11.80%. Threshold0.3 is selected by the prior SR/SPL rule. At0.7,OSR67.97% but SR18.75% shows arrival is often lost after goal/mode changes; calibration alone is insufficient. Failed runs are preserved without rerunning Unseen. The next3 diagnostic controls fix the initial A3 Top1 goal and isolate learned trajectory versus Stop; no-stop/direct-waypoint variants are explicitly incomplete ablations.

Real Large P2 current-observation interface smoke passed on2 generated native384 current views with2 named Rail regions each,finite[2,256] vectors and geometric/depth-confidence0.64–0.66. This verifies only the actual encoder/current-view/cache/history interface; it does not establish all-category semantic alignment or dynamic-belief gains. Source snapshots now isolate all queued Python imports.101 regression tests pass. All source changes, raw/calibrated failures and reports remain on the independent experiment branch.

## Planning controls and visual-binding corrections — 2026-10-10

Three fixed-goal controls confirm the first learned path decoder is harmful:fixed-goal+learned path/Stop SR30.47%,fixed-goal+learned path/no Stop33.59%,versus direct-goal+learned Stop38.28% (same as preserved A3 quick baseline). The direct-goal control reduces movements9.625→6.719 and modestly improves SPL37.61→37.75%,but its False-Stop Rate53.52% remains substantial. It is selected for full Seen validation; learned-path benefit is not claimed. See `docs/TRAJECTORY_STOP_SEEN_DIAGNOSTIC_20261010.md`.

P3 review found visual/geometry association loss from separately linear-averaged node features; joint nonlinear fusion before pooling now preserves each visual-to-landmark binding. Auxiliary current-visible named landmarks outside the static referenced set can affect all784 candidates. Low registration quality attenuates residual amplitude instead of merely changing normalized weights. History now separates accumulated averaging mass from bounded registration quality, preventing biased means and overconfidence from repeated views. All are untrained corrections covered by regression tests; no visual-belief gain has been measured.

104 tests pass. Stage6 queues frozen selected-Large vectors from trusted actual Train Seen/Val Seen frames,2470 public Seen A3 fields and the selected direct-goal/Stop full Seen control. No Unseen visual data/gallery/cache or model-selection run is queued. A public-query preparation smoke initially guessed physical directory `revised` and failed before writing rows; fixed to load `data.version=processed_citynav` from the exact SBF config. Failed attempt retained; corrected v2 source SHA matches the verified Seen annotation and keeps2470 separate labels/public rows.

## P3 observed-state data and precision guard — 2026-10-10

Selected P2 head encoded17493 trusted Train ROI records in139.22s and6170 Seen records in80.50s,peak3.68/3.71GiB;9188 Train patches reused from immutable caches. No Unseen visual cache is assembled. Current/past evidence contains2635 training states from1673 approved episodes/1671 public priors,925 Seen states from570 episodes; these are explicitly along-human-observation subsets, not full navigation results or initial-only RGB. History is limited to the same episode and observations≤current step, with query-family split and goal labels separated from forward inputs.

A3 dense-cache audit found FP16 goal-logit rounding can create ties and change ranking. No P3 training had started. New FP32 score caches are created separately, while old head inputs/results remain immutable. Dynamic datasets reject rounded legacy goal-score caches and verify vector/geometry/query/chunk SHA. Cached candidate-context FP16 storage is now explicitly matched in the trained-head runtime.108 tests pass, including future-history rejection and GT-label mutation leaving every dynamic forward input identical.

Full Seen direct-goal/learned-Stop control:SR40.8502%,SPL39.7991%,OSR47.1255%,NE31.5249m,6.802 movements. Compared with A3 bounded Top1,net−1 successful episode and+0.031pp SPL,while movements drop28.1%. Individual successes both gained/lost from early Stop are retained, not described as an SR improvement. Best geometry baseline is still preserved A3. P3 residual training and actual RGB/history closed-loop integration remain outstanding; no GraphNav Unseen or acceptance claim yet.

## P3 residual training recipe frozen — 2026-10-10

Exact FP32 dense-score cache recreation is complete for Train and Seen; original rounded caches/head results remain untouched. Dynamic trainer reuses frozen BERT/A3/P2 features, with real current/past ROI vectors and source-relative geometry, no encoder re-execution each epoch. Checkpoints use query-family-disjoint internal-dev query-macro R@1 then NE, not Loss alone. Per-query macro metrics keep variable prefix counts from changing query mass; these along-human-prefix diagnostics are explicitly separate from full navigation.

Three main recipes are predeclared in `reports/sbf_protocol/phase8_plan.json`:graph-only residual,current-visual-only residual,graph+visual+causal history. Same batch64,BF16,workers4,seed0,5 epochs; head LR2e-4,gate LR2e-3,Gaussian20m target loss with0.05 KL preservation. A3 remains frozen and gates initialize zero. A real CUDA one-batch forward/backward smoke must pass before the3 full training runs. History deletion of the joint model provides the graph+visual/no-history ablation later.109 tests currently pass, including a real cached visual residual optimizer step proving A3 remains frozen and all GT/future labels stay outside forward inputs.

## Visual node association audit and v2 repair — 2026-10-10

First P3 runs are retained in `reports/dynamic_v1_20261010/`:Seen prefix query-macro R@1 gains are graph+1.23pp,visual+0.96pp,joint history+0.58pp,but NE worsens1.36–1.76m. These are925 human-prefix states/570 queries,not navigation metrics. Audit found all current regions had remained auxiliary visual nodes,so GAT reference states had not received images. v1 joint results cannot be presented as validated visually updated GAT.

v2 associates observations by public source name and full contour (never destination ID),removes duplicate auxiliary references and injects confidence-weighted visual state before GAT message passing. Regression tests prove changing actual evidence changes the graph residual even with visual output gate disabled,while unknown evidence preserves static graph outputs. The same3 recipes will be retrained in separate v2 outputs using immutable frozen features.

Online cache now persists only256D vectors keyed by exact current pixels,head SHA and actual backbone/precision protocol; large per-ROI patch tokens are not saved on every navigation visit. History is episode-local; cache reuse occurs only after observing the current pixels. Tests verify repeated pixels avoid encoding and only owned transient PNGs are removed.112 tests pass. The initial one-shot head/encoder results remain untouched; new real cache/node-state smoke must pass before v2 full training.

Internal P3 dev is disjoint for the residual stage only:the preserved A3 had already trained on the original full Train Seen. Val Seen remains the independent architecture/acceptance selection set. No new Unseen model evaluation or tuning has occurred.

## 2026-10-10：真实节点关联与闭环接口

V2 三组五轮训练完成；所有实际视觉输入均按当前/历史 cutoff 构造，Seen 前缀约45%状态包含已观测引用地标。指标见 `reports/dynamic_v2_20261010/comparison.csv`，尚不代表闭环增益。实际四 Episode CUDA 在线接口已通过；加入当前384px RGB/Depth环境边界、公共名称+完整轮廓节点关联、全784候选残差与按像素哈希持久化视觉缓存。114单测通过。下一步：四组动态图 Seen128 与三个预测目标收敛/Stop门控对照，不依据 Unseen 选择。

## 2026-10-10：Seen128模块消融、轨迹收敛及吞吐

统一128 Episode（Manifest SHA完全一致）比较已完成。A3 SR38.28125%/SPL37.60669%；GAT SR41.40625%/SPL40.66671%；Visual SR41.40625%/SPL40.52153%；Joint-current SR41.40625%/SPL39.92132%；Joint-history SR41.40625%/SPL40.60267%。RGB/History尚无额外SR增益，结果见 `reports/closedloop_seen128_20261010/`。

三种收敛控制保持同样49个成功：完整学习路径SPL36.13009%，25%学习残差SPL37.54340%，直接目标SPL37.60669%。25%路径残差仍真正使用学习的多模态路径/模式，平均步数7.875 vs基线9.625；预测目标到达门控不读取GT，只修复早停/到达后路径漂移。Stop仍有高错误率，不能声称已经解决目标确认。

真实A3 BF16前后向（相同1024 Train Query、BERT96缓存、workers4）batch64：535.17 samples/s、3.72GiB；batch128：728.78 samples/s、7.01GiB。测的是热缓存小规模训练吞吐，不是模型准确性、完整epoch时间或视觉速度。原训练/Checkpoint均未覆盖。

补充输入审计：Unseen没有96Token截断；17%–21% Query可被现有具名规则解析。约68%–69% Query包含left/right/front等词，83%–88%包含颜色/屋顶/形状词；这些词频不是人工标注的语义类别/解析准确率，不能机械将left/right变成east/west。沿途地标匹配不能替代目标外观与语言的条件匹配。

下一步：同一25%学习轨迹+Stop接口的全量Seen五组确认，锁定候选后才开始新的Unseen全量评估。真实A3训练及动态视觉+轨迹+Stop CUDA smoke已通过，115单元测试通过。

## 2026-10-10：闭环渲染瓶颈修复

慢速全量GAT运行在318个Episode后主动中断，原预测/源码快照完整保留，未覆盖或当作完整指标。根因是无视觉分支仍渲染native384，且每步对整幅源高度栅格重复构造有效性/填充数组。修复缓存不变栅格、GAT静态字段，并只为真实视觉分支生成native384；8个实际起点逐像素RGB/Depth/有效mask/footprint完全一致，128 Episode所有轨迹/终止/SR/SPL/NE完全一致，GAT评估69.12s降至14.13s（策略延迟6.93→3.67ms）。修复后的全量运行另起 `_optimized` 目录；单位测试115通过。旧运行中断报告和精确复核记录见 `reports/sbf_protocol/`。新增10组实际成功/失败轨迹图，其中目标坐标、人类未来与全图背景仅用于离线诊断。

## 2026-10-10：全量Seen已完成部分与Stop暴露偏差对照

同一修复轨迹/Stop接口，全量Seen2470：A3 SR40.8502%/SPL39.6138%；GAT SR42.0648%/SPL40.7956%，NE32.2201m；Visual SR41.5385%/SPL39.3911%，NE31.7168m，评估1951.0s（32.52min）。GAT相对A3 SR+1.2146pp、SPL+1.1818pp，但NE变差；纯视觉SR小幅增加而SPL回退，尚不是全面增益。Joint/History继续全量Seen，不开始新的Unseen选择。318个带学习路径的优化前后轨迹完全一致，记录已保存。

118单元测试通过。新增仅Stop MLP的候选终点/10m扰动困难负样本训练：位置来自公开预测Top8和起点、标签只用于20m到达BCE；保持Trajectory/Rank/StateFusion完全冻结。只比较旧Stop、恢复Stop、恢复Stop+Top4受限探索三方案；探索不删除未知区域信念，且所有选择仍仅用Train/Seen。完整训练/Seen128队列等待当前视觉全量评估结束。

额外SBF检查：原当前工作树90测试通过、1个既有配置glob测试错误（将两份评估suite清单当模型配置）；冻结历史源代码71个兼容测试通过，另外3项新接口测试与历史源代码不兼容。初次路径/版本收集失败也保留，见 `reports/sbf_protocol/unit_tests/`，不修改SBF原分支或历史Baseline。实际使用的SBF main配置合法、5167历史轨迹精确复现。不能把这些外部检查宣称为“所有SBF测试通过”。

## 2026-10-10：配对/地图不确定性与目标可见范围审计

全量Seen初步独立报告：`P3_P4_FULL_SEEN_INITIAL_20261010.md`；逐Episode/地图CSV与2000次配对Bootstrap见`reports/closedloop_full_seen_initial_20261010/`。GAT新增132成功、丢失102，净增30；Episode加权SR差95%CI[+0.0405,+2.3887]pp，但地图等权SR差CI[-1.2057,+2.6361]pp。NE显著变差，不能称为稳定地图收益。视觉净增17、SPL回退且CI含零。119单测通过，包括配对起点/分母保护。

仅Train/Seen的GT离线诊断脚本不会进入模型：现有Seen P0实际6242帧、960Episode，目标点44.25%帧处于几何FOV、42.68%帧处有非黑源像素；初始观测仅23.85%/22.40%，人类前缀任一帧79.38%/76.77%。这是抽样/步长/截断的人类轨迹诊断，不是全量GraphNav沿途可见率，更不是语义可见/遮挡真值。远于100m时几何FOV仅14.23%，因此不能只依赖目标附近视觉。报告：`reports/p0_target_exposure_seen_20261010/`。

三组独立A3从头训练Seed0/1/2已预注册排队（`phase14_plan.json`），保持历史A3结构/Batch16/BF16/lr2e-4/5轮，仅使用Train与Seen选择Best，复用冻结BERT缓存。它们只验证几何主干的训练随机性；不能伪称为三组独立完整系统训练。队列等待Stop对照，避免GPU竞争和覆盖历史结果。

## 2026-10-10：五组全量Seen结束、P1标签修复与性能检查失败

五组相同学习轨迹/Stop的Seen2470对照完整：A3 SR40.8502/SPL39.6139%；GAT42.0648/40.7956%；Visual41.5385/39.3911%；GAT+Visual-current42.4291/40.4542%；GAT+Visual-history41.9433/40.5579%。History没有进一步收益。完整逐Episode/地图/配对CI：`reports/closedloop_full_seen_modules_20261010/`。新GraphNav Unseen仍未评估。

Stop修复训练5轮完成，314544训练/35280内部Dev候选终点与扰动样本。全量Seen仅新Stop：SR42.0648%、SPL40.7956%、7.3607步；加Top4探索：42.1457%、40.6910%、7.4866步。旧Stop同GAT为42.0648%、40.7956%、7.8138步。Stop修复缩短等待，不解决大多数目标识别错误；探索仅净增2成功且SPL下降，无全面收益。结果保留于`reports/stop_recovery_20261010/`及本地`runs/stop_recovery_full_seen/`。

P1检查发现顶点平均代替面积中心：原1150唯一目标中34点不在自身Polygon内，最大面积中心偏差16.01m。仅修复新P1生成器：面积中心在区域外时使用内部代表点；双向目标到引用轮廓的有符号边界距离均需>=5m；逆位移损失使用正确公开轮廓中心而非顶点均值。冻结A3特征约定、旧数据/权重/结果完全保留。新18244方向样本/9122含增强对/4100原始关系家族、24真实Train地图，拒绝76边界不合法对。参见`p1_v2_label_audit.json`/`p1_v2_pair_manifest.json`；h128、BERT96、非relative架构的P1/匹配G0-G1正在独立对照，不读取新Unseen。

反事实脚本也修复硬编码h256及全局方向词替换：按Checkpoint真实结构加载，仅交换关系短语，保留地标名称中的方向词。反射点明确只是人工反事实诊断标签，不是真实地标/导航GT。Seen网格Oracle审计：2426去重Query全部在原28x28候选中具有20m内点（平均最近4.96m，最大9.66m）；Seen不能归咎于候选网格表示能力不足，不新增更大网格搜索。

新视觉性能优化（跳过不用的SBF9通道canvas、复用纯公开候选—轮廓几何）的全量vs128子集等价检查失败：10/128轨迹不完全相同，SR全部一致，但最大路径差4.64m、部分NE差<0.00015m。原报告保留，尚不宣称性能优化通过；同一128顺序的旧源码对照排队，后续视觉机制运行必须等待布尔验收True。队列增加显式JSON检查，文件存在但验收失败不会继续依赖实验。纯GAT Stop及P1训练独立于该视觉路径，继续正常执行。

GAT固定权重Seen128：完整边53成功、自环52、清零边特征53；小样本不能证明几何边特征有稳定增益。当前视觉的原向量/零向量/同类别当前帧内交换对照已预注册并等待等价验证；区分真实像素向量与可见子集/几何效应。GPU采样与阶段窗口报告`reports/gpu_initial_snapshot_20261010/`；短epoch的2秒采样可能只有0–1点，不能据此断言训练利用率。原外部SBF配置测试错误通过只读schema审计核清：当前23模型配置+2评估suite全部变体合法，冻结源16模型配置合法；旧失败记录保持不变。

## 2026-10-10：同序等价通过、机制负结果与完整系统随机种子

同一Seen128顺序的优化前后所有位姿、终止与导航指标完全一致，`reports/sbf_protocol/native_boundary_same128_exact_equivalence.json`验收True。先前全量vs子集的10个细微差异原样保留，说明存在Episode顺序/缓存相关数值变化；具体编码批次机制尚未直接证明，不能把猜测写成已确认原因。

固定联合权重、同一当前帧ROI/几何/置信度的视觉机制测试：真实向量/清零向量/同类别帧内交换，SR均40.625%，SPL分别39.0822/39.1238/39.1488%。真实向量没有胜过控制，尚不能把联合分支的Seen改善归因于RGB外观识别。报告`reports/visual_mechanism_seen128_20261010/`；GAT完整边/自环/零边特征报告`reports/gat_mechanism_seen128_20261010/`。

P1 v2公平h128/BERT96/非relative、同batch16/seed0/5轮：Scratch Seen R@1=40.8904%、R@8=75.0618%、NE31.5274m；初始化组39.1179%、74.4435%、31.4696m。Seen128闭环SR41.4063%→38.2813%。P1方向交换响应改善但真实任务无收益，不采用其初始化。保存`reports/p1_v2_matched_20261010/`，未读取新Unseen。补充第三且最后P1主要方案：同地图/类别、相似距离/面积、严格错误方向的实际地标作为仅Loss困难负样本，不将其他合法解误作负样本。后续须真GPU验证与公平Seen对照，不能只凭新损失宣布有效。

三组Scratch A3已完成，Seen R@1 Seed0/1/2分别40.8904/42.5392/40.9728%，随机性幅度与当前GAT增益同量级。`phase20_plan.json`预注册三个完整下游系统：各自A3、动态图残差、Trajectory/Stop独立训练，固定query split seed0、相同5轮/配置；BERT及已验证P2 Large编码器/Head共同冻结，因此随机种子结论条件于这个共享视觉编码器。沿用先前Seen选择的当前帧联合推理、25%轨迹残差、Stop .3、无探索；先完成Seen，再冻结最终候选进行Unseen。最新单元测试133通过，1个既有rasterio弃用提示。

## 2026-10-10：最后P1困难负样本、冻结最终评估准备

P1 v3只给v2的相同18244行添加私有Loss标签：9262行有严格错误方向的同地图/同类别真实地标，合计25260负标签；距离比0.6–1.4、面积比0.5–2、正负中心距>40m。其他满足方向关系的多解绝不作为负样本；坐标增强同步平移并过滤越界负标签。仅增加0.1权重margin，不改变模型forward输入、原始关系家族划分与匹配A3精调配置。计划`phase21b_plan.json`等待三种子Seen完成，先真实CUDA smoke再完整5轮。最初依赖状态文件仍为running而被严格验收拒绝的排队尝试保留`phase21_prerequisite_failure.json`；新队列改为等待最终完成产物，无GPU竞争。

公开Polygon边界框索引减少每步无关地标扫描；16个实际native384帧的ROI顺序、像素、轮廓、置信度与质量逐项完全一致，`spatial_index_actual_native_equivalence.json`。这是公开地图广阶段筛选，不缓存任何未观测证据，不声称改变准确率。

`freeze_final_eval.py`将等待三种子Seen和最后P1控制完成，按Seen SR/SPL/NE预先选定主候选，并记录所有最终模块/种子/负P1候选及权重SHA；生成后必须先Commit，再运行新Unseen。若P1 v3出现真实Seen Top1改善，会拒绝直接冻结Scratch系统，先验证下游整合。`evaluate_frozen_static.py`统一去重的2426 Seen/2679 Unseen静态R@1/4/8和NE；明确区别于2470/2697完整闭环Episode分母。

远程连接恢复：原代理规则将GitHub指定DIRECT，直连不可用；独立本任务代理端口17890经可用节点成功Push到`e9fa714`，原代理配置未修改。新代码/报告会继续提交并Push，不合并main。

## 2026-10-10：真实Stop高度错误与统一最终Runtime修复

独立全量Seen动作审计`reports/rollout_constraints_seen_20261010/`：SBF/A3/GAT/Joint无新增XY或Z非法状态、无50m/10m步长违反、无horizon超限。原源数据1020个起点高度非法、1个XY非法保持不变，只允许受限非恶化恢复。发现Graph三个Head控制均有4个Episode在恢复到合法高度前就Stop，SBF为0。具体索引763/776/810/821；不能静默忽略。

修复仅用公开当前高度和地图ground：Stop需ground+10..100m；不改XY20m成功判定、不修正GT路径、不改原始起点。增加真实Runtime回归，测试原本高置信且到预测目标时非法高度不Stop、合法边界可Stop；137单元测试通过。已在飞的seed0快照不被改动；seed1/2后续源快照包含修复，`phase22_plan.json`将另起所有旧模块与seed0的完整Seen，保证最终Seen/Unseen版本一致。旧指标/失败轨迹完整保留。

`run_frozen_final_stage.py`已实现顺序门禁：必须phase20/21b/22全完成、P1公平控制结束，才依据Seen冻结最终清单并Commit，随后仅一次执行预声明Unseen模块/种子/静态对照。前置失败会显式阻断；Push错误只记录并保留本地Commit，不静默跳过实验或伪称已推送。最终性能尚未评估，目标继续active。

## 2026-10-10：完整Seed0完成、补齐公平P1导航表与表面净空诊断

新完整Seed0（独立Scratch A3+动态图残差+Trajectory/Stop，共享冻结视觉编码器）Seen2470：SR41.7004%、SPL40.3309%、NE31.3305m、7.7547步、False-Stop57.9984%。其源码快照在公开高度Stop修复之前冻结，另行phase22复核，不能与修复后Unseen直接混作同版本结果。Seed1正在实际全量Seen。当前仍无新GraphNav Unseen指标。

补齐最终公平P1导航行：phase22b将分别为G1 v2/v3缓存本模型字段并训练相同五轮Trajectory/Stop；与G0 Seed0保持同结构/样本/划分/batch/seed/epochs，唯一A3初始化差异，并报告额外P1计算预算。不能偷用另一A3的Head或把静态Recall代替闭环增益。旧最终控制器尚处waiting_seen、未写最终清单或启动Unseen，已取消并记录`final_controller_v1_cancellation.json`；v2控制器增加phase22b完成门禁后重新等待。

源DEM点位私有离线净空代理：原SBF Seen204 Episode有后续点低于表面，A3 6、Joint 8；对应1102/10/12点。原始起点没有已知表面以下情况，但源黑像素/无效高度明确计为未知。数据`seen_surface_clearance_proxy_audit.json`；复现脚本`audit_surface_clearance.py`。这是最近DEM像素代理，不是swept路径碰撞/语义或真实飞行安全真值，不进入policy/计分/成功判定。公开动作约束没有新增非法状态的结论仍保持；复杂真实可飞空间/遮挡需另有环境支持，不能伪称已具备。

增加在线真实ROI向量缓存计数，分别记录当前观测数、内存命中、持久化命中、新向量请求及实际backbone编码数，解释冷/热缓存延迟差异。只改变日志，不改变输入或网络计算。`summarize_final_system.py`仅汇总冻结候选，核验权重SHA和相同Episode分母，给出逐Episode/地图配对Bootstrap及三种子均值/标准差/范围；数值Level2/3门槛与完整pipeline/Stop/安全验收分开，不能仅凭阈值自动宣布完成。137单元测试通过，日志已保存。

## 2026-10-10：第二完整种子完成（Unseen仍冻结未开）

完整Seed1、公开高度Stop修复后的Seen2470：SR42.6721%、SPL40.8949%、NE30.7278m、nDTW29.8195%、sDTW15.2045%、8.0263步、False-Stop56.9490%。评估实际22,257次当前观测、83,696个ROI，65,809次新视觉编码、17,290持久化命中、97内存命中。真实实际视觉microbatch常远小于16，单Episode顺序因果执行限制GPU利用率；这不表示模型没运行，也不能通过虚构未来帧凑batch。Seed0/1完整阶段JSON临时独立保存在`reports/full_system_seed_progress_20261010/`，Seed0的旧高度Stop版本会另作一致性复核。

两个完整种子Seen结果尚不构成三种子稳定性或最终性能验收。Seed2继续字段缓存/训练/Seen；之后P1最后方案、公开高度修复完整确认、公平P1轨迹对照、冻结清单Commit与统一Unseen顺序执行，无需新增调参或覆盖旧结果。主候选只依据修复后Seen；新Unseen尚未执行。最终报告保持work in progress。

## Full seed1 闭环动作约束核验（2026-10-10）

实际2470个Seen Episode、19825次运动均通过50m XY / 10m Z /20步检查；新增非法XY/高度状态、越界恶化和结束时非法高度均为0。原始1020个非法初始高度与1个非法XY起点保留，采用有界恢复。详见`reports/full_seed1_constraints_20261010/summary.json`与逐Episode CSV。False-Stop的实际NE中位数36.42m，790次在20–40m、523次在40–100m、72次超过100m；仍主要表现为错误目标信念，不能将动作合法性当成导航成功。第三组及冻结前对照继续运行，未开始新GraphNav Unseen评估。

## 三种子训练计算预算汇总（2026-10-10）

A3、Dynamic Belief、Trajectory/Stop的三组五轮训练均完成；统一逐轮CSV和配置/输入/权重SHA汇总保存于`reports/full_system_training_20261010`。该表仅统计实际epoch训练与验证耗时，缓存/视觉编码单独计入队列预算。已完成Trajectory/Stop旧记录无绝对起止时间，不能恢复逐轮GPU利用率；后续运行已补充wall-clock字段，并兼容旧Checkpoint恢复后的混合历史CSV。此改动只补记录，不改变Loss/梯度/模型配置。第三组完整Seen评估仍在运行，尚无新的GraphNav Unseen结果。

## 三组Full系统Seen完成（2026-10-10）

Seed2全量2470个Seen Episode：SR41.5789%、SPL39.2060%、OSR51.9838%、NE31.3440m、nDTW29.6769%、sDTW14.6229%、平均8.0636次运动、策略延迟68.5132ms、False-Stop58.3437%，实际评估1631.885s。Seed1当前SR42.6721%、SPL40.8949%较高；Seed0仍须用公开高度Stop修复后的Runtime重新确认，不能混合旧Runtime结果构造最终三种子表。三组原始配置/逐轮记录/全量Seen结果都保存在`reports/full_system_seed_progress_20261010`。phase20完整结束，phase21b困难负样本P1训练已启动；新GraphNav Unseen尚未开始。

## P1 v3困难负样本公平对照完成（2026-10-10）

完整Seen最佳R@1：Scratch40.8904%→P1v3 36.8508%；R@8：75.0618%→73.2894%；NE31.5274→32.6913m。Seen128有界导航SR41.4063%→36.7188%、SPL40.6522%→36.1198%。预训练67.42s+微调269.32s未带来增益，不提升为主干，保留全部负结果。独立报告`docs/P1_MATCHED_V3_HARD_20261010.md`及CSV/JSON已完成。phase21b全部结束，phase22公开高度Stop修复后的完整Seen确认已启动；新GraphNav Unseen尚未开始。

## 公开高度Stop修复实际A3确认（2026-10-10）

A3完整Seen2470评估完成：SR40.8502%、SPL39.6139%、NE31.4762m，显示精度下与修复前相同。旧四个非法结束高度Episode（763/776/810/821）全部通过有界额外运动恢复到合法高度，全量非法结束高度为0，各case成功与失败判定均保持。详见`reports/sbf_protocol/altitude_stop_guard_actual_a3_seen.json`。该修复解决动作/Stop合规问题，不声称定位提升；后续GAT/视觉/历史/Seed0确认继续，Unseen尚未开始。

## Seen Stop失败机会分解（2026-10-10）

Full seed1的1416次失败中1193次从未在已观测位置到目标20m内、208次曾在合法高度经过后离开、15次仅非法高度经过。主要瓶颈仍是目标信念，不能全部归因于Stop；合法经过失败中的26个Episode曾达到Stop概率阈值，但模型预测目标到达门控等仍拒绝。私有标签仅用于完成推理后的诊断，不改模型。独立报告`docs/STOP_FAILURE_DIAGNOSIS_SEEN_20261010.md`及全部CSV/JSON保留。诊断脚本兼容原生SBF无Head字段，并增加4个回归测试，当前全套141通过。

## 两路独立测试准备（2026-10-10）

用户授权并行测试。完成共享ROI与固定tmp并发风险修复、两路评估/训练屏障调度及并发时延标注。全部144项回归测试通过。当前运行phase22不重启，仅接管尚未启动phase22b；模型参数、候选选择及Unseen冻结协议不变。说明见`docs/PARALLEL_EVALUATION_20261010.md`。

## 并行调度等待队列交接完成（2026-10-10）

仅终止没有任何子进程或已启动job的phase22b旧waiting调度器（PID751669）；原状态/SHA/Manifest SHA保留于`reports/sbf_protocol/phase22b_parallel_waiting_handover.json`，未中断模型推理或训练。新调度器已实际启动、保持waiting且evaluation_workers=2；仍等待phase22全量确认完成。当前phase22已有模型进程不重启。最终冻结清单也将使用两路独立评估，并标注并发时延。

## 四组修复后Seen确认与缓存实测（2026-10-10）

A3/GAT/Visual/Joint-current均完成相同2470个Seen Episode；原始JSON和前四组CSV保存于`reports/altitude_guard_seen_20261010`。Joint-current SR42.4291%、SPL40.4542%、NE31.5900m，单进程实测642.306s（10.7min）、策略22.0814ms；80716个实际当前ROI中磁盘命中80112、内存命中601、实际仅新增Backbone编码2张。不能把这次热缓存时间与首次/大量未命中Full种子27–28min当成同条件算法加速。尚待History及Full seed0确认；phase22b已配置两路独立评估、训练串行。Unseen尚未开始。

## 冻结后评估优先顺序（2026-10-10）

最终清单生成时，将仅由Seen确定的主候选与A3学习轨迹基线放在前两路，尽早报告实际Unseen验收结果；其它固定消融仍全部执行，稳定保留原顺序。没有减少或增加候选，没有读取新Unseen结果，仍需所有Seen前置确认完成、生成完整候选权重SHA清单并Commit后才能启动。当前纯评估剩26批次（含进行中的History）、另有2缓存/2轨迹训练准备，随后完成审计与报告。

## 修复后五组Seen确认及动作审计完成（2026-10-10）

History完整Seen2470：SR41.9433%、SPL40.5579%、NE32.0475m，热缓存644.74s，80408个当前ROI无新增Backbone编码。五组同A3/Trajectory模型的配对bootstrap、地图级统计与动作审计已保存；全部无新增越界/高度/步长违规，原始非法起点完整保留。GAT较A3增加SR1.2146个百分点但NE恶化0.744m，地图级CI仍含零；Visual降低SPL，History未提高当前证据模型的SR。独立报告`docs/P3_P4_ALTITUDE_GUARD_SEEN_20261010.md`。Full seed0修复后确认仍运行；后续P1公平轨迹对照等待其完成，新GraphNav Unseen仍未开始。

## 实际闭环视觉证据覆盖核验（2026-10-10）

新增纯离线公共诊断汇总，不读取目标/未来帧、不改变模型。Full seed1/2的实际22257/22332次已观测决策中，70.02%/69.22%有可信区域，35.78%/35.33%能更新指令引用地标；约98.2%的Episode至少一次有可信区域。累计网格FOV约32%，不等同目标可见或已搜索飞行空间。History增加累计证据保留，却没有提高对应Seen SR。独立报告`docs/ONPOLICY_VISUAL_EVIDENCE_SEEN_20261010.md`及CSV/JSON已保存；不能把human-prefix覆盖率当作实际策略覆盖率。后续冻结评估不因该诊断改变参数。

## 六组Seen确认完成，P1公平轨迹队列实际启动（2026-10-10）

修复后Full seed0 SR41.7004%、SPL40.3309%、NE31.3305m，热缓存601.05s；全2470 Episode/19163次运动无步长、高度、边界、horizon违规或非法结束高度。六组修复后CSV与三种子均值/样本标准差已保存于`reports/altitude_guard_seen_20261010`；三种子SR41.9838±0.5991个百分点、SPL40.1440±0.8598个百分点（共享冻结P2，非置信区间）。所有历史结果保留。phase22结束，phase22b已实际运行P1v2本模型字段缓存，之后同配置五轮Head训练及最多两路Seen评估。预选主候选仍为Seen较优的Full seed1；完整冻结清单须待P1公平对照完成才Commit并启动新Unseen。


## 2026-10-10：完整联合方案第一轮（最新用户要求）

原有测试队列保持取消。SBFNav/HETT均不重跑。先完成可训练A3、多尺度观测视觉匹配、历史信念、轨迹及Stop联合训练的**第1轮全量Train Seen + Seen128闭环验证**，再返回指标与实测耗时，后续轮次等待用户审阅。20轮调度已配置，最终Checkpoint使用20轮中Seen SR最好、同SR比较SPL的Best，之后才冻结评估Unseen。

174项测试及真实模型/闭环Smoke已通过；完整第1轮指标尚未产生，没有性能提升声明。详见`docs/JOINT_ACTIVE_FIRST_EPOCH_20261010.md`及`reports/active_perception_20261010/joint_protocol_reference.json`。原A3检查点和已有验证结果保持不变。

第一轮工作流已实际启动：源代码Commit `7804ace`，工作流PID895677。已核验24张训练地图RGB/Height SHA与历史SBF输入一致；正在准备冻结特征，完整Epoch1尚未开始。实时状态以`runs/joint_active_first_epoch_20261010/workflow_status.json`为准；任务使用`--first-epoch-only`，完成第1轮与Seen128后停止，不启动后续19轮或Unseen。

用户随后授权硬件支持时并行RGB测试。已启动独立Seen32诊断PID903027，对比原生SigLIP2单尺度、多尺度裁剪及多尺度Patch池化，包含真实位姿下降的配对观测和同类别困难负样本。两个实际进程同时运行时总显存峰值8.70GiB、采样GPU平均36.5%，无显存不足。诊断在主工作流进入训练阶段时退出；主模型训练参数不改，第一轮仍完整训练后审阅。结果目录`runs/rgb_matching_parallel_seen32_20261010`，这属于纯视觉条件检索，不是导航SR/SPL。


并行Seen32 RGB诊断已完成：193.78秒，164观测中18合格、6去重Query。单尺度/裁剪多尺度/原生Patch多尺度条件R@1为50.00%/44.44%/8.33%；7组相同候选下降配对未显示稳定增益。报告`docs/RGB_PARALLEL_DIAGNOSTIC_20261010.md`，完整CSV/JSON及两组代表图已归档，18组图保留本地SHA清单。主任务PID895722仍在准备冻结特征，尚未产生Epoch1指标；保持第一轮后审阅、不自动进入后续19轮和Unseen。


第一轮评估输入预核验：从当前Revised Seen原始数据与当前配置重新生成固定128 Episode清单，与历史SBF/A3清单逐项一致（Episode ID、索引、起点），SHA256为`ebef159ca4c9a0c7e9a3fcfdf83a413a7e34809166ba548c30340b6cd7ecc695`。20步、水平50m、垂直10m、高度10–100m、20m成功半径及二维距离声明协议一致。证据：`reports/active_perception_20261010/epoch1_seen128_protocol_preflight.json`。未重跑基线；历史控制器版本与SBF曾按Unseen选择权重的差异仍需披露，不能仅由清单相同宣称所有实现完全一致。第一轮输出产生后还须核验实际输出清单及协议。


用户授权继续并行测试后，完整CPU测试177项通过（5.54秒，1项warning，详见保存日志）。在正在生成缓存的已提交分片中固定抽取首部、四分位、中部、四分之三及尾部5片共320训练状态，SHA、每Episode四状态、历史步不超过cutoff、候选索引唯一合法、特征有限值/维度、无观测区域零特征全部通过。该抽样不等于全缓存验证，也不单独证明渲染来源或无所有形式泄漏。结果`reports/active_perception_20261010/parallel_cache_audit/`。主准备进程未中断，未运行Unseen或重跑基线。


用户重新授权运行1轮：完整87512状态Epoch1及Seen128已完成。训练524.28秒、166.92状态/秒、峰值分配显存1.73GiB；验证89.00秒。SR28.9063%、SPL22.9716%、OSR67.9688%、NE38.3867m；最终动态定位R@1/4/8=34.375/60.1563/68.75%。False-Stop Rate68.4685%，错误Stop占全部Episode59.375%；当前低于已有Seen128基线，无提升声明。Checkpoint、CSV/JSON保存，未执行Epoch2或Unseen。报告docs/JOINT_ACTIVE_EPOCH1_RESULTS_20261010.md。

## A3 / 空间关系辅助监督 / GAT 可控实验（2026-10-10）

按用户新指令暂停上一组视觉/History迁移测试，保留所有已完成及部分输出，未提交的旧模块工作存入命名 Git stash。当前仅开展静态语言—几何定位；不改变视觉/History/主动观察。三Seed的A3均第5轮Seen最佳，Seed0/1/2 R@1=40.8904/42.5392/40.9728%，尚不能确定超过5轮的收益。E1新增独立续训入口：恢复Seed0第5轮的模型、AdamW动量、RNG及输入SHA，明确将学习率2e-4降低为5e-5，保留原batch16/BF16/数据/结构；另存8/10/15轮，不覆盖原Checkpoint。182单元测试通过，真实CUDA一批续训Smoke通过。E2/E3及闭环增益尚未验证，计划在`reports/a3_relation_gat_20261010/plan.json`；原约41%强基线仍保留。
