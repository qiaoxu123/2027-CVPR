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
