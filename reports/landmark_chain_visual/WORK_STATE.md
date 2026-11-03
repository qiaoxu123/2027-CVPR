# Active Goal：原始GeomBelief的地标关系链视觉分支

用户Goal：基于地标→中间区域/对象→目标关系链设计并验证原始视觉分支；不使用AerialVG。固定yaw采集；以真实闭环SR为准，冻结方案后先128条Val-Seen，SR严格超过65%（至少84/128）且有配对收益才扩大完整评测。可以重训，不限残差形式。

## 已完成
- 原始工程定位：/home/20261024_1/Workspace/2027-CVPR/GraphNav；原始论文GeomBelief/GraphNav-V1，SigLIP2而非AerialVG。
- 主权重runs/graphnav_v1_stage1/best.pt，SHA00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a。标准论文28×28候选、384px观测；root MAIN_SCHEME曾写384×384稠密候选，这是另一个推理设定，必须先实际核对，不能混用。
- TRAIN22,002指令/21,303唯一map+text、24图分析完成，分支research/instruction-landmark-association-20261103，Commit2303ce122762d5c977ffe88ad03f1b7fb4a8357e已推送旧仓库。
- 完整自动初筛：直接几何9356、自名称1、间接7182、无明确锚点325、未判明5138。公开名称可绑定20959（95.26%）。本会话101条分层语义复核校正估计：直接50.79%、间接45.70%、无明确锚点1.18%、不确定2.33%；不是人类金标准，直接/间接区间约±8pp，不可当精确全量真值。
- 车辆自动解析间接5205>直接3144，建筑直接5184>间接1544。需要区域/参照实例表示，不能把地标中心当目标。语言频率不证明SR因果收益。
- 旧主工作树有3个修改与恢复脚本等未提交工件，保持不动；新Goal在此独立worktree/分支实施。

## 当前未完成
- 尚无新视觉模块设计定案、训练或导航结果；不要声称新SR。
- 需先读取graphnav/stage1.py、stage1_runtime.py、原SigLIP2 ROI/记忆实现与已有CVR/局部语义负结果，确认瓶颈。
- 审查原始模型实际对图像依赖（Real/Zero/Shuffle/Visual-off）、语言角色与目标/参照文本，及公共地图可用对象边界。不得用匿名目标实例Contour构造推理ROI。

## 下一阶段执行顺序
1. 原始代码/缓存/模型及控制器协议审计；补固定yaw显式实验适配，不改变旧默认；固定源SHA、数据物理实例分组。
2. 基于失败证据确定最小方案（候选方向：地标几何→未知中间区域→目标视觉匹配；是否非残差由测量决定），先TRAIN机制对照，比较geometry/text-only、Real、Zero/Shuffle，证明视觉贡献。
3. 明确方案后真实GPU Smoke/吞吐、seed0小规模训练及独立TRAIN确认。不重复已失败的全局倍率/单质量规则，不盲增大Attention。
4. 冻结模型/配置/128Episode清单后，运行真实闭环Original与新方案配对Val-Seen128。严格>65%且有可信收益才全量；不在同一128反复调参后宣称泛化。
5. 正式Test只在最终冻结后，遵守freeze manifest，不依据Test反馈改模型。没有独立收益则保留负结果、停止扩大。

约束：原始指令保留；GT仅TRAIN损失/事后评分，不进入正常策略；不使用未来图像；既有暂停任务不自动恢复。固定yaw0表示朝东非北朝上，XYZ不变；原生栅格信息上限不因放大而增加。语言left/right/front/behind参考系未知时不硬编世界方向。

## 2026-11-03第二阶段权威更新（覆盖上文“尚未采集”旧状态）

- 固定yaw适配器graphnav/fixed_yaw_observation.py已完成，原正常入口未改。
- 24物理实例/16图/87参考轨迹帧已真实采集；数据server目录：GraphNav/runs/landmark_chain_visual/fixed_yaw_pilot_24_s0。相机yaw0、agent姿态不变。目标标签独立private文件，仅监督/事后评分。
- 同RGB冻结SigLIP2原ROI32/64/128及小ROI8/16/32编码都完成，目录fixed_yaw_features_24_s0与fixed_yaw_features_smallroi_24_s0；大文件不提交。
- 指令角色由当前会话AI整理，非Qwen。任务显示类别元数据，非严格盲解析金标准；保留原文/不确定方向。
- 原ROI22帧无任何ROI，小ROI13帧；GT点几何/深度代理支持但无ROI10→4。近20m候选支持25→38。
- 87帧空间Top1全文3→11，小ROI目标文本13、分尺度11；小ROI全文shuffle11、zero10。视觉性能增益未证明；新分尺度规则不作为已有效方法。共同25帧NE全文77.58→74.42，目标文本90.30、分尺度88.78。
- reports/landmark_chain_visual/pilot24/REPORT_zh.md、所有逐帧JSON、SHAmanifest及24实例图册cases/index.html已生成。相关17 tests passed，源码compile通过。
- 下一步：观测内细粒度位置候选/直接分布头+独立TRAIN训练与确认；不要继续搜索本24例规则权重。尚无新任务训练权重、Goal Belief指标或SR，不得声称>65%。

## 2026-11-04第三阶段更新（训练已实际完成）
- 新模块：graphnav/local_chain_position.py、local_chain_observation.py；scripts/cache_local_chain_training.py、parse_local_chain_roles.py、train_local_chain_position.py。原模型/默认入口不变，SigLIP2冻结，无AerialVG。
- Server root GraphNav/runs/landmark_chain_visual：fixed_yaw_train400_s1（400目标、1454帧，排除24开发目标），chain_train400_features（完整输入SHA、地图划分），chain_head_pilot_s0（best/last/训练日志/一次确认），chain_head_smoke_s0（8batch Smoke且恢复到第2轮）。所有任务已终止正常完成，不要当仍在训练。
- 18fit地图351目标1271帧；3dev地图25目标92帧；3confirm地图24目标91帧。确认标签单独CLI路径，训练未读取；确认前freeze_for_confirm已封存。
- Main训练seed0 lr.001 batch8三轮，最佳1轮devR1 81.52 vs冻结A3 78.26；11tensor更新；25相关tests passed。
- 新头将784全局原A3假设与144当前观测局部位置假设联合归一化，无旧±.197上限；局部3窗口1/3/7patch，不是高分辨率重新编码。
- 一次确认：Real GoalR1 59.34 NE24.761 vs A3 53.85 NE34.074，纠正9误改4；Shuffle60.44 NE24.803，Zero56.04，NoGeom56.04，FullText59.34。目标组95CI含0。无法证明视觉对应或关系语义真实增益；Car净收益0。不要把GoalR1写SR。
- best SHA a2a850160452ecc13f7b7b15402075b7e2f30f287181a7254a5186e9fa762ad5。原主权重仍00f2acc...未变。reports/landmark_chain_visual/train400保存完整中文报告/实际日志/权重说明/图表/确认json。
- 下一步最有证据假说：局部损失仍可借几何/先验捷径，需分离视觉专属分数并加独立视觉世界位置监督（仍弱监督，不能称像素实例标签），在TRAIN开发并用新物理目标确认。不要重复使用当前confirm24调参。必须补完整原Stage1 RGB+历史对照与控制器runtime；当前仅A3基线，尚无新闭环SR。只有冻结方案ValSeen128至少84成功且配对改善才全量。

导航端提醒：Goal要求最终导航，不能长期停留离线。下一阶段还须优先接入原控制器并补完整原Stage1 RGB+历史同观测对照，可先以冻结TRAIN小闭环检查执行机制（不作为正式性能），用真实动作/Stop结果判断局部候选离线收益是否可执行。当前未实现该runtime、未跑新闭环；不可声称原系统SR被改善。

## 2026-11-04第四阶段更新（导航已实际完成）
- 新graphnav/local_chain_runtime.py与scripts/evaluate_local_chain_navigation.py；原正常入口未改，shared rollout使用实际预测轨迹采观测，不读参考未来。28相关tests passed。
- Server navigation_train24完成Original/A3/Real/Zero/Shuffle各24条，共120真实闭环；navigation_smoke2仅2条接口检查。无存活训练/导航任务要等待；不要重启已完成任务。
- 原完整RGB+历史13/24 SR54.17 SPL.5173 NE34.265；A3同13/24；Real15/24 SR62.5 SPL.3990 NE28.550，Zero14/24，Shuffle15/24。Real/Shuffle逐任务成功集合完全一样。SR只按原工程最终XY20m，不强制Stop；Real无Stop，不能称停止已解决。
- Real全20步，无Stop；last预测Goal距离中位2.932m(原.000137)，连续Goal位移中位3.639m，Goal丢失10轨迹15次；Shuffle14轨迹27次。9失败末次Goal全错、7从未正确、2曾正确。不能只修Stop；不要调阈值。
- 真实120轨迹、参数/代码SHA、归因、6实例18决策位姿重建PNG/HTML、报告与图表已保存reports/landmark_chain_visual/navigation24。重入resume验证所有已完成Episode跳过，轨迹/指标SHA一致。大文件server保留；图册RGB是实际已访问决策姿态确定性重建，原runtime未记录逐像素SHA，已诚实标明。
- 下一阶段优先：把联合全局/局部位置概率投回原固定世界784格，保持原控制器候选/上下文/熵支持，避免相机移动后局部点集变化导致到达距离无法收敛；这是位置表示改进，不改Stop、不锁上一帧Goal。先TRAIN开发，不用旧24确认组调参。
- 同时针对图像捷径：独立视觉位置评分/弱世界Goal监督，分离几何/先验；从实际语法路径抽中间参照NP而不是全文context。全fit1256有效帧局部特征paircos中位.446，并非全同，先不要盲目多crop/换大模型。新物理目标确认。
- 后续最终必须新冻结方案128 Val-Seen配对，严格至少84/128且有相对收益才全量。当前TRAIN24不是Val门槛，无新Val/Test。

重要确定性修正：Direct候选网格不含相机光心，最近局部点距agent约16√2·h/383，合法h≥10时至少.5908m>.001到达Guard；所以选局部点时几何上无法Stop。原控制器没问题，是新接口设计错误。下一步必须将概率投回原固定世界784格，Direct仅保留负对照；不用放宽Stop或上帧锁定掩盖。已记录~/.Codex/lessons.md。

## 2026-11-04第五阶段更新（表示错误已修复）
- 新graphnav/local_chain_raster.py：联合概率可微双线性投回原784格，mass守恒、原dense_context/坐标/entropy支持，local_chain_runtime默认field_mode='grid'，direct仅显式负对照。原Stop与控制器不改；32 tests passed。
- Server navigation_dev25_grid完成Original/Real/Zero/Shuffle各25条；navigation_dev25_direct完成Real25，共新125轨迹，全部正常结束，无存活job需等待。
- 同25 TRAIN开发目标（不是原确认24）：Original21/25=.84 SPL.8059 steps7 Stop25；Direct18/25=.72 SPL.4925 steps19 Stop2；Grid20/25=.80 SPL.7234 steps8.16 Stop25，Zero19/25，Shuffle18/25。Grid相对Direct+3/-1，修复有效；相对Original+0/-1，仍不能部署为提升方案。不要把TRAIN80%当Val65门槛。
- 逐决策现在保存observed_rgb_sha256/camera_pose/Footprint；case6797重建Direct/Grid帧SHA与实际输入内容日志一致。reports/landmark_chain_visual/world_grid_fix含全轨迹/冻结清单/图/QA/报告。
- 原nav24空间支持后验分析：Real7neverCorrect失败中6几何/5深度/6近20m空间候选，Original11中10/9/9。仅代理，不能等于真实可见/实例命中；不能假定neverCorrect全是未观测。
- 下一步：先输入级Gray RGB与像素块Shuffle控制（候选/真实深度/质量保持不变），因为featureShuffle也动位置编码；再独立视觉得分弱世界定位监督/必要时视觉语言对比，切断几何/先验/相机中心捷径。参考分支应提取实际中间对象/区域NP，目前只是全文context不能称完整链。最终Goal损失应在投回的固定世界格上，保真/argmax目标对齐。新方案用新物理目标确认，别再调旧confirm24。
- 全目标仍要求最终导航：未跑Val128，需冻结后至少84/128且有配对收益才全量；无Test访问。原RGB/SigLIP/A3/控制权重均保留不改。

2026-11-04 continuation: 19 related tests passed. Both V2 full/reference 3epoch fits complete; reference output visual_chain_v2_reference_s0_r1, best sha5a33d705d3ca95e8700052b314ffbce6d47fb5e2640f738209b3bae0589111e2. Reference dev92 R1 .8043478 NE18.4114 +6/-4; full epoch3 .7934783 NE18.7539 +5/-4. See V2_PROGRESS_zh.md. Next actual V2 dev25 closedloop/content ablations, no Val yet. Input content controls completed: gray19/25, patchshuffle20/25 versus real20/25. All new training jobs terminal.

V2 real closedloop completed (125 trajectories total, 5x25): Original21/25, reference19/25, full-context19/25, gray17/25, pixelshuffle17/25. Original->reference +0/-2, same two harms8123/8125 as full-context; reference4neverGoalcorrectfailures/2oncecorrectlost. No Val. New reports v2_navigation25, reusable summarize_chain_navigation.py validates identical first poses and observed RGB SHA across all five. Next inspect actual failure8123/8125 observations/localmass; training Goal precision/preservation hypothesis must be evidence-backed before further iteration.

2026-11-04 next continuation: audited two added harms8123/8125 with actual decision-pose RGB reconstructed and raw SHA matched every selected frame. Both Ground. 8123 lossstep1 mass.2203 nearestGTpoint50.73m but targetContour17.1% overlaps (22670m2 extended area); cannot call unseen target. 8125 lossstep3 mass.3014 nearest102.59m contour overlap0, laterlocalmass.865 whilewrongGoal171m. Actual images show similar grass regions. New posthoc script audit_chain_goal_losses.py and v2_goal_losses reports/HTML/PNGs. Next priority: TRAIN-fit on-policy hard negative observation acquisition versus teacher-path cache, then region-versus-point evidence learning; do not simply add another gate or fit these dev cases. No Val, no active training jobs.

2026-11-04 on-policy augmentation complete:32TRAIN-fit targets8/category15maps ->314actual decision images with rawSHA equality.101noGoal20support/213supported; oldfit542/1271unsupported -> not evidence original lacked negatives. Frozen SigLIP cache onpolicy_fit32_features complete partition allfit inherited. ReinitializedV2 reference+314 mix,477optimizersteps seed0 bestR1.793478 NE18.8927; actualdev25closedloop18/25=.72 OSR.84 SPL.6301 NE17.0688, +0/-3 vsOriginal (8123,8125,14739), previousV2.76. Reject scaling this augmentation. Reports onpolicy_fit32.22tests passed. Next principled audit local/global probability area/density duplicate counting before further training; no Val, all jobs finished.

2026-11-04 world likelihoodV3 implemented+trained+100closedloops complete. New world_likelihood_evidence.py makes zero scores exactlyneutral, averages splattedlogenergy(noaddedpriorcopy), ignoresoutofbounds. SyntheticV2zeroevidence counterexample changespeak0->400 localmass.4395; artificial repeatedpointnotactualfrequency. V3sameoriginal1271fit/noextra cache,3epochs477updates,bestepoch1SHA5a77073db071de5f46ec4a275d9401919767b19cc7a2aba7fac16115b96735fa. DevR1.8043478 NE17.1936+7/-5. Navdev25Original21,V3real20,gray19,pixelshuffle18; real0correction1harm8146, former8123/8125nowpreserved. StillbelowOriginal -> noVal.25relatedtestspass. Reports world_likelihood_v3. Next quantify TRAIN-fit visual localization precision/support bycategory and semantics, notmorethreshold/devepisodepatches; eventualfreshconfirmation required. Allcurrentjobscomplete.

2026-11-04 precision/preservation continuation complete: audit_chain_fit_precision.py exposes fit351targets1271frames only363support5m,729support20; trainedvisualTop1within5=91 vsnativecos104. New TRAIN-only precision_preserve_loss.py sameV3arch/data477updates: fineCar94rows/epoch sigma3 +GTloss-only priorcorrect807 KL. FitCar5top1 47->50/102, harms81->64; Dev92R1.815217 NE17.8618 bestepoch1. Realdev25SR20/25 sameV3,Original21/25; harm8146 still, no recoveries. Gray19,pixelshuffle18. NoSRgain->noextraepochs/noVal. New confirmation map physicalgroup inventory afterexclude424:65Building61Car3Ground0Parking; onlycounts checked no predictions. Next audit native pixel scale and fineROI encoding viability, notmorelossweights. Server ckpt world_likelihood_v3_precision_s0 sha6a36eaa63926e8a0e0625041c611eb3a1ef7da2b4cde3c14faf91b513ce4843d. Alljobscomplete.

2026-11-04 major data finding: teacher cache only checkedaboveground. Fit1271 includes197<10m,174>100noninitial. Dev92=13+11outofrange. Filtering poseonly retaininginitial yields900fit, notoverwritingcache. SameV3architecture159replacement updatesx3 baselineunfiltered controlvsfiltered: realdev25Original21,control20,filtered22 (+1/-0,14744correct),gray21(+1/-1),shuffle19(0/-2). Filterbestepoch1SHAe5a43a59d3695397e0453a677a7aebd63552b55ae7c5fa354c9f5a88a63b2435. Devselectionstillfixed92incloutofrange explicitly exploratory. ROI24samephysicaltargets originallatest4/6Carbelow10=>notdeployment evidence. RepeatedlatestenvelopeROI24 support5=14/support20=16; nativefineTop1_20=10,crop48=9,crop112=10, noROIbenefit. 31tests passed. Fresh64TRAIN confirm prepared/frozen excluding424oldgroups on3heldoutmaps31Car30Building3Ground0Parking; publicroles64 parsed noGT; no confirm results yet. Next run frozen Original/real/gray/shuffle64 via server run_navigation_fresh_confirm64.sh; no fitting on these, noVal yet. Reports deployment_envelope,roi_encoding_fit24,roi_encoding_envelope24. All priorjobscomplete.

2026-11-04 fresh64 frozen confirmation completed 256 real closedloops. Original47/64=.734375 OSR50, real45/64=.703125 OSR56 (+2/-4), gray49/64, shuffle48/64. RealSRdelta-3.125pp targetbootstrapCI[-10.94,4.69] mapCI[-9.80,3.50], only3maps. Reject dev88% claim as independent gain; noVal. Real19fails:9neverGoalcorrect10oncecorrect. 11OSRonly allfinalGoalwrong,10evercorrect,10policyStop1horizon; original3OSRonly. Original hasCandidateMemory while newhead currentonly, butgraybetterreal warnsagainstblindhistoryaccumulation. New report+33tests+6caseRGB SHAverified via audit_chain_goal_losses --case-type. Next TRAIN-only datahypothesis: heightfilter removedCar5msupport102->60; replace discardednoninitial teacherheights with renderat10..100 sameXY/cameraYaw0 (explicit simulatedTRAINview, not realvisited), preserving originaldata. Must nottuneonfresh64; itisconsumed. Nojobsrunning aftercaseexportfinishes.

2026-11-04 aligned-height experiment complete:371TRAIN-fit noninitial views re-rendered at10..100m,XY/agentYaw unchanged,cameraYaw0, simulatedNOTvisited. old900+new371 same1271logicalframes canonical ordermatchesoldbatchrandomdraws. world_v3_aligned_s0 3epochs477updates bestepoch1SHA72789ec236a61f4285369454f4f34230794291c2c3873cede8f9e0350f8e193b DevR1.847826NE17.4922+7/-1; actualdev25 real/gray/shuffle ALL21/25 identicalsuccesssets +14744/-14739. NoRGBgain,nofresh64reuse,noVal. New training_instance_labels.py loss-only proxy and audit_chain_instance_supervision.py:311Carframes,61targetcore,54target+othercore;20m7731points includes139targetcore885otherCarcore (remainingunknown),5m1155includes139target159other. All registryIDs restricted actualTRAIN-fit annotations; NOTpixelgold. Private losslabels onlyprepared nottrained. Centerbias Car geometricFOV153 central20%box50=32.7% vsuniformarea4%,hypothesisnotcausalproof. FullTRAINresourceaudit processed21878,fit20190tasks3641physical,80698frames~66.49GiBfeatures,477.5GiBfree,no missingA3queries; notlaunched. Next: samearchitecture test instance-aware loss+unknown masking and TRAIN-only offset sampling tobreakcenter shortcut; showrealRGBadvantagebeforefullscale/newindependentconfirmation.37tests passed. Allstagejobscomplete. Reports height_aligned_training incl6pairedsimviewcases.
