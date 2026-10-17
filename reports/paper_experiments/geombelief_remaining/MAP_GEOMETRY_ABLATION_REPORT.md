# GeomBelief: Language-Grounded Map Geometry


Baseline: 34d297425ac5c48c036055af62f325f20126964b. Archived Polygon experiments remain read-only. Model capacity, frozen BERT, loss, seeds 0/1/2, initialization, Train-Fit/Dev object split, 15 epochs and per-epoch sample order match the archived Full Geometry runs exactly. Correct Geometry reuses those three runs; no checkpoint is overwritten.

|Path|Correct Geometry|Name/Type Only|Distance / Direction / Both|Shuffled Geometry|
|---|---|---|---|---|
|Candidate–landmark 10-D|Original|All zero|Only channel 2 / 3,4 / 2,3,4|Donor spatial bundle|
|Physical AND relation node geometry|Original|All zero|All zero|Donor spatial bundle|
|prior / prior_buildings|Original|Zero|Zero|Donor spatial bundle|
|rules / relation_fields|Original|Zero|Zero|Donor spatial bundle|
|Edges / adjacency|Zero (geometry variant, no GAT)|Zero|Zero|Zero|
|Candidate positional encoding|Absolute fixed grid|Same absolute grid|Same absolute grid|Same absolute grid|
|Language, landmark names/types, masks|Original|Original|Original|Recipient original|

The architecture has relative_position=False. Thus grid positional features cannot recover landmark centers after erasing node geometry. Grid coordinates remain necessary output coordinates and permit an unconditional spatial prior; Name/Type Only means no supplied landmark spatial geometry, not no spatial output representation. Learned name-location memorization on Seen maps remains possible and is evaluated on held-out maps.

All relative XY magnitudes, boundary distance, Inside and Extent are removed from the simple distance/direction arms. Distance is unsigned center distance; Direction is the normalized vector. This avoids signed boundary leaking inside and raw XY leaking both distance and direction. Distances to several landmarks can themselves constrain a position; that is legitimate distance information.

Shuffle uses a fixed seeded permutation of COMPLETE geometry tensor bundles (nodes, pair descriptors, priors and rule fields), stratified by map, exact node type/validity/entity schema and Train-Fit/Dev partition. It retains every spatial tensor's empirical joint distribution and landmark count. Language/name embeddings stay with the recipient. Rule fields deliberately become spatial context from a different instruction; this is an all-spatial-grounding intervention, not a pure within-query name-binding test. Singleton strata cannot be shuffled without breaking matching constraints and remain unchanged, with exact counts in manifests. This limitation reduces intervention strength and must be reported. No GT coordinate enters partition, shuffle or network.

Correct retains all geometry paths; simple arms clear other paths. Therefore Correct-minus-simple combines descriptor and rule/prior contributions, while Distance+Direction-minus-Name isolates adding those simple descriptors. It must not be interpreted as a fair test of Polygon boundary shape.

The fixed controller is the archived pretrained trajectory/Stop head. Newly trained locator contexts are a transfer distribution; navigation cannot isolate localization independently of that shift. Formal-model controller attribution uses the complete frozen C checkpoint separately, including its original visual module and trajectory weights.

Data: 2426/2679 unique Seen/Unseen queries; navigation 2470/2697. No Test inference, training or tuning. Model selection only fixed Train-Dev R@1@20m, then NE, then earliest tied epoch. Paired 10,000-draw map bootstrap and seed means use all seeds; four Unseen map blocks do not establish broad cross-city generalization.

Navigation donor cache access is whitelisted to nodes/prior/prior_buildings/rules/relation_fields/pair_geometry before constructing policy inputs. Cached target labels are never exposed through the donor policy dictionary. Dataset targets and expert trajectories remain in the outer evaluator solely for post-rollout scoring. Oracle coordinates are an explicit exception only in separately labeled controller diagnostics.

Metrics: localization recall uses a 20 m radius; navigation SR uses final XY NE ≤20 m regardless of termination. False-Stop Rate = failed policy_stop episodes / all policy_stop episodes; if no policy_stop occurs it is undefined (empty/None), not zero. SPL uses Euclidean start-to-goal length and actual XY path length.

## Per-seed results

|endpoint|split|arm|seed|status|R@1|R@4|R@8|SR|SPL|OSR|NE|False_Stop_Rate|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|localization|val_seen|Language + Correct Geometry|0|Done|0.42003297609233303|0.6343775762572135|0.738252267106348||||31.27512907537172||
|localization|val_seen|Language + Correct Geometry|1|Done|0.40189612530915086|0.6281945589447651|0.7374278647980215||||31.84552330830229||
|localization|val_seen|Language + Correct Geometry|2|Done|0.40272052761747734|0.6240725474031328|0.7440230832646332||||31.408823947201853||
|localization|val_seen|Language + Name/Type Only|0|Done|0.3112118713932399|0.5004122011541632|0.6075845012366035||||53.89767077029587||
|localization|val_seen|Language + Name/Type Only|1|Done|0.3268755152514427|0.5103050288540808|0.6203627370156637||||49.47470051900277||
|localization|val_seen|Language + Name/Type Only|2|Done|0.320280296784831|0.510717230008244|0.6236603462489695||||52.332834669606115||
|localization|val_seen|Language + Shuffled Geometry|0|Done|0.3194558944765045|0.5288540807914263|0.6541632316570486||||44.01846308543087||
|localization|val_seen|Language + Shuffled Geometry|1|Done|0.3396537510305029|0.5449299258037923|0.6628194558944766||||41.77280650938334||
|localization|val_seen|Language + Shuffled Geometry|2|Done|0.3536685902720528|0.5507007419620775|0.6706512778235779||||42.620866088879964||
|localization|val_seen|Language + Distance Only|0|Done|0.3763396537510305|0.6063478977741138|0.7209398186314921||||33.52842312574583||
|localization|val_seen|Language + Distance Only|1|Done|0.3944765045342127|0.5939818631492169|0.7159934047815334||||32.351348476894415||
|localization|val_seen|Language + Distance Only|2|Done|0.37551525144270403|0.5923330585325639|0.7230008244023083||||32.59589959478339||
|localization|val_seen|Language + Direction Only|0|Done|0.3882934872217642|0.5964550700741962|0.7032151690024732||||35.9375088175318||
|localization|val_seen|Language + Direction Only|1|Done|0.3437757625721352|0.5589447650453421|0.6694146743610883||||37.02651619712184||
|localization|val_seen|Language + Direction Only|2|Done|0.3763396537510305|0.5882110469909316|0.7036273701566365||||35.956238245433354||
|localization|val_seen|Language + Distance + Direction|0|Done|0.3920032976092333|0.596042868920033|0.7139323990107173||||33.327402060226476||
|localization|val_seen|Language + Distance + Direction|1|Done|0.3668590272052762|0.5939818631492169|0.7197032151690025||||32.586544404271606||
|localization|val_seen|Language + Distance + Direction|2|Done|0.3557295960428689|0.5981038746908491|0.717230008244023||||33.338891232446116||
|localization|val_unseen|Language + Correct Geometry|0|Done|0.1862635311683464|0.3445315416200075|0.4494214259051885||||53.380361738868864||
|localization|val_unseen|Language + Correct Geometry|1|Done|0.20119447555057857|0.3684210526315789|0.48114968271743186||||53.745322871848956||
|localization|val_unseen|Language + Correct Geometry|2|Done|0.2123926838372527|0.3602090332213512|0.46061963419186264||||51.84313869600912||
|localization|val_unseen|Language + Name/Type Only|0|Done|0.004479283314669653|0.011944755505785741|0.018290406868234417||||215.96237479274333||
|localization|val_unseen|Language + Name/Type Only|1|Done|0.007838745800671893|0.013064576334453154|0.02127659574468085||||217.6819793836267||
|localization|val_unseen|Language + Name/Type Only|2|Done|0.0059723777528928705|0.010824934677118328|0.014557670772676373||||223.40876955788454||
|localization|val_unseen|Language + Shuffled Geometry|0|Done|0.04479283314669653|0.10377006345651363|0.1362448674878686||||158.85024251560526||
|localization|val_unseen|Language + Shuffled Geometry|1|Done|0.021649869354236656|0.05860395670026129|0.09742441209406495||||167.288694839773||
|localization|val_unseen|Language + Shuffled Geometry|2|Done|0.04329973870847331|0.08846584546472565|0.11720791340052258||||174.3689818659039||
|localization|val_unseen|Language + Distance Only|0|Done|0.20082120194102276|0.3572228443449048|0.45576707726763716||||54.42500253531772||
|localization|val_unseen|Language + Distance Only|1|Done|0.20044792833146696|0.3471444568868981|0.4665920119447555||||53.339065017563144||
|localization|val_unseen|Language + Distance Only|2|Done|0.18962299365434865|0.3355729749906682|0.44270250093318403||||54.136186459207764||
|localization|val_unseen|Language + Direction Only|0|Done|0.06942889137737962|0.13064576334453154|0.17431877566256065||||149.98608967295866||
|localization|val_unseen|Language + Direction Only|1|Done|0.05449794699514744|0.10339678984695783|0.13960432997387084||||148.5256889395057||
|localization|val_unseen|Language + Direction Only|2|Done|0.07017543859649122|0.12131392310563643|0.16312056737588654||||147.71287881104348||
|localization|val_unseen|Language + Distance + Direction|0|Done|0.19970138111235536|0.3527435610302352|0.43822321761851435||||54.226480926315745||
|localization|val_unseen|Language + Distance + Direction|1|Done|0.1787980589772303|0.3467711832773423|0.4583799925345278||||53.21550590242633||
|localization|val_unseen|Language + Distance + Direction|2|Done|0.18066442702500934|0.3232549458753266|0.4240388204553938||||55.62651592379588||
|navigation|val_seen|Language + Correct Geometry|0|Done||||0.4202429149797571|0.40695244965783456|0.5016194331983805|31.231796742570143|0.5797570850202429|
|navigation|val_seen|Language + Correct Geometry|1|Done||||0.40850202429149796|0.39571671287508237|0.5020242914979757|31.623739089877162|0.5912408759124088|
|navigation|val_seen|Language + Correct Geometry|2|Done||||0.4093117408906883|0.39785699943384206|0.4983805668016194|31.301594577689816|0.5904298459042985|
|navigation|val_seen|Language + Name/Type Only|0|Pending|||||||||
|navigation|val_seen|Language + Name/Type Only|1|Pending|||||||||
|navigation|val_seen|Language + Name/Type Only|2|Pending|||||||||
|navigation|val_seen|Language + Shuffled Geometry|0|Pending|||||||||
|navigation|val_seen|Language + Shuffled Geometry|1|Pending|||||||||
|navigation|val_seen|Language + Shuffled Geometry|2|Pending|||||||||
|navigation|val_seen|Language + Distance Only|0|Pending|||||||||
|navigation|val_seen|Language + Distance Only|1|Pending|||||||||
|navigation|val_seen|Language + Distance Only|2|Pending|||||||||
|navigation|val_seen|Language + Direction Only|0|Pending|||||||||
|navigation|val_seen|Language + Direction Only|1|Pending|||||||||
|navigation|val_seen|Language + Direction Only|2|Pending|||||||||
|navigation|val_seen|Language + Distance + Direction|0|Pending|||||||||
|navigation|val_seen|Language + Distance + Direction|1|Pending|||||||||
|navigation|val_seen|Language + Distance + Direction|2|Pending|||||||||
|navigation|val_unseen|Language + Correct Geometry|0|Done||||0.1846496106785317|0.17902334271543008|0.2810530218761587|53.61244037808063|0.8153503893214683|
|navigation|val_unseen|Language + Correct Geometry|1|Done||||0.19799777530589543|0.19199550414963018|0.28846866889136075|53.95016535406546|0.8017817371937639|
|navigation|val_unseen|Language + Correct Geometry|2|Done||||0.20986281053021877|0.20444530777231737|0.310715609936967|51.99855859739478|0.7887271369914147|
|navigation|val_unseen|Language + Name/Type Only|0|Pending|||||||||
|navigation|val_unseen|Language + Name/Type Only|1|Pending|||||||||
|navigation|val_unseen|Language + Name/Type Only|2|Pending|||||||||
|navigation|val_unseen|Language + Shuffled Geometry|0|Pending|||||||||
|navigation|val_unseen|Language + Shuffled Geometry|1|Pending|||||||||
|navigation|val_unseen|Language + Shuffled Geometry|2|Pending|||||||||
|navigation|val_unseen|Language + Distance Only|0|Pending|||||||||
|navigation|val_unseen|Language + Distance Only|1|Pending|||||||||
|navigation|val_unseen|Language + Distance Only|2|Pending|||||||||
|navigation|val_unseen|Language + Direction Only|0|Pending|||||||||
|navigation|val_unseen|Language + Direction Only|1|Pending|||||||||
|navigation|val_unseen|Language + Direction Only|2|Pending|||||||||
|navigation|val_unseen|Language + Distance + Direction|0|Pending|||||||||
|navigation|val_unseen|Language + Distance + Direction|1|Pending|||||||||
|navigation|val_unseen|Language + Distance + Direction|2|Pending|||||||||

## Three-seed means and sample standard deviations

|endpoint|split|arm|R@1_mean|R@1_std|SR_mean|SR_std|SPL_mean|SPL_std|NE_mean|
|---|---|---|---|---|---|---|---|---|---|
|localization|val_seen|Language + Correct Geometry|0.40821654300632043|0.010241629652972898|||||31.50982544362529|
|localization|val_seen|Language + Name/Type Only|0.3194558944765045|0.007864296796512323|||||51.901735319634916|
|localization|val_seen|Language + Shuffled Geometry|0.33759274525968674|0.01719921355731664|||||42.80404522789805|
|localization|val_seen|Language + Distance Only|0.3821104699093157|0.010717230008244013|||||32.82522373247455|
|localization|val_seen|Language + Direction Only|0.3694696345149766|0.023040287003916835|||||36.306754420029|
|localization|val_seen|Language + Distance + Direction|0.37153064028579275|0.018582608345223153|||||33.084279232314735|
|localization|val_unseen|Language + Correct Geometry|0.19995023018539257|0.013108938351191661|||||52.98960776890899|
|localization|val_unseen|Language + Name/Type Only|0.006096802289411472|0.0016831839316248207|||||219.01770791141817|
|localization|val_unseen|Language + Shuffled Geometry|0.03658081373646883|0.012952110170136224|||||166.83597307376073|
|localization|val_unseen|Language + Distance Only|0.1969640413089461|0.006360272698336215|||||53.96675133736287|
|localization|val_unseen|Language + Direction Only|0.06470075898967276|0.008843775360914201|||||148.7415524745026|
|localization|val_unseen|Language + Distance + Direction|0.18638795570486502|0.011567467505165074|||||54.35616758417931|
|navigation|val_seen|Language + Correct Geometry|||0.4126855600539811|0.006557371451150517|0.40017538732225305|0.0059658726992511775|31.385710136712373|
|navigation|val_unseen|Language + Correct Geometry|||0.19750339883821533|0.012613868072844474|0.1918213848791259|0.012711876925878031|53.18705477651363|

## Paired map-cluster intervals

|comparison|metric|delta_pp|map_CI95_pp|maps|
|---|---|---|---|---|
|localization/val_seen/correct-minus-name_type/seed_mean|R@1|8.876064852981589|[6.028308887963896, 11.775807541384054]|23|
|localization/val_seen/correct-minus-name_type/seed_mean|R@4|12.173674086287441|[8.88624297401625, 15.752324133137904]|23|
|localization/val_seen/correct-minus-name_type/seed_mean|R@8|12.269854355592194|[9.145057398543944, 15.775242026916212]|23|
|localization/val_seen/correct-minus-shuffled/seed_mean|R@1|7.06237977466337|[3.867322447500874, 10.405765533251396]|23|
|localization/val_seen/correct-minus-shuffled/seed_mean|R@4|8.738664468260511|[5.3628990150221165, 12.502850520840864]|23|
|localization/val_seen/correct-minus-shuffled/seed_mean|R@8|7.735641659796647|[5.0443249554718985, 10.631858686160639]|23|
|localization/val_seen/distance_direction-minus-name_type/seed_mean|R@1|5.207474580928826|[2.807389035683661, 7.795033056519261]|23|
|localization/val_seen/distance_direction-minus-name_type/seed_mean|R@4|8.889804891453695|[6.512175461307453, 11.3525807245393]|23|
|localization/val_seen/distance_direction-minus-name_type/seed_mean|R@8|9.975267930750205|[7.028074740275658, 13.383054836191796]|23|
|localization/val_seen/distance-minus-name_type/seed_mean|R@1|6.265457543281121|[3.8509483958493176, 8.686256005946218]|23|
|localization/val_seen/distance-minus-name_type/seed_mean|R@4|9.04094531464688|[6.869348012056118, 11.21201461164365]|23|
|localization/val_seen/distance-minus-name_type/seed_mean|R@8|10.277548777136575|[7.1910800383176205, 13.749940178071665]|23|
|localization/val_seen/direction-minus-name_type/seed_mean|R@1|5.001374003847211|[2.8480718075028206, 6.968312348781848]|23|
|localization/val_seen/direction-minus-name_type/seed_mean|R@4|7.405880736466061|[4.80465938365413, 9.939156622763507]|23|
|localization/val_seen/direction-minus-name_type/seed_mean|R@8|7.488320967298708|[4.466769249725452, 10.595000933846102]|23|
|localization/val_seen/distance_direction-minus-distance/seed_mean|R@1|-1.0579829623522945|[-2.4686316855969705, 0.4529669782481737]|23|
|localization/val_seen/distance_direction-minus-distance/seed_mean|R@4|-0.15114042319318485|[-0.9776550820292539, 0.7965852180222265]|23|
|localization/val_seen/distance_direction-minus-distance/seed_mean|R@8|-0.3022808463863699|[-1.2882773637922376, 0.6441403554175291]|23|
|localization/val_seen/distance_direction-minus-direction/seed_mean|R@1|0.20610057708161572|[-1.635571446393786, 2.5454710324481256]|23|
|localization/val_seen/distance_direction-minus-direction/seed_mean|R@4|1.483924154987634|[-0.48223933820528686, 3.8652194711448082]|23|
|localization/val_seen/distance_direction-minus-direction/seed_mean|R@8|2.4869469634514973|[0.6279536123120423, 4.647356474851695]|23|
|localization/val_unseen/correct-minus-name_type/seed_mean|R@1|19.385342789598106|[11.407160699417155, 26.10441767068271]|4|
|localization/val_unseen/correct-minus-name_type/seed_mean|R@4|34.577578698519346|[20.954759922286968, 45.64727108705461]|4|
|localization/val_unseen/correct-minus-name_type/seed_mean|R@8|44.568868980963046|[29.225645295587015, 57.036535859269335]|4|
|localization/val_unseen/correct-minus-shuffled/seed_mean|R@1|16.336941644892374|[8.298640022203719, 22.868741542625187]|4|
|localization/val_unseen/correct-minus-shuffled/seed_mean|R@4|27.4107253950479|[14.016097696364135, 38.29499323410016]|4|
|localization/val_unseen/correct-minus-shuffled/seed_mean|R@8|34.67711832773423|[19.15070774354705, 47.29364005412722]|4|
|localization/val_unseen/distance_direction-minus-name_type/seed_mean|R@1|18.029115341545353|[9.297807382736606, 25.124041497519165]|4|
|localization/val_unseen/distance_direction-minus-name_type/seed_mean|R@4|32.897847455518225|[17.790729947266165, 45.17365809652688]|4|
|localization/val_unseen/distance_direction-minus-name_type/seed_mean|R@8|42.21724524076148|[25.58978628920344, 55.72846188543078]|4|
|localization/val_unseen/distance-minus-name_type/seed_mean|R@1|19.086723901953462|[12.295309464335269, 24.605322507893554]|4|
|localization/val_unseen/distance-minus-name_type/seed_mean|R@4|33.470200323503796|[19.78906466833194, 44.5872801082544]|4|
|localization/val_unseen/distance-minus-name_type/seed_mean|R@8|43.69789722533284|[27.25506522342493, 57.05908885881825]|4|
|localization/val_unseen/direction-minus-name_type/seed_mean|R@1|5.860395670026129|[1.7762975298362478, 10.24757529351711]|4|
|localization/val_unseen/direction-minus-name_type/seed_mean|R@4|10.650740325992285|[3.6081043574798763, 18.861664114344038]|4|
|localization/val_unseen/direction-minus-name_type/seed_mean|R@8|14.097299987557546|[5.578684429641964, 24.527820316488018]|4|
|localization/val_unseen/distance_direction-minus-distance/seed_mean|R@1|-1.0576085604081122|[-2.9975020815986673, 1.1613067891781514]|4|
|localization/val_unseen/distance_direction-minus-distance/seed_mean|R@4|-0.5723528679855667|[-1.9983347210657785, 0.702811244979919]|4|
|localization/val_unseen/distance_direction-minus-distance/seed_mean|R@8|-1.4806519845713575|[-1.885410946366466, -0.6216006216006215]|4|
|localization/val_unseen/distance_direction-minus-direction/seed_mean|R@1|12.168719671519224|[7.52150985290036, 15.944970681100596]|4|
|localization/val_unseen/distance_direction-minus-direction/seed_mean|R@4|22.247107129525943|[14.182625589786287, 29.58553791887119]|4|
|localization/val_unseen/distance_direction-minus-direction/seed_mean|R@8|28.119945253203927|[20.011101859561464, 36.37566137566136]|4|

The three predeclared primary contrasts are Correct−Name/Type, Correct−Shuffle and Distance+Direction−Name/Type. Other simple-geometry contrasts and R@4/R@8 are secondary. Rates are fractions in per-seed tables; effects in the CI table are percentage points.

Confidence intervals are pointwise, not multiplicity-adjusted. Episode bootstrap, equal-map estimates, paired map differences and leave-one-map-out are in paired_statistics.json; absolute per-map per-seed metrics are in map_geometry_per_map.csv. Four Unseen maps are map blocks, not four independent cities. No missing or partial result is treated as zero.

## Conclusions

- correct − name_type, Unseen R@1: 19.385 pp, 95% CI [11.407, 26.104]; positive map-cluster support.
- correct − name_type, Unseen SR: Pending complete three-seed paired results.
- correct − shuffled, Unseen R@1: 16.337 pp, 95% CI [8.299, 22.869]; positive map-cluster support.
- correct − shuffled, Unseen SR: Pending complete three-seed paired results.
- distance_direction − name_type, Unseen R@1: 18.029 pp, 95% CI [9.298, 25.124]; positive map-cluster support.
- distance_direction − name_type, Unseen SR: Pending complete three-seed paired results.

Correct-minus-Name tests all map geometry together, not uniquely language–geometry interaction. A language-shuffle factorial would be needed to isolate that interaction; none is claimed here. Fixed-head transfer and possibly memorized Seen map/name priors limit causal interpretation of navigation.
