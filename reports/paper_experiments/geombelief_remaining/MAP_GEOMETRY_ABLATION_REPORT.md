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

## Reference-name permissions

The navigation loader uses the supplied processed CityRefer description's `landmarks` annotation. `CityReferCatalog.referenced_landmark_names` indexes it by map/object ID/description ID, then resolves the provided names by normalized string/edit distance against the map catalog. Object ID is an annotation lookup key; it is not a locator input, and the target coordinate/teacher route is not used to choose references. This is annotation-assisted landmark linking, not a newly demonstrated raw-text-only entity recognizer. Every map-input arm shares this same language/name/type interface. Comparisons to methods without supplied reference-name annotations must disclose that input privilege.

## Observation support permissions

Native visual/control experiments use actual CityNav orthophoto crops at executed poses. The supplied surface-height raster produces a current-crop vertical-depth validity proxy; this is not a perspective occlusion simulation. The policy does not receive the whole raster or the surface_height_m array. Candidate pooling uses the current crop’s depth_valid mask and known footprint/pose for support and geometric ROI quality. Therefore RGB semantic encodings share extra support/pose metadata; the new residual RGB+Text mask is not a depth-free, geometry-free full-system baseline. Map-only locator experiments use none of these visual observations.

The generic dataset implementation also contains geometry-dependent ranking of unmentioned external roads. That path is inactive here: the frozen configuration has max_roads=0 and roads_json=null for every arm and Seed. Named road-like CityRefer references remain in the common supplied semantic reference set; their numeric geometry is masked under Name/Type Only.

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
|navigation|val_seen|Language + Name/Type Only|0|Done||||0.31821862348178137|0.3076295150766381|0.41578947368421054|53.503671883430584|0.6814303128809427|
|navigation|val_seen|Language + Name/Type Only|1|Done||||0.3299595141700405|0.3186638376445185|0.42105263157894735|48.97929812868214|0.6691480562448304|
|navigation|val_seen|Language + Name/Type Only|2|Done||||0.3186234817813765|0.30880192701541403|0.4121457489878543|51.99856057518638|0.6838203227213258|
|navigation|val_seen|Language + Shuffled Geometry|0|Done||||0.32348178137651823|0.3130194226163005|0.4174089068825911|44.00318235499995|0.656312625250501|
|navigation|val_seen|Language + Shuffled Geometry|1|Done||||0.3412955465587045|0.32978620714697104|0.4352226720647773|41.415683258193845|0.6300164925783397|
|navigation|val_seen|Language + Shuffled Geometry|2|Done||||0.35303643724696354|0.3416933611087723|0.45101214574898785|42.45049555609738|0.6426131511528608|
|navigation|val_seen|Language + Distance Only|0|Done||||0.38421052631578945|0.37153645649378697|0.4805668016194332|33.20422176875359|0.6157894736842106|
|navigation|val_seen|Language + Distance Only|1|Done||||0.3939271255060729|0.38152330127434453|0.4744939271255061|32.30207642611071|0.6045565500406835|
|navigation|val_seen|Language + Distance Only|2|Done||||0.37732793522267205|0.3656427438934909|0.45991902834008097|32.499802166520915|0.6214459788789602|
|navigation|val_seen|Language + Direction Only|0|Done||||0.38785425101214577|0.3751809133447137|0.4785425101214575|35.804902678548906|0.6121457489878542|
|navigation|val_seen|Language + Direction Only|1|Done||||0.3445344129554656|0.3331283165197529|0.4348178137651822|37.04800678580465|0.6547122602168474|
|navigation|val_seen|Language + Direction Only|2|Done||||0.3769230769230769|0.3659935702789309|0.4716599190283401|35.877167713214426|0.6162937948501478|
|navigation|val_seen|Language + Distance + Direction|0|Done||||0.3923076923076923|0.38005369546538226|0.4805668016194332|33.22588978698227|0.6076923076923076|
|navigation|val_seen|Language + Distance + Direction|1|Done||||0.37206477732793525|0.3608831666712084|0.4546558704453441|32.42975201877778|0.6270292207792207|
|navigation|val_seen|Language + Distance + Direction|2|Done||||0.3639676113360324|0.3529208283044412|0.4477732793522267|32.978220596847144|0.635884973673552|
|navigation|val_unseen|Language + Correct Geometry|0|Done||||0.1846496106785317|0.17902334271543008|0.2810530218761587|53.61244037808063|0.8153503893214683|
|navigation|val_unseen|Language + Correct Geometry|1|Done||||0.19799777530589543|0.19199550414963018|0.28846866889136075|53.95016535406546|0.8017817371937639|
|navigation|val_unseen|Language + Correct Geometry|2|Done||||0.20986281053021877|0.20444530777231737|0.310715609936967|51.99855859739478|0.7887271369914147|
|navigation|val_unseen|Language + Name/Type Only|0|Done||||0.005190952910641453|0.004858250027607215|0.04338153503893215|215.7895407900259|0.9950645406226272|
|navigation|val_unseen|Language + Name/Type Only|1|Done||||0.00778642936596218|0.007573781476040722|0.047830923248053395|217.95788063843818|0.9920634920634921|
|navigation|val_unseen|Language + Name/Type Only|2|Done||||0.006674082313681869|0.006204432249137821|0.04449388209121246|223.58773921825406|0.9938603223330775|
|navigation|val_unseen|Language + Shuffled Geometry|0|Done||||0.04449388209121246|0.04339952200354367|0.09788654060066741|158.42897193486652|0.9524539877300614|
|navigation|val_unseen|Language + Shuffled Geometry|1|Done||||0.021505376344086023|0.02081599776937692|0.07601038190582128|167.30181898967737|0.9761781460383221|
|navigation|val_unseen|Language + Shuffled Geometry|2|Done||||0.04338153503893215|0.0420331303224342|0.09899888765294772|173.66827888583364|0.9610894941634242|
|navigation|val_unseen|Language + Distance Only|0|Done||||0.19985168705969597|0.1940598665491841|0.29551353355580273|54.382673306983584|0.8001483129403041|
|navigation|val_unseen|Language + Distance Only|1|Done||||0.20133481646273638|0.1953885265394018|0.28883945124212085|53.342213422225356|0.7983993902439024|
|navigation|val_unseen|Language + Distance Only|2|Done||||0.19132369299221358|0.18516319490191574|0.2918057100482017|54.00958324938964|0.8097928436911488|
|navigation|val_unseen|Language + Direction Only|0|Done||||0.06970708194289951|0.06767940030061965|0.12643678160919541|150.3043404640036|0.9303538175046555|
|navigation|val_unseen|Language + Direction Only|1|Done||||0.05190952910641453|0.049942170554153155|0.10456062291434928|148.76362532048245|0.947242206235012|
|navigation|val_unseen|Language + Direction Only|2|Done||||0.06896551724137931|0.06727538679756438|0.12347052280311457|147.44690754451202|0.9304446978335233|
|navigation|val_unseen|Language + Distance + Direction|0|Done||||0.20059325176121617|0.19402286756641923|0.28846866889136075|54.602396386732586|0.7994067482387839|
|navigation|val_unseen|Language + Distance + Direction|1|Done||||0.18020022246941045|0.1752738169942236|0.27437893956247683|53.300849031024626|0.8199772123053551|
|navigation|val_unseen|Language + Distance + Direction|2|Done||||0.18057100482017055|0.17472096711570012|0.27771598071931775|55.85832454975294|0.8212269476853594|

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
|navigation|val_seen|Language + Name/Type Only|||0.32226720647773277|0.006664808758362084|0.3116984265788569|0.006060639481422492|51.493843529099706|
|navigation|val_seen|Language + Shuffled Geometry|||0.3392712550607288|0.014880952185821123|0.32816633029068126|0.014405439526084281|42.62312038976372|
|navigation|val_seen|Language + Distance Only|||0.3851551956815114|0.008339818923336254|0.37290083388720746|0.008027712621625414|32.66870012046174|
|navigation|val_seen|Language + Direction Only|||0.3697705802968961|0.022528221152698556|0.3581009333811325|0.02210939874422285|36.24335905918932|
|navigation|val_seen|Language + Distance + Direction|||0.37611336032388665|0.014597373584874443|0.3646192301470106|0.013946926106305137|32.877954134202405|
|navigation|val_unseen|Language + Correct Geometry|||0.19750339883821533|0.012613868072844474|0.1918213848791259|0.012711876925878031|53.18705477651363|
|navigation|val_unseen|Language + Name/Type Only|||0.0065504881967618335|0.0013021448217590827|0.0062121545842619185|0.001357782194500529|219.11172021557272|
|navigation|val_unseen|Language + Shuffled Geometry|||0.03646026449141021|0.012963249539193955|0.035416216698451596|0.012662604437446276|166.4663566034592|
|navigation|val_unseen|Language + Distance Only|||0.19750339883821533|0.005402915089456035|0.19153719599683386|0.005559878758652443|53.911489992866194|
|navigation|val_unseen|Language + Direction Only|||0.06352737609689778|0.010068180375437547|0.06163231921744574|0.010125980868788645|148.83829110966602|
|navigation|val_unseen|Language + Distance + Direction|||0.18712149301693237|0.011668358180825986|0.18133921722544763|0.010987841017509641|54.58718998917005|

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
|localization/val_seen/correct-minus-distance/seed_mean|R@1|2.610607309700467|[-0.029850281511609508, 5.107208539312615]|23|
|localization/val_seen/correct-minus-distance/seed_mean|R@4|3.1327287716405605|[1.076276159863891, 5.09260963835706]|23|
|localization/val_seen/correct-minus-distance/seed_mean|R@8|1.9923055784556196|[0.20653565618410272, 3.4718227763557916]|23|
|localization/val_seen/correct-minus-direction/seed_mean|R@1|3.874690849134377|[2.3476290648889035, 5.631760088776432]|23|
|localization/val_seen/correct-minus-direction/seed_mean|R@4|4.767793349821379|[2.906487862124755, 7.043608583714262]|23|
|localization/val_seen/correct-minus-direction/seed_mean|R@8|4.781533388293488|[3.100402036243435, 6.6421408738123935]|23|
|localization/val_seen/correct-minus-distance_direction/seed_mean|R@1|3.6685902720527617|[1.1622818142426792, 6.292460261942812]|23|
|localization/val_seen/correct-minus-distance_direction/seed_mean|R@4|3.2838691948337453|[1.2640690999052768, 5.178137245280088]|23|
|localization/val_seen/correct-minus-distance_direction/seed_mean|R@8|2.2945864248419894|[0.5317348958870518, 3.7269868385688554]|23|
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
|localization/val_unseen/correct-minus-distance/seed_mean|R@1|0.2986188876446435|[-0.8881487649181233, 2.169474221541603]|4|
|localization/val_unseen/correct-minus-distance/seed_mean|R@4|1.107378375015553|[-1.8518518518518516, 3.0241480478447307]|4|
|localization/val_unseen/correct-minus-distance/seed_mean|R@8|0.87097175563021|[-1.759259259259258, 3.100518592077681]|4|
|localization/val_unseen/correct-minus-direction/seed_mean|R@1|13.524947119571978|[9.630863169580904, 16.68921966621561]|4|
|localization/val_unseen/correct-minus-direction/seed_mean|R@4|23.92683837252706|[17.346655564807097, 29.37610229276892]|4|
|localization/val_unseen/correct-minus-direction/seed_mean|R@8|30.471568993405494|[23.35648148148149, 36.9708994708995]|4|
|localization/val_unseen/correct-minus-distance_direction/seed_mean|R@1|1.356227448052756|[0.5842151675485007, 2.12962962962963]|4|
|localization/val_unseen/correct-minus-distance_direction/seed_mean|R@4|1.6797312430011195|[-0.3605102606766498, 3.994262385523559]|4|
|localization/val_unseen/correct-minus-distance_direction/seed_mean|R@8|2.3516237402015676|[-0.85949177877429, 4.865938430983118]|4|
|navigation/val_seen/correct-minus-name_type/seed_mean|SR|9.04183535762483|[6.186655836316228, 11.897035362278492]|23|
|navigation/val_seen/correct-minus-name_type/seed_mean|SPL|8.847696074339613|[6.07548933549009, 11.6207628850333]|23|
|navigation/val_seen/correct-minus-shuffled/seed_mean|SR|7.341430499325235|[4.177091629572806, 10.676008470696912]|23|
|navigation/val_seen/correct-minus-shuffled/seed_mean|SPL|7.20090570315717|[4.137419818950739, 10.446784826290225]|23|
|navigation/val_seen/distance_direction-minus-name_type/seed_mean|SR|5.384615384615385|[3.011493857719107, 7.7709326386579445]|23|
|navigation/val_seen/distance_direction-minus-name_type/seed_mean|SPL|5.292080356815373|[2.9687223194991734, 7.63348963173476]|23|
|navigation/val_seen/distance-minus-name_type/seed_mean|SR|6.288798920377866|[4.03231353216296, 8.499907622038211]|23|
|navigation/val_seen/distance-minus-name_type/seed_mean|SPL|6.120240730835057|[3.8926322658119017, 8.27977729942574]|23|
|navigation/val_seen/direction-minus-name_type/seed_mean|SR|4.750337381916329|[2.5679480379891326, 6.71393826856117]|23|
|navigation/val_seen/direction-minus-name_type/seed_mean|SPL|4.640250680227565|[2.493242213858105, 6.5513321893946275]|23|
|navigation/val_seen/distance_direction-minus-distance/seed_mean|SR|-0.9041835357624832|[-2.1905327858064174, 0.4491493068440308]|23|
|navigation/val_seen/distance_direction-minus-distance/seed_mean|SPL|-0.8281603740196857|[-2.08379493473653, 0.49086671846636837]|23|
|navigation/val_seen/distance_direction-minus-direction/seed_mean|SR|0.6342780026990552|[-1.4208953858681797, 3.2335344296647377]|23|
|navigation/val_seen/distance_direction-minus-direction/seed_mean|SPL|0.6518296765878072|[-1.3427579006802555, 3.19484660820936]|23|
|navigation/val_seen/correct-minus-distance/seed_mean|SR|2.753036437246963|[0.13444589103246965, 5.322448540119342]|23|
|navigation/val_seen/correct-minus-distance/seed_mean|SPL|2.727455343504554|[0.16799379907672995, 5.202254799092401]|23|
|navigation/val_seen/correct-minus-direction/seed_mean|SR|4.291497975708501|[2.650498204666092, 6.195079487999641]|23|
|navigation/val_seen/correct-minus-direction/seed_mean|SPL|4.207445394112047|[2.6389663678707374, 6.0723319207237125]|23|
|navigation/val_seen/correct-minus-distance_direction/seed_mean|SR|3.6572199730094463|[1.0226281678974105, 6.3415112035029235]|23|
|navigation/val_seen/correct-minus-distance_direction/seed_mean|SPL|3.5556157175242395|[0.9770120052885702, 6.160400658881202]|23|
|navigation/val_unseen/correct-minus-name_type/seed_mean|SR|19.095291064145346|[11.046672190002763, 25.955466932535714]|4|
|navigation/val_unseen/correct-minus-name_type/seed_mean|SPL|18.560923029486396|[10.75466773965391, 25.20061467697313]|4|
|navigation/val_unseen/correct-minus-shuffled/seed_mean|SR|16.104313434680513|[8.008837337752004, 22.66219239373603]|4|
|navigation/val_unseen/correct-minus-shuffled/seed_mean|SPL|15.640516818067427|[7.776586986289521, 22.01082776233316]|4|
|navigation/val_unseen/distance_direction-minus-name_type/seed_mean|SR|18.057100482017056|[9.279204639602318, 25.167785234899327]|4|
|navigation/val_unseen/distance_direction-minus-name_type/seed_mean|SPL|17.512706264118574|[9.057066767807392, 24.362341748714275]|4|
|navigation/val_unseen/distance-minus-name_type/seed_mean|SR|19.095291064145346|[12.234189450428053, 24.65324384787472]|4|
|navigation/val_unseen/distance-minus-name_type/seed_mean|SPL|18.532504141257196|[11.889623642961746, 23.913683175782438]|4|
|navigation/val_unseen/direction-minus-name_type/seed_mean|SR|5.697688790013595|[1.4913007456503728, 10.086407444333672]|4|
|navigation/val_unseen/direction-minus-name_type/seed_mean|SPL|5.542016463318381|[1.4637452262204482, 9.751509224811112]|4|
|navigation/val_unseen/distance_direction-minus-distance/seed_mean|SR|-1.0381905821282906|[-2.9549848108257395, 0.8973080757726813]|4|
|navigation/val_unseen/distance_direction-minus-distance/seed_mean|SPL|-1.019797877138624|[-2.832556875154359, 0.8413527396500629]|4|
|navigation/val_unseen/distance_direction-minus-direction/seed_mean|SR|12.359411692003459|[7.787903893951949, 16.06263982102909]|4|
|navigation/val_unseen/distance_direction-minus-direction/seed_mean|SPL|11.97068980080019|[7.593321541586944, 15.51665187386757]|4|
|navigation/val_unseen/correct-minus-distance/seed_mean|SR|0.0|[-1.1875172604252966, 1.9376899696048628]|4|
|navigation/val_unseen/correct-minus-distance/seed_mean|SPL|0.028418888229197883|[-1.1349559033078378, 1.9016418229147605]|4|
|navigation/val_unseen/correct-minus-direction/seed_mean|SR|13.397602274131748|[9.555371444352387, 16.510067114093964]|4|
|navigation/val_unseen/correct-minus-direction/seed_mean|SPL|13.018906566168015|[9.290922513433463, 16.038823849154994]|4|
|navigation/val_unseen/correct-minus-distance_direction/seed_mean|SR|1.038190582128291|[0.06009615384615375, 1.7931034482758628]|4|
|navigation/val_unseen/correct-minus-distance_direction/seed_mean|SPL|1.048216765367822|[0.16339995809140467, 1.764916347652575]|4|

The three predeclared primary contrasts are Correct−Name/Type, Correct−Shuffle and Distance+Direction−Name/Type. Other simple-geometry contrasts and R@4/R@8 are secondary. Correct versus individual simple-geometry arms is an exploratory completeness analysis added after static means were inspected; it does not affect checkpoint choice or hyperparameters. Rates are fractions in per-seed tables; effects in the CI table are percentage points.

Confidence intervals are pointwise, not multiplicity-adjusted. Episode bootstrap, equal-map estimates, paired map differences and leave-one-map-out are in paired_statistics.json; absolute per-map per-seed metrics are in map_geometry_per_map.csv. Four Unseen maps are map blocks, not four independent cities. No missing or partial result is treated as zero.

## Actual shuffle corruption

|split|queries|singleton_schema_queries|effective_changed_queries|effective_unchanged_queries|
|---|---|---|---|---|
|train_seen|21166|988|11490|9676|
|val_seen|2426|318|1068|1358|
|val_unseen|2679|107|1607|1072|

Different query indices can have identical map bundles. These are counted as unchanged, not falsely described as corrupted geometry. This distribution-preserving shuffle is weaker than corrupting every landmark.

## Fixed-controller input distribution

|split|arm|seed|top1_context_norm|top1_context_cosine_to_head_training|same_grid_context_relative_L2|
|---|---|---|---|---|---|
|val_seen|correct|0|4.135250091552734|0.5842485427856445|0.815494954586029|
|val_seen|correct|1|4.752623081207275|0.030523385852575302|1.2564646005630493|
|val_seen|correct|2|4.579168319702148|0.003247434739023447|1.3023921251296997|
|val_seen|name_type|0|6.475388050079346|0.2605606019496918|1.4767948389053345|
|val_seen|name_type|1|6.803019046783447|-0.018316248431801796|1.657137393951416|
|val_seen|name_type|2|5.959340572357178|-0.017941413447260857|1.5557527542114258|
|val_seen|shuffled|0|11.265719413757324|0.14403966069221497|2.1965198516845703|
|val_seen|shuffled|1|10.476790428161621|-0.010616729035973549|2.2046468257904053|
|val_seen|shuffled|2|11.44680404663086|0.032965924590826035|2.3091790676116943|
|val_seen|distance|0|3.4161853790283203|0.23428048193454742|1.1021243333816528|
|val_seen|distance|1|3.6112873554229736|-0.04418962076306343|1.3037739992141724|
|val_seen|distance|2|3.8883931636810303|-0.09208496659994125|1.3561676740646362|
|val_seen|direction|0|5.782371520996094|0.21191568672657013|1.4471282958984375|
|val_seen|direction|1|6.422529697418213|0.007232255302369595|1.6993153095245361|
|val_seen|direction|2|5.856410026550293|-0.01893831416964531|1.6053931713104248|
|val_seen|distance_direction|0|3.1029698848724365|0.24979768693447113|1.0624785423278809|
|val_seen|distance_direction|1|3.2952561378479004|-0.05536067858338356|1.2174954414367676|
|val_seen|distance_direction|2|3.5097031593322754|-0.08796827495098114|1.2777494192123413|
|val_unseen|correct|0|4.322559356689453|0.5651780366897583|0.8198104500770569|
|val_unseen|correct|1|5.082489490509033|0.025200646370649338|1.2618919610977173|
|val_unseen|correct|2|5.0772385597229|0.020028632134199142|1.308500051498413|
|val_unseen|name_type|0|7.200801372528076|0.18856051564216614|1.5312130451202393|
|val_unseen|name_type|1|7.447281837463379|-0.0046983580105006695|1.6838688850402832|
|val_unseen|name_type|2|6.641241550445557|-0.009504317305982113|1.592075228691101|
|val_unseen|shuffled|0|11.650854110717773|0.11952793598175049|2.1419780254364014|
|val_unseen|shuffled|1|10.506009101867676|0.006935334764420986|2.120816707611084|
|val_unseen|shuffled|2|11.515247344970703|0.04430103674530983|2.200901508331299|
|val_unseen|distance|0|3.519676685333252|0.23822695016860962|1.0430757999420166|
|val_unseen|distance|1|4.0379638671875|-0.04453714191913605|1.2905758619308472|
|val_unseen|distance|2|4.254784107208252|-0.069894939661026|1.3383914232254028|
|val_unseen|direction|0|6.02636194229126|0.18855057656764984|1.3814818859100342|
|val_unseen|direction|1|6.954392433166504|-0.009526433423161507|1.6811515092849731|
|val_unseen|direction|2|6.692106246948242|-0.014088057912886143|1.584171175956726|
|val_unseen|distance_direction|0|3.2714200019836426|0.2564171850681305|1.008936882019043|
|val_unseen|distance_direction|1|3.7022812366485596|-0.04979289695620537|1.2068696022033691|
|val_unseen|distance_direction|2|3.994097948074341|-0.06959918141365051|1.273622751235962|

These latent diagnostics compare against the actual head-training locator. Different independently trained latent bases make cosine/L2 descriptive, not a semantic performance criterion. No controller was adapted.

## Conclusions

- correct − name_type, Unseen R@1: 19.385 pp, 95% CI [11.407, 26.104]; positive map-cluster support.
- correct − name_type, Unseen SR: 19.095 pp, 95% CI [11.047, 25.955]; positive map-cluster support.
- correct − shuffled, Unseen R@1: 16.337 pp, 95% CI [8.299, 22.869]; positive map-cluster support.
- correct − shuffled, Unseen SR: 16.104 pp, 95% CI [8.009, 22.662]; positive map-cluster support.
- distance_direction − name_type, Unseen R@1: 18.029 pp, 95% CI [9.298, 25.124]; positive map-cluster support.
- distance_direction − name_type, Unseen SR: 18.057 pp, 95% CI [9.279, 25.168]; positive map-cluster support.

Correct-minus-Name tests all map geometry together, not uniquely language–geometry interaction. A language-shuffle factorial would be needed to isolate that interaction; none is claimed here. Fixed-head transfer and possibly memorized Seen map/name priors limit causal interpretation of navigation.
