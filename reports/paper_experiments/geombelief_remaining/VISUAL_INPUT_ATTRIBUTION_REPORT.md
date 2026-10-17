# GeomBelief visual input attribution

Original complete Real/Zero/Shuffle/No-belief Seen/Unseen experiments are reused by path and checkpoint SHA. RGB+Geometry+Pose and Current+History are aliases of that original Real input configuration, not extra evaluations.

New Geometry+Pose, RGB+Text and Current-RGB controls are **frozen-weight inference interventions**, not fair retraining. All retain the same frozen geometric baseline goal belief; labels refer to inputs to the residual correction module. RGB+Text Only therefore does not mean a full map-free navigation model.

Geometry+Pose masks both projected RGB/text interactions, every image-quality channel and current/history appearance score. RGB+Text masks candidate geometric-language context, static relative logits/probability and pose distance. Both preserve common observed-support indicators and the residual visibility gate; thus this isolates RGB semantic descriptors beyond geometry/pose, not the ability to infer physical visibility from actual observation support. Current RGB disables history features, validity, quality and fusion consistently. All parameters, candidate grid, Stop0.3 and controller remain unchanged.

Zero versus No-belief equality is enforced by the existing residual's meaningful-RGB gate. It cannot be reused as a fair trained Geometry+Pose model: zeroing RGB disables the residual outright. A favorable inference mask is also not evidence a model trained from scratch on that subset would perform best.

Fair retraining of these three residual input schemes remains Pending, with estimated additional 8 GPU-hours from the actual archived 3-epoch runtime (9565.5s per scheme), plus navigation. No Unseen/Test tuning or weight selection is permitted. Existing calibrated checkpoint selected on Train-Dev is used throughout the new diagnostic inference.

Rates below are fractions, NE is meters. Pointwise paired bootstrap uses 10,000 draws and map clusters; only four Unseen map blocks. Existing primary Real/No-belief and Real/Shuffle intervals are preserved in ../../../../visual_residual_unseen_validation/bootstrap_results.json and not rerun.

|arm|split|status|kind|episodes|SR|SPL|OSR|NE|False_Stop_Rate|
|---|---|---|---|---|---|---|---|---|---|
|real|val_seen|Done|archived_full_validation|2470|0.4870445344129555|0.46494088505594233|0.5817813765182186|28.930326471488257|0.5129554655870445|
|zero|val_seen|Done|archived_full_validation|2470|0.47368421052631576|0.4595848256433525|0.5538461538461539|29.253030150070142|0.5263157894736842|
|shuffle|val_seen|Done|archived_full_validation|2470|0.46923076923076923|0.43419287830228337|0.5955465587044534|29.038146390951603|0.5305791818550021|
|no-belief|val_seen|Done|archived_full_validation|2470|0.47368421052631576|0.4595848256433525|0.5538461538461539|29.253030150070142|0.5263157894736842|
|geometry_pose|val_seen|Pending|frozen_input_intervention_not_fair_retraining|||||||
|rgb_text|val_seen|Pending|frozen_input_intervention_not_fair_retraining|||||||
|current_rgb|val_seen|Pending|frozen_input_intervention_not_fair_retraining|||||||
|RGB + Geometry + Pose|val_seen|Done|exact_alias_of_archived_real_not_new_experiment|2470|0.4870445344129555|0.46494088505594233|0.5817813765182186|28.930326471488257|0.5129554655870445|
|Current RGB + History|val_seen|Done|exact_alias_of_archived_real_not_new_experiment|2470|0.4870445344129555|0.46494088505594233|0.5817813765182186|28.930326471488257|0.5129554655870445|
|real|val_unseen|Done|archived_full_validation|2697|0.23952539859102706|0.2276250823849327|0.3455691509084168|50.681843726310845|0.7601188265874489|
|zero|val_unseen|Done|archived_full_validation|2697|0.22135706340378197|0.21529588731760127|0.3129403040415276|51.75753258960896|0.7784786641929499|
|shuffle|val_unseen|Done|archived_full_validation|2697|0.22617723396366332|0.20805578992057383|0.3474230626622173|51.6328583728313|0.7734026745913819|
|no-belief|val_unseen|Done|archived_full_validation|2697|0.22135706340378197|0.21529588731760127|0.3129403040415276|51.75753258960896|0.7784786641929499|
|geometry_pose|val_unseen|Pending|frozen_input_intervention_not_fair_retraining|||||||
|rgb_text|val_unseen|Pending|frozen_input_intervention_not_fair_retraining|||||||
|current_rgb|val_unseen|Pending|frozen_input_intervention_not_fair_retraining|||||||
|RGB + Geometry + Pose|val_unseen|Done|exact_alias_of_archived_real_not_new_experiment|2697|0.23952539859102706|0.2276250823849327|0.3455691509084168|50.681843726310845|0.7601188265874489|
|Current RGB + History|val_unseen|Done|exact_alias_of_archived_real_not_new_experiment|2697|0.23952539859102706|0.2276250823849327|0.3455691509084168|50.681843726310845|0.7601188265874489|
|geometry_pose|Train-Seen/Train-Dev then Val-Seen/Val-Unseen|Pending|fair_retraining|||||||
|rgb_text|Train-Seen/Train-Dev then Val-Seen/Val-Unseen|Pending|fair_retraining|||||||
|current_rgb|Train-Seen/Train-Dev then Val-Seen/Val-Unseen|Pending|fair_retraining|||||||

The original full validation did not meet its predeclared Unseen positive-SR/map-CI criterion: {'reference': 'Real minus No-belief, full Val-Unseen', 'positive_point_and_positive_map_cluster_lower': False, 'criterion': 'episode-weighted Unseen delta_SR > 0 and paired map-cluster bootstrap 95% CI lower > 0', 'no_retraining_or_unseen_selection': True}
