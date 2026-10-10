Data preparation contract
=========================

`data.py` prepares full Train Seen and Val Seen metadata from the original
processed CityNav annotations. It does not access Val Unseen. Each source
family has a forward and reverse episode. Reversed XYZ samples are the exact
measured forward samples in reverse order; these are synthetic return
traversals of real visited positions, not recorded return flights. Yaw is
recomputed from the next nonstationary displacement; endpoint heading retains
the last incoming direction. Original altitude is preserved.

Five fixed quantiles 0/.25/.5/.75/1 include the actual endpoint. When source
trajectories are exceptionally short, identical quantile indices are deduplicated
and each forward state still receives exactly two E0 sampling-budget entries.
E1 receives one forward and one reverse state. Trainer epochs and steps must
remain equal. No future observation is inserted to fill a missing view.

`episodes.jsonl` includes full trajectory supervision and `private_labels`.
Those trajectories, target IDs, and target coordinates are not public model
inputs. `reverse_geometry_description` is audit metadata: only its instruction
and `public_anchor` are public; fields beginning `label_` are private.

`prefixes.jsonl` has observable instruction, current pose, ground level, cutoff,
and forward current/history camera crops. Targets, remaining human waypoints,
and arrival labels are exclusively under `labels`. Reverse observation time is
`length - 1 - source_step`; changed heading never aliases a forward current
crop. `reverse_camera_reuse_candidates` records source camera views requiring
a camera turn or a fresh render before claiming them as reverse observations.
`pilot_frame_plan.jsonl` explicitly plans exact direction-specific current views;
nonpositive source altitude must remain an invalid/Unknown observation, not be
silently raised to a new flight altitude.

S object supervision requires exactly one valid Building/Parking/Ground polygon
containing the launch XY at least two meters inside. Ambiguous overlapping
annotations remain unknown. Launch point language uses actual launch XY, never
a large region's centroid. A named anchor must support the existing strict
cardinal relation (30--220 m axis displacement, transverse displacement at most
min(20 m, 20% of axis displacement)). Distance rounding error is at most 5 m.
A launch point without a specific object can still support verified point-anchor
geometry language, but cannot supply an object/appearance pair. With no anchor
relation, generic return language is trajectory-only. Names remain original
human map metadata; they do not validate colors mentioned within names.

Possible same-class positives intersect the approximate launch-language
relation region. Geometry negatives require the entire polygon to lie at least
five meters on the opposite directional side; all other objects remain unknown.
No synthetic color or visual attribute annotation is created. Existing benchmark
attribute fields are explicitly language-weak, not image-verified labels.

Train components connect source families, source/target object identities, and
all visited 200 m spatial cells. In the actual dataset this produces one connected
component per map, with 19,567 train and 2,311 internal-dev families and zero
family/object/visited-region overlap. CityNav's original map splits are retained.
Pilot selection hashes family IDs then uses round-robin maps, selecting 2,048
train-partition families and 128 Seen families before any model results.

`visual_pairs.jsonl` contains pilot endpoint-near actual observed nadir ROIs,
with native minimum bbox side 24 px and camera XY within 50 m of the direction's
goal. Reverse pairs describe a visited reverse position using explicit source
camera orientation; their `requires_reverse_current_render` prevents treating
these source camera crops as recomputed-yaw current observations.
`data_pair_images.py` saves `visual_pairs_materialized.jsonl` and reuses prior
SigLIP2 feature rows only by exact source/crop SHA and frozen image ordering.
Images not in the existing cache require independent encoding; no encoding is
claimed by this data preparation tool.

`data_checks.py` checks every full trajectory reversal, every causal prefix,
remaining waypoint and Stop target, matched budgets, and Train isolation.
`tests/test_bidirectional_core_data.py` checks stationary/vertical path reversal,
endpoint sampling, ambiguous polygon binding, multi-positive geometry and
connected-component isolation.
