# CityNav Instruction Relation Graph — prototype and training protocol

Status: **UNTRAINED research prototype**. No parser gold labels, no GPU
checkpoint and no navigation SR results are claimed. This branch deliberately
does not modify the original Geometry/Relation/PGVB model, Stop threshold,
runtime policy or main branch.

## Research question

Convert goal-directed UAV navigation instructions into **separate target,
intermediate reference, and named public landmark nodes** linked by directed
spatial relation edges. Distinguish 1-hop (direct) from 2+-hop (indirect)
target-to-landmark paths. Provide a typed geometric/semantic vector while
preserving unknown locations rather than inventing GT positions.

Example (illustrative, no inference claim):

> "The blue car to the right of a black car in the parking lot behind One Stop"

Target blue car -->right_of--> black car -->inside--> parking lot
-->behind--> One Stop (public Polygon)

- The car, black car and parking lot have UNKNOWN world coordinates until
  actually grounded in observed RGB/semantic map.
- One Stop has known PUBLIC Polygon geometry.
- "right_of" / "behind" have unresolved **reference frames**; NEVER
  automatically convert them to east/west or world XYZ.
- An indirect path is NOT equivalent to a direct near(blue car, One Stop).
- The graph expresses constraints, not a proven image instance localization.

## Existing repo reused

- graphnav.structured_navigation.parse_instruction: existing spaCy-derived,
  **unverified** candidate target, entity mentions, relation paths.
- graphnav.unified_instruction: existing direct/indirect adapter; do not
  turn its diagnostic status into "approved_for_inference".
- graphnav.semantic_graph: static **map** topology, different from this
  goal-centric **instruction** relation graph.
- graphnav.candidate_geometry: existing candidate-to-public-Polygon geometry.

## Added components

1. **graphnav/instruction_relation_graph.py**:
   \`compile_relation_graph(parsed, public_objects, world_bounds)\` consumes an
   already parsed, GT-free instruction. Produces source-text spans, node
   roles, typed predicate edges, native edge direction, explicit one-hop /
   multi-hop landmark paths and ambiguity flags. Map polygon bbox proxies
   are written ONLY on verified public nodes. Unmapped target/intermediates
   have geometry_valid=False. All numeric geometry feature channels are
   accompanied by validity masks; no silent truncation.
2. **graphnav/instruction_graph_parser_head.py**:
   trainable BERT-span role / ordered-pair predicate / reference-frame
   classifiers. This prototype assumes candidate noun-phrase spans from
   an external proposer/teacher, so it is **not fully end-to-end span
   detection from arbitrary raw text**.
3. **graphnav/instruction_relation_encoder.py**: trainable text+geometry
   graph message passing with target-node readout (2 layers default).
   Accepts separately encoded node phrases and edge predicates, not a
   single mixed sentence embedding. Predicates and relation frames are
   distinct embeddings; unknown physical locations have masked geometry.
   It is a downstream encoder, not a standalone learned parser.
4. **scripts/prepare_instruction_graph_annotations.py**: turn the EXISTING
   parser's JSONL proposals into \`reviewed=false\` human annotation worklist.
   NEVER auto-approve pseudo labels.
5. **scripts/train_instruction_relation_parser.py**: frozen BERT + trainable
   supervised span graph parser head, reviewed JSONL fit/dev, 5 epochs
   default, seed42, map-disjoint, loss and edge-triple F1 reporting.
6. CPU test modules: \`tests/test_instruction_relation_graph.py\`,
   \`tests/test_instruction_graph_parser_head.py\`.

## Representation

For node \`v\`:

- Role: target / intermediate / public_landmark / reference_object.
- Language: original phrase/span, attributes, frozen BERT phrase features.
- Public geometric [8]: normalized bounding-center (2), extent (2),
  log polygon-area ratio (1), number-of-public-fragments scaled (1),
  aspect ratio (1), presence (1); accompanied by node_geo_mask.
- Do not pretend a public Polygon is a verified pixel segmentation mask.

For edge \`e=(subject,predicate,object)\`:

- Predicate ID in a named relation vocabulary, source span and text.
- Reference-frame label: topological / metric_or_adjacency /
  reference_frame_unknown / world_cardinal / unknown.
- 2D language direction only when it is a cardinal relation and an
  independent world-north alignment has been verified. In all other cases,
  direction_xy_valid=False, numerical direction zero.
- Public-pair geometric [9]: relative world delta (2), distance (1),
  unit direction (2), endpoint bbox extents (4); edge_geo_mask True
  ONLY if both endpoint physical locations are public and known.
- Semantics and geometry are deliberately SEPARATE; an unobserved car
  gets no physical relative geometry before candidate binding.

Graph encoder predicts a 128-d target graph state and node/edge states.
It is suitable for later candidate-to-constraint scores but **has not
been trained or connected to PGVB in this PR**.

## Human-review dataset (REQUIRED BEFORE TRAINING)

Generate proposals from current parser JSONL:

\`\`\`bash
python -m scripts.prepare_instruction_graph_annotations \
  --parsed-jsonl /path/to/public_parser_proposals.jsonl \
  --output /path/to/graph_annotation_worklist.jsonl
\`\`\`

Inspect and correct the target role, all object spans, predicate scope,
subject/object direction, hop chains, named public binding, relation frame,
and coreference. Each approved row must explicitly have \`"reviewed":true\`.
No script sets this value automatically.

Reviewed annotation line example:

\`\`\`json
{"instruction":"A blue car near One Stop.","map_name":"cambridge_block_4","reviewed":true,"spans":[{"start":7,"end":10,"role":"target"},{"start":16,"end":24,"role":"public_landmark"}],"relations":[{"source":0,"target":1,"predicate":"near","frame":"metric_or_adjacency"}]}
\`\`\`

\`near\` is a *semantic relation label*, NOT a hard-coded 30m distance.
A reviewer must check token spans and role scopes; multiple references
in complex commands require additional nodes and edges. Include
negative examples (unresolvable target, wrong landmark, ambiguous relations)
in a later production data schema; the current supervised training entry
expects a known target and valid spans, not open-set rejection.

### Supervised parser training (only after labels exist)

\`\`\`bash
python -m scripts.train_instruction_relation_parser \
  --fit-annotated /path/to/cambridge_reviewed.jsonl \
  --dev-annotated /path/to/birmingham_reviewed.jsonl \
  --output runs/instruction_graph_s42 \
  --bert bert-base-uncased \
  --epochs 5 --batch-size 8 --seed 42
\`\`\`

Requires existing cached local HuggingFace BERT. The tokenizer is fast for
character-to-token span alignment; missing/truncated spans fail closed.
A shared map or duplicate instruction across fit and dev is rejected.
The parser trains role + directed pair predicate + frame heads using
\`L=CE_role+CE_edge+0.3*CE_frame\`. Edge absent is a separate class and
downweighted in training. No checkpoint is shipped with this PR.

### Pre-navigation verification gates

1. Human review quality: target identification, reference identity, relation
   triplet Precision/Recall/F1, direct/indirect path exact match, unknown
   reference-frame recall, counterfactual predicate swaps. Pilot **300–500
   stratified** direct/indirect/no-public/unresolved cases, then expand if good.
2. Keep true labels separate from heuristic parser proposals. Report
   automatic coverage and rejection counts; do not silently score only
   easy sentences.
3. Fix the text graph and test candidate geometry *without RGB* first:
   teacher graph vs predicted graph; Geometry only vs graph constraints.
   Candidate scoring must never depend on GT/future RGB.
4. Then register *causally observed* visual object nodes and the same public
   map Polygons to this graph, with explicit unknown-visible masks.
   Only then train a relation-aware candidate reranker, with protected
   correct Geometry hypotheses and hard negative object instances.
5. Leave 2470 Val-Seen untouched until parser and TRAIN-Dev candidate
   metrics are stable. No Val-selected checkpoints, no Stop changes.

## Closest academic prior work

- **GC-VLN (CoRL 2025)** graph constraint decomposition of navigation
  language and constraint solving:
  https://proceedings.mlr.press/v305/yin25a.html
  code: https://github.com/bagh2178/GC-VLN
- **ORD (CVPR 2026)** explicit target-anchor geometry and predicate-entity
  decoupling:
  https://openaccess.thecvf.com/content/CVPR2026/html/Huang_ORD_Object-Relation_Decoupling_for_Generalized_3D_Visual_Grounding_CVPR_2026_paper.html
- **LSVG (Pattern Recognition 2026)** task-specific language-guided graph
  between target and referred objects:
  https://www.sciencedirect.com/science/article/pii/S0031320325015894
- **AerialVG (ICCV 2025)** aerial object-pair positional grounding:
  https://openaccess.thecvf.com/content/ICCV2025/html/Liu_AerialVG_A_Challenging_Benchmark_for_Aerial_Visual_Grounding_by_Exploring_ICCV_2025_paper.html
- **GROKE (ACL 2026)** map/graph-based instruction evaluation:
  https://aclanthology.org/2026.acl-long.1874/
- **3D-DRES (AAAI 2026)** phrase-to-instance supervision:
  https://ojs.aaai.org/index.php/AAAI/article/view/37288

No claim of a new paradigm is justified by this prototype; CityNav
multihop, frame ambiguity, missing visual intermediates and causal graph
updates would need dedicated experiments to establish a contribution.
