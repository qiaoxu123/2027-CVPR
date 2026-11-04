# Polygon Direction Region Implementation Plan

> **For agentic workers:** execute sequentially in this session; no delegation is needed.

**Goal:** Rescan all 22,002 TRAIN instructions and compare full-Polygon directional probability regions with the existing frozen 32-parameter model.

**Architecture:** Candidate retrieval reads all original text plus existing text-only entity spans, followed by explicit per-candidate scope decisions. Prediction only accepts public Polygon, public bounds and a cardinal word. A front-envelope ray region and a 45-degree cone region retain distances and disjoint components instead of translating a centroid. Private labels are isolated from inference.

**Tech Stack:** Python, NumPy, Shapely, PyTorch, Matplotlib; existing graphnav geometry.

**Spec:** latest user request in this session, based on 70f8e39.

## Global Constraints
- TRAIN only; unchanged engineering compass, not verified geographic axes.
- No PGVB/navigation changes or checkpoints; no indirect/appearance model, no closed-loop evaluation.
- No GT-based direction rotation, Anchor selection or sample filtering.
- Persist text-review decisions and all exclusions; no claim of human annotation accuracy.
- Use new held-out physical-landmark/Episode groups excluding previous direction-experiment exposure where feasible; explicitly disclose any map overlap.
- Baseline first. Training is conditional, not automatic.

## Review Focus
- Parked target vs target inside a parking region with its own relation.
- Large and multi-fragment polygons, holes, map-edge directions with no legal support.
- Compound/proper-name cardinal words vs independent direction predicates.
- Repeated physical landmarks or targets crossing split boundaries.
- Broad candidate support vs sharp accurate prediction: report both probability mass and peak error.

## Tasks
- [x] Scan every TRAIN original instruction, union lexical and dependency candidates, record evidence spans. Explicitly review each candidate; bind only public names/unique documented aliases, retain all legal fragments.
- [x] Seal cohort/groups before model scoring; use opaque physical grouping only for split metadata. Select new confirmation by public group hashes, with no target coordinate access.
- [x] Write failing tests for Polygon ray/cone geometry, multi-fragment support, four-word axes, zero support abstention and no-target inference inputs. Implement minimal NumPy/Shapely baseline.
- [x] Freeze main baseline = full-Polygon front-envelope 45-degree cone, distance weight 1/(1+(distance/scale)^2), scale=max(20m,sqrt(public union area)/2). Strict ray version is a predeclared diagnostic, not a tuned replacement.
- [x] Run original32, NoDirection, Anchor, PolygonRay, PolygonCone with identical grids/Anchors. Seal arrays before opening GT labels in a separate scoring process.
- [x] Development training gate: PolygonCone must have GT20 probability mass > original32, R20 >= original32 and NE <= original32. If any fails, stop before new fitting; retain negative evidence. If all pass and sufficient fit/groups exist, train a small convex mixture of predeclared public geometry bases on fit only; freeze before confirmation. Do not pick an alternative rule based on confirmation.
- [x] Once strategy frozen, score untouched confirmation once; render real orthophotos/Polygon/probabilities/GT, per-word tables and Chinese report.
- [x] Run targeted tests, verify source/checkpoint SHA and split disjointness; update CHANGELOG for new experimental geometry component, commit/push.

## Actual decision
All three predeclared development criteria failed. No new training was started; final public-input strategy was frozen before one-shot confirmation. 30 confirmation pairs/13 groups were scored once. Preserve the negative result. Map overlap is explicitly disclosed; do not reuse confirmation to tune a road-specific variant.
