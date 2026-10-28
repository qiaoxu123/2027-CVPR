# A2 revision: full-pool grounding and bounded belief correction

User goal authorizes implementing the explained improvements and iterating on TRAIN evidence toward actual A2>A0. No success is assumed.

## Immutable baseline and execution
A0 including RGB residual, WALDO/BERT/SigLIP2, controller and28x28 grid remain frozen. Preserve A0/A1 and A2 training_collected/calibration_collected. Add revision_* modules/output revision_v1, seed0, full20 ranking+5 belief-refinement epochs. Only one GPU job under shared gpu.lock. Reuse collected snapshot; do not restart full extraction.

## Data
Preserve snapshot target-disjoint train31415/dev4618 observations and raw bytes. Freshly initialize ALL added A2 parameters. Extend trusted negative identities across detector categories ONLY if trusted association, original seed0 TRAIN partition and derived target partition agree. Unknown identities remain masked for rank/trust; no all-unknown-negative softmax.
Add WORLD-GRID proposal localization loss using offline TRAIN target coordinates, separately from object-identity labels. This supervises spatial suitability of the full public candidate distribution without inventing object identities. GT category only supervises instruction classifier; runtime category consistency is predicted from instruction, never private GT.
Old calibration targets already examined are development evidence, not untouched confirmation. Before deployment reserve unused TRAIN target instances, freeze candidate, independently calibrate/verify and run true closed-loop pilot. Any necessary new collection is limited confirmation only. No benchmark tuning or repeated Unseen selection.

## Model
Reuse trainable per-Landmark attention/relation/scorer architecture, reinitialized. Add instruction-only Car/Building soft category consistency. Add per-box current-A0 features: support log mass, support/global peak gap, distance to A0 peak, entropy, top2 margin, support size. New ranking/trust and global benefit heads use these label-free features. Trust estimates reachable match; benefit estimates whether correcting A0 helps. Offset is exactly0 in this revision because old correction worsened heldout error.

## Fusion
Q is the normalized ranked/soft-trusted proposal on actual observed candidate support with original4m kernels. P'=(1-g)P_A0+gQ, g in[0,.25]. Zero gate/no candidates/no valid references/no support returns EXACT original logits. Compute in log space. This bounded global-mass transfer deliberately replaces old support-mass conservation: normalized outside probabilities shrink by1-g, although unnormalized outside scores can be retained up to a shared constant. It allows observed correct-region evidence to replace a wrong A0 peak while bounding intervention mass. Grid/controller unchanged.

## Training
Ranking20: original known-only rank/relation/trust losses plus instruction-category CE and full-proposal grid localization CE. Unknown object IDs never become invented rank/trust negatives; map supervision uses actual private TRAIN target coordinates only in loss.
Refinement5: start from best ranking checkpoint; continue earlier losses, add mixed-map localization CE, protection against breaking correct A0 states, and a benefit target from detached proposal-vs-A0 expected target distance. Loss is separate from forward inputs. Fresh optimizer at stage boundary. No shortened formal run.
Stage1 selection: full-proposal R@1 within20m then NE. Stage2: net corrected-minus-broken observations, fewer breaks, lower NE. Known-only MRR is diagnostic. Process every train/dev row each epoch; per-epoch weights/history/TB, atomic provenance-bound resume. Final selection only after all25.

## Validation and final success
Tests cover target-disjoint cross-category labels, unknown masks, provenance, no GT in forward, outside-peak correction, exact fallback, valid-support normalization, finite gradients, zero offsets. CPU TRAIN oracle and genuine CUDA backward before training, independent review, frozen signatures and serial execution.
Internal iterations use TRAIN dev/dedicated calibration/true paired pilot only. Preserve risk checks; no forced enabling. Final frozen A0/A2 evaluation uses allSeen2470+Unseen2697 same episode/controller. Report SR/OSR/SPL/NE/nDTW/sDTW/FalseStop/Recall, repairs/breaks/fallback/coverage/latency, maps/paired differences, curves/logs/checkpoints and original A0/A1 conclusion. Positive full-split SR differences on BOTH splits with paired uncertainty and SPL/NE tradeoffs are necessary evidence; cached proxies, subset gains and OFF parity cannot complete the goal. Unseen is final confirmation, never tuning/selection. If evidence fails the goal stays unachieved.

## Independent design review rulings
- Reserve256 never-observed original dev/calibration target identities BEFORE any revision labels/training; manifest SHA saved. Exclude all older dev/calib cache targets. OriginalTRAIN-only negative guard also excludes these reserved identities. Reserve includes no RGB collection or model results yet.
- Recompute trust labels for zero-offset proposals: trusted matching identity plus observed support reaching a <=20m target grid cell, rather than blindly reuse old offset_valid.
- Proposal localization CE uses a normalized target Gaussian restricted to supported cells within20m; rows with no compatible support skip proposal CE and report coverage. Mixed-map CE remains finite from original A0 support; benefit learns OFF on unhelpful proposals.
- CPU oracle uses the exact max gate .25 and reports reachable correction coverage; bounded fusion cannot repair arbitrary highly confident wrong priors.
