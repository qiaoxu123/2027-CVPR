## [0.1.0] - 2026-10-27
### Features
- Added independent language-conditioned local box/reference relation network, observed RGB-D extraction, private-label association, two-stage training, calibration and original28×28 runtime adapter.
- Added masked uncertainty handling, local probability mass conservation, real-data verification and relation visualization.
### Design Rationale
- Freeze original localization and encoders; isolate trainable relation reasoning from flight control and preserve comparable paper settings.
- Keep full RGB/token features and actual box-reference distance/angle/weak surface evidence rather than one similarity scalar.
### Notes & Caveats
- DSM is a 2.5D observed-surface proxy. Annotation matching remains a quality-audit requirement.
- Smoke/unqualified calibration cannot enable navigation feedback; continuous offsets do not imply subgrid flight control.
