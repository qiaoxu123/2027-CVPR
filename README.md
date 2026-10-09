# GraphNav — Graph-First Aerial Vision-and-Language Navigation

Independent research implementation for CityNav / CityRefer. **Not based on the HETT network**.

GraphNav uses deterministic name-to-contour binding, an edge-conditioned landmark/UAV graph, dense geometric goal belief, observed-only RGB evidence, and goal-conditioned multi-modal waypoint prediction. Data interfaces remain compatible with CityNav annotations, but model modules and training are newly designed.

> Status: research baseline. CPU unit tests and synthetic smoke training are provided. Closed-loop CityNav SR/SPL are **not** claimed until an evaluator is connected and full experiments are run.

See [architecture](docs/ARCHITECTURE.md), [data conversion](docs/DATA.md), and [experiments](docs/EXPERIMENTS.md).

Quick start after implementation:
```bash
pip install -e '.[dev]'
pytest -q
python -m graphnav.train --smoke --epochs 1
```
