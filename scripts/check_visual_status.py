"""Read progress without computing a premature full-split SR."""
import json
from pathlib import Path
for split,total in [('val_seen',2470),('val_unseen',2697),('test_unseen',5281)]:
    root=Path('full_resolution_384')/split
    rows=sum(1 for _ in (root/'predictions.jsonl').open()) if (root/'predictions.jsonl').exists() else 0
    record=dict(split=split,completed=rows,total=total)
    if (root/'metrics.json').exists():record['metrics']=json.loads((root/'metrics.json').read_text())['metrics']
    print(json.dumps(record))
for model in ('calibrated_residual','absolute_residual'):
    root=Path(model)
    if (root/'history.json').exists():print(json.dumps(dict(model=model,history=json.loads((root/'history.json').read_text()))))
