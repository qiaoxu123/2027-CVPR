"""CLI for weak GT-free instruction-graph preparation, full TRAIN only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphnav.unified_instruction_graph_data import prepare_graphs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl",type=Path,required=True)
    p.add_argument("--public-objects",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected-records",type=int,default=21878)
    p.add_argument("--spacy-model",default="en_core_web_sm")
    args=p.parse_args()
    result=prepare_graphs(args.train_jsonl,args.public_objects,
                          args.output,expected_records=args.expected_records,
                          spacy_model=args.spacy_model)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
