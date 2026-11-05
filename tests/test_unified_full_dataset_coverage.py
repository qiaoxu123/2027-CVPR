"""All-instruction TRAIN adapter must not apply the old 573-query eligibility."""
from argparse import Namespace
import json
import torch

from scripts.train_unified_full_relation import full_loader
from graphnav.full_relation_training import full_relation_inputs
from graphnav.aerialvg_inspired_relation import CandidateAnchorRelation


class FakeFrozenBERT:
    def encode_tokens(self, texts):
        return torch.randn(len(texts), 5, 768), torch.ones(len(texts), 5, dtype=torch.bool)

    def encode(self, texts):
        return torch.randn(len(texts), 768)


def test_all_instructions_survive_converter_dataset_and_batching(tmp_path):
    common = dict(
        map_name="city_block_1", dataset_version="revised",
        match_mode="fuzzy", world_bounds=[0, 0, 400, 400],
        physical_map_sizes=[400, 400],
        target_xy=[0.5, 0.5],
    )
    refs = [dict(id="a", name="School", object_type="Building",
                 contour=[[.1,.1],[.2,.1],[.2,.2],[.1,.2]])]
    records = [
        dict(common, instruction="A red vehicle beside School.", landmarks=refs,
             episode_id="0"),
        dict(common, instruction="A car inside a parking area past School.",
             landmarks=refs, episode_id="1"),
        dict(common, instruction="The third blue car in the far row.",
             landmarks=[], episode_id="2"),
    ]
    path = tmp_path/"all_train.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in records))
    args = Namespace(max_landmarks=2, max_roads=0,
                     max_relations=2, prior_sigma_m=25.,
                     rule_sigma_m=25., batch_size=3)
    dataset, loader = full_loader(path, args, shuffle=False)
    assert len(dataset) == 3
    batch = next(iter(loader))
    assert batch["text"] == [r["instruction"] for r in records]
    assert batch["entity_valid"].shape == (3, 4)
    assert not batch["entity_valid"][2].any()
    observable = full_relation_inputs(batch, FakeFrozenBERT(), "cpu")
    assert set(observable) == {
        "static_inputs", "tokens", "token_mask", "node_emb"
    }
    assert "target_xy" not in observable
    head = CandidateAnchorRelation(
        context_dim=16, text_dim=768, hidden=24, heads=4,
    ).eval()
    # Full novel relation support without a named-landmark/direction parser.
    dense = torch.randn(3, 16, 16)
    geometry = torch.randn(3, 4, 16, 10)
    mask = batch["entity_valid"]
    logits = head(dense, observable["tokens"], observable["token_mask"],
                  geometry, torch.zeros(3, 4, dtype=torch.long), mask)
    assert logits.shape == (3, 16)
    assert torch.equal(logits[2], torch.zeros(16))
