"""Query count/order must not change an existing point's frozen score."""
import torch
from graphnav.full_localization import FullTokenGeometryLocalizer


def test_geometry_queries_are_independent():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = FullTokenGeometryLocalizer(hidden=16, text_dim=8, heads=2,
                                      max_relations=2, top_k=2, dropout=0).eval()
    batch = dict(prior=torch.randn(1, 9), rules=torch.randn(1, 9),
                 relation_fields=torch.randn(1, 9, 2), grid_xy=torch.rand(1, 9, 2),
                 nodes=torch.rand(1, 3, model.gnn.node_geometry[0].in_features),
                 node_types=torch.zeros(1, 3, dtype=torch.long),
                 node_valid=torch.ones(1, 3, dtype=torch.bool),
                 entity_valid=torch.ones(1, 3, dtype=torch.bool),
                 pair_geometry=torch.randn(1, 9, 3, 10))
    tokens = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)
    names = torch.randn(1, 3, 8)
    indices = torch.tensor([5, 0, 5, 8])
    query_fields = {'prior', 'rules', 'relation_fields', 'grid_xy', 'pair_geometry'}
    subset = {k: (v[:, indices] if k in query_fields else v) for k, v in batch.items()}
    with torch.no_grad():
        full = model(batch, variant='geometry', tokens=tokens, token_mask=mask, node_emb=names)
        partial = model(subset, variant='geometry', tokens=tokens, token_mask=mask, node_emb=names)
    torch.testing.assert_close(full.belief_logits[:, indices], partial.belief_logits)
    torch.testing.assert_close(full.dense_context[:, indices], partial.dense_context)
