import torch
from experiments.a2_waldo_landmark_belief.batching import collate_views


def test_padding_respects_candidate_reference_and_instruction_axes():
    def row(n,l,t):
        return dict(public=dict(network=dict(box_rgb=torch.ones(n,16),box_valid=torch.ones(n,dtype=torch.bool),
                    instruction_tokens=torch.ones(t,12),token_valid=torch.ones(t,dtype=torch.bool),
                    mention_mask=torch.ones(l,t,dtype=torch.bool),landmark_name_tokens=torch.ones(l,3,12),
                    name_valid=torch.ones(l,3,dtype=torch.bool),link_valid=torch.ones(l,dtype=torch.bool),
                    pair_geometry=torch.ones(n,l,14),pair_valid=torch.ones(n,l,dtype=torch.bool),offset_limit=torch.ones(n,2)),
                    spatial=dict(candidate_support=torch.ones(n,5,dtype=torch.bool))),
                    labels=dict(ranking=torch.ones(n),rank_label_mask=torch.ones(n,dtype=torch.bool),
                                angle=torch.ones(l,2),angle_valid=torch.ones(l,dtype=torch.bool),distance=torch.ones(l),
                                offset=torch.zeros(n,2),offset_valid=torch.ones(n,dtype=torch.bool),
                                match=torch.tensor(1.),match_valid=torch.tensor(True)))
    batch=collate_views([row(2,3,7),row(4,1,5)])
    x=batch['network']
    assert x['pair_geometry'].shape==(2,4,3,14)
    assert not x['box_valid'][0,2:].any() and not x['link_valid'][1,1:].any()
    assert not x['mention_mask'][1,:,5:].any()
    assert not batch['labels']['rank_label_mask'][0,2:].any()
