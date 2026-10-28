"""Opt-in live reference wrapper sharing the exact offline feature adapter."""
from .public_road_references import resolve_road_references
from .reference_rebuild import rebuild_reference_pack


def bind_public_road_references(policy, episode, catalog):
    """Set a separate public reference list without destination/controller access."""
    named = [item for item in catalog.objects[episode.map_name].values() if item.get('name')]
    policy.a2_road_references = resolve_road_references(episode.map_name,episode.reference_names,named)


class ReferencePerception:
    def __init__(self, legacy_perception):
        self.legacy = legacy_perception

    def extract(self, policy, state, field):
        if policy.bert.max_length != 96 or not hasattr(policy,'a2_road_references'):
            raise ValueError('Explicit episode-bound references and frozen96-token BERT required')
        public,meta = self.legacy.extract(policy,state,field)
        instruction = policy.record['instruction']
        offsets = policy.bert.tokenizer(instruction,truncation=True,max_length=96,
                                        return_offsets_mapping=True)['offset_mapping']
        return rebuild_reference_pack(public,meta,instruction=instruction,offsets=offsets,
            references=policy.a2_road_references,encode_names=policy.bert.encode_tokens)
