"""Opt-in PR23 text/public-binding adapter, inherited frozen navigation unchanged."""
import json
from pathlib import Path
from .unified_full_relation_runtime import UnifiedPGVBFullRelationPolicy
from .unified_text_ablation import public_text_record
from scripts.analyze_instruction_landmark_association import PublicNames


class UnifiedTextAblationPolicy(UnifiedPGVBFullRelationPolicy):
    def __init__(self,*args,instruction_mode,public_objects,**kwargs):
        if instruction_mode not in ('full_text','landmark_relation'):raise ValueError('Cancelled or unknown text arm')
        if kwargs.get('visual_control','off')!='off':raise ValueError('PR23 isolates static text/Polygon inference')
        kwargs['visual_control']='off'
        super().__init__(*args,**kwargs)
        self.instruction_mode=instruction_mode
        self.public_objects=json.loads(Path(public_objects).read_text())
        self.public_name_matcher=PublicNames(self.public_objects)
        self.original_instruction=None

    def set_episode(self,episode,transform,catalog):
        # Evaluator passes empty references. Never call target-indexed catalog
        # reference resolution or accept an externally selected target anchor.
        if episode.references or episode.reference_names:raise ValueError('PR23 must bind its own public references from original text')
        super().set_episode(episode,transform,catalog)
        bounds=[transform.x_min,transform.y_min,transform.x_max,transform.y_max]
        self.original_instruction=episode.instruction
        self.record,audit=public_text_record(episode.map_name,episode.instruction,bounds,self.public_objects,self.public_name_matcher,mode=self.instruction_mode,max_landmarks=self.args.max_landmarks)
        self.relation_coverage={**audit,'visual_anchor_available':False}

    def predict(self,state,instruction):
        if instruction!=self.original_instruction:raise ValueError('Original public instruction changed')
        result=super().predict(state,self.record['instruction'])
        result.diagnostics['instruction_mode']=self.instruction_mode
        result.diagnostics['model_instruction']=self.record['instruction']
        return result
