"""Original full RGB policy behind the shared landmark regression interface."""
from .stage1_runtime import GraphNavStage1Policy


class OriginalRGBRegressionPolicy(GraphNavStage1Policy):
    def __init__(self,checkpoint,unused_visual_checkpoint,directory):
        super().__init__(checkpoint,directory)

    def predict(self,state,instruction,*,use_visual=True):
        # The shared runner flag must not disable original RGB in this baseline.
        return super().predict(state,instruction)
