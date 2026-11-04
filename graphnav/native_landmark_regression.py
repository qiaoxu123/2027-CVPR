"""Untrained shared-space score with exactly the new fusion/runtime interface."""
from .geometry_guided_runtime import GeometryGuidedLandmarkPolicy
from .geometry_guided_visual import LandmarkVisualBelief


class NativeLandmarkRegressionPolicy(GeometryGuidedLandmarkPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Initial adapter output is exactly zero, independent of random hidden
        # weights: only frozen native .25-name/.75-type similarity remains.
        self.learned_visual=LandmarkVisualBelief().to(self.device).eval().requires_grad_(False)
