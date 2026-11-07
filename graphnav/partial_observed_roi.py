"""Experimental visible-intersection crops; normal navigation is unchanged."""
import numpy as np
from .observed_candidate_rois import camera_support_mask


def observed_intersection_crop(image, roi_quad, camera_footprint):
    """Return observed-only pixels, their original ROI bounds and coverage.

    Source ROI images already carry depth/RGB invalidity as zero. This function
    additionally enforces the current camera footprint before removing margins.
    It neither renders pixels nor repairs holes nor reads target annotations.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.shape[0] != image.shape[1]:
        raise ValueError('Square source ROI RGB required')
    support = camera_support_mask(roi_quad, camera_footprint, image.shape[0])
    fraction = float(support.mean())
    yy, xx = np.nonzero(support)
    if len(xx) == 0:
        return None, None, fraction
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()+1), int(yy.min()), int(yy.max()+1)
    bounds = (x0,y0,x1,y1)
    if x1-x0 < 2 or y1-y0 < 2:
        return None, bounds, fraction
    observed = image.copy()
    observed[~support] = 0
    return observed[y0:y1,x0:x1].copy(), bounds, fraction
