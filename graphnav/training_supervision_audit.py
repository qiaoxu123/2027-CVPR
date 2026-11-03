"""Private posthoc loss-label audits; never imported by normal inference."""
import numpy as np


def position_region_counts(distance, valid, region_labels, category):
    if category not in ('Building', 'Parking', 'Ground'):
        return dict(applicable=False, reason='vehicle/object identity is not land-use region type')
    expected = ('Building', 'Parking', 'Ground').index(category)
    positive = np.asarray(valid, bool) & (np.asarray(distance) <= 20)
    label = np.asarray(region_labels)
    return dict(applicable=True, positive_points=int(positive.sum()),
        same_region=int((positive & (label == expected)).sum()),
        other_region=int((positive & (label >= 0) & (label != expected)).sum()),
        unknown_region=int((positive & (label == -1)).sum()))


def car_core_position_counts(distance, valid, core_labels):
    distance = np.asarray(distance); valid = np.asarray(valid, bool); label = np.asarray(core_labels)
    positive = valid & (distance <= 20)
    weights = np.exp(-.5*(distance/10)**2)*positive
    denominator = weights.sum()
    mass = lambda mask: float(weights[mask].sum()/denominator) if denominator else 0.
    return dict(position_positive_points=int(positive.sum()), target_core_points=int((positive & (label == 1)).sum()),
        other_car_core_points=int((positive & (label == 0)).sum()), unknown_points=int((positive & (label == -1)).sum()),
        target_core_gaussian_mass=mass(label == 1), other_car_core_gaussian_mass=mass(label == 0),
        unknown_gaussian_mass=mass(label == -1))
