import numpy as np

from cohortcraft.segmentation.kmeans import assign_segments, choose_k, segment_customers
from cohortcraft.segmentation.profile import profile_segments


def test_segmentation_and_profiles(frame):
    result = segment_customers(frame, k_min=2, k_max=5, seed=0)
    assert 2 <= result.k <= 5
    assert len(np.unique(result.labels)) == result.k
    assert len(result.k_search) == 4
    df = frame.assign(segment=result.labels)
    prof = profile_segments(df)
    assert prof["customers"].sum() == len(frame)
    assert np.isclose(prof["share"].sum(), 1.0)
    assert prof["name"].is_unique
    assert (assign_segments(result.model, frame) == result.labels).all()


def test_choose_k_prefers_min_k():
    import pandas as pd

    ks = pd.DataFrame({"k": [2, 3, 4], "silhouette": [0.6, 0.4, 0.5], "inertia": [3, 2, 1]})
    assert choose_k(ks, min_k=3) == 4
    assert choose_k(ks, min_k=2) == 2
