import numpy as np
import pandas as pd
from features.hashing import hash_features


def test_feature_hashing_consistency():
    X = pd.Series(["electronics", "electronics"])
    hashed = hash_features(X)

    assert np.array_equal(hashed[0], hashed[1])


def test_feature_hashing_shape():
    X = pd.Series(["fashion", "books"])
    hashed = hash_features(X)

    assert hashed.shape == (2, 16)
