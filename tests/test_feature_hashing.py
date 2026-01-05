from sklearn.feature_extraction import FeatureHasher
import numpy as np


def test_feature_hashing_consistency():
    hasher = FeatureHasher(n_features=16, input_type="string")

    input_data = [["electronics"], ["electronics"]]
    hashed = hasher.transform(input_data).toarray()

    # Same input → same hash
    assert np.array_equal(hashed[0], hashed[1])

def test_feature_hashing_shape():
    hasher = FeatureHasher(n_features=16, input_type="string")

    input_data = [["fashion"], ["books"]]
    hashed = hasher.transform(input_data).toarray()

    assert hashed.shape == (2, 16)
