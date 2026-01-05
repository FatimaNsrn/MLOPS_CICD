#BROKEN CODE (Intentional Bug)

from sklearn.feature_extraction import FeatureHasher


def hash_features(X, n_features=16):
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    hashed = hasher.transform(X.astype(str).values).toarray()
    return hashed[0]   # ❌ INTENTIONAL BUG
