

from sklearn.feature_extraction import FeatureHasher


def hash_features(X, n_features=16):
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    X_iterable = X.astype(str).apply(lambda x: [x])
    return hasher.transform(X_iterable).toarray()
