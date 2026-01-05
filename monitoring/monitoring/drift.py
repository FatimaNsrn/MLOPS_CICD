# Author: Mezred Mohamed Wassim
# Role: MLOps / Monitoring Engineer

import numpy as np
from scipy.spatial.distance import jensenshannon

def compute_numeric_drift(ref, cur):
    return abs(ref.mean() - cur.mean()) / (ref.std() + 1e-6)

def compute_categorical_drift(ref, cur):
    ref_dist = ref.value_counts(normalize=True)
    cur_dist = cur.value_counts(normalize=True)
    idx = ref_dist.index.union(cur_dist.index)
    return jensenshannon(
        ref_dist.reindex(idx, fill_value=0),
        cur_dist.reindex(idx, fill_value=0)
    )

def compute_drift(reference, incoming, threshold):
    alerts = []
    for col in reference.columns:
        if col not in incoming.columns:
            continue
        if reference[col].dtype == "object":
            score = compute_categorical_drift(reference[col], incoming[col])
        else:
            score = compute_numeric_drift(reference[col], incoming[col])

        if score > threshold:
            alerts.append((col, float(score)))

    return alerts
