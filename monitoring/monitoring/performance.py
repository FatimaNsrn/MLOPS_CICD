# Author: Mezred Mohamed Wassim
# Role: MLOps / Monitoring Engineer

from sklearn.metrics import f1_score, recall_score

def compute_metrics(y_true, y_pred):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro")
    }
