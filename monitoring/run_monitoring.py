# Author: Mezred Mohamed Wassim
# Role: MLOps / Monitoring Engineer

import pandas as pd
import yaml
from monitoring.schema_utils import harmonize_columns
from monitoring.data_validation import validate_incoming
from monitoring.drift import compute_drift
from monitoring.performance import compute_metrics
from monitoring.actions import trigger_retrain, trigger_fallback

config = yaml.safe_load(open("config.yaml"))

reference = harmonize_columns(pd.read_csv("reference.csv"))
incoming = harmonize_columns(pd.read_csv("incoming.csv"))

validation = validate_incoming(incoming, config["data"]["required_columns"])

if not validation["ok"]:
    trigger_fallback()
    print("Validation failed")
    exit()

drift_alerts = compute_drift(
    reference,
    incoming,
    config["drift"]["alert_threshold"]
)

if drift_alerts:
    trigger_retrain()
    print("Drift detected:", drift_alerts)
