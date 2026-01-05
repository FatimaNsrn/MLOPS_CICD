from prefect import flow, task
import pandas as pd
import great_expectations as ge
import numpy as np
from pathlib import Path


@task
def validate_incoming_data(csv_path: str) -> bool:
    df = pd.read_csv(csv_path)

    ge_df = ge.from_pandas(df)

    result = ge_df.expect_column_values_to_be_between(
        column="discount",
        min_value=0.0,
        max_value=0.8
    )

    print("Validation success:", result["success"])
    return result["success"]


@task
def detect_discount_drift(
    reference_csv: str,
    incoming_csv: str,
    threshold: float = 0.2
) -> bool:
    ref = pd.read_csv(reference_csv)["discount_rate"]
    inc = pd.read_csv(incoming_csv)["discount"]

    drift_score = abs(ref.mean() - inc.mean())

    print(f"Drift score: {drift_score:.3f}")

    return drift_score > threshold

@task
def request_retraining(reason: str):
    Path("signals").mkdir(exist_ok=True)
    signal_file = Path("signals/retrain_requested.txt")
    signal_file.write_text(f"RETRAIN REQUESTED\nReason: {reason}\n")

    print("🚨 Retrain triggered")

@task
def fallback_discount_rule():
    print("⚠️ Model unhealthy → using fallback rules")
    print("➡ Applying max 10% discount")
    print("➡ Holiday discount capped at 5%")


@flow(name="monitoring-flow")
def monitoring_flow():
    incoming_csv = "../data/raw/monitoring_data.csv"
    reference_csv = "../data/processed/profit_dataset_processed.csv"

    # Step 1: Validation
    is_valid = validate_incoming_data(incoming_csv)

    # Step 2: Drift
    has_drift = detect_discount_drift(reference_csv, incoming_csv)

    # Step 3: Decision
    if is_valid and not has_drift:
        print("✅ Data healthy → send to ML pipeline")
    else:
        fallback_discount_rule()
        request_retraining("monitoring_failure")


if __name__ == "__main__":
    monitoring_flow()
