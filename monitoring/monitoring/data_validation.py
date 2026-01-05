# Author: Mezred Mohamed Wassim
# Role: MLOps / Monitoring Engineer

def validate_incoming(df, required_columns):
    errors = []

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    if df.empty:
        errors.append("Incoming dataset is empty")

    if "discount_rate" in df.columns:
        if ((df["discount_rate"] < 0) | (df["discount_rate"] > 1)).any():
            errors.append("discount_rate outside [0,1]")

    return {
        "ok": len(errors) == 0,
        "errors": errors
    }
