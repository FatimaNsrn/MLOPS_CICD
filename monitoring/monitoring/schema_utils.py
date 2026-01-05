# Author: Mezred Mohamed Wassim
# Role: MLOps / Monitoring Engineer

def harmonize_columns(df):
    rename_map = {
        "discount": "discount_rate",
        "avg_units_sold": "avg_units_sold_30d",
        "avg_customer_inflow": "avg_customers_30d",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
