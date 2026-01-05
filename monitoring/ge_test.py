import great_expectations as ge
import pandas as pd

df = pd.DataFrame({
    "discount_rate": [0.1, 0.2, 0.15],
    "profit": [1, 0, 1]
})

ge_df = ge.from_pandas(df)

result = ge_df.expect_column_values_to_be_between(
    "discount_rate", min_value=0, max_value=0.5
)

print(result["success"])
