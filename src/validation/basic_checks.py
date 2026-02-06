from pathlib import Path
import pandas as pd
import great_expectations as ge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

trans = pd.read_csv(DATA_DIR / "train_transaction.csv")
ident = pd.read_csv(DATA_DIR / "train_identity.csv")

df = trans.merge(ident, on="TransactionID", how="left")

ge_df = ge.from_pandas(df)

# FIXED: correct table expectation
ge_df.expect_table_row_count_to_be_between(min_value=500_000)

ge_df.expect_column_to_exist("TransactionID")
ge_df.expect_column_to_exist("isFraud")

ge_df.expect_column_values_to_not_be_null("isFraud")
ge_df.expect_column_values_to_be_in_set("isFraud", [0, 1])

ge_df.expect_column_values_to_be_unique("TransactionID")

ge_df.expect_column_values_to_be_between(
    "TransactionAmt",
    min_value=0,
    mostly=0.999
)

results = ge_df.validate()

if not results["success"]:
    raise ValueError("Data validation failed")

print("Data validation passed")
