from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

trans = pd.read_csv(DATA_DIR / "train_transaction.csv")
ident = pd.read_csv(DATA_DIR / "train_identity.csv")

df = trans.merge(ident, on="TransactionID", how="left")


BASE_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionAmt",
    "TransactionDT",
    "ProductCD"
]

df = df[BASE_COLUMNS]

import numpy as np

df["log_transaction_amt"] = np.log1p(df["TransactionAmt"])

SECONDS_IN_DAY = 24 * 60 * 60
df["transaction_day"] = df["TransactionDT"] // SECONDS_IN_DAY

df["ProductCD"] = df["ProductCD"].astype("category").cat.codes

FEATURE_COLUMNS = [
    "log_transaction_amt",
    "transaction_day",
    "ProductCD"
]

X = df[FEATURE_COLUMNS]
y = df["isFraud"]

print(X.head())
print(y.value_counts(normalize=True))

