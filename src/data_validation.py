# src/data_validation.py

import pandas as pd
import pandera.pandas as pa

# Load CSV
df = pd.read_csv("data/loan.csv")

# Cast Loan_Amount to float
df["Loan_Amount"] = df["Loan_Amount"].astype(float)

# Define schema
schema = pa.DataFrameSchema({
    "Loan_Amount": pa.Column(float, checks=pa.Check.ge(0)),   # >=0
    "Employment_Status": pa.Column(str, checks=pa.Check(lambda s: s.notnull()))
})

# Validate
validated_df = schema.validate(df)

print("✅ Data validation passed!")