import pandas as pd

print("Loading dataset...", flush=True)

df = pd.read_csv("df_final.csv")

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
