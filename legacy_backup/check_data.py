
import pandas as pd

df = pd.read_csv("df_final.csv")

# Parse the date column
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

print("Dataset shape:", df.shape)
print("\nDate range:")
print("Start:", df["DATE"].min())
print("End:", df["DATE"].max())

print("\nFirst 10 observations:")
print(df[["DATE", "POWER_DEMAND", "temp", "humidity"]].head(10))

print("\nDemand statistics:")
print(df["POWER_DEMAND"].describe())

print("\nDuplicate dates:", df["DATE"].duplicated().sum())
print("\nDemand missing values:", df["POWER_DEMAND"].isna().sum())
