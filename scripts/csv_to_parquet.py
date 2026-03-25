import pandas as pd

print("Reading CSV...")
df = pd.read_csv("data/processed/training_dataset.csv")
print(f"Shape: {df.shape}")

print("Saving parquet...")
df.to_parquet("data/processed/training_dataset.parquet", index=False)
print("Done.")