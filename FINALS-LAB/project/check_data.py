import polars as pl

# CSV
csv_df = pl.read_csv("data/raw/students.csv")
print("CSV rows:", csv_df.height)

# Parquet
parquet_df = pl.read_parquet("data/students.parquet")
print("Parquet rows:", parquet_df.height)