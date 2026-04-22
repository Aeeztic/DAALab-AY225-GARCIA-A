import polars as pl
from pathlib import Path

# Get correct base directory
BASE_DIR = Path(__file__).resolve().parent

# Paths
csv_path = BASE_DIR / "data" / "raw" / "students.csv"
parquet_path = BASE_DIR / "data" / "students.parquet"

print("Reading CSV... (this may take a bit)")
df = pl.read_csv(csv_path)

print("Rows loaded:", df.height)
print("Columns:", len(df.columns))

print("Writing to Parquet...")
df.write_parquet(parquet_path)

print("✅ Clean parquet created")