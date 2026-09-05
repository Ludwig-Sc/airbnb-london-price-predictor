import pandas as pd
from pathlib import Path

# 01: Adjusting Datapath


# Run from the repo root: python scripts/01_clean_data.py
input_file = Path("data/raw/listings.csv")                 # or "listings.csv.gz"
output_file = Path("data/processed/listings_cleaned.csv")  # new file, original untouched


# 02: Load Data 

df = pd.read_csv(
    input_file,
    compression="infer",
    low_memory=False
)

print("Original Shape:", df.shape)


# 03: Boolean-Columns t/f -> 1/0


boolean_cols = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "has_availability",
    "instant_bookable"
]

for col in boolean_cols:
    if col in df.columns:
        df[col] = df[col].map({
            "t": 1,
            "f": 0,
            True: 1,
            False: 0
        }).astype("Int64")



# 04: Percent-Columns cleaned
# Example: "96%" -> 0.96

percent_cols = [
    "host_response_rate",
    "host_acceptance_rate"
]

for col in percent_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .replace(["nan", "N/A", "None", ""], pd.NA)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100



# 05: Price-Columns cleaned
# Example: "$1,234.00" -> 1234.00

if "price" in df.columns:
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace(["nan", "N/A", "None", ""], pd.NA)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")



# 06: Date-Columns uniformed
# Format stays: YYYY-MM-DD


date_cols = [
    "last_scraped",
    "host_since",
    "calendar_last_scraped",
    "first_review",
    "last_review"
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")


# 07: Empty Columns dropped


df = df.dropna(axis=1, how="all")



# 08: Saving new CSV 
# sep=";"       
# decimal=","  

df.to_csv(
    output_file,
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig"
)

print("Bereinigte Datei gespeichert als:", output_file)
print("Cleaned Shape:", df.shape)
