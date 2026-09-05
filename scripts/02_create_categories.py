import pandas as pd
from pathlib import Path

# 01: Path

# Run from the repo root: python scripts/02_create_categories.py
input_file = Path("data/processed/listings_cleaned.csv")
output_dir = Path("data/processed/category_csvs")
output_dir.mkdir(parents=True, exist_ok=True)


# 02: Load cleaned Data
# sep=";" and decimal=","

df = pd.read_csv(
    input_file,
    sep=";",
    decimal=",",
    low_memory=False
)

print("Loaded File:", df.shape)


# 03: Define Categories

categories = {
    "01_target_price.csv": [
        "id",
        "price"
    ],

    "02_listing_basic.csv": [
        "id",
        "name",
        "description",
        "neighborhood_overview",
        "property_type",
        "room_type",
        "accommodates",
        "bathrooms",
        "bathrooms_text",
        "bedrooms",
        "beds",
        "amenities"
    ],

    "03_location.csv": [
        "id",
        "neighbourhood",
        "neighbourhood_cleansed",
        "neighbourhood_group_cleansed",
        "latitude",
        "longitude"
    ],

    "04_host.csv": [
        "id",
        "host_id",
        "host_name",
        "host_since",
        "host_location",
        "host_about",
        "host_response_time",
        "host_response_rate",
        "host_acceptance_rate",
        "host_is_superhost",
        "host_neighbourhood",
        "host_listings_count",
        "host_total_listings_count",
        "host_verifications",
        "host_has_profile_pic",
        "host_identity_verified"
    ],

    "05_booking_rules_availability.csv": [
        "id",
        "minimum_nights",
        "maximum_nights",
        "minimum_minimum_nights",
        "maximum_minimum_nights",
        "minimum_maximum_nights",
        "maximum_maximum_nights",
        "minimum_nights_avg_ntm",
        "maximum_nights_avg_ntm",
        "has_availability",
        "availability_30",
        "availability_60",
        "availability_90",
        "availability_365",
        "availability_eoy",
        "instant_bookable"
    ],

    "06_reviews_ratings.csv": [
        "id",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        "number_of_reviews_ly",
        "first_review",
        "last_review",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "reviews_per_month"
    ],

    "07_host_calculated_counts.csv": [
        "id",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms"
    ],

    "08_technical_links.csv": [
        "id",
        "listing_url",
        "scrape_id",
        "last_scraped",
        "source",
        "picture_url",
        "host_url",
        "host_thumbnail_url",
        "host_picture_url",
        "calendar_last_scraped"
    ],

    "09_possible_leakage_or_optional.csv": [
        "id",
        "estimated_occupancy_l365d",
        "estimated_revenue_l365d",
        "license",
        "calendar_updated"
    ]
}



# 04: Create seperate CSV for every Categorie


used_columns = set()

for filename, cols in categories.items():
    existing_cols = [col for col in cols if col in df.columns]

    if "id" not in existing_cols and "id" in df.columns:
        existing_cols = ["id"] + existing_cols

    category_df = df[existing_cols].copy()

    output_path = output_dir / filename

    category_df.to_csv(
        output_path,
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig"
    )

    used_columns.update(existing_cols)

    print(f"{filename}: {category_df.shape[1]} Columns saved")



# 05: Save remaining Columns



remaining_cols = [col for col in df.columns if col not in used_columns]

if remaining_cols:
    remaining_df = df[["id"] + remaining_cols].copy()

    remaining_df.to_csv(
        output_dir / "99_remaining_columns.csv",
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8-sig"
    )

    print("99_remaining_columns.csv gespeichert mit:", remaining_cols)
else:
    print("No remaining columns left.")



# 6. Summary as Controll 


summary_rows = []

for filename, cols in categories.items():
    for col in cols:
        summary_rows.append({
            "category_file": filename,
            "column": col,
            "exists_in_dataset": col in df.columns
        })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    output_dir / "00_column_category_overview.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("Done. Files are in the folder.:", output_dir)
