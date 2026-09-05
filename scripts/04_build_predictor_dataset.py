import pandas as pd
import numpy as np
from pathlib import Path


class PredictorWithIDCreator:
    def __init__(
        self,
        category_dir="data/processed/category_csvs",
        output_file="data/processed/predictor_dataset_with_id.csv",
        sep=";",
        decimal=",",
        encoding="utf-8-sig"
    ):
        self.category_dir = Path(category_dir)
        self.output_file = Path(output_file)
        self.sep = sep
        self.decimal = decimal
        self.encoding = encoding

    def read_csv(self, filename):
        path = self.category_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")

        df = pd.read_csv(
            path,
            sep=self.sep,
            decimal=self.decimal,
            encoding=self.encoding,
            low_memory=False
        )

        df.columns = (
            df.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        return df

    def create(self):
        price = self.read_csv("01_target_price.csv")
        basic = self.read_csv("02_listing_basic.csv")
        location = self.read_csv("03_location.csv")

        needed_basic = [
            "id",
            "bedrooms",
            "accommodates",
            "beds",
            "room_type"
        ]

        needed_location = [
            "id",
            "neighbourhood_cleansed"
        ]

        needed_reviews = [
            "id",
            "review_scores_rating",
            "number_of_reviews"
        ]

        df = price[["id", "price"]].copy()

        df = df.merge(
            basic[[col for col in needed_basic if col in basic.columns]],
            on="id",
            how="left"
        )

        df = df.merge(
            location[[col for col in needed_location if col in location.columns]],
            on="id",
            how="left"
        )



        # Clean Numeric Columns
        numeric_cols = [
            "price",
            "bedrooms",
            "accommodates",
            "beds",
            "review_scores_rating",
            "number_of_reviews"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean Target-Variables
        df = df[
            df["price"].notna() &
            (df["price"] > 0)
        ].copy()

        upper_price = df["price"].quantile(0.99)
        df = df[df["price"] <= upper_price].copy()

        # is_studio
        if "bedrooms" in df.columns:
            df["is_studio"] = (
                df["bedrooms"].fillna(0) == 0
            ).astype(int)

            df["bedrooms_log"] = np.log1p(
                df["bedrooms"].fillna(0)
            )
        else:
            df["is_studio"] = 0
            df["bedrooms_log"] = 0

        final_cols = [
            "id",
            "price",
            "bedrooms",
            "is_studio",
            "bedrooms_log",
            "accommodates",
            "beds",
            "room_type",
            "neighbourhood_cleansed",
            "review_scores_rating",
            "number_of_reviews"
        ]

        final_cols = [col for col in final_cols if col in df.columns]
        df = df[final_cols].copy()

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            self.output_file,
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding
        )

        print("Neue Predictor-Datei gespeichert:")
        print(self.output_file)
        print("Shape:", df.shape)
        print("Spalten:", df.columns.tolist())

        return df


if __name__ == "__main__":
    # Run from the repo root: python scripts/04_build_predictor_dataset.py
    creator = PredictorWithIDCreator(
        category_dir="data/processed/category_csvs",
        output_file="data/processed/predictor_dataset_with_id.csv"
    )

    creator.create()
