import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer


class AmenitiesPriceAnalyzer:
    def __init__(
        self,
        category_dir="category_csvs",
        output_dir="amenities_outputs",
        min_count=30,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        remove_top_price_outliers=True,
        price_upper_quantile=0.99
    ):
        self.category_dir = Path(category_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.min_count = min_count
        self.sep = sep
        self.decimal = decimal
        self.encoding = encoding

        self.remove_top_price_outliers = remove_top_price_outliers
        self.price_upper_quantile = price_upper_quantile

        self.df = None
        self.amenities_binary = None
        self.importance_table = None


    def load_csv(self, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path.resolve()}")

        df = pd.read_csv(
            file_path,
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



    @staticmethod
    def parse_amenities(value):
        if pd.isna(value):
            return []

        value = str(value).strip()

        if value == "" or value == "[]":
            return []

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip() != ""
                ]

            return []

        except Exception:
            value = value.replace("[", "").replace("]", "")
            parts = value.split(",")

            return [
                item.strip().strip('"').strip("'")
                for item in parts
                if item.strip() != ""
            ]



    @staticmethod
    def clean_column_name(amenity):
        amenity = amenity.lower()
        amenity = amenity.replace("&", "and")
        amenity = re.sub(r"[^a-z0-9]+", "_", amenity)
        amenity = re.sub(r"_+", "_", amenity)
        amenity = amenity.strip("_")

        return "amenity_" + amenity

    @staticmethod
    def make_unique_column_names(names):
        """
        Appends _2, _3 etc. if duplicate column names arise after cleaning.
        """
        seen = {}
        unique_names = []

        for name in names:
            if name not in seen:
                seen[name] = 1
                unique_names.append(name)
            else:
                seen[name] += 1
                unique_names.append(f"{name}_{seen[name]}")

        return unique_names



    def load_data(self):
        listing_file = self.category_dir / "02_listing_basic.csv"
        price_file = self.category_dir / "01_target_price.csv"

        listing_df = self.load_csv(listing_file)
        price_df = self.load_csv(price_file)

        if "id" not in listing_df.columns:
            raise KeyError("Column 'id' missing in 02_listing_basic.csv.")

        if "amenities" not in listing_df.columns:
            raise KeyError("Column 'amenities' missing in 02_listing_basic.csv.")

        if "id" not in price_df.columns:
            raise KeyError("Column 'id' missing in 01_target_price.csv.")

        if "price" not in price_df.columns:
            raise KeyError("Column 'price' missing in 01_target_price.csv.")

        df = listing_df[["id", "amenities"]].merge(
            price_df[["id", "price"]],
            on="id",
            how="inner"
        )

        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        df = df[
            df["price"].notna() &
            (df["price"] > 0)
        ].copy()

        if self.remove_top_price_outliers:
            upper_limit = df["price"].quantile(self.price_upper_quantile)
            df = df[df["price"] <= upper_limit].copy()
            print(f"Top outliers removed. Price upper threshold at {self.price_upper_quantile:.2f}-quantile: {upper_limit:.2f}")

        df["amenities_list"] = df["amenities"].apply(self.parse_amenities)

        self.df = df

        print("Loaded and merged data:", self.df.shape)

        return self.df



    def create_binary_amenities(self):
        if self.df is None:
            self.load_data()

        mlb = MultiLabelBinarizer()
        matrix = mlb.fit_transform(self.df["amenities_list"])

        original_amenities = [
            self.remove_invalid_unicode(a)
            for a in list(mlb.classes_)
        ]

        cleaned_names = [
            self.clean_column_name(a)
            for a in original_amenities
        ]
        cleaned_names = self.make_unique_column_names(cleaned_names)

        binary_df = pd.DataFrame(
            matrix,
            columns=cleaned_names,
            index=self.df.index
        )

        binary_df.insert(0, "id", self.df["id"].values)
        binary_df.insert(1, "price", self.df["price"].values)

        # Mapping von originalem Amenity-Namen zu Spaltenname
        mapping_df = pd.DataFrame({
            "amenity_original": original_amenities,
            "amenity_column": cleaned_names
        })

        # Calculate frequencies
        counts = binary_df.drop(columns=["id", "price"]).sum().sort_values(ascending=False)

        valid_cols = counts[counts >= self.min_count].index.tolist()

        print(f"Total amenities: {len(counts)}")
        print(f"Amenities with at least {self.min_count} occurrences: {len(valid_cols)}")

        filtered_binary_df = binary_df[["id", "price"] + valid_cols].copy()

        self.amenities_binary = filtered_binary_df

        # Save files
        binary_path = self.output_dir / f"amenities_binary_min{self.min_count}.csv"
        mapping_path = self.output_dir / "amenities_column_mapping.csv"

        filtered_binary_df.to_csv(
            binary_path,
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding
        )

        mapping_df.to_csv(
            mapping_path,
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding
        )

        print("Binary amenities file saved:", binary_path)
        print("Amenity mapping saved:", mapping_path)

        return filtered_binary_df



    def calculate_price_importance(self):
        if self.amenities_binary is None:
            self.create_binary_amenities()

        df = self.amenities_binary.copy()

        global_median_price = df["price"].median()
        global_mean_price = df["price"].mean()

        amenity_cols = [
            col for col in df.columns
            if col not in ["id", "price"]
        ]

        rows = []

        for col in amenity_cols:
            with_amenity = df[df[col] == 1]
            without_amenity = df[df[col] == 0]

            count_with = len(with_amenity)
            count_without = len(without_amenity)

            if count_with < self.min_count:
                continue

            median_with = with_amenity["price"].median()
            median_without = without_amenity["price"].median()

            mean_with = with_amenity["price"].mean()
            mean_without = without_amenity["price"].mean()

            median_diff = median_with - median_without
            mean_diff = mean_with - mean_without

            if median_without != 0:
                median_diff_percent = median_diff / median_without * 100
            else:
                median_diff_percent = np.nan

            if mean_without != 0:
                mean_diff_percent = mean_diff / mean_without * 100
            else:
                mean_diff_percent = np.nan

            rows.append({
                "amenity_column": col,
                "count_with_amenity": count_with,
                "count_without_amenity": count_without,
                "share_with_amenity_percent": count_with / len(df) * 100,

                "median_price_with_amenity": median_with,
                "median_price_without_amenity": median_without,
                "median_price_difference": median_diff,
                "median_price_difference_percent": median_diff_percent,

                "mean_price_with_amenity": mean_with,
                "mean_price_without_amenity": mean_without,
                "mean_price_difference": mean_diff,
                "mean_price_difference_percent": mean_diff_percent,

                "absolute_median_price_difference": abs(median_diff),
                "global_median_price": global_median_price,
                "global_mean_price": global_mean_price
            })

        importance = pd.DataFrame(rows)

        importance = importance.sort_values(
            by="absolute_median_price_difference",
            ascending=False
        )

        self.importance_table = importance

        output_path = self.output_dir / f"amenities_price_importance_min{self.min_count}.csv"

        importance.to_csv(
            output_path,
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding
        )

        print("Amenity importance file saved:", output_path)

        return importance


    def print_top_results(self, top_n=20):
        if self.importance_table is None:
            self.calculate_price_importance()

        print(f"\nTop {top_n} Amenities nach absolutem Medianpreis-Unterschied:")
        print(
            self.importance_table[
                [
                    "amenity_column",
                    "count_with_amenity",
                    "median_price_with_amenity",
                    "median_price_without_amenity",
                    "median_price_difference",
                    "median_price_difference_percent"
                ]
            ].head(top_n)
        )

        positive = self.importance_table.sort_values(
            "median_price_difference",
            ascending=False
        )

        negative = self.importance_table.sort_values(
            "median_price_difference",
            ascending=True
        )

        print(f"\nTop {top_n} Amenities mit positivem Preisunterschied:")
        print(
            positive[
                [
                    "amenity_column",
                    "count_with_amenity",
                    "median_price_with_amenity",
                    "median_price_without_amenity",
                    "median_price_difference",
                    "median_price_difference_percent"
                ]
            ].head(top_n)
        )

        print(f"\nTop {top_n} Amenities mit negativem Preisunterschied:")
        print(
            negative[
                [
                    "amenity_column",
                    "count_with_amenity",
                    "median_price_with_amenity",
                    "median_price_without_amenity",
                    "median_price_difference",
                    "median_price_difference_percent"
                ]
            ].head(top_n)
        )



    def run(self):
        self.load_data()
        self.create_binary_amenities()
        self.calculate_price_importance()
        self.print_top_results()

        print("\nFertig. Ergebnisse liegen im Ordner:", self.output_dir)

        return self.importance_table

    @staticmethod
    def remove_invalid_unicode(value):
        if pd.isna(value):
            return ""

        value = str(value)

        return (
            value
            .encode("utf-8", errors="ignore")
            .decode("utf-8", errors="ignore")
        )


# Run from the repo root: python scripts/07_amenities_price_analysis.py
# Note: this only builds a full per-amenity binary matrix (hundreds of
# columns) — it is NOT the script that produced the grouped ~10-feature
# amenities_reduced_grouped.csv used by the final model. That grouping
# script is missing from the project (see README known issues).
analyzer = AmenitiesPriceAnalyzer(
    category_dir="data/processed/category_csvs",
    output_dir="outputs/amenities_outputs",
    min_count=30
)

importance_table = analyzer.run()