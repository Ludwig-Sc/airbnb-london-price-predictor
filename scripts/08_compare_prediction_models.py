import json
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent


class AirbnbPredictionModelRunner:
    """
    Create two test runs:

    Test run 1:
    - predictor/listings_final_bedrooms.csv
    - amenities_reduced_grouped.csv

    Test run 2:
    - predictor/listings_final_bedrooms.csv
    - amenities_reduced_grouped.csv
    - additional features from category_csvs/, provided they are not already present

    Output:
    - Model comparison
    - CV results
    - Predictions
    - Charts
    - Feature importance
    - Best model parameters
    - Saved models
    """

    def __init__(
        self,
        # Defaults below are only used if this class is imported directly;
        # __main__ at the bottom of this file overrides them with paths
        # relative to the repo root (BASE_DIR).
        predictor_file="data/processed/predictor_dataset_with_id.csv",
        amenities_file="data/processed/amenities_reduced_grouped.csv",
        category_dir="data/processed/category_csvs",
        output_base_dir="outputs/model_results",
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        test_size=0.2,
        random_state=42,
        cv=3,
        search_type="random",   # "random" or "grid"
        n_iter=15,
        sample_size=None,       # e.g. 30000 for rapid testing, None for full dataset
        price_upper_quantile=0.99,
        use_log_price_feature=True,
        save_models=True,
        permutation_importance_sample_size=3000
    ):
        self.predictor_file = Path(predictor_file)
        self.amenities_file = Path(amenities_file)
        self.category_dir = Path(category_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_base_dir = Path(output_base_dir) / timestamp
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.sep = sep
        self.decimal = decimal
        self.encoding = encoding

        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.search_type = search_type
        self.n_iter = n_iter
        self.sample_size = sample_size
        self.price_upper_quantile = price_upper_quantile
        self.use_log_price_feature = use_log_price_feature
        self.save_models = save_models
        self.permutation_importance_sample_size = permutation_importance_sample_size

        self.run_results = []

    
    # 01: Basic functions
    

    def read_csv_safely(self, path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path.resolve()}")

        try:
            df = pd.read_csv(
                path,
                sep=self.sep,
                decimal=self.decimal,
                encoding=self.encoding,
                encoding_errors="ignore",
                low_memory=False
            )
        except TypeError:
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

    def save_json(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False, default=str)

    @staticmethod
    def make_one_hot_encoder():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def clean_object_columns(df):
        """
        Converts obvious booleans and percentage values.
        Preserves standard categories such as room_type and neighbourhood_cleansed.
        """
        df = df.copy()

        for col in df.columns:
            if df[col].dtype == "object":
                lowered = df[col].astype(str).str.lower().str.strip()

                unique_non_null = set(lowered[df[col].notna()].unique())

                # Boolean-like Columns
                if unique_non_null.issubset({"t", "f", "true", "false", "1", "0", "nan", "none", ""}):
                    df[col] = lowered.map({
                        "t": 1,
                        "true": 1,
                        "1": 1,
                        "f": 0,
                        "false": 0,
                        "0": 0
                    })
                    continue

                # Percent-Values like "96%"
                percent_share = lowered.str.contains("%", regex=False, na=False).mean()
                if percent_share > 0.5:
                    cleaned = (
                        lowered
                        .str.replace("%", "", regex=False)
                        .str.replace(",", ".", regex=False)
                    )
                    df[col] = pd.to_numeric(cleaned, errors="coerce") / 100
                    continue

        return df

    
    # 02: Load Data

    def load_predictor_data(self):
        predictor_df = self.read_csv_safely(self.predictor_file)

        if "price" not in predictor_df.columns:
            raise KeyError("Column 'price' missing in predictor dataset.")

        predictor_df = self.clean_object_columns(predictor_df)

        predictor_df["price"] = pd.to_numeric(predictor_df["price"], errors="coerce")

        print("Predictor data loaded:", predictor_df.shape)

        return predictor_df

    def load_amenities_data(self):
        if not self.amenities_file.exists():
            possible_paths = [
                Path("amenities_reduced_grouped.csv"),
                Path("../amenities_reduced_outputs/amenities_reduced_grouped.csv"),
                Path("category_csvs/amenities_reduced_grouped.csv")
            ]

            found = None
            for path in possible_paths:
                if path.exists():
                    found = path
                    break

            if found is None:
                raise FileNotFoundError(
                    "amenities_reduced_grouped.csv not found. "
                    "Please set amenities_file path correctly."
                )

            self.amenities_file = found

        amenities_df = self.read_csv_safely(self.amenities_file)
        amenities_df = self.clean_object_columns(amenities_df)

        if "id" not in amenities_df.columns:
            raise KeyError("Column 'id' missing in amenities_reduced_grouped.csv.")

        if "price" in amenities_df.columns:
            amenities_df = amenities_df.drop(columns=["price"])

        print("Amenities data loaded:", amenities_df.shape)

        return amenities_df

    def merge_predictor_and_amenities(self):
        predictor_df = self.load_predictor_data()
        amenities_df = self.load_amenities_data()

        # Case 1: Predictor file has an ID -> clean merge
        if "id" in predictor_df.columns:
            duplicate_cols = [
                col for col in amenities_df.columns
                if col in predictor_df.columns and col != "id"
            ]

            if duplicate_cols:
                amenities_df = amenities_df.drop(columns=duplicate_cols)

            merged_df = predictor_df.merge(
                amenities_df,
                on="id",
                how="left"
            )

            print("Predictor + Amenities merged via ID:", merged_df.shape)
            return merged_df

        # Case 2: Predictor file has no ID -> Row-order merge
        print(
            "\nWARNING: listings_final_bedrooms.csv does not contain an id column.\n"
            "Attempting to join Predictor and Amenities based on row order.\n"
            "This is only correct if both files were created from the same source file "
            "in exactly the same order.\n"
        )

        if len(predictor_df) != len(amenities_df):
            raise ValueError(
                "Row-order merge not possible: The files have a different number of rows.\n"
                f"Predictor rows: {len(predictor_df)}\n"
                f"Amenities rows: {len(amenities_df)}\n"
                "Please add an ID column to listings_final_bedrooms.csv."
            )

        # Safety check: compare 'price' line by line if both files contain 'price'
        if "price" in predictor_df.columns and "price" in amenities_df.columns:
            predictor_price = pd.to_numeric(predictor_df["price"], errors="coerce").round(2)
            amenities_price = pd.to_numeric(amenities_df["price"], errors="coerce").round(2)

            price_match_rate = (predictor_price == amenities_price).mean()

            print(f"Price match rate between Predictor and Amenities: {price_match_rate:.4f}")

            if price_match_rate < 0.99:
                raise ValueError(
                    "Row-order merge aborted due to price mismatch.\n"
                    f"Match rate: {price_match_rate:.2%}\n"
                    "Please ensure row orders are aligned or provide an ID column."
                )

        # id from amenities 
        if "id" not in amenities_df.columns:
            raise KeyError(
                "amenities_reduced_grouped.csv does not contain an id column. "
                "Cannot perform clean merge with category_csvs."
            )

        # Remove 'price' from 'amenities' because 'predictor_df' already contains 'price'.
        amenities_without_price = amenities_df.drop(
            columns=[col for col in ["price"] if col in amenities_df.columns]
        ).copy()

        # Avoid double columns
        duplicate_cols = [
            col for col in amenities_without_price.columns
            if col in predictor_df.columns
        ]

        if duplicate_cols:
            amenities_without_price = amenities_without_price.drop(columns=duplicate_cols)

        # Merge line by line
        merged_df = pd.concat(
            [
                amenities_without_price[["id"]].reset_index(drop=True),
                predictor_df.reset_index(drop=True),
                amenities_without_price.drop(columns=["id"]).reset_index(drop=True)
            ],
            axis=1
        )

        print("Predictor + Amenities concatenated by row order:", merged_df.shape)
        print("ID retained from amenities_reduced_grouped.csv.")

        return merged_df

    
    # 03: Load additional category_csvs features
   
    def load_category_extra_features(self, existing_columns):
        """
        Loads additional columns from category_csvs, but only those
        that are not already present in the predictor dataset.
        """
        extra_specs = {
            "02_listing_basic.csv": [
                "id",
                "property_type",
                "bathrooms"
            ],
            "03_location.csv": [
                "id",
                "latitude",
                "longitude"
            ],
            "04_host.csv": [
                "id",
                "host_response_time",
                "host_response_rate",
                "host_acceptance_rate",
                "host_is_superhost",
                "host_listings_count",
                "host_total_listings_count",
                "host_has_profile_pic",
                "host_identity_verified"
            ],
            "05_booking_rules_availability.csv": [
                "id",
                "minimum_nights",
                "maximum_nights",
                "availability_30",
                "availability_60",
                "availability_90",
                "availability_365",
                "instant_bookable"
            ],
            "06_reviews_ratings.csv": [
                "id",
                "number_of_reviews_ltm",
                "number_of_reviews_l30d",
                "number_of_reviews_ly",
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
            ]
        }

        merged_extra = None
        used_columns = []

        for filename, wanted_cols in extra_specs.items():
            path = self.category_dir / filename

            if not path.exists():
                print(f"Extra file not found, skipping: {filename}")
                continue

            temp = self.read_csv_safely(path)
            temp = self.clean_object_columns(temp)

            available_cols = [col for col in wanted_cols if col in temp.columns]

            # Only take columns that do not already exist
            selected_cols = [
                col for col in available_cols
                if col == "id" or col not in existing_columns
            ]

            if selected_cols == ["id"] or "id" not in selected_cols:
                continue

            temp = temp[selected_cols].copy()

            if merged_extra is None:
                merged_extra = temp
            else:
                duplicate_cols = [
                    col for col in temp.columns
                    if col in merged_extra.columns and col != "id"
                ]

                if duplicate_cols:
                    temp = temp.drop(columns=duplicate_cols)

                merged_extra = merged_extra.merge(temp, on="id", how="outer")

            used_columns.extend([col for col in selected_cols if col != "id"])

        if merged_extra is None:
            print("No additional category_csvs features found.")
            return None, []

        print("Additional category_csvs features:", merged_extra.shape)
        print("Columns used:", used_columns)

        return merged_extra, used_columns

    def build_dataset_run1(self):
        df = self.merge_predictor_and_amenities()
        return df, "run1_predictor_plus_amenities"

    def build_dataset_run2(self):
        df = self.merge_predictor_and_amenities()

        extra_df, used_extra_cols = self.load_category_extra_features(
            existing_columns=set(df.columns)
        )

        if extra_df is not None:
            duplicate_cols = [
                col for col in extra_df.columns
                if col in df.columns and col != "id"
            ]

            if duplicate_cols:
                extra_df = extra_df.drop(columns=duplicate_cols)

            df = df.merge(extra_df, on="id", how="left")

        return df, "run2_predictor_plus_amenities_plus_category"

    
    # 04: Prepare model data
    

    def prepare_model_data(self, df, run_dir):
        df = df.copy()

        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        df = df[
            df["price"].notna() &
            (df["price"] > 0)
        ].copy()

        
        upper_price = df["price"].quantile(self.price_upper_quantile)
        df = df[df["price"] <= upper_price].copy()

        
        if self.use_log_price_feature:
            if "number_of_reviews" in df.columns:
                df["number_of_reviews_log"] = np.log1p(
                    pd.to_numeric(df["number_of_reviews"], errors="coerce")
                )

            if "reviews_per_month" in df.columns:
                df["reviews_per_month_log"] = np.log1p(
                    pd.to_numeric(df["reviews_per_month"], errors="coerce")
                )

        
        if "number_of_reviews" in df.columns:
            df["has_reviews"] = (
                pd.to_numeric(df["number_of_reviews"], errors="coerce").fillna(0) > 0
            ).astype(int)

        
        if self.sample_size is not None and len(df) > self.sample_size:
            df = df.sample(
                n=self.sample_size,
                random_state=self.random_state
            ).copy()

        # Do not use as a feature
        drop_cols = [
            "price"
        ]

        # Retain the ID for predictions, but do not use it as a feature.
        if "id" in df.columns:
            ids = df["id"].copy()
            drop_cols.append("id")
        else:
            ids = pd.Series(range(len(df)), name="id")

        feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns]).copy()
        y = df["price"].copy()

        # Remove infinite values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # Clean up numeric columns
        for col in feature_df.columns:
            if feature_df[col].dtype == "object":
                continue
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

        numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

        
        for col in numeric_features:
            lower = feature_df[col].quantile(0.001)
            upper = feature_df[col].quantile(0.999)

            if pd.notna(lower) and pd.notna(upper) and lower < upper:
                feature_df[col] = feature_df[col].clip(lower=lower, upper=upper)

        dataset_info = {
            "rows_after_cleaning": int(len(feature_df)),
            "features_total": int(feature_df.shape[1]),
            "numeric_features": int(len(numeric_features)),
            "categorical_features": int(len(categorical_features)),
            "price_upper_quantile": self.price_upper_quantile,
            "price_upper_limit": float(upper_price),
            "sample_size": self.sample_size,
            "feature_columns": feature_df.columns.tolist(),
            "numeric_feature_columns": numeric_features,
            "categorical_feature_columns": categorical_features
        }

        self.save_json(dataset_info, run_dir / "dataset_info.json")

        model_dataset = pd.concat(
            [
                ids.reset_index(drop=True).rename("id"),
                y.reset_index(drop=True).rename("price"),
                feature_df.reset_index(drop=True)
            ],
            axis=1
        )

        model_dataset.to_csv(
            run_dir / "model_dataset_used.csv",
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding,
            errors="ignore"
        )

        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            feature_df,
            y,
            ids,
            test_size=self.test_size,
            random_state=self.random_state
        )

        print("Rows:", len(feature_df))
        print("Features:", feature_df.shape[1])
        print("Numeric:", len(numeric_features))
        print("Categorical:", len(categorical_features))

        return X_train, X_test, y_train, y_test, id_train, id_test, numeric_features, categorical_features

    
    # 05: Preprocessing and Models


    def build_preprocessor(self, numeric_features, categorical_features):
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", self.make_one_hot_encoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ],
            remainder="drop"
        )

        return preprocessor

    def get_models_and_params(self):
        models = {
            "Dummy_Baseline": {
                "estimator": DummyRegressor(),
                "params": {
                    "model__strategy": ["median", "mean"]
                }
            },
            "Ridge_Regression": {
                "estimator": Ridge(random_state=self.random_state),
                "params": {
                    "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]
                }
            },
            "ElasticNet": {
                "estimator": ElasticNet(
                    random_state=self.random_state,
                    max_iter=5000
                ),
                "params": {
                    "model__alpha": [0.001, 0.01, 0.1, 1.0],
                    "model__l1_ratio": [0.1, 0.5, 0.9]
                }
            },
            "Random_Forest": {
                "estimator": RandomForestRegressor(
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                "params": {
                    "model__n_estimators": [150, 300],
                    "model__max_depth": [15, 30, None],
                    "model__min_samples_leaf": [1, 3, 10],
                    "model__max_features": ["sqrt", 0.7]
                }
            },
            "Extra_Trees": {
                "estimator": ExtraTreesRegressor(
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                "params": {
                    "model__n_estimators": [150, 300],
                    "model__max_depth": [15, 30, None],
                    "model__min_samples_leaf": [1, 3, 10],
                    "model__max_features": ["sqrt", 0.7]
                }
            },
            "HistGradientBoosting": {
                "estimator": HistGradientBoostingRegressor(
                    random_state=self.random_state
                ),
                "params": {
                    "model__max_iter": [100, 200],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__max_leaf_nodes": [31, 63],
                    "model__l2_regularization": [0.0, 0.1]
                }
            }
        }

        return models

  
    # 06: Training and Evaluation
  

    def train_and_evaluate_run(self, df, run_name):
        print("\n" + "=" * 80)
        print(f"START: {run_name}")
        print("=" * 80)

        run_dir = self.output_base_dir / run_name
        tables_dir = run_dir / "tables"
        plots_dir = run_dir / "plots"
        models_dir = run_dir / "models"

        tables_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        (
            X_train,
            X_test,
            y_train,
            y_test,
            id_train,
            id_test,
            numeric_features,
            categorical_features
        ) = self.prepare_model_data(df, run_dir)

        preprocessor = self.build_preprocessor(numeric_features, categorical_features)
        models = self.get_models_and_params()

        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }

        result_rows = []
        best_estimators = {}

        for model_name, model_info in models.items():
            print(f"\nTraining model: {model_name}")

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_info["estimator"])
            ])

            params = model_info["params"]

            param_count = 1
            for values in params.values():
                param_count *= len(values)

            if self.search_type == "grid" or param_count <= self.n_iter:
                search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=params,
                    scoring=scoring,
                    refit="mae",
                    cv=self.cv,
                    n_jobs=-1,
                    verbose=1,
                    return_train_score=True,
                    error_score=np.nan
                )
            else:
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=params,
                    n_iter=self.n_iter,
                    scoring=scoring,
                    refit="mae",
                    cv=self.cv,
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.random_state,
                    return_train_score=True,
                    error_score=np.nan
                )

            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            y_pred = np.clip(y_pred, 0, None)

            test_mae = mean_absolute_error(y_test, y_pred)
            test_rmse = self.rmse(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)

            cv_results = pd.DataFrame(search.cv_results_)

            if "mean_test_mae" in cv_results.columns:
                cv_results["mean_test_mae_positive"] = -cv_results["mean_test_mae"]

            if "mean_test_rmse" in cv_results.columns:
                cv_results["mean_test_rmse_positive"] = -cv_results["mean_test_rmse"]

            cv_results.to_csv(
                tables_dir / f"cv_results_{model_name}.csv",
                sep=self.sep,
                decimal=self.decimal,
                index=False,
                encoding=self.encoding,
                errors="ignore"
            )

            best_idx = search.best_index_

            cv_mae = np.nan
            cv_rmse = np.nan
            cv_r2 = np.nan

            if "mean_test_mae" in cv_results.columns:
                cv_mae = -cv_results.loc[best_idx, "mean_test_mae"]

            if "mean_test_rmse" in cv_results.columns:
                cv_rmse = -cv_results.loc[best_idx, "mean_test_rmse"]

            if "mean_test_r2" in cv_results.columns:
                cv_r2 = cv_results.loc[best_idx, "mean_test_r2"]

            predictions = pd.DataFrame({
                "id": id_test.values,
                "actual_price": y_test.values,
                "predicted_price": y_pred,
                "residual": y_test.values - y_pred,
                "absolute_error": np.abs(y_test.values - y_pred),
                "model": model_name
            })

            predictions.to_csv(
                tables_dir / f"predictions_{model_name}.csv",
                sep=self.sep,
                decimal=self.decimal,
                index=False,
                encoding=self.encoding,
                errors="ignore"
            )

            best_params_json = json.dumps(search.best_params_, ensure_ascii=False)

            result_rows.append({
                "run": run_name,
                "model": model_name,
                "cv_mae": cv_mae,
                "cv_rmse": cv_rmse,
                "cv_r2": cv_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "best_params": best_params_json
            })

            best_estimators[model_name] = search.best_estimator_

            if self.save_models:
                joblib.dump(
                    search.best_estimator_,
                    models_dir / f"best_estimator_{model_name}.joblib"
                )

            print(f"{model_name} fertig.")
            print(f"Test MAE:  {test_mae:.2f}")
            print(f"Test RMSE: {test_rmse:.2f}")
            print(f"Test R²:   {test_r2:.4f}")

        results = pd.DataFrame(result_rows).sort_values("test_mae")

        results.to_csv(
            tables_dir / "model_comparison_results.csv",
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding,
            errors="ignore"
        )

        best_row = results.iloc[0]
        best_model_name = best_row["model"]
        best_estimator = best_estimators[best_model_name]

        best_summary = {
            "run": run_name,
            "best_model": best_model_name,
            "best_result": best_row.to_dict()
        }

        self.save_json(best_summary, tables_dir / "best_model_summary.json")

        joblib.dump(
            best_estimator,
            models_dir / "best_overall_estimator.joblib"
        )

        print("\nBest model for this run:", best_model_name)
        print(best_row)

        self.create_visualizations(
            results=results,
            best_estimator=best_estimator,
            best_model_name=best_model_name,
            X_test=X_test,
            y_test=y_test,
            id_test=id_test,
            tables_dir=tables_dir,
            plots_dir=plots_dir
        )

        self.run_results.append(best_row.to_dict())

        return results

    
    # 07: Visualization
    

    def create_visualizations(
        self,
        results,
        best_estimator,
        best_model_name,
        X_test,
        y_test,
        id_test,
        tables_dir,
        plots_dir
    ):
        # Model comparison MAE
        df_mae = results.sort_values("test_mae")

        plt.figure(figsize=(10, 6))
        plt.barh(df_mae["model"][::-1], df_mae["test_mae"][::-1])
        plt.xlabel("Test MAE in £")
        plt.ylabel("Model")
        plt.title("Model Comparison: MAE")
        plt.tight_layout()
        plt.savefig(plots_dir / "01_model_comparison_mae.png", dpi=150)
        plt.close()

        # Model Comparison RMSE
        df_rmse = results.sort_values("test_rmse")

        plt.figure(figsize=(10, 6))
        plt.barh(df_rmse["model"][::-1], df_rmse["test_rmse"][::-1])
        plt.xlabel("Test RMSE in £")
        plt.ylabel("Model")
        plt.title("Model Comparison: RMSE")
        plt.tight_layout()
        plt.savefig(plots_dir / "02_model_comparison_rmse.png", dpi=150)
        plt.close()

        # Model Comparison R²
        df_r2 = results.sort_values("test_r2", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(df_r2["model"][::-1], df_r2["test_r2"][::-1])
        plt.xlabel("Test R²")
        plt.ylabel("Model")
        plt.title("Model Comparison: R²")
        plt.tight_layout()
        plt.savefig(plots_dir / "03_model_comparison_r2.png", dpi=150)
        plt.close()

        # Best Model Predictions
        y_pred = best_estimator.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        best_predictions = pd.DataFrame({
            "id": id_test.values,
            "actual_price": y_test.values,
            "predicted_price": y_pred,
            "residual": y_test.values - y_pred,
            "absolute_error": np.abs(y_test.values - y_pred)
        })

        best_predictions.to_csv(
            tables_dir / "best_model_predictions.csv",
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding,
            errors="ignore"
        )

        # Actual vs Predicted
        plt.figure(figsize=(7, 7))
        plt.scatter(
            best_predictions["actual_price"],
            best_predictions["predicted_price"],
            alpha=0.25,
            s=10
        )

        max_val = max(
            best_predictions["actual_price"].max(),
            best_predictions["predicted_price"].max()
        )

        plt.plot([0, max_val], [0, max_val], linestyle="--")
        plt.xlabel("Actual Price in £")
        plt.ylabel("Predicted Price in £")
        plt.title(f"Actual vs. Predicted: {best_model_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / "04_actual_vs_predicted_best_model.png", dpi=150)
        plt.close()

        # Residuals
        plt.figure(figsize=(10, 6))
        plt.hist(best_predictions["residual"], bins=80)
        plt.xlabel("Residual: actual - predicted")
        plt.ylabel("Number of Listings")
        plt.title(f"Residual Distribution: {best_model_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / "05_residual_distribution_best_model.png", dpi=150)
        plt.close()

        # Errors by price range
        bins = [0, 50, 100, 150, 200, 300, 500, 1000, np.inf]
        labels = ["0-50", "50-100", "100-150", "150-200", "200-300", "300-500", "500-1000", "1000+"]

        best_predictions["price_bucket"] = pd.cut(
            best_predictions["actual_price"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        bucket_error = (
            best_predictions
            .groupby("price_bucket", observed=False)
            .agg(
                count=("absolute_error", "count"),
                mae=("absolute_error", "mean"),
                median_error=("residual", "median"),
                mean_actual_price=("actual_price", "mean"),
                mean_predicted_price=("predicted_price", "mean")
            )
            .reset_index()
        )

        bucket_error.to_csv(
            tables_dir / "error_by_price_bucket.csv",
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding,
            errors="ignore"
        )

        plt.figure(figsize=(10, 6))
        plt.bar(bucket_error["price_bucket"].astype(str), bucket_error["mae"])
        plt.xlabel("Actual price bucket in £")
        plt.ylabel("MAE in £")
        plt.title("Prediction Error by Price Bucket")
        plt.tight_layout()
        plt.savefig(plots_dir / "06_error_by_price_bucket.png", dpi=150)
        plt.close()

        # Feature Importance
        self.save_feature_importance(
            best_estimator=best_estimator,
            best_model_name=best_model_name,
            X_test=X_test,
            y_test=y_test,
            tables_dir=tables_dir,
            plots_dir=plots_dir
        )

    
    # 08: Feature Importance
    

    def get_feature_names_from_pipeline(self, pipeline):
        preprocessor = pipeline.named_steps["preprocessor"]

        try:
            names = preprocessor.get_feature_names_out()
            return list(names)
        except Exception:
            return [f"feature_{i}" for i in range(len(pipeline.named_steps["model"].feature_importances_))]

    def save_feature_importance(
        self,
        best_estimator,
        best_model_name,
        X_test,
        y_test,
        tables_dir,
        plots_dir
    ):
        model = best_estimator.named_steps["model"]
        feature_names = self.get_feature_names_from_pipeline(best_estimator)

        # Model-native feature importance if available
        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_

            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance_values,
                "importance_type": "model_feature_importance"
            }).sort_values("importance", ascending=False)

            importance_df.to_csv(
                tables_dir / "best_model_feature_importance.csv",
                sep=self.sep,
                decimal=self.decimal,
                index=False,
                encoding=self.encoding,
                errors="ignore"
            )

            top = importance_df.head(30).sort_values("importance")

            plt.figure(figsize=(10, 9))
            plt.barh(top["feature"], top["importance"])
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")
            plt.title(f"Top 30 Feature Importances: {best_model_name}")
            plt.tight_layout()
            plt.savefig(plots_dir / "07_best_model_feature_importance.png", dpi=150)
            plt.close()

        elif hasattr(model, "coef_"):
            coef = np.ravel(model.coef_)

            coef_df = pd.DataFrame({
                "feature": feature_names,
                "coefficient": coef,
                "absolute_coefficient": np.abs(coef),
                "importance_type": "coefficient"
            }).sort_values("absolute_coefficient", ascending=False)

            coef_df.to_csv(
                tables_dir / "best_model_coefficients.csv",
                sep=self.sep,
                decimal=self.decimal,
                index=False,
                encoding=self.encoding,
                errors="ignore"
            )

            top = coef_df.head(30).sort_values("absolute_coefficient")

            plt.figure(figsize=(10, 9))
            plt.barh(top["feature"], top["coefficient"])
            plt.xlabel("Coefficient")
            plt.ylabel("Feature")
            plt.title(f"Top 30 Coefficients: {best_model_name}")
            plt.tight_layout()
            plt.savefig(plots_dir / "07_best_model_coefficients.png", dpi=150)
            plt.close()

        # Additionally, permutation importance on the test sample.
        if self.permutation_importance_sample_size is not None and len(X_test) > self.permutation_importance_sample_size:
            X_perm = X_test.sample(
                n=self.permutation_importance_sample_size,
                random_state=self.random_state
            )
            y_perm = y_test.loc[X_perm.index]
        else:
            X_perm = X_test
            y_perm = y_test

        try:
            perm = permutation_importance(
                best_estimator,
                X_perm,
                y_perm,
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=self.random_state,
                n_jobs=-1
            )

            perm_df = pd.DataFrame({
                "feature": X_perm.columns,
                "permutation_importance_mae_increase": perm.importances_mean,
                "permutation_importance_std": perm.importances_std
            }).sort_values("permutation_importance_mae_increase", ascending=False)

            perm_df.to_csv(
                tables_dir / "best_model_permutation_importance.csv",
                sep=self.sep,
                decimal=self.decimal,
                index=False,
                encoding=self.encoding,
                errors="ignore"
            )

            top_perm = perm_df.head(30).sort_values("permutation_importance_mae_increase")

            plt.figure(figsize=(10, 9))
            plt.barh(
                top_perm["feature"],
                top_perm["permutation_importance_mae_increase"]
            )
            plt.xlabel("MAE increase after permutation")
            plt.ylabel("Feature")
            plt.title(f"Permutation Importance: {best_model_name}")
            plt.tight_layout()
            plt.savefig(plots_dir / "08_best_model_permutation_importance.png", dpi=150)
            plt.close()

        except Exception as e:
            print("Could not compute permutation importance:", e)

    
    # 09: Overall comparison between Run 1 and Run 2
    

    def create_overall_summary(self):
        if not self.run_results:
            return None

        summary = pd.DataFrame(self.run_results).sort_values("test_mae")

        summary.to_csv(
            self.output_base_dir / "overall_best_models_summary.csv",
            sep=self.sep,
            decimal=self.decimal,
            index=False,
            encoding=self.encoding,
            errors="ignore"
        )

        plt.figure(figsize=(10, 6))
        plt.barh(summary["run"][::-1], summary["test_mae"][::-1])
        plt.xlabel("Best model Test MAE in £")
        plt.ylabel("Run")
        plt.title("Best Model Comparison: Run 1 vs Run 2")
        plt.tight_layout()
        plt.savefig(self.output_base_dir / "overall_run_comparison_mae.png", dpi=150)
        plt.close()

        self.save_json(
            {
                "best_overall": summary.iloc[0].to_dict(),
                "all_runs": summary.to_dict(orient="records")
            },
            self.output_base_dir / "overall_best_model_summary.json"
        )

        return summary


    # 10: Execute all


    def run_all(self):
        config = {
            "predictor_file": str(self.predictor_file),
            "amenities_file": str(self.amenities_file),
            "category_dir": str(self.category_dir),
            "output_base_dir": str(self.output_base_dir),
            "test_size": self.test_size,
            "random_state": self.random_state,
            "cv": self.cv,
            "search_type": self.search_type,
            "n_iter": self.n_iter,
            "sample_size": self.sample_size,
            "price_upper_quantile": self.price_upper_quantile,
            "use_log_price_feature": self.use_log_price_feature
        }

        self.save_json(config, self.output_base_dir / "run_config.json")

        df_run1, run1_name = self.build_dataset_run1()
        self.train_and_evaluate_run(df_run1, run1_name)

        df_run2, run2_name = self.build_dataset_run2()
        self.train_and_evaluate_run(df_run2, run2_name)

        overall_summary = self.create_overall_summary()

        print("\n" + "=" * 80)
        print("COMPLETED")
        print("=" * 80)
        print("Results saved in:")
        print(self.output_base_dir)

        if overall_summary is not None:
            print("\nOverall Summary:")
            print(overall_summary[["run", "model", "test_mae", "test_rmse", "test_r2"]])

        return overall_summary


if __name__ == "__main__":
    # Run from the repo root: python scripts/08_compare_prediction_models.py
    runner = AirbnbPredictionModelRunner(
        predictor_file=BASE_DIR / "data" / "processed" / "predictor_dataset_with_id.csv",
        amenities_file=BASE_DIR / "data" / "processed" / "amenities_reduced_grouped.csv",
        category_dir=BASE_DIR / "data" / "processed" / "category_csvs",
        output_base_dir=BASE_DIR / "outputs" / "model_results",

        
        sample_size=None,

        search_type="random",
        n_iter=15,
        cv=3,
        test_size=0.2,
        random_state=42,
        price_upper_quantile=0.99,
        save_models=True
    )

    runner.run_all()
