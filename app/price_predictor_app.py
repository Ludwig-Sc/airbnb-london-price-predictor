import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class AirbnbPricePredictionApp:
    def __init__(
        self,
        model_path,
        model_dataset_path,
        predictions_path=None,
        sep=";",
        decimal=",",
        encoding="utf-8-sig"
    ):
        self.model_path = Path(model_path)
        self.model_dataset_path = Path(model_dataset_path)
        self.predictions_path = Path(predictions_path) if predictions_path else None

        self.sep = sep
        self.decimal = decimal
        self.encoding = encoding

        self.model = None
        self.reference_df = None
        self.predictions_df = None
        self.feature_columns = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modell nicht gefunden: {self.model_path}")

        if not self.model_dataset_path.exists():
            raise FileNotFoundError(f"Modelldatensatz nicht gefunden: {self.model_dataset_path}")

        self.model = joblib.load(self.model_path)

        self.reference_df = pd.read_csv(
            self.model_dataset_path,
            sep=self.sep,
            decimal=self.decimal,
            encoding=self.encoding,
            low_memory=False
        )

        self.reference_df.columns = (
            self.reference_df.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        self.feature_columns = [
            col for col in self.reference_df.columns
            if col not in ["id", "price"]
        ]

        # Load test predictions to calculate the forecast interval
        if self.predictions_path and self.predictions_path.exists():
            self.predictions_df = pd.read_csv(
                self.predictions_path,
                sep=self.sep,
                decimal=self.decimal,
                encoding=self.encoding,
                low_memory=False
            )

            self.predictions_df.columns = (
                self.predictions_df.columns
                .astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.strip()
            )

            needed_cols = ["actual_price", "predicted_price"]
            missing = [col for col in needed_cols if col not in self.predictions_df.columns]

            if missing:
                st.warning(f"Prediction-Interval nicht verfügbar. Fehlende Spalten: {missing}")
                self.predictions_df = None
            else:
                self.predictions_df["actual_price"] = pd.to_numeric(
                    self.predictions_df["actual_price"],
                    errors="coerce"
                )

                self.predictions_df["predicted_price"] = pd.to_numeric(
                    self.predictions_df["predicted_price"],
                    errors="coerce"
                )

                self.predictions_df = self.predictions_df.dropna(
                    subset=["actual_price", "predicted_price"]
                )

                self.predictions_df["residual"] = (
                    self.predictions_df["actual_price"]
                    - self.predictions_df["predicted_price"]
                )

        return self

    def get_default_value(self, col):
        if col not in self.reference_df.columns:
            return 0

        series = self.reference_df[col]

        if pd.api.types.is_numeric_dtype(series):
            value = series.median()
            if pd.isna(value):
                return 0
            return float(value)

        mode = series.dropna().mode()
        if len(mode) > 0:
            return str(mode.iloc[0])

        return ""

    def create_default_input_row(self):
        row = {}

        for col in self.feature_columns:
            row[col] = self.get_default_value(col)

        return row

    def predict(self, input_data):
        input_df = pd.DataFrame([input_data])
        input_df = input_df[self.feature_columns]

        prediction = self.model.predict(input_df)[0]
        prediction = max(0, prediction)

        return prediction

    def predict_with_interval(self, input_data, interval=0.90, nearest_n=1000):
        """
        Creates a local prediction interval based on similar test predictions
        
        """

        prediction = self.predict(input_data)

        if self.predictions_df is None or len(self.predictions_df) < 50:
            return {
                "prediction": prediction,
                "lower": None,
                "upper": None,
                "interval": interval,
                "method": "no_interval_available"
            }

        df = self.predictions_df.copy()

        # Search for similar forecasts
        df["prediction_distance"] = (
            df["predicted_price"] - prediction
        ).abs()

        local_errors = (
            df
            .sort_values("prediction_distance")
            .head(min(nearest_n, len(df)))
            ["residual"]
        )

        alpha = 1 - interval
        lower_error = local_errors.quantile(alpha / 2)
        upper_error = local_errors.quantile(1 - alpha / 2)

        lower = max(0, prediction + lower_error)
        upper = max(0, prediction + upper_error)

        return {
            "prediction": prediction,
            "lower": lower,
            "upper": upper,
            "interval": interval,
            "method": f"local_residual_interval_nearest_{nearest_n}"
        }



# Streamlit UI


st.set_page_config(
    page_title="London Airbnb Price Predictor",
    layout="wide"
)

st.title("London Airbnb Price Predictor")
st.write(
    "Gib Eigenschaften eines Airbnb ein. "
    "Das Modell schätzt daraus den erwarteten Preis pro Nacht."
)



# Points at the final run (run2: predictor + amenities + category features,
# HistGradientBoosting) — see outputs/model_results/final_run/ for the full
# comparison this was picked from.
MODEL_PATH = BASE_DIR / "models" / "best_overall_estimator.joblib"
MODEL_DATASET_PATH = BASE_DIR / "data" / "processed" / "model_dataset_used_run2.csv"
PREDICTIONS_PATH = BASE_DIR / "outputs" / "model_results" / "final_run" / "tables" / "best_model_predictions.csv"

app = AirbnbPricePredictionApp(
    model_path=MODEL_PATH,
    model_dataset_path=MODEL_DATASET_PATH,
    predictions_path=PREDICTIONS_PATH
).load()

default_row = app.create_default_input_row()



# Input form


st.sidebar.header("Listing-Eigenschaften")

input_data = default_row.copy()



# Main Features


st.sidebar.subheader("Grunddaten")

if "bedrooms" in app.feature_columns:
    input_data["bedrooms"] = st.sidebar.number_input(
        "Bedrooms",
        min_value=0.0,
        max_value=20.0,
        value=float(default_row.get("bedrooms", 1)),
        step=1.0
    )

if "is_studio" in app.feature_columns:
    input_data["is_studio"] = st.sidebar.selectbox(
        "Studio?",
        options=[0, 1],
        index=int(default_row.get("is_studio", 0)) if default_row.get("is_studio", 0) in [0, 1] else 0
    )

if "bedrooms_log" in app.feature_columns:
    input_data["bedrooms_log"] = np.log1p(input_data.get("bedrooms", 0))

if "accommodates" in app.feature_columns:
    input_data["accommodates"] = st.sidebar.number_input(
        "Accommodates",
        min_value=1.0,
        max_value=30.0,
        value=float(default_row.get("accommodates", 2)),
        step=1.0
    )

if "beds" in app.feature_columns:
    input_data["beds"] = st.sidebar.number_input(
        "Beds",
        min_value=0.0,
        max_value=30.0,
        value=float(default_row.get("beds", 1)),
        step=1.0
    )

if "bathrooms" in app.feature_columns:
    input_data["bathrooms"] = st.sidebar.number_input(
        "Bathrooms",
        min_value=0.0,
        max_value=20.0,
        value=float(default_row.get("bathrooms", 1)),
        step=0.5
    )



# Categories


# Categories / Location


st.sidebar.subheader("Typ und Lage")

if "room_type" in app.feature_columns:
    room_types = sorted(app.reference_df["room_type"].dropna().astype(str).unique())
    default_room = str(default_row.get("room_type", room_types[0] if room_types else ""))

    input_data["room_type"] = st.sidebar.selectbox(
        "Room Type",
        options=room_types,
        index=room_types.index(default_room) if default_room in room_types else 0
    )

if "property_type" in app.feature_columns:
    property_types = sorted(app.reference_df["property_type"].dropna().astype(str).unique())
    default_property = str(default_row.get("property_type", property_types[0] if property_types else ""))

    input_data["property_type"] = st.sidebar.selectbox(
        "Property Type",
        options=property_types,
        index=property_types.index(default_property) if default_property in property_types else 0
    )



# Neighbourhood


if (
    "neighbourhood_cleansed" in app.feature_columns
    and "latitude" in app.feature_columns
    and "longitude" in app.feature_columns
):
    geo_df = app.reference_df[
        ["neighbourhood_cleansed", "latitude", "longitude"]
    ].copy()

    geo_df["latitude"] = pd.to_numeric(geo_df["latitude"], errors="coerce")
    geo_df["longitude"] = pd.to_numeric(geo_df["longitude"], errors="coerce")

    geo_df = geo_df.dropna(subset=["neighbourhood_cleansed", "latitude", "longitude"])

    neighbourhood_geo = (
        geo_df
        .groupby("neighbourhood_cleansed", as_index=False)
        .agg(
            latitude_median=("latitude", "median"),
            longitude_median=("longitude", "median"),
            listings_count=("latitude", "count")
        )
        .sort_values("neighbourhood_cleansed")
    )

    neighbourhoods = neighbourhood_geo["neighbourhood_cleansed"].astype(str).tolist()

    default_neighbourhood = str(
        default_row.get(
            "neighbourhood_cleansed",
            neighbourhoods[0] if neighbourhoods else ""
        )
    )

    selected_neighbourhood = st.sidebar.selectbox(
        "Neighbourhood",
        options=neighbourhoods,
        index=neighbourhoods.index(default_neighbourhood)
        if default_neighbourhood in neighbourhoods
        else 0
    )

    input_data["neighbourhood_cleansed"] = selected_neighbourhood

    selected_geo = neighbourhood_geo[
        neighbourhood_geo["neighbourhood_cleansed"].astype(str) == selected_neighbourhood
    ].iloc[0]

    auto_latitude = float(selected_geo["latitude_median"])
    auto_longitude = float(selected_geo["longitude_median"])
    listings_count = int(selected_geo["listings_count"])

    st.sidebar.caption(
        f"Koordinaten automatisch gesetzt auf den Mittelpunkt von "
        f"{selected_neighbourhood} basierend auf {listings_count} Listings."
    )

    manual_coordinates = st.sidebar.checkbox(
        "Koordinaten manuell überschreiben",
        value=False
    )

    if manual_coordinates:
        input_data["latitude"] = st.sidebar.number_input(
            "Latitude",
            min_value=51.0,
            max_value=52.0,
            value=auto_latitude,
            step=0.001,
            format="%.6f"
        )

        input_data["longitude"] = st.sidebar.number_input(
            "Longitude",
            min_value=-1.0,
            max_value=1.0,
            value=auto_longitude,
            step=0.001,
            format="%.6f"
        )

        st.sidebar.warning(
            "Achtung: Manuelle Koordinaten können der ausgewählten Neighbourhood widersprechen."
        )

    else:
        input_data["latitude"] = auto_latitude
        input_data["longitude"] = auto_longitude

        st.sidebar.number_input(
            "Latitude automatisch",
            value=auto_latitude,
            format="%.6f",
            disabled=True
        )

        st.sidebar.number_input(
            "Longitude automatisch",
            value=auto_longitude,
            format="%.6f",
            disabled=True
        )


# If the model uses the neighborhood but not coordinates
elif "neighbourhood_cleansed" in app.feature_columns:
    neighbourhoods = sorted(
        app.reference_df["neighbourhood_cleansed"]
        .dropna()
        .astype(str)
        .unique()
    )

    default_neighbourhood = str(
        default_row.get(
            "neighbourhood_cleansed",
            neighbourhoods[0] if neighbourhoods else ""
        )
    )

    input_data["neighbourhood_cleansed"] = st.sidebar.selectbox(
        "Neighbourhood",
        options=neighbourhoods,
        index=neighbourhoods.index(default_neighbourhood)
        if default_neighbourhood in neighbourhoods
        else 0
    )



# Reviews


st.sidebar.subheader("Reviews")

if "review_scores_rating" in app.feature_columns:
    input_data["review_scores_rating"] = st.sidebar.slider(
        "Review Score Rating",
        min_value=0.0,
        max_value=5.0,
        value=float(default_row.get("review_scores_rating", 4.5)),
        step=0.1
    )

if "number_of_reviews" in app.feature_columns:
    input_data["number_of_reviews"] = st.sidebar.number_input(
        "Number of Reviews",
        min_value=0.0,
        max_value=2000.0,
        value=float(default_row.get("number_of_reviews", 10)),
        step=1.0
    )

if "number_of_reviews_log" in app.feature_columns:
    input_data["number_of_reviews_log"] = np.log1p(input_data.get("number_of_reviews", 0))

if "has_reviews" in app.feature_columns:
    input_data["has_reviews"] = int(input_data.get("number_of_reviews", 0) > 0)



# Host Features


st.sidebar.subheader("Host")

host_numeric_features = [
    "host_response_rate",
    "host_acceptance_rate",
    "host_listings_count",
    "host_total_listings_count",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms"
]

for col in host_numeric_features:
    if col in app.feature_columns:
        max_value = float(app.reference_df[col].quantile(0.99)) if col in app.reference_df.columns else 100.0
        default_value = float(default_row.get(col, 0))

        input_data[col] = st.sidebar.number_input(
            col,
            min_value=0.0,
            max_value=max(1.0, max_value),
            value=min(default_value, max(1.0, max_value)),
            step=1.0 if "count" in col else 0.01
        )

host_binary_features = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified"
]

for col in host_binary_features:
    if col in app.feature_columns:
        input_data[col] = int(
            st.sidebar.checkbox(
                col,
                value=bool(default_row.get(col, 0))
            )
        )



# Amenities


st.sidebar.subheader("Amenities")

amenity_features = [
    col for col in app.feature_columns
    if col.startswith("has_")
    and col not in ["has_reviews"]
]

for col in amenity_features:
    input_data[col] = int(
        st.sidebar.checkbox(
            col.replace("has_", "").replace("_", " ").title(),
            value=bool(default_row.get(col, 0))
        )
    )



# Prediction


prediction_result = app.predict_with_interval(
    input_data=input_data,
    interval=0.90,
    nearest_n=1000
)

prediction = prediction_result["prediction"]
lower = prediction_result["lower"]
upper = prediction_result["upper"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Geschätzter Preis pro Nacht", f"£{prediction:,.2f}")

with col2:
    if lower is not None and upper is not None:
        st.metric("90%-Prognoseintervall", f"£{lower:,.0f} – £{upper:,.0f}")
    else:
        st.metric("90%-Prognoseintervall", "Nicht verfügbar")

with col3:
    st.metric("Modell", "London Airbnb Price Estimator")
if lower is not None and upper is not None:
    st.info(
        f"Das Modell schätzt einen Preis von **£{prediction:,.2f}**. "
        f"Basierend auf ähnlichen Testfällen liegt ein realistischer Bereich "
        f"ungefähr zwischen **£{lower:,.0f} und £{upper:,.0f}**."
    )
else:
    st.info(
        f"Das Modell schätzt einen Preis von **£{prediction:,.2f}**. "
        "Ein Prognoseintervall konnte nicht berechnet werden, weil keine Test-Predictions geladen wurden."
    )








with st.expander("Verwendete Eingabedaten anzeigen"):
    input_df_display = pd.DataFrame([input_data]).T
    input_df_display.columns = ["value"]
    st.dataframe(input_df_display)
