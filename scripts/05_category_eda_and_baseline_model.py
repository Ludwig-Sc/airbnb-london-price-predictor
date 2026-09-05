import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Run from the repo root: python scripts/05_category_eda_and_baseline_model.py
DATA_PATH = "data/processed/category_csvs/"
FIGURE_PATH = "outputs/eda_figures/"


# Helpers — define BEFORE use

def to_numeric_eu(series):
    """Convert European decimal strings (1,5 → 1.5) to numeric."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(',', '.', regex=False)
        .pipe(pd.to_numeric, errors='coerce')
    )

def map_boolean(series):
    """Handle t/f, true/false, 1/0, 1,0/0,0 (European decimal booleans)."""
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map({'t': 1, 'f': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})
    mask = mapped.isna() & (s != 'nan')
    if mask.any():
        numeric = s[mask].str.replace(',', '.', regex=False).pipe(pd.to_numeric, errors='coerce')
        mapped[mask] = (numeric >= 0.5).astype(float)
        mapped[mask & numeric.isna()] = np.nan
    return mapped


# Loading CSVs

target      = pd.read_csv(f"{DATA_PATH}01_target_price.csv", sep=';')
listing     = pd.read_csv(f"{DATA_PATH}02_listing_basic.csv", sep=';', quotechar='"', escapechar='\\', on_bad_lines='skip', engine='python')
location    = pd.read_csv(f"{DATA_PATH}03_location.csv", sep=';')
host        = pd.read_csv(f"{DATA_PATH}04_host.csv", sep=';', quotechar='"', escapechar='\\', on_bad_lines='skip', engine='python')
booking     = pd.read_csv(f"{DATA_PATH}05_booking_rules_availability.csv", sep=';')
reviews     = pd.read_csv(f"{DATA_PATH}06_reviews_ratings.csv", sep=';')
host_counts = pd.read_csv(f"{DATA_PATH}07_host_calculated_counts.csv", sep=';')
leakage     = pd.read_csv(f"{DATA_PATH}09_possible_leakage_or_optional.csv", sep=';')

# Normalise all id columns to Int64 before any merges
for frame in [target, listing, location, host, booking, reviews, host_counts, leakage]:
    if 'id' in frame.columns:
        frame['id'] = pd.to_numeric(frame['id'].astype(str).str.strip(), errors='coerce').astype('Int64')


# Cleaning


# 01: Target 
target['price'] = to_numeric_eu(target['price'].astype(str).str.replace(r'[\$£]', '', regex=True))
target = target[(target['price'] >= 10) & (target['price'] <= 5000)].copy()
target['log_price'] = np.log1p(target['price'])

# 02: Listing Basic 
# Convert ALL numeric columns using to_numeric_eu (handles European decimals)
for col in ['bedrooms', 'beds', 'accommodates', 'bathrooms']:
    if col in listing.columns:
        listing[col] = to_numeric_eu(listing[col])

listing['bathrooms_numeric'] = listing['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
listing['bedrooms']          = listing['bedrooms'].fillna(listing['bedrooms'].median())
listing['beds']              = listing['beds'].fillna(listing['beds'].median())
listing['bathrooms_numeric'] = listing['bathrooms_numeric'].fillna(listing['bathrooms_numeric'].median())

def count_amenities(a):
    try: return len(ast.literal_eval(a))
    except: return 0

listing['amenity_count'] = listing['amenities'].apply(count_amenities)
high_value = ['parking', 'gym', 'pool', 'dishwasher', 'washer', 'dryer', 'elevator']
for amenity in high_value:
    listing[f'has_{amenity}'] = listing['amenities'].str.lower().str.contains(amenity, na=False).astype(int)

listing['room_type_enc'] = LabelEncoder().fit_transform(listing['room_type'].fillna('Unknown'))

# 03: Location — drop neighbourhood_enc, will be created after merge 
# (neighbourhood_group_cleansed is missing from this dataset)

# 04: Host 
for col in ['host_response_rate', 'host_acceptance_rate']:
    host[col] = (
        host[col].astype(str)
        .str.replace(r'%', '', regex=True)
        .str.replace(',', '.', regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
        / 100
    )
for col in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']:
    if col in host.columns:
        host[col] = map_boolean(host[col])

host['host_since'] = pd.to_datetime(host['host_since'], errors='coerce')
host['host_tenure_years'] = (pd.Timestamp('today') - host['host_since']).dt.days / 365
host['host_response_rate']   = host['host_response_rate'].fillna(host['host_response_rate'].median())
host['host_acceptance_rate'] = host['host_acceptance_rate'].fillna(host['host_acceptance_rate'].median())
host['host_tenure_years']    = host['host_tenure_years'].fillna(host['host_tenure_years'].median())

# 05: Booking 
booking['instant_bookable'] = map_boolean(booking['instant_bookable'])
booking['has_availability']  = map_boolean(booking['has_availability'])
for col in ['availability_30', 'availability_60', 'availability_90', 'availability_365',
            'minimum_nights', 'maximum_nights']:
    if col in booking.columns:
        booking[col] = to_numeric_eu(booking[col])

# 06: Reviews
review_numeric_cols = [
    'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
    'number_of_reviews_ly', 'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location',
    'review_scores_value', 'reviews_per_month'
]
for col in review_numeric_cols:
    if col in reviews.columns:
        reviews[col] = to_numeric_eu(reviews[col])

for col in ['first_review', 'last_review']:
    reviews[col] = pd.to_datetime(reviews[col], errors='coerce')

reviews['days_since_last_review']  = (pd.Timestamp('today') - reviews['last_review']).dt.days
reviews['days_since_first_review'] = (pd.Timestamp('today') - reviews['first_review']).dt.days

for col in ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
            'review_scores_checkin', 'review_scores_communication',
            'review_scores_location', 'review_scores_value']:
    reviews[col] = reviews[col].fillna(reviews[col].median())

reviews['reviews_per_month'] = reviews['reviews_per_month'].fillna(0)
reviews['has_reviews'] = (reviews['number_of_reviews'] > 0).astype(int)

# 07: Host counts
for col in host_counts.columns:
    if col != 'id':
        host_counts[col] = to_numeric_eu(host_counts[col])

# 09: Leakage (inspect only)
print("Leakage preview:")
print(leakage[['id', 'estimated_occupancy_l365d']].describe())


# Merge

df = (
    target
    .merge(listing,     on='id', how='left')
    .merge(location,    on='id', how='left')
    .merge(host,        on='id', how='left')
    .merge(booking,     on='id', how='left')
    .merge(reviews,     on='id', how='left')
    .merge(host_counts, on='id', how='left')
)

print(f"Merged shape: {df.shape}")
df = df.dropna(subset=['log_price'])

# Re-apply boolean fix post-merge (safety net)
for col in ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')


# Post-merge encoding (must happen after merge so value_counts sees full data)

top_props = df['property_type'].value_counts().nlargest(8).index
df['property_type_clean'] = df['property_type'].where(df['property_type'].isin(top_props), 'Other')

top_neighbourhoods = df['neighbourhood_cleansed'].value_counts().nlargest(30).index
df['neighbourhood_enc'] = df['neighbourhood_cleansed'].where(
    df['neighbourhood_cleansed'].isin(top_neighbourhoods), 'Other'
)

df = pd.get_dummies(df, columns=['property_type_clean', 'neighbourhood_enc'], drop_first=True)

# Sanity check
print(df[['bedrooms', 'beds', 'review_scores_rating', 'host_is_superhost', 'reviews_per_month']].describe())


# EDA


# Price distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['price'], bins=100, color='steelblue', edgecolor='white')
axes[0].set_title('Price Distribution (Raw £)')
axes[1].hist(df['log_price'], bins=100, color='darkorange', edgecolor='white')
axes[1].set_title('Price Distribution (Log-transformed)')
plt.tight_layout(); plt.savefig(FIGURE_PATH + 'price_distribution.png', dpi=150); plt.show()

# Median price by neighbourhood (top 20)
top20 = df.groupby('neighbourhood_cleansed')['price'].median().nlargest(20)
plt.figure(figsize=(14, 6))
top20.sort_values().plot(kind='barh', color='teal')
plt.title('Top 20 Neighbourhoods by Median Price')
plt.xlabel('Median Price (£)')
plt.tight_layout(); plt.savefig(FIGURE_PATH + 'neighbourhood_prices.png', dpi=150); plt.show()

# Price by room type
plt.figure(figsize=(8, 5))
df.groupby('room_type')['price'].median().sort_values().plot(kind='barh', color='salmon')
plt.title('Median Price by Room Type')
plt.xlabel('Median Price (£)')
plt.tight_layout(); plt.savefig(FIGURE_PATH + 'room_type_prices.png', dpi=150); plt.show()

# Correlation heatmap
corr_features = [
    'log_price', 'accommodates', 'bedrooms', 'beds', 'bathrooms_numeric',
    'amenity_count', 'review_scores_rating', 'review_scores_cleanliness',
    'review_scores_location', 'review_scores_value', 'host_tenure_years',
    'host_is_superhost', 'availability_365', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count', 'instant_bookable'
]
corr_features = [c for c in corr_features if c in df.columns]
plt.figure(figsize=(14, 10))
sns.heatmap(df[corr_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix – Key Features vs log(Price)')
plt.tight_layout(); plt.savefig(FIGURE_PATH + 'correlation_heatmap.png', dpi=150); plt.show()

# Superhost price premium
superhost_price = df.groupby('host_is_superhost')['price'].median().dropna()
if len(superhost_price) > 0:
    plt.figure(figsize=(7, 5))
    superhost_price.plot(kind='bar', color=['steelblue', 'gold'], rot=0)
    plt.xticks(range(len(superhost_price)),
               ['Regular Host' if x == 0 else 'Superhost' for x in superhost_price.index])
    plt.title('Median Price: Superhost vs Regular')
    plt.ylabel('Median Price (£)')
    plt.tight_layout(); plt.savefig(FIGURE_PATH + 'superhost_premium.png', dpi=150); plt.show()


# Modelling

feature_cols = (
    ['accommodates', 'bedrooms', 'beds', 'bathrooms_numeric', 'amenity_count',
     'host_tenure_years', 'host_is_superhost', 'host_identity_verified',
     'host_response_rate', 'host_acceptance_rate',
     'review_scores_rating', 'review_scores_cleanliness',
     'review_scores_location', 'review_scores_value',
     'reviews_per_month', 'days_since_last_review', 'has_reviews',
     'availability_365', 'minimum_nights', 'instant_bookable',
     'calculated_host_listings_count',
     'calculated_host_listings_count_entire_homes',
     'room_type_enc']
    + [c for c in df.columns if c.startswith('property_type_clean_')]
    + [c for c in df.columns if c.startswith('neighbourhood_enc_')]
    + [f'has_{a}' for a in high_value]
)
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].fillna(0)
y = df['log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=10)':  Ridge(alpha=10),
    'Lasso (alpha=0.01)': Lasso(alpha=0.01)
}
for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
    print(f"{name}: R²={r2:.3f} | RMSE=£{rmse:.2f}")

# Ridge coefficient plot
ridge_pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=10))])
ridge_pipe.fit(X_train, y_train)
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': ridge_pipe.named_steps['model'].coef_
}).sort_values('coefficient', key=abs, ascending=False).head(20)

plt.figure(figsize=(10, 7))
plt.barh(coef_df['feature'], coef_df['coefficient'], color='steelblue')
plt.title('Top 20 Ridge Coefficients')
plt.xlabel('Coefficient (log price scale)')
plt.gca().invert_yaxis()
plt.tight_layout(); plt.savefig(FIGURE_PATH + 'ridge_coefficients.png', dpi=150); plt.show()
