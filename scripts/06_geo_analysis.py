import pandas as pd
import numpy as np

# Run from the repo root: python scripts/06_geo_analysis.py
DATA_PATH = "data/processed/category_csvs/"
MAP_PATH = "outputs/geo_maps/"
FIGURE_PATH = "outputs/eda_figures/"
OUTPUT_PATH = "data/processed/"

# 01: Load
df_price = pd.read_csv(f'{DATA_PATH}01_target_price.csv', sep=';')
df_geo = pd.read_csv(f'{DATA_PATH}03_location.csv', sep=';')

# 02: Merge via ID
df = df_geo.merge(df_price, on='id')

# 03: Convert decimal comma to decimal point
df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
df['price'] = df['price'].astype(str).str.replace(',', '.').astype(float)

print(df['price'].isna().sum())

print(df[df['price'].isna()]['neighbourhood_cleansed'].value_counts().head(10))


df = df.dropna(subset=['price'])

df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
df['price'] = df['price'].astype(str).str.replace(',', '.').astype(float)

# 04: Check results
print(df.shape)
print(df[['latitude', 'longitude', 'price']].head())
print(df['price'].describe())

# 05: Check price outliers
print(df[df['price'] > 10000].shape[0], "Listings über 10.000")
print(df[df['price'] > 1000].shape[0], "Listings über 1.000")
print(df[df['price'] > 500].shape[0], "Listings über 500")

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

obergrenze = Q3 + 3 * IQR  

print(f"Obergrenze: {obergrenze:.2f}")

df = df[df['price'] <= obergrenze]
print(df.shape)
print(df['price'].describe())

print(f"Obergrenze: {obergrenze:.2f}")

import folium
from folium.plugins import HeatMap

# 06: Create map – centered on London
karte = folium.Map(location=[51.5074, -0.1278], zoom_start=11)

heat_data = df[['latitude', 'longitude', 'price']].values.tolist()
HeatMap(heat_data, min_opacity=0.3, radius=8).add_to(karte)

karte.save(MAP_PATH + 'london_heatmap.html')
print("Karte gespeichert!")

# 07: Load 
import requests
import json

url = "https://raw.githubusercontent.com/radoi90/housequest-data/master/london_boroughs.geojson"
response = requests.get(url)
london_geo = response.json()

# 08: Calculate the average price per district
preis_pro_stadtteil = df.groupby('neighbourhood_cleansed')['price'].mean().reset_index()
preis_pro_stadtteil.columns = ['neighbourhood', 'avg_price']

print(preis_pro_stadtteil.sort_values('avg_price', ascending=False).head(10))

# 09: Create a choropleth map

import folium

choropleth_karte = folium.Map(location=[51.5074, -0.1278], zoom_start=10)

folium.Choropleth(
    geo_data=london_geo,
    data=preis_pro_stadtteil,
    columns=['neighbourhood', 'avg_price'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name='Durchschnittspreis (£)'
).add_to(choropleth_karte)

choropleth_karte.save(MAP_PATH + 'london_choropleth.html')
print("Choropleth map saved!")

# 10: Caclucate Distances

from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


df['dist_center_km'] = df.apply(
    lambda r: haversine(r['latitude'], r['longitude'], 51.5080, -0.1281), axis=1
)


print(df['dist_center_km'].describe())
print(df[['neighbourhood_cleansed', 'dist_center_km', 'price']].head(10))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['dist_center_km'], df['price'], alpha=0.1, s=5)
plt.xlabel('Distanz zum Zentrum (km)')
plt.ylabel('Preis (£)')
plt.title('Preis vs. Distanz zum Zentrum - London Airbnb')
plt.savefig(FIGURE_PATH + 'preis_vs_distanz.png', dpi=150)
plt.show()
print("Plot gespeichert!")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 11: Scale coordinates 
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(df[['latitude', 'longitude']])

# 12: K-Means with 8 Clusters
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['geo_cluster'] = kmeans.fit_predict(coords_scaled)


print(df.groupby('geo_cluster')['price'].mean().sort_values(ascending=False))

import matplotlib.pyplot as plt


farben = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown', 'cyan']

plt.figure(figsize=(12, 8))
for cluster in range(8):
    mask = df['geo_cluster'] == cluster
    plt.scatter(
        df[mask]['longitude'],
        df[mask]['latitude'],
        c=farben[cluster],
        label=f'Cluster {cluster} (Ø {df[mask]["price"].mean():.0f}£)',
        alpha=0.3,
        s=5
    )

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geo-Cluster London Airbnb')
plt.legend(loc='upper left', markerscale=3)
plt.savefig(FIGURE_PATH + 'geo_cluster.png', dpi=150)
plt.show()
print("Cluster-Plot gespeichert!")

# 13: Calculate the average price per district
stadtteil_preis = df.groupby('neighbourhood_cleansed')['price'].mean()
df['neighbourhood_avg_price'] = df['neighbourhood_cleansed'].map(stadtteil_preis)

print(df[['neighbourhood_cleansed', 'neighbourhood_avg_price', 'price']].head())

# 14: Overview of the new geo-features
print(df[['latitude', 'longitude', 'dist_center_km', 'geo_cluster', 'neighbourhood_avg_price', 'price']].describe())

# 15: Save Data
df.to_csv(OUTPUT_PATH + 'london_geo_features.csv', index=False, sep=';')
print("Record saved!")
print(f"Final columns: {df.columns.tolist()}")
