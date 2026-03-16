import folium
from folium.plugins import HeatMap
import pandas as pd


def generate_heatmap(spark_df):

    # Convert Spark dataframe to Pandas
    df = spark_df.toPandas()

    # For demo we map state/district to approximate coordinates
    # (Later you can replace this with real coordinates dataset)
    
    location_data = {
        "DELHI": [28.61, 77.23],
        "MAHARASHTRA": [19.07, 72.87],
        "UTTAR PRADESH": [26.85, 80.95],
        "BIHAR": [25.59, 85.13],
        "RAJASTHAN": [26.91, 75.78],
        "TAMIL NADU": [13.08, 80.27],
        "KARNATAKA": [12.97, 77.59]
    }

    heat_data = []

    for _, row in df.iterrows():

        state = row["STATE/UT"]

        if state in location_data:
            lat, lon = location_data[state]

            heat_data.append([lat, lon, row["total_crime"]])

    india_map = folium.Map(location=[22.97, 78.65], zoom_start=5)

    HeatMap(heat_data).add_to(india_map)

    india_map.save("crime_heatmap.html")

    print("crime_heatmap.html generated")