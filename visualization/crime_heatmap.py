import pandas as pd
import folium
import json
import os

def generate_crime_heatmap(csv_path="data/predicted_hotspots.csv", 
                           topojson_path="data/india_district.json", 
                           output_path="crime_heatmap.html"):
    
    # 1. Load your AI Prediction Results
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run your Spark pipeline first.")
        return
    df = pd.read_csv(csv_path)

    # 2. Standardize Casing for Matching
    # Your JSON uses "Balrampur", but CSV might have "BALRAMPUR"
    df['DISTRICT'] = df['DISTRICT'].astype(str).str.strip().str.title()

    # 3. Load and Parse TopoJSON
    with open(topojson_path, 'r') as f:
        topo_data = json.load(f)
    
    # Extract geometries from the specific layer in your file
    # Internal key from your file: 'india-districts-2019-734'
    layer_name = 'india-districts-2019-734'
    geometries = topo_data['objects'][layer_name]['geometries']
    
    # Convert TopoJSON objects to a FeatureCollection for Folium compatibility
    geo_json_features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": g, # TopoJSON geometries work directly if structured this way
                "properties": g['properties']
            } for g in geometries
        ]
    }

    # 4. Initialize Folium Map (Centered on India)
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    # 5. Create the Choropleth Layer
    folium.Choropleth(
        geo_data=geo_json_features,
        name="Crime Risk Hotspots",
        data=df,
        columns=["DISTRICT", "predicted_risk_level"],
        key_on="feature.properties.district", # Matches 'district' in your JSON snippet
        fill_color="YlOrRd", # Yellow to Red (Hotspot) gradient
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="AI Predicted Risk Level (0: Low - 4: High)",
        nan_fill_color="#eeeeee" # Light grey for districts with no data
    ).add_to(m)

    # 6. Save and Finish
    m.save(output_path)
    print(f" Success! Heatmap generated at: {output_path}")

if __name__ == "__main__":
    generate_crime_heatmap()