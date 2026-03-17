import pymongo
import pandas as pd
import folium
import json

def get_data_from_atlas():
    # Use the SAME string you used in load_data.py
    uri = "mongodb+srv://ayushjha:ayushjha007@cluster0.aoh7j5n.mongodb.net/"
    client = pymongo.MongoClient(uri)
    db = client["crime_db"]
    collection = db["hotspots"]
    
    # Fetch data from cloud
    data = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(data)

def generate_hotspot_map():
    print("Fetching AI results from MongoDB Atlas...")
    df = get_data_from_atlas()
    
    # Standardize for map matching
    df['DISTRICT'] = df['DISTRICT'].str.title()
    
    # Load your map file
    with open("data/india_district.json", 'r') as f:
        topo_data = json.load(f)
    
    layer_key = 'india-districts-2019-734'
    features = topo_data['objects'][layer_key]['geometries']
    geo_json_data = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": g, "properties": g['properties']} for g in features]}

    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=geo_json_data,
        data=df,
        columns=["DISTRICT", "predicted_risk_level"],
        key_on="feature.properties.district",
        fill_color="YlOrRd",
        legend_name="Cloud-Synced Crime Risk Level",
        bins=[0, 1, 2, 3, 4, 5]
    ).add_to(m)

    m.save("crime_hotspot_cloud_map.html")
    print("Heatmap generated from Cloud Data: crime_hotspot_cloud_map.html")

if __name__ == "__main__":
    generate_hotspot_map()