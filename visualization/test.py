import pandas as pd
import json

# Load CSV and JSON
df = pd.read_csv("data/predicted_hotspots.csv")
df['DISTRICT'] = df['DISTRICT'].str.title()

with open("data/india_district.json") as f:
    topo_data = json.load(f)
layer = topo_data['objects']['india-districts-2019-734']['geometries']
json_names = {g['properties']['district'] for g in layer}

# Find mismatches
missing_in_json = set(df['DISTRICT']) - json_names
print(f"Districts in your CSV NOT found in Map: {list(missing_in_json)[:10]}")