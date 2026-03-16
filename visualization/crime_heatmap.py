import folium

def generate_map():

    india_map = folium.Map(location=[22.97,78.65], zoom_start=5)

    folium.CircleMarker(
        location=[28.61,77.23],
        radius=8,
        color="red"
    ).add_to(india_map)

    india_map.save("crime_heatmap.html")