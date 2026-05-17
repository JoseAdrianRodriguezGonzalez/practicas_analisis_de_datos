import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

"""
TASK 1: 
Interactive visualization of geographic points with Folium.

OBJECTIVE:
Represent georeferenced data of Ecobici CDMX stations, clean data, generate clusters and 
heatmaps.
"""

def load_and_clean_data(filepath):
    """
    Load the dataset and perform initial cleaning.
    """
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath, encoding="latin1")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    print(f"Original records: {len(df)}")

    # Clean data: remove records without valid coordinates
    if(("latitud" not in df.columns) or ("longitud" not in df.columns)):
        raise ValueError("No coordinate columns (latitud/longitud) found.")

    df_clean = df.dropna(subset=["latitud", "longitud"])
    
    # Filter out incorrect coordinates (outside approximate CDMX range)
    # CDMX approx: Lat 19.0-19.6, Lon -99.3 - -99.0
    df_clean = df_clean[
        (df_clean["latitud"] > 19.0) & (df_clean["latitud"] < 19.6) &
        (df_clean["longitud"] > -99.3) & (df_clean["longitud"] < -99.0)
    ]

    print(f"Records after cleaning: {len(df_clean)}")
    return df_clean

def generate_kmeans_clusters(df, n_clusters=5):
    """
    Generate spatial clusters using K-Means to group stations.
    """
    print(f"Generating {n_clusters} K-Means clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df[["latitud", "longitud"]])
    return df

def get_cluster_color(cluster_id, total_clusters):
    """
    Generate a color list to differentiate clusters.
    """    
    colors = list(mcolors.TABLEAU_COLORS.values())
    return colors[cluster_id % len(colors)]

def create_interactive_map(df):
    """
    Generates the HTML map with markers, clusters and density zones.
    """
    # Central coordinates of CDMX to initialize the map
    cdmx_center = [19.4326, -99.1332]
    
    map_folium = folium.Map(location=cdmx_center, zoom_start=13, tiles="CartoDB positron")

    # Add marker clusters for each group
    marker_clusters_group = MarkerCluster(name="Dynamic Grouping").add_to(map_folium)

    for _, row in df.iterrows():
        radius = 5
        popup_info = f"Station ID: {row.get('num_cicloe', 'N/A')}"

        # If there is a column with bike availability, adjust circle size
        col_bikes = next((c for c in df.columns if "bike" in c.lower() or "bici" in c.lower()), None)
        if(col_bikes):
            try:
                bike_count = float(row[col_bikes])
                radius = max(3, bike_count / 2)
                popup_info += f"<br>Available Bikes: {int(bike_count)}"
            except:
                pass
        
        cluster_color = get_cluster_color(row["cluster"], 5)

        folium.CircleMarker(
            location=[row["latitud"], row["longitud"]],
            radius=radius,
            popup=folium.Popup(popup_info, max_width=200),
            color=cluster_color,
            fill=True,
            fill_color=cluster_color,
            fill_opacity=0.7
        ).add_to(marker_clusters_group)

    # Add heatmap layer
    heat_data = [[row["latitud"], row["longitud"]] for _, row in df.iterrows()]
    HeatMap(heat_data, name="Density Map", radius=15).add_to(map_folium)

    folium.LayerControl().add_to(map_folium)

    return map_folium

if __name__ == "__main__":
    FILE_ECOBICI = "data/cicloestaciones_ecobici.csv"
    
    df_ecobici = load_and_clean_data(FILE_ECOBICI)

    if df_ecobici is not None:
        df_ecobici = generate_kmeans_clusters(df_ecobici)
        map_ecobici = create_interactive_map(df_ecobici)
        
        output_file = "results/ecobici_cdmx_map.html"
        map_ecobici.save(output_file)
        print(f"Map saved to: {output_file}")
        print("Open the file in your browser to view the interactive map.")