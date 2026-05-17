import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

"""
TASK 4: 
Routes and optimal paths in graphs (OpenFlights).

OBJECTIVE:
1. Build and directed graph of airports and routes.
2. Dijkstra algorithm for shortest route (fewest stops).
3. Fastest route (distance/weight).
4. Diameter and connectivity.
5. Visualization.
"""

# URLs to "raw" data on GitHub
URL_AIRPORTS = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
URL_ROUTES = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance in kilometers between two geographic points).
    Used to estimate the 'weight' (duration/cost) of the flight.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def load_and_build_graph():
    """
    Load data from URL and build a directed graph (DiGraph) with weights.
    """
    print("Loading OpenFlights data")

    # 1. Load Airports (Nodes)
    cols_airports = ['ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source']
    df_airports = pd.read_csv(URL_AIRPORTS, header=None, names=cols_airports, index_col='ID')
    print(f"Airports loaded: {len(df_airports)}")

    # 2. Load Routes (Edges)
    cols_routes = ['Airline', 'Airline ID', 'SourceAirport', 'SourceID', 'DestAirport', 'DestID', 'Codeshare', 'Stops', 'Equipment']
    df_routes = pd.read_csv(URL_ROUTES, header=None, names=cols_routes, na_values='\\N')

    # Clean routes with null or invalid IDs
    df_routes = df_routes.dropna(subset=['SourceID', 'DestID'])
    df_routes['SourceID'] = df_routes['SourceID'].astype(int)
    df_routes['DestID'] = df_routes['DestID'].astype(int)
    print(f"Routes loaded: {len(df_routes)}")

    # 3. Build the graph
    G = nx.DiGraph()
    
    print("Building graph and calculating distances")
    for _, row in df_routes.iterrows():
        source_id = row['SourceID']
        dest_id = row['DestID']

        if((source_id in df_airports.index) and (dest_id in df_airports.index)):
            lat1, lon1 = df_airports.loc[source_id, ['Latitude', 'Longitude']]
            lat2, lon2 = df_airports.loc[dest_id, ['Latitude', 'Longitude']]
            
            distance_km = haversine(lon1, lat1, lon2, lat2)
            G.add_edge(source_id, dest_id, weight=distance_km)

    # Add node attributes
    for node_id in G.nodes():
        if(node_id in df_airports.index):
            G.nodes[node_id]['name'] = df_airports.loc[node_id, 'Name']
            G.nodes[node_id]['iata'] = df_airports.loc[node_id, 'IATA']
            G.nodes[node_id]['pos'] = (df_airports.loc[node_id, 'Longitude'], df_airports.loc[node_id, 'Latitude'])
        
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, df_airports

def analyze_routes(G, origin_code, dest_code, df_airports):
    """
    Run Dijkstra for shortest (fewest stops) and fastest (distance) routes.
    Accepts either IATA or ICAO codes automatically.
    """
    def find_node(code):
        # Try IATA
        match = df_airports[df_airports['IATA'] == code]
        if not match.empty:
            return match.index[0]
        # Try ICAO
        match = df_airports[df_airports['ICAO'] == code]
        if not match.empty:
            return match.index[0]
        # Try City or Name
        match = df_airports[df_airports['City'].str.contains(code, case=False, na=False)]
        if not match.empty:
            return match.index[0]
        match = df_airports[df_airports['Name'].str.contains(code, case=False, na=False)]
        if not match.empty:
            return match.index[0]
        return None

    source_node = find_node(origin_code)
    dest_node = find_node(dest_code)

    if source_node is None or dest_node is None:
        print(f"Error: Could not find airport code {origin_code} or {dest_code} (IATA/ICAO/City/Name).")
        return None

    print(f"\n--- Analyzing route: {origin_code} -> {dest_code} ---")

    # Shortest route in stops
    try:
        path_stops = nx.shortest_path(G, source=source_node, target=dest_node)
        print(f"Route with fewest stops ({len(path_stops)-1} flights):")
        print(" -> ".join([G.nodes[n].get('iata', G.nodes[n].get('name', str(n))) for n in path_stops]))
    except nx.NetworkXNoPath:
        print("No route exists between these airports.")
        return None

    # Shortest route in distance
    try:
        path_dist = nx.dijkstra_path(G, source=source_node, target=dest_node, weight='weight')
        total_dist = nx.dijkstra_path_length(G, source=source_node, target=dest_node, weight='weight')
        print(f"Shortest distance route ({total_dist:.2f} km):")
        print(" -> ".join([G.nodes[n].get('iata', G.nodes[n].get('name', str(n))) for n in path_dist]))
    except nx.NetworkXNoPath:
        print("No weighted route exists.")
        return None

    return path_dist

def global_metrics(G):
    """
    Analyze connectivity and diameter.
    """
    print("\nGlobal Graph Metrics")

    is_strongly = nx.is_strongly_connected(G)
    print(f"Strongly connected: {'YES' if is_strongly else 'NO'}")

    is_weakly = nx.is_weakly_connected(G)
    print(f"Weakly connected: {'YES' if is_weakly else 'NO'}")

    largest_cc = max(nx.strongly_connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    
    print(f"Calculating diameter of largest component ({len(subgraph)} nodes)")
    try:
        diameter = nx.diameter(subgraph)
        print(f"Diameter of main network: {diameter} hops (maximum flights needed between farthest points in the network)")
    except Exception as e:
        print(f"Could not calculate exact diameter: {e}")

def vizualize_routes(G, path_nodes):
    """
    Visualize the graph and a specific route.
    """
    print("\nGenerating Visualization")
    plt.figure(figsize=(12, 8))
    
    pos = nx.get_node_attributes(G, 'pos')
    pos = {k:v for k,v in pos.items() if v}

    nx.draw_networkx_nodes(G, pos, node_size=5, node_color='lightgray', alpha=0.5)
    
    path_edges = list(zip(path_nodes, path_nodes[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_size=50, node_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2, arrows=True)

    labels = {n: G.nodes[n]['iata'] for n in path_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black', verticalalignment='bottom')
    
    plt.title("Optimal Route in the OpenFlights Network")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("results/graph_routes.png")
    print("Visualization saved as 'graph_routes.png'")

if __name__ == "__main__":
    graph_air, df_airports = load_and_build_graph()
    optimal_route = analyze_routes(graph_air, "MMMX", "KJFK", df_airports)
    
    if(optimal_route):
        global_metrics(graph_air)
        vizualize_routes(graph_air, optimal_route)