import rdflib
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, XSD
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta 
import pandas as pd 
import re
import unicodedata


"""
TASK 2:
Knowledge Graphs with RDFLib.

OBJECTIVE:
1. Build an RDF graph about Mexican universities.
2. Define triples (Subject, Predicate, Object).
3. SPARQL queries (Location and Year of foundation).
4. Serialize in Turtle format (.ttl).
5. Visualize with Matplotlib.
"""
def make_uri_safe(text):
    if text is None:
        return "Unknown"
    text = re.sub(r'<.*?>', '', text)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^A-Za-z0-9]+', '_', text)
    text = text.strip('_')
    return text or "Unknown"
def extract_data():
    """
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   universidad_id               3388 non-null   int64  
 1   universidad_nombre           3388 non-null   object 
 2   universidad_fecha_fundacion  2425 non-null   float64
 3   universidad_adscripcion      2545 non-null   object 
 4   estado_id                    3388 non-null   int64  
 5   nom_ent                      3388 non-null   object 
 6   municipio_id                 3388 non-null   int64  
 7   nom_mun                      3388 non-null   object 
 8   localidad_id                 3388 non-null   int64  
 9   nom_loc                      3388 non-null   object 
 10  universidad_calle_numero     3379 non-null   object 
 11  universidad_colonia          3101 non-null   object 
 12  universidad_cp               3303 non-null   float64
 13  email                        2901 non-null   object 
 14  longitud                     3388 non-null   float64
 15  latitud                      3388 non-null   float64
 16  link_sic                     3388 non-null   object 
 17  fecha_mod                    3388 non-null   int64  
 18  nom_ent_corto                3388 non-null   object 
 19  fn_universidad_colonia       1 non-null      float64
 20  nom_ent_etq                  3388 non-null   object 
"""
    dataset=pd.read_csv("data/data-2026-02-19.csv") 
    print(dataset["universidad_fecha_fundacion"].isna().sum())
    dataset.dropna(subset=["universidad_fecha_fundacion"],inplace=True)
    print(dataset["universidad_fecha_fundacion"].isna().sum())
    epoch=datetime(1970,1,1)
    dataset.loc[:,"fundacion_estimada"]=dataset["universidad_fecha_fundacion"].apply(lambda x: epoch+timedelta(days=x)) 
    dataset["anio"]=dataset["fundacion_estimada"].apply(lambda x:x.year)
    dataset["ciudad"]=dataset.apply(
        lambda row:row["nom_ent"] if row["nom_ent"]=="Ciudad de México" 
            else row["nom_mun"],
            axis=1 
    )
    df=pd.DataFrame(dataset[["universidad_nombre","ciudad","anio"]])
    print(df)
    return df 
def create_knowledge_graph(dataset):
    """
    Build the RDF graph and define triples manually.
    """
    print("Creating Knowledge Graph")
    
    g = Graph()
    EX = Namespace("http://example.org/universidades/")
    g.bind("ex", EX)

    # Properties (Predicates)
    located_in = EX.ubicada_en
    founded_in = EX.fundada_en
    rdf_type = RDF.type
    class_uni = EX.Universidad

    # Define that they are Universities
    for _, row in dataset.iterrows(): 
        name=row["universidad_nombre"]
        uri_name=make_uri_safe(name)
        
        uni=EX[uri_name]
        g.add((uni,RDF.type,class_uni))
        if pd.notna(row["ciudad"]):
            g.add((uni,located_in,Literal(row["ciudad"])))
        if pd.notna(row["anio"]):
            g.add((uni,founded_in,Literal(int(row["anio"]),datatype=XSD.integer)))

    # Years of foundation
    print(f"Graph created with {len(g)} triples.")
    return g, EX

def sparql_queries(g, EX):
    """
    Run the SPARQL queries requested in the task.
    """
    print("\nRunning SPARQL Queries")

    # Query 1: List all universities in CDMX
    query_cdmx = """
    SELECT ?name
    WHERE {
        ?uni ex:ubicada_en "Ciudad de México" .
        ?uni rdf:type ex:Universidad .
        BIND(STRAFTER(STR(?uni), "http://example.org/universidades/") AS ?name)
    }
    """
    print(">>> Universities in Ciudad de Mexico:")
    results = g.query(query_cdmx)
    for row in results:
        print(f" - {row.name}")

    # Query 2: Universities founded before 1950
    query_year = """
    SELECT ?name ?year
    WHERE {
        ?uni ex:fundada_en ?year .
        FILTER (?year < 1950)
        BIND(STRAFTER(STR(?uni), "http://example.org/universidades/") AS ?name)
    }
    """
    print("\n>>> Universities founded before 1950:")
    results = g.query(query_year)
    for row in results:
        print(f" - {row.name} (Founded in {row.year})")

def serialize_and_save(g):
    """
    Save the graph in Turtle (.ttl) format.
    """
    file = "results/universidades_mexico.ttl"
    g.serialize(destination=file, format="turtle")
    print(f"\nGraph serialized and saved in: {file}")

def visualize_graph(g,nodes=10,ciudad=None):
    """
    Convert RDF triples to a NetworkX graph for visualization.
    """
    print("\nGenerating Graph Visualization")
    
    nx_graph = nx.MultiDiGraph()
    count=0 
    uni=set()
    if ciudad:
        for s,p,o in g:
            if str(p).endswith("ubicada_en") and str(o)==ciudad:
                uni.add(s)
    for s, p, o in g:
        if count>= nodes:
            break
        subject = s.split('/')[-1]
                  
        obj = o.split('/')[-1] if isinstance(o, URIRef) else o
        if ciudad and s not in uni:
            continue
        # Simplify predicate names
        if p == RDF.type:
            predicate = "es_una"
        elif str(p).endswith("ubicada_en"):
            predicate = "ubicada_en"
        elif str(p).endswith("fundada_en"):
            predicate = "fundada_en"
        else:
            predicate = p.split('/')[-1]
        nx_graph.add_edge(subject, obj, relation=predicate)
        count+=1 
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(nx_graph, k=0.5, seed=42)

    nx.draw_networkx_nodes(nx_graph, pos, node_size=2000, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(nx_graph, pos, width=1.5, alpha=0.6, edge_color="gray", arrowsize=20)

    edge_labels = nx.get_edge_attributes(nx_graph, 'relation')
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Knowledge Graph: Universidades Mexicanas")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("results/knowledge_graph.png")
    print("Visualization saved as 'knowledge_graph.png'")

if __name__ == "__main__":
    df=extract_data()    
    graph, namespace = create_knowledge_graph(df)
    sparql_queries(graph, namespace)
    serialize_and_save(graph)
    visualize_graph(graph,ciudad="Ciudad de México")
