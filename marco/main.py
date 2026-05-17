import pandas as pd
import numpy as np 
import folium
from pandas.core.arrays import categorical
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import geopandas as gpd 
class primer_problema:
    def __init__(self,path_dataframe):
        self.path_dataframe=path_dataframe
        self.df=self.load_dataset(self.path_dataframe)
    def load_dataset(self,path):
        return pd.read_csv(path,encoding="latin-1")
    def limpieza(self):
        print(self.df[["latitud","longitud"]].isna().sum())
        self.df_clean=self.df.dropna(subset=["latitud","longitud"])
        print(self.df_clean.isna().sum())
        #mantener un solo valor
        print(self.df_clean["num_cicloe"].dtype)
        self.df_clean.loc[:,"num_cicloe"]=self.df_clean["num_cicloe"].apply(self.rango_a_numero)
        print(self.df_clean["num_cicloe"].dtype)
        print(self.df_clean["num_cicloe"])
    def rango_a_numero(self,x):
        if isinstance(x,str) and "-" in x:
            a,b=x.split("-")
            return (int(a)+int(b))//2
        try:
            return int(x)
        except:
            return np.nan 
    def clusterizado(self):
        coords=self.df_clean[["latitud","longitud"]].to_numpy()
        coords_radian=np.radians(coords)
        model=DBSCAN(
            eps=0.000075,
            min_samples=5,
            metric="haversine"
        )
        clusters=model.fit_predict(coords_radian)
        self.df_clean.loc[:,"cluster"]=clusters 
        print(clusters)
        print(f" cantidad {self.df_clean['cluster'].value_counts()}")
    def mapear_puntos(self):
        colores=["red","blue","green","purple","orange","darkred","cadetblue"]
        centro=[self.df_clean["latitud"].mean(),self.df_clean["longitud"].mean()]
        mapa=folium.Map(location=centro,zoom_start=13)
        for _,fila in self.df_clean.iterrows():
            cluster=fila["cluster"]
            color= "black" if cluster ==-1 else colores[int(cluster) %len(colores)]
            folium.CircleMarker(location=[fila["latitud"],fila["longitud"]],
                                radius=fila["num_cicloe"]*0.01,
                                color="gray",
                                fill=True,
                                fill_opacity=0.4).add_to(mapa)
        for cluster_id in self.df_clean["cluster"].unique():
            if cluster_id==-1:
                continue
            puntos_cluster = self.df_clean[self.df_clean["cluster"] == cluster_id][["latitud", "longitud"]].values
            if len(puntos_cluster)>=3:
                hull=ConvexHull(puntos_cluster)
                vertices=puntos_cluster[hull.vertices]
                color_cluster=colores[int(cluster_id)%len(colores)]
                folium.Polygon(
                    locations=vertices,
                    color=color_cluster,
                    fill=True,
                    fill_color=color_cluster,
                    fill_opacity=0.3,
                    popup=f"ZOna de densidad {cluster_id}"
                ).add_to(mapa)
        mapa.save("results/mapa_ecobici_dbscan.html")
class segundo_problema:
    def __init__(self,path_first,path_second):
        self.gdf_hospitales=gpd.read_file(path_second)
        self.gdf_colonias=gpd.read_file(path_first)
        self.gdf_colonias=self.gdf_colonias.to_crs("EPSG:4326")
         
        print(self.gdf_colonias.crs)
        print(self.gdf_hospitales.crs)
        print("Ver si hay ajuste")
        print(self.gdf_colonias.total_bounds)
        print(self.gdf_hospitales.total_bounds)
    
    def graficar(self,colonias_m,hospitales_m,buffers):
        colonias_plot=colonias_m.to_crs("EPSG:4326")
        hospitales_plot=hospitales_m.to_crs("EPSG:4326")
        buffers_plot=buffers.to_crs("EPSG:4326")
        mapa=colonias_plot.explore(
            column="cubierta",
            cmap=["red","green"],
            categorical=True,
            legend=True,
            name="Colonias"
        )
        buffers_plot.explore(
            m=mapa,
            color="blue",
            style_kwds={"fill_opacity":0.1},
            name="Area cobertura de 1 km"
        )
        hospitales_plot.explore(
            m=mapa, 
            color="black",
            marker_kwds={"radius":4},
            name="hospitales"
        )
        return mapa 
    def buffer(self,radius=1000):
        colonias_m=self.gdf_colonias.to_crs("EPSG:32614")
        hospitales_m=self.gdf_hospitales.to_crs("EPSG:32614")
        buffers = hospitales_m.copy()
        buffers["geometry"] = buffers.geometry.buffer(radius)
        hospitales_m["buffer"]=hospitales_m.buffer(radius)
        colonias_m["cubierta"]=colonias_m.geometry.apply(
            lambda g: hospitales_m["buffer"].intersects(g).any() 
        )
        return colonias_m,hospitales_m,buffers 
def pasos_primer_problema():
    path_ciclo="data/cicloestaciones_ecobici.csv"
    parte_1=primer_problema(path_ciclo)
    parte_1.limpieza()
    parte_1.clusterizado()
    parte_1.mapear_puntos()
def pasos_segundo_problema():
    path_hospital="data/Centros_de_salud/Centros_de_salud.shp"
    path_colonias="data/unidades_habitacionales_cdmx/Unidades_Habitacionales.shp"
    parte_2=segundo_problema(path_colonias,path_hospital)
    col,hos,b=parte_2.buffer()
    mapa=parte_2.graficar(col,hos,b)
    mapa.save("results/mapa_segundo_problema.html")
def main():
    pasos_primer_problema()
    pasos_segundo_problema()
#    pasos_tercer_problema()
main()
