import pandas as pd 
import os 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import  pearsonr
def read_files(src_main):
    for folder in os.listdir(src_main):
        path_folder=os.path.join(src_main,folder)
        subpath_folder=os.path.join(path_folder,"horarios","2022")
        df_list=[pd.read_csv(os.path.join(subpath_folder,csv_file),encoding="latin-1",skiprows=3)for csv_file in os.listdir(subpath_folder)]
        df_submaster=pd.concat(df_list,ignore_index=True)
        df_submaster.to_csv(os.path.join(path_folder,"horarios","master.csv"),index=False)
def read_masters(src_main):
    rename_map = {
        "O3 [ppm]": "O3",
    "NO2 [ppm]": "NO2",
    "SO2 [ppm]": "SO2",
    "CO [ppm]": "CO",
    "PM10 [µg/m³]": "PM10",
    "PM2.5 [µg/m³]": "PM25",
    "TEMP": "TEMP",
    "HR": "HUMIDITY",
    "WS": "WIND_SPEED",
    "WD": "WIND_DIR"
    }
    
    for folder in os.listdir(src_main):
        path_folder=os.path.join(src_main,folder,"horarios","master.csv")
        df=pd.read_csv(path_folder)
        df = df.rename(columns=rename_map)
        info_general(df)
        df=clean(df)
        print("*"*20)
        print("Info limpia")
        print("*"*20)
        plots_(df)
        df.to_csv(path_folder,index=False)
def info_general(df:pd.DataFrame):
    print(df.columns)
    print(df.info())
    print(df.isna().sum())
    latex_table = df.describe().to_latex()
    print(latex_table)
def clean(df):
    df_clean = df.dropna()
    return df_clean
def plots_(df:pd.DataFrame):
    columnas=df.columns.to_list() 
    sns.pairplot(df[columnas])
    plt.show()
def correlation_map(src_main,method):
    for folder in os.listdir(src_main):
        path_folder=os.path.join(src_main,folder,"horarios","master.csv")
        df=pd.read_csv(path_folder)
        df=df.drop(columns="MES")
        corr_matrix=df.corr(method=method)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Matriz de correlación ({method})")
        plt.show()
def correlation_simple(df, x, y):
    return df[[x, y]].dropna().corr(method="pearson").iloc[0, 1]
import pingouin as pg
def partial_correlation(df, x, y, z):
    result = pg.partial_corr(data=df, x=x, y=y, covar=z)
    return result["r"].values[0]
def analyze_case(df, x, y, z):
    r_simple = correlation_simple(df, x, y)
    r_partial = partial_correlation(df, x, y, z)
    print(f"\nCaso: {x} vs {y} controlando {z}")
    print(f"Correlación simple: {r_simple:.4f}")
    print(f"Correlación parcial: {r_partial:.4f}")
    print(f"Cambio: {r_partial - r_simple:.4f}")

def full_analysis(src_main):
    for folder in os.listdir(src_main):
        path_folder=os.path.join(src_main,folder,"horarios","master.csv")
        df=pd.read_csv(path_folder)
        df=df.drop(columns="MES")
        pm25_case="PM25" if "PM25" in df.columns else  "PM10"
        analyze_case(df, pm25_case, "NO2", "TEMP")
        analyze_case(df, "PM10", "NO2", "HUMIDITY")
        analyze_case(df, pm25_case, "CO", "WIND_SPEED")
        analyze_case(df, "NO2", "O3", "TEMP")
def pearson_full(df, x, y):
    data = df[[x, y]].dropna()
    r, p = pearsonr(data[x], data[y])
    return r, p
def pearson_ana(src_main):
    for folder in os.listdir(src_main):
        path_folder=os.path.join(src_main,folder,"horarios","master.csv")
        df=pd.read_csv(path_folder)
        df=df.drop(columns="MES")
        if "PM25" not in df.columns:
            continue
        pearson_cases(df) 
def pearson_cases(df):
    cases = [
        ("PM25", "PM10"),
        ("PM25", "HUMIDITY"),
        ("PM25", "TEMP"),
        ("NO2", "O3")
    ]

    for x, y in cases:
        r, p = pearson_full(df, x, y)
        print(f"{x} vs {y} -> r={r:.4f}, p={p:.4e}")
read_files("data/Salamanca")
read_masters("data/Salamanca")
correlation_map("data/Salamanca","pearson")
pearson_ana("data/Salamanca")
correlation_map("data/Salamanca","spearman")
full_analysis("data/Salamanca")
