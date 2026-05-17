from json import load
import pandas as pd 
def load_dataset(path):
    return pd.read_excel(path)
def traduccion(df):
    df.columns = [
        "id",
        "fecha",
        "nombre_producto",
        "precio",
        "ventas_30d_botellas",
        "numero_resenas",
        "marca",
        "numero_busquedas"
    ]
    return df
def intercambio_de_datos(df):
    df["fecha"]=pd.to_datetime(df["fecha"],format="%Y%m")
    return df
def main():
    df=load_dataset("wine_sales_data_20182019.xlsx")
    df=traduccion(df)
    print(df.head())
    print(df.info())
    df=intercambio_de_datos(df)
    print(df["fecha"].min())
main()
