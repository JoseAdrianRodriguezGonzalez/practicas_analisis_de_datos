import pandas as pd 
import os 
def variable_identifier(df:pd.DataFrame,max_n=7):
    for columns in df.columns:
        
        if isinstance(df[columns].dtype,pd.CategoricalDtype):
            if df[columns].ordered:
                print(f"{columns}: Cualitativa ordinal")
            else:
                print(f"{columns}: Cualitativa nominal")
        elif pd.api.types.is_object_dtype(df[columns]):
            print(f"{columns}: cualitativa nominal")
        elif pd.api.types.is_float_dtype(df[columns]):
            print(f"{columns}: Cuantitativa continua")
        elif pd.api.types.is_integer_dtype(df[columns]):
            if df[columns].nunique()<=max_n:
                print(f"{columns}: Cualitativa ordinal")
            else:
                print(f"{columns}: Cuantitativa discreta")
def read_csv(file_name):
    try:
        df = pd.read_csv(file_name, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, encoding="latin1")
    print("Descripción: \n")
    print(df.describe())
    print("Información:\n")
    print(df.info())
    print("Cabecera de datos ")
    variable_identifier(df)
    

def read_files(folder: str):
    for fname in os.listdir(folder):
        print(f"\narchivo: {fname}")
        path = os.path.join(folder, fname)
        read_csv(path)
read_files("data")

