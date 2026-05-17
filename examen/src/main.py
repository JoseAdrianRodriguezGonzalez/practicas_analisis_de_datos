
import pandas as pd 
import os

#################

#    Etapa 1    #

#################
#Auxiliar functions
def standardize_pm25(df:pd.DataFrame)->pd.DataFrame:
    for col in df.columns:
        col_lower = col.lower()        
        if "pm2.5" in col_lower or "pm25" in col_lower or "mpm2.5" in col_lower:
            return df.rename(columns={col: "pm25"}) 
    return df
def read_csv_fix(path:str)->pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", header=None)
    for i in range(10):
        row = df.iloc[i].astype(str).str.lower()
        if any("pm" in cell for cell in row):
            return pd.read_csv(path, encoding="latin-1", skiprows=i)
    return pd.read_csv(path, encoding="latin-1")
def create_folder(src:str):
    os.makedirs(src,exist_ok=True)
class first_stage:
    def __init__(self,root_path:str,out:str,variable:str) -> None:
        self.root_path=root_path
        self.out=out 
        self.variable=variable
    def read_and_join_dataframes(self,root_path:str="",tipo:str="horarios")->dict[str,dict[str,list[pd.DataFrame]]]:
        if len(self.root_path)==0:
            self.root_path=root_path
        #reading files
        dict_per_station={}
        for folder in os.listdir(self.root_path):
            path_first=os.path.join(self.root_path,folder)
            years=dict()
            for year in sorted(os.listdir(os.path.join(path_first,tipo))):
                path_second_year=os.path.join(path_first,tipo,year)
                list_df=[]
                for file in sorted(os.listdir(path_second_year)):
                    df=read_csv_fix(os.path.join(path_second_year,file))
                    df=standardize_pm25(df)
                    list_df.append(df)
                years[year]=list_df
            dict_per_station[folder]=years
        return dict_per_station 
    def join_datasets_according_variable(self,dict_per_station:dict[str,dict[str,list[pd.DataFrame]]],variable:str="")->dict[str,dict[str,list[pd.DataFrame]]]:
        if len(self.variable)==0:
            self.variable=variable
        return {
            station:{
                years:[df for df in list_df
                        if self.variable in df.columns]
                for years,list_df in dictionaries.items()
            }
            for station,dictionaries in dict_per_station.items()
        }

    def concat_dataframes_according_years(self,dict_per_station:dict[str,dict[str,list[pd.DataFrame]]])->dict[str,dict[str,pd.DataFrame]]:
        return {
            station:{
                years:pd.concat(list_df,ignore_index=True)
                for years,list_df in dictionaries.items()
                if len(list_df)>0
            }
            for station,dictionaries in dict_per_station.items()
        }
    def concat_dataframes_according_station(self,dict_per_station:dict[str,dict[str,pd.DataFrame]])->dict[str,pd.DataFrame]:
        result={}
        for station,dictionaries in dict_per_station.items():
            dfs=[]
            for years,dataframe in dictionaries.items():
                temp=dataframe.assign(anio=years)
                dfs.append(temp)
            if len(dfs)!=0:
                result[station]=pd.concat(dfs,ignore_index=True)
        return result
    def concat_dataframe_all(self,dict_per_station:dict[str,pd.DataFrame])->pd.DataFrame:
        dfs=[]
        for station,dataframe in dict_per_station.items():
            temp=dataframe.assign(station=station)
            dfs.append(temp)
        return pd.concat(dfs,ignore_index=True)
    def filter_by_variable(self,df:pd.DataFrame,variable:str="")->pd.DataFrame:
        if len(self.variable)==0:
            self.variable=variable
        df_result = df.filter(items=["anio","MES","DIA","HORA", self.variable,"station"])
        return df_result
    def save_cache(self,df:pd.DataFrame):
        df.to_csv(self.out,index=False)

    def run_pipeline(self, tipo="horarios"):
        d = self.read_and_join_dataframes(tipo=tipo)
        d = self.join_datasets_according_variable(d)
        d = self.concat_dataframes_according_years(d)
        d = self.concat_dataframes_according_station(d)
        d = self.concat_dataframe_all(d)
        d = self.filter_by_variable(d)
        return d

################
#              #
#   Etapa 2    #
#              #
################


def main():
    etapa = first_stage("data/Salamanca", "data/processed/poblacion_completa.csv", "pm25")
    create_folder("data/processed/dataframe")
    df_fijo = etapa.run_pipeline(tipo="horarios")
    etapa.save_cache(df_fijo)
main()
