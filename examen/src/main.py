import pandas as pd 
import os
##########################

#    preprocesamiento    #

###########################
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

#####################

#    creacion de    #
#    datasets       #

#####################
################
#              #
#   Etapa 2    #
#              #
################
import numpy as np 
class second_stage:
    def __init__(self, src_dataframe:str) -> None:
        self.src=src_dataframe
    def create_random_sample(self,out_dataframe:str,n:int=1740):
        df=pd.read_csv(self.src)
        df_sample=df.sample(n,random_state=42)
        df_sample.to_csv(out_dataframe,index=False)
    def  create_random_sample_prop(self,out_dataframe:str,n:int=1740):
        df=pd.read_csv(self.src)
        total=len(df)
        samples_per_station=(
            df["station"].value_counts(normalize=True)*n
        ).round().astype(int)
        df_prop=pd.concat([
            df[df["station"]==station].sample(
                n=min(n,len(df[df["station"]==station])),random_state=42
            )
            for station,n in samples_per_station.items()
        ])
        df_prop.to_csv(out_dataframe,index=False)
    def create_systematic_random(self,out_dataframe:str,n:int=1740):
        df=pd.read_csv(self.src)
        df=df.sort_values(["station","anio","MES","DIA","HORA"])
        total=len(df)
        proportions=df["station"].value_counts(normalize=True)
        samples_per_station=(proportions*n).round().astype(int)
        diff=n-samples_per_station.sum()
        if diff!=0:
            idx_max=samples_per_station.idxmax()
            samples_per_station[idx_max]+=diff
        result=[]
        for station, n_station in samples_per_station.items():
            group=df[df["station"]==station]
            if len(group)==0 or n_station==0:
                continue 
            step=len(group)/n_station
            indices=(np.arange(n_station)*step).astype(int)
            sampled=group.iloc[indices]
            result.append(sampled)
        df_final=pd.concat(result).reset_index(drop=True)
        df_final.to_csv(out_dataframe,index=False)
        return df_final
    def create_politic_sample(self, out_dataframe:str,n:int=1740):
        df=pd.read_csv(self.src)
        df_antes = df[(df["anio"] >= 2017) & (df["anio"] <= 2020)]
        df_despues = df[(df["anio"] >= 2021) & (df["anio"] <= 2023)]
        df_antes["key"] = df_antes["MES"].astype(str) + "_" + df_antes["DIA"].astype(str) + "_" + df_antes["HORA"].astype(str) + "_" + df_antes["station"]
        df_despues["key"] = df_despues["MES"].astype(str) + "_" + df_despues["DIA"].astype(str) + "_" + df_despues["HORA"].astype(str) + "_" + df_despues["station"]
        df_merge = pd.merge(df_antes, df_despues, on=["key","station"], suffixes=("_antes", "_despues"))
        df_pairs=df_merge[["station","pm25_antes","pm25_despues"]]
        df_pairs=df_pairs.sample(n=n,random_state=42)
        df_pairs["reduccion_porcentual"]=(
            (df_pairs["pm25_antes"]-df_pairs["pm25_despues"])/df_pairs["pm25_antes"]
        )*100
        df_pairs.to_csv(out_dataframe,index=False)

########################

#     Etapa 1          #
# Carga y exploracion  #
# Incial               #

########################
import matplotlib.pyplot as plt 
import scipy.stats as stats 
class analysis_1:
    def __init__(self,root_source:str) -> None:
        self.src=root_source
        self.load_files()
    def load_files(self):
        self.dataframes={
            os.path.splitext(file)[0]:pd.read_csv(os.path.join(self.src,file)) 
            for file in sorted(os.listdir(self.src))}
    def calculate_statistics_all(self)->pd.DataFrame:
        stats_dict= {
            file:{
                "mean":dataframe["pm25"].mean(),
                "median":dataframe["pm25"].median(),
                "std":dataframe["pm25"].std(),
                "IQR":dataframe["pm25"].quantile(q=0.75)-dataframe["pm25"].quantile(q=0.25)
            }
            for file,dataframe in self.dataframes.items()
            if "politica" in file
        }
        df_stats=pd.DataFrame(stats_dict).T
        return df_stats
    def calculate_statistics(self,dataframe:str)->pd.DataFrame:
        stats_dict={
                "mean":self.dataframes[dataframe]["pm25"].mean(),
                "median":self.dataframes[dataframe]["pm25"].median(),
                "std":self.dataframes[dataframe]["pm25"].std(),
                "IQR":self.dataframes[dataframe]["pm25"].quantile(q=0.75)-self.dataframes[dataframe]["pm25"].quantile(q=0.25)
        }
        df_stats=pd.DataFrame([stats_dict],index=[dataframe])
        return df_stats 
    def visualize_distributions(self,file:str,column:str="pm25",title="Descripcion",out:str="plots"):
        data=self.dataframes[file][column].dropna()
        fix,axes=plt.subplots(1,3,figsize=(18,5))
        axes[0].hist(data,bins=30)
        axes[0].set_title(f"{title} - Histograma")
        
        #boxplot 
        axes[1].boxplot(data,vert=True)
        axes[1].set_title(f"{title} - BoxPlot")

        stats.probplot(data,dist="norm",plot=axes[2])
        axes[2].set_title(f"{title} - Q-Q Plot")
        create_folder(out)
        file_name=f"{file}_{column}.png"
        full_path=os.path.join(out,file_name)
        plt.savefig(full_path,dpi=300)
        plt.tight_layout()
        plt.show()
        print(f"Gráfica guardada en: {full_path}")
    def normality_shapiro(self,file:str,column:str="pm25",alpha:float=0.05)->tuple[float,float]:
        data=self.dataframes[file][column].dropna()
        data_sampled=data.sample(5000,random_state=42)
        stat,p =stats.shapiro(data_sampled)
        print(f"Estadistico : {stat}")
        print(f"Estadistico : {p:.5f}")
        if p>alpha :
            print("No se rechaza H0: por lo tanto es distribucion normal")
        else:
            print("Se rechaza H0, por lo tanto no es normal")
        return stat,p 
#################

#   Etapa 2     #
#   Prueba t    #
#   para una    #
#   muestra     #

#################
class analysis_2:
    def __init__(self,dataframe_complete:dict[str,pd.DataFrame]):
        self.data=dataframe_complete
    def one_sample_t_test(self,file:str,column:str="pm25",mu:float=45.2):
        data=self.data[file][column].dropna()
        mean=data.mean()
        t_stat,p_value=stats.ttest_1samp(data,popmean=mu)
        confidence=0.95
        n=len(data)
        se=stats.sem(data)
        h=se*stats.t.ppf((1+confidence)/2,n-1)
        ci_lower=mean-h
        ci_upper=mean+h
        reject=p_value<0.05 
        conclusion = "Se rechaza H0" if reject else "No se rechaza H0"
        return {
            "mean":mean,
            "t_stat":t_stat,
            "p_value":p_value,
            "ci_lower":ci_lower,
            "ci_uppper":ci_upper,
            "reject_H0":reject,
            "conclusion":conclusion 
        }
    def complete_table(self,list_files:list[str],column:str="pm25",mu:float=45.2):
        results=[self.one_sample_t_test(file,column,mu) for file in list_files]
        return pd.DataFrame(results,index=list_files)
  ######################

#   Etapa 3            #
#   Prueba t           #
#   para muestras      #
#   independientes     #

########################
class analysis_3:
    def __init__(self,dataframe_complete:dict[str,pd.DataFrame]) :
        self.data=dataframe_complete
    def independent_t_test(self,file1:str,file2:str,column:str="pm25"):
        data1=self.data[file1][column]
        data2=self.data[file2][column]
        stat_leven,p_levene=stats.levene(data1,data2)
        equal_var=p_levene>=0.05
        t_stat,pvalue=stats.ttest_ind(data1,data2,equal_var=equal_var)
        type_test="t-test" if equal_var else "Welch"
        return {
            "mean_1":float(data1.mean()),
            "mean_2":float(data2.mean()),
            "levene_p":float(p_levene),
            "type_test":type_test,
            "t_stat":float(t_stat),
            "p_value":float(pvalue),
            "reject_H0":bool(pvalue<0.05)
        }
    def create_table(self):
        pairs=[("A","B"),("A","C"),("B","C")]
        results=[]
        for f1, f2 in pairs:
            res = self.independent_t_test(f1, f2)
        
            results.append({
                "comparison": f"{f1} vs {f2}",
                "mean_1": res["mean_1"],
                "mean_2": res["mean_2"],
                "levene_p": res["levene_p"],
                "type_test":res["type_test"],
                "t_stat": res["t_stat"],
                "p_value": res["p_value"],
                "reject_H0": res["reject_H0"],
                "conclusion": "Diferentes" if res["reject_H0"] else "No diferentes"
            })
        
        return pd.DataFrame(results)
                
def preprocesamiento(src_poblacion_completa,src_A,src_B,src_C,src_politica):
    """
    Primera etapa 
    """
    etapa = first_stage("data/Salamanca", src_poblacion_completa, "pm25")
    create_folder("processed/")
    df_fijo = etapa.run_pipeline(tipo="horarios")
    df_fijo=df_fijo.dropna()
    etapa.save_cache(df_fijo)

    """
        Segunda etapa 
    """
    etapa_2=second_stage(src_poblacion_completa)
    etapa_2.create_random_sample(src_A)
    etapa_2.create_random_sample_prop(src_B)
    etapa_2.create_systematic_random(src_C)
    etapa_2.create_politic_sample(src_politica)

def main():
    src_poblacion_completa="processed/poblacion_completa.csv"
    src_A="processed/A.csv"
    src_B="processed/B.csv"
    src_C="processed/C.csv"
    src_politica="processed/politica.csv"
#    preprocesamiento(src_poblacion_completa,src_A,src_B,src_C,src_politica)
    #Primer punto 
    a1=analysis_1("processed")
    df_statistics_complete_population=a1.calculate_statistics("poblacion_completa")
    print(df_statistics_complete_population)
    a1.visualize_distributions("poblacion_completa",title="Poblacion Completa")
    stats,p=a1.normality_shapiro("poblacion_completa")
    #segundo punto 
    a2=analysis_2(a1.dataframes)
    df_ttest_sample=a2.complete_table(list_files=["A","B","C"],mu=df_statistics_complete_population["mean"])
    print(df_ttest_sample)
    #tercer punto
    a3=analysis_3(a1.dataframes)
    df_ttest_independent=a3.create_table()
    print(df_ttest_independent)
main()
