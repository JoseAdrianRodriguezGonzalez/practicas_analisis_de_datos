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
                
  ##########################

#   Etapa 4                #
#   Prueba                 #
#   wilcoxon para          #
#   politica ambiental     #

############################
class analysis_4:
    def __init__(self,dataframe_complete:dict[str,pd.DataFrame]):
        self.data=dataframe_complete["politica"]
    def prueba_comparativa(self)->pd.DataFrame:
        diff=self.data["pm25_despues"]-self.data["pm25_antes"]
        stat,pvalue=stats.wilcoxon(diff)
        t_stat,p_t=stats.ttest_rel(self.data["pm25_despues"],self.data["pm25_antes"])
        return pd.DataFrame([
            {
                "w_stat":stat,
                "w_p":pvalue,
                "t_stat":t_stat,
                "t_p":p_t
            }
        ])
                        
  ##########################

#   Etapa 5             #
#   Prueba              #
#   ANOVA de un factor  #

############################
class analysis_5:
    def __init__(self,dataframe_complete:dict[str,pd.DataFrame]) :
        self.data=dataframe_complete
    def check_homogeneity(self,column="pm25"):
        groups=["A","B","C"]
        data_groups=[self.data[sample][column].dropna() for sample in groups]
        stat,p=stats.levene(*data_groups)
        return pd.DataFrame([{
            "levene_stat":float(stat),
            "levene_p":float(p),
            "equal_variance":bool(p>=0.05)
        }])
    def check_normality(self,column="pm25"):
        groups=["A","B","C"]
        results={}
        for k in groups:
            stat,p=stats.shapiro(self.data[k][column].dropna())
            results[k]={
                "shapiro_stat":float(stat),
                "shapiro_p":float(p),
                "normal":bool(p>=0.05)
            }
        return pd.DataFrame.from_dict(results,orient="index")

    def anova_one_way(self,column="pm25"):
        groups=["A","B","C"]
        data_groups=[self.data[sample][column].dropna() for sample in groups]
        stat,p=stats.f_oneway(*data_groups)
        return pd.DataFrame([{
            "f_stat":float(stat),
            "f_p":float(p),
            "reject_HO":bool(p<0.05)
        }])
    def get_residuals(self,column="pm25"):
        groups=["A","B","C"]
        residuals={}
        for k in groups:
            values=self.data[k][column].dropna()
            mean=values.mean()
            residuals[k]=values-mean
        return residuals
    def check_residual_normality(self,column="pm25"):
        residuals=self.get_residuals(column)
        results={}
        for k,res in residuals.items():
            stat,p=stats.shapiro(res)
            results[k]={
                "shapiro_stat":float(stat),
                "shapiro_p":float(p),
                "normal":bool(p>=0.05)
            }
        return pd.DataFrame.from_dict(results,orient="index")
    def full_analysis(self,column="pm25"):
        normality=self.check_normality(column)
        homogenity=self.check_homogeneity(column)
        anova=self.anova_one_way(column)
        residual_normality=self.check_residual_normality(column)
        return {
            "normality":normality,
            "homogenity":homogenity,
            "anova":anova,
            "residual_normality":residual_normality
        }

  ##########################

#   Etapa 6             #
#   Prueba              #
#   ANOVA de un factor  #

############################
import scikit_posthocs as sp
class analysis_6:
    def __init__(self,dataframe_complete:dict[str,pd.DataFrame]) :
        self.data=dataframe_complete
    def kruskal_wallis(self,column="pm25"):
        groups=["A","B","C"]
        data_groups=[self.data[sample][column].dropna() for sample in groups]
        stat,p=stats.kruskal(*data_groups)
        return pd.DataFrame([
            {
                "h_stat":float(stat),
                "p_value":float(p),
                "reject_H0":bool(p<0.05)
            }
        ])
    def dunn_posthocs(self,column="pm25",p_adjust="bonferroni"):
        data=[]
        group=["A","B","C"]
        for k in group:
            temp=pd.DataFrame({
                "value":self.data[k][column].dropna(),
                "group":k
            })
            data.append(temp)
        full_df=pd.concat(data)
        posthoc=sp.posthoc_dunn(
            full_df,val_col="value",
            group_col="group",
            p_adjust=p_adjust
        )
        return posthoc 
    def full_analysis(self,column="pm25"):
        kruskal=self.kruskal_wallis(column)
        dunn=self.dunn_posthocs(column)
        return {
            "kruskal":kruskal,
            "dunn":dunn
        }
   ##########################

#   Etapa 7             #
#   Prueba              #
#   chi cuadrada        #

############################

class analysis_7:
    def __init__(self,dataframe_complete:dict[str,pd.DataFrame],groups:list[str]=["A","B","C"],column:str="pm25"):
        self.list_groups=groups
        self.data=dataframe_complete
        self.column=column
    def create_contigency_table(self):
        table={}
        for g in self.list_groups:
            df=self.data[g]
            classification=df[self.column].apply(
                lambda x: "si" if x>=50 else "no"
            )
            counts=classification.value_counts()
            table[g]={
                "si":int(counts.get("si",0)),
                "no":int(counts.get("no",0))
            }
        return pd.DataFrame.from_dict(table,orient="index")
            
    def chi_square_test(self):
        table=self.create_contigency_table()
        chi2,p,dof,expected=stats.chi2_contingency(table)
        expected_df = pd.DataFrame(
           expected,
            index=table.index,
            columns=table.columns
        )
        result={
            "chi2": float(chi2),
            "p_value":float(p),
            "dof":int(dof),
            "reject_H0":bool(p<0.05),
        }
        return{
            "frequency":expected_df,
            "contingency":table,
            "result":pd.DataFrame([result]) 
        }
####################

#   Etapa 8        #
#   dashboard      #

####################

class analysis_8:
    def __init__(self,dataframe_complete:dict[str,pd.DataFrame],groups:list[str]=["poblacion_completa","A","B","C"],column:str="pm25"):
        self.list_groups=groups
        self.data=dataframe_complete
        self.column=column
    def dashboard(self,out="plot"):
        groups=self.list_groups
        fig=plt.figure(figsize=(16,12))
        #boxplot
        ax1=plt.subplot(2,2,1)
        data_box=[self.data[g][self.column].dropna() for g in groups]
        labels=groups
        ax1.boxplot(data_box,tick_labels=labels)
        ax1.set_title("Boxplots comparativos")
        # q-q plots
        ax2=plt.subplot(2,2,2)
        for g in groups:
            if "poblacion_completa" in g:
                continue
            stats.probplot(self.data[g][self.column].dropna(),dist="norm",plot=ax2)
        ax2.set_title("Q-Q plots (todas las muestras")
        #histograma densidad
        ax3=plt.subplot(2,2,3)
        colors = [
            "#1f77b4",  # azul fuerte
            "#2ca02c",  # verde
            "#d62728",  # café
            "#bcbd22",  # amarillo oscuro
            "#7f7f7f",  # gris
            "#17becf"   # cyan
        ] 
        population=self.data["poblacion_completa"][self.column].dropna()
        ax3.hist(population,bins=30,density=True, alpha=0.4,label="Poblacion",color=colors[0])
        i=1
        for g in groups:
            if "poblacion_completa" in g:
                continue
            ax3.hist(self.data[g][self.column].dropna(),bins=30,density=True,alpha=0.4,label=f"Muestra {g}",color=colors[i%len(colors)])
            i+=1
        ax3.set_title("Distribucion de densidad")
        ax3.legend()

        #Medias + IC 95
        ax4 = plt.subplot(2,2,4)
        means = []
        ci_lower = []
        ci_upper = []
        plot_groups = [g for g in groups if g != "poblacion_completa"]
        for g in plot_groups:

            data = self.data[g][self.column].dropna()
            mean = np.mean(data)
            sem = stats.sem(data)
            ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
            means.append(mean)
            ci_lower.append(mean - ci[0])
            ci_upper.append(ci[1] - mean)

        ax4.errorbar(plot_groups, means, yerr=[ci_lower, ci_upper], fmt='o', capsize=5)
        ax4.set_title("Medias con IC 95%")        

        create_folder(out)
        file_name=f"dashborad_{self.column}.png"
        full_path=os.path.join(out,file_name)
        plt.savefig(full_path,dpi=300)
        plt.tight_layout()
        plt.show()

####################

#     Etapa 9      #
#     ranking      #

####################


class analysis_9:
    def __init__(self,dataframe_complete:dict[str,pd.DataFrame],
                 groups:list[str]=["A","B","C"],
                 column:str="pm25"):

        self.data=dataframe_complete
        self.groups=groups 
        self.column=column
    def mean_error(self):
        pop_mean=self.data["poblacion_completa"][self.column].mean()
        return { g: abs(self.data[g][self.column].mean()-pop_mean) for g in self.groups}
    def ks_test(self):
        pop=self.data["poblacion_completa"][self.column].dropna()
        scores={}
        for g in self.groups:
            stat,p=stats.ks_2samp(pop,self.data[g][self.column].dropna())
            scores[g]=p
        return scores
    def chi_square(self):
        scores={}
        for g in self.groups:
            data=self.data[g][self.column]
            cat=data.apply(lambda x:"si" if x>=50 else "no")
            obs=cat.value_counts()

            pop=self.data["poblacion_completa"][self.column]
            pop_cat=pop.apply(lambda x:"si" if x>=50 else "no")
            exp=pop_cat.value_counts(normalize=True)
            expected=exp*len(cat)
            chi2,p=stats.chisquare(f_obs=obs,f_exp=expected)
            scores[g]=p
        return scores
    def compute_ranking(self):
        mean_err=self.mean_error()
        ks=self.ks_test()
        chi=self.chi_square()
        results=[]
        for g in self.groups:
            mean_score=-mean_err[g]
            ks_score=ks[g]
            chi_score=chi[g]
            final_score=mean_score+ks_score+chi_score
            results.append({
                "strategy":g,
                "mean_error":mean_err[g],
                "ks_pvalue":ks[g],
                "chi_pvalue":chi[g],
                "final_score":final_score
            })
        df=pd.DataFrame(results)
        df["rank"]=df["final_score"].rank(ascending=False,method="dense")
        return df.sort_values("rank")

  ##########################

#   Etapa 10             #
#   Prueba              #
#   ANOVA de un factor  #

############################
def auditar_muestreo(data_frame:pd.DataFrame,estrategias_dict:dict[str,pd.DataFrame],columna_medicion:str="pm25",limite_saludable:int=50):
    data={"poblacion_completa":data_frame}
    data.update(estrategias_dict)
    #prueba t
    prueba_t=analysis_2(data)
    media_poblacional=data_frame[columna_medicion].mean()
    df_t=prueba_t.complete_table(list_files=["A","B","C"],mu=media_poblacional)
    mean_muestreal = df_t["mean"].to_dict()
    pvalue_t = df_t["p_value"].to_dict()
    # prueb chi 
    pruebas_restantes=analysis_9(data)
    chi=pruebas_restantes.chi_square()
    ks=pruebas_restantes.ks_test()
    results=[]
    for g in pruebas_restantes.groups:
        decision="Buena representacion"
        score = (int(pvalue_t[g] > 0.05) +int(ks[g] > 0.05) +int(chi[g] > 0.05))
        if (pvalue_t[g]<0.05)or (ks[g]<0.05) or (chi[g]<0.05):
            decision="Posible sesgo"
        results.append({
            "estrategia":g,
            "media_muestreal":mean_muestreal[g],
            "media_poblacional":media_poblacional,
            "p_value_t":pvalue_t[g],
            "p_value_ks":ks[g],
            "p_value_chi":chi[g],
            "recomendacion":decision ,
            "score":score
        })
    df_final =pd.DataFrame(results)
    return df_final
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
    diccionario_dataframes={}
    preprocesamiento(src_poblacion_completa,src_A,src_B,src_C,src_politica)
    #Primer punto 
    a1=analysis_1("processed")
    df_statistics_complete_population=a1.calculate_statistics("poblacion_completa")
    diccionario_dataframes["a1"]=df_statistics_complete_population
    print(df_statistics_complete_population)
    a1.visualize_distributions("poblacion_completa",title="Poblacion Completa")
    stats,p=a1.normality_shapiro("poblacion_completa")

    #segundo punto 
    a2=analysis_2(a1.dataframes)
    df_ttest_sample=a2.complete_table(list_files=["A","B","C"],mu=df_statistics_complete_population["mean"])
    print(df_ttest_sample)
    diccionario_dataframes["ttest_sample"]=df_ttest_sample

    #tercer punto
    a3=analysis_3(a1.dataframes)
    df_ttest_independent=a3.create_table()
    diccionario_dataframes["ttest_independent"]=df_ttest_independent
    print(df_ttest_independent)
    #cuarto punto
    a4=analysis_4(a1.dataframes)
    df_wt_test=a4.prueba_comparativa()
    diccionario_dataframes["wilcoxon_pol"]=df_wt_test
    print(df_wt_test)
    #quinto punto
    a5=analysis_5(a1.dataframes)
    dictionary_results=a5.full_analysis()
    for titulo, df in dictionary_results.items():
        diccionario_dataframes[titulo]=df
        print(f"****{titulo}****")
        print(df)
    a6=analysis_6(a1.dataframes)
    dictionary_results_a6=a6.full_analysis()
    for titulo, df in dictionary_results_a6.items():
        diccionario_dataframes[titulo]=df
        print(f"****{titulo}****")
        print(df)
    #Fase 7
    a7=analysis_7(a1.dataframes)
    dictionary_results_a7=a7.chi_square_test()
    for titulo, df in dictionary_results_a7.items():
        diccionario_dataframes[titulo]=df
        print(f"****{titulo}****")
        print(df)
    #fase 8 
    a8=analysis_8(a1.dataframes)
    a8.dashboard()
    #fase 9
    a9=analysis_9(a1.dataframes)
    df_ranking=a9.compute_ranking()
    
    diccionario_dataframes["ranking"]=df_ranking
    print(df_ranking)
    df_final=auditar_muestreo(a1.dataframes["poblacion_completa"],{k:v for k,v in a1.dataframes.items() if k in ["A","B","C"]},
                              columna_medicion="pm25",limite_saludable=50)
    print(df_final)
    diccionario_dataframes["auditar"]=df_final 
    ###Final
    ##Crear las tablas
    save_tables(diccionario_dataframes)
def save_tables(dict_dfs,folder="tables"):
    create_folder(folder)
    for name,df in dict_dfs.items():
        with open(f"{folder}/{name}.tex","w") as f:
            f.write(df.to_latex()) 
main()
