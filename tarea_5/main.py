import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency,chisquare
import matplotlib.pyplot as plt 
def problema1(src):
    print("---"*20)
    print("Problema 1") 
    print("---"*20)
    df= pd.read_csv(src)
    tab=pd.crosstab(df["error_type"],df["predicted_class"])
    print(tab)
    tab_latex=tab.to_latex(index=True)
    print(tab_latex)
    chi2,p,dof,expected=chi2_contingency(tab)
    results={"chi":chi2,
             "p-value":p,
             "dof":dof}
    df_results=pd.DataFrame([results])
    table_latex=df_results.to_latex(index=False)
    print(table_latex)
    print(f"Chi2: {chi2}")
    print(f"p-value: {p}")
    print(f"dof: {dof}")
    print(f"Chi2 expected: {expected}")

def problema2(src):
    df=pd.read_csv(src)
    print("---"*20)
    print("Problema 2") 
    print("---"*20)
    print(df)
    tab=pd.crosstab(df["age_group"],df["accepted"])
    print(tab)
    chi2,p,dof,expected=chi2_contingency(tab)
    n=tab.values.sum()
    r,c=tab.shape
    crammers_v=np.sqrt(chi2/n*(min(r,c)-1))
    print(f"Chi2: {chi2}")
    print(f"p-value: {p}")
    print(f"dof: {dof}")
    print(f"Crammer's V: {crammers_v}")
    results={"chi":chi2,
             "p-value":p,
             "dof":dof,
             "Crammer":crammers_v}
    df_results=pd.DataFrame([results])
    table_latex=df_results.to_latex(index=False)
    print(table_latex)
def problema3(src):
    df=pd.read_csv(src)
    print("---"*20)
    print("Problema 3") 
    print("---"*20)
    print(df)

    tab=pd.crosstab(df["sensor_type"],df["failure_type"])
    print(tab)
    chi2,p,dof,expected=chi2_contingency(tab)
    n=tab.values.sum()
    r,c=tab.shape
    crammers_v=np.sqrt(chi2/n*(min(r,c)-1))
    print(f"Chi2: {chi2}")
    print(f"p-value: {p}")
    print(f"dof: {dof}")
    print(f"Crammer's V: {crammers_v}")
    results={"chi":chi2,
             "p-value":p,
             "dof":dof,
             "crammer":crammers_v}
    df_results=pd.DataFrame([results])
    table_latex=df_results.to_latex(index=False)
    print(table_latex)
    tab_latex=tab.to_latex(index=True)
    print(tab_latex)
def problema4(src):
    df=pd.read_csv(src)
    print("---"*20)
    print("Problema 4") 
    print("---"*20)
    print(df)
    tab=pd.crosstab(df["company"],df["architecture"])
    print(tab)
    chi2,p,dof,expected=chi2_contingency(tab)
    n=tab.values.sum()
    r,c=tab.shape
    crammers_v=np.sqrt(chi2/n*(min(r,c)-1))
    print(f"Chi2: {chi2}")
    print(f"p-value: {p}")
    print(f"dof: {dof}")
    print(f"Crammer's V: {crammers_v}")
    results={"chi":chi2,
             "p-value":p,
             "dof":dof,
             "crammer":crammers_v}
    df_results=pd.DataFrame([results])
    table_latex=df_results.to_latex(index=False)
    print(table_latex)
    residuals = (tab - expected) / np.sqrt(expected)
    tab_latex=tab.to_latex(index=True)
    print(tab_latex)
    res_df = pd.DataFrame(residuals, 
                      index=tab.index, 
                      columns=tab.columns)

    print(res_df)
    print(res_df.to_latex(index=True))
def problema5(src):
    df=pd.read_csv(src)
    print("---"*20)
    print("Problema 5") 
    print("---"*20)
    print(df)
    obs=df["observed"].values
    exp=df["expected_count"].values
    chi2,p_value=chisquare(f_obs=obs,f_exp=exp)
    print("Chi2:", chi2)
    print("p-value:", p_value)
def problema6(src):
    df=pd.read_csv(src)
    print("---"*20)
    print("Problema 6") 
    print("---"*20)
    obs=df["requests"].values 
    
    n=obs.sum()
    k=len(obs)
    expected=np.full(k,n/k)
    chi2,p_value=chisquare(f_exp=expected,f_obs=obs)
    print("Chi2:", chi2)
    print("p-value:", p_value)
    plt.bar(range(len(obs)),obs) 
    plt.xlabel("Categoria")
    plt.ylabel("Frecuencias")
    plt.title("Frecuencias observadas")
    plt.tight_layout()
#    plt.show()
def problema7(src):
    print("---"*20)
    print("Problema 7") 
    print("---"*20)
    df=pd.read_csv(src)
    tab=pd.crosstab(df["algorithm"],df["stability"])
    print(tab)
    chi2,p,dof,expected=chi2_contingency(tab)
    print(f"Chi2: {chi2}")
    print(f"p-value: {p}")
    print(f"dof: {dof}")
    residuals = (tab - expected) / np.sqrt(expected)
    res_df = pd.DataFrame(residuals, 
                      index=tab.index, 
                      columns=tab.columns)


    print(res_df.to_latex(index=True))
    results={"chi":chi2,
             "p-value":p,
             "dof":dof,
             }
    df_results=pd.DataFrame([results])
    table_latex=df_results.to_latex(index=False)
    print(table_latex)
def problema8(src):
    print("---"*20)
    print("Problema 8") 
    print("---"*20)
    df=pd.read_csv(src)
    tab = pd.crosstab(df["platform"], df["sentiment"])
    chi2, p, dof, expected = chi2_contingency(tab)
    print("Chi2:", chi2)
    print("p-value:", p)
    residuals = (tab - expected) / np.sqrt(expected)
    res_df = pd.DataFrame(residuals,
                      index=tab.index,
                      columns=tab.columns)

    print(res_df)
    print(res_df.to_latex(index=True))
    results={"chi":chi2,
             "p-value":p,
             "dof":dof,
             }
    df_results=pd.DataFrame([results])
    table_latex=df_results.to_latex(index=False)
    print(res_df.to_latex(index=True))
    results={"chi":chi2,
             "p-value":p,
             "dof":dof,
             }
    df_results=pd.DataFrame([results])
    table_latex=df_results.to_latex(index=False)
    print(table_latex)
    print(table_latex)
def problema9(src):
    print("---"*20)
    print("Problema 9") 
    print("---"*20)
    df=pd.read_csv(src)
    tab = pd.crosstab(df["region"], df["approved"])
    chi2, p, dof, expected = chi2_contingency(tab)
    n=tab.values.sum()
    r,c=tab.shape
    crammers_v=np.sqrt(chi2/n*(min(r,c)-1))
    print(f"Chi2: {chi2}")
    print(f"p-value: {p}")
    print(f"dof: {dof}")
    print(f"Crammer's V: {crammers_v}")
    results={"chi":chi2,
             "p-value":p,
             "dof":dof,
             "Crammer":crammers_v}
    df_results=pd.DataFrame([results])
    table_latex=df_results.to_latex(index=False)
    print(table_latex)
def problema10(src):
    print("---"*20)
    print("Problema 10") 
    print("---"*20)
    df=pd.read_csv(src)
    obs=df["generated_class"].value_counts().sort_index()
    print(obs)

    probs=np.array([0.25,0.2,0.18,0.22,0.15])
    n=obs.sum()
    expected=probs*n
    chi2,p=chisquare(f_exp=expected,f_obs=obs)
    print("Chi2:", chi2)
    print("p-value:", p)
def main():
    problema1("datos/1_independencia_error_vs_clase.csv")
    problema2("datos/2_independencia_aceptacion_vs_edad.csv")
    problema3("datos/3_independencia_falla_vs_sensor.csv")
    problema4("datos/4_homogeneidad_arquitectura_por_empresa.csv")
    problema5("datos/5_bondad_ajuste_confusion_teorica.csv")
    problema6("datos/6_bondad_ajuste_trafico_horario.csv")
    problema7("datos/7_independencia_clustering_vs_estabilidad.csv")
    problema8("datos/8_homogeneidad_sentimiento_por_plataforma.csv")
    problema9("datos/9_independencia_credito_por_region.csv")
    problema10("datos/10_bondad_ajuste_modelo_generativo.csv")
if __name__=="__main__":
    main()
