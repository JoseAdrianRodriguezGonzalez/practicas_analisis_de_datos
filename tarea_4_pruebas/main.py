import pandas as pd
import numpy as np 
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib.pyplot as plt 
def tester_2sample(func,sample1,sample2,**kwargs):
    stats,p=func(sample1,sample2,**kwargs)
    H0=True if p>=0.05 else False 
    mensaje="Se rechaza la hipotesis nula" if p <0.05 else "Se acepta la hipotesis nula"
    print(mensaje)
    if func.__name__ == 'mannwhitneyu':
        n1, n2 = len(sample1), len(sample2)
        mu_U = n1 * n2 / 2
        sigma_U = ((n1 * n2 * (n1 + n2 + 1)) / 12) ** 0.5
        Z = (stats - mu_U) / sigma_U
    elif func.__name__=="wilcoxon":
        n=len(sample1)
        mu = n*(n+1)/4
        sigma = np.sqrt(n*(n+1)*(2*n+1)/24)
        Z = (stats - mu) / sigma
    else:
        Z = None
    return {"estadistico":stats,
            "p-value":p,
            "H0":H0,
            "Z":Z}
def delta_cliff(A,B):
    n1,n2=len(A),len(B)
    more,less=0,0
    for a in A:
        for b in B:
            if a<b:
                less+=1 
            elif a>b:
                more+=1
    return (more-less)/(n1*n2)
def r_value(A,B,Z):
    n1,n2=len(A),len(B)
    n=n1+n2
    r=Z/(n1+n2)**(1/2) 
    return r 
def save_box_plot(A,B,labels,title,axis_labels,src):
    plt.figure()
    plt.boxplot([A,B],tick_labels=labels)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.tight_layout() 
    plt.savefig(src,dpi=300)
    plt.close() 
def save_line_scatter(A,B,title,axis_labels,src):
    plt.figure()
    plt.scatter(A,B)
    maximo=max(max(A),max(B))
    plt.plot([0,maximo],[0,maximo]) 
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.tight_layout() 
    plt.savefig(src,dpi=300)
    plt.close() 
def problema_1(src):
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 1")
    print("-"*23)
    A=df[df["algoritmo"]=="A"]["f1_score"]
    B=df[df["algoritmo"]=="B"]["f1_score"]
    results=tester_2sample(mannwhitneyu,A,B,alternative="less")    
    delta=delta_cliff(A,B)
    r=r_value(A,B,results["Z"])
    print(f"U:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    print(f"delta: {delta}")
    print(f"r: {r}")
    results["r"]=r 
    results["delta"]=delta
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
def problema_2(src):
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 2")
    print("-"*23)
    A=df[df["marca"]=="A"]["activaciones"]
    B=df[df["marca"]=="B"]["activaciones"]
    results=tester_2sample(mannwhitneyu,A,B,alternative="two-sided")
    save_box_plot(A,B,["A","B"],"Boxplot de sensores",["sensores","mediciones"],"boxplot.png")
    print(f"U:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
def problema_3(src):
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 3")
    print("-"*23)
    A=df[df["region"]=="Norte"]["salario_anual_mxn"]
    B=df[df["region"]=="Sur"]["salario_anual_mxn"] 
    results=tester_2sample(mannwhitneyu,A,B,alternative="less")    
    print(f"U:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
def problema_4(src):
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 4")
    print("-"*23)
    A=df[df["gpu"]=="A"]["tiempo_ms"]
    B=df[df["gpu"]=="B"]["tiempo_ms"] 
    results=tester_2sample(mannwhitneyu,A,B,alternative="less")    
    print(f"U:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    delta=delta_cliff(A,B)
    r=r_value(A,B,results["Z"])
    print(f"delta: {delta}")
    print(f"r: {r}")
    results["r"]=r 
    results["delta"]=delta
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
def problema_5(src):
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 5")
    print("-"*23)
    A=df[df["metodo"]=="X"]["mae"]
    B=df[df["metodo"]=="Y"]["mae"]
    results=tester_2sample(mannwhitneyu,A,B,alternative="two-sided")
    print(f"U:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    median_A=A.median()
    median_B=B.median()
    mensaje="El metodo X es mejor" if median_A<median_B else "el metodo Y es mejor"
    print(mensaje)
    results["median_A"]=median_A
    results["median_B"]=median_B
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
def problema_6(src):
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 6")
    print("-"*23)
    A=df["ctr_pre"]
    B=df["ctr_post"]
    results=tester_2sample(wilcoxon,A,B,alternative="less")    
    print(f"w:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    low_users=A<np.percentile(A,25)
    results_low_users=tester_2sample(wilcoxon,A[low_users],B[low_users],alternative="less")    
    print(f"w:{results_low_users["estadistico"]}")
    print(f"p-value:{results_low_users["p-value"]}")
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
    
    df_res_2=pd.DataFrame([results_low_users])
    latex_table_2=df_res_2.to_latex(index=False)
    print("Latex table")
    print(latex_table_2)
def problema_7(src):
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 7")
    print("-"*23)
    A=df["error_pre_m"]
    B=df["error_post_m"]
    results=tester_2sample(wilcoxon,A,B,alternative="greater")    
    print(f"w:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
    save_line_scatter(A,B,"Linea de error",["Pre","Pos"],"scatter.png")
def problema_8(src):
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 8")
    print("-"*23)
    A=df["consumo_pre_mJ"]
    B=df["consumo_post_mJ"]
    results=tester_2sample(wilcoxon,A,B,alternative="greater")    
    print(f"w:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    r=r_value(A,B,results["Z"])
    print(f"Efecto del cambio: {r}")
    results["r"]=r 
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
def problema_9(src): 
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema 9")
    print("-"*23)
    A=df["ssim_pre"]
    B=df["ssim_post"]
    results=tester_2sample(wilcoxon,A,B,alternative="less")    
    print(f"w:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    
    cantidad_buena=(B>A).sum()
    cantidad_mala=(B<A).sum()
    cantidad_igual=(A==B).sum()
    print(f"Cantidad de fotos con mejora: {cantidad_buena}")
    print(f"Cantidad de fotos con empeora: {cantidad_mala}")
    print(f"Cantidad de fotos sin mejora: {cantidad_igual}")
    results["Buena"]=cantidad_buena
    results["mala"]=cantidad_mala
    results["igual"]=cantidad_igual
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
def problema_10(src): 
    df=pd.read_csv(src)
    print("-"*23)
    print("Problema problema_10")
    print("-"*23)
    A=df["lat_pre_ms"]
    B=df["lat_post_ms"]
    results=tester_2sample(wilcoxon,A,B,alternative="greater")    
    print(f"w:{results["estadistico"]}")
    print(f"p-value:{results["p-value"]}")
    median_A=A.median()
    median_B=B.median()
    diferencias=A-B
    median_d=diferencias.median()
    mensaje="El backend estaba mejor antes" if median_A<median_B else "serverless mejoro"
    print(mensaje)
    results["median_A"]=median_A
    results["median_B"]=median_B
    results["median_d"]=median_d
    df_res=pd.DataFrame([results])
    latex_table=df_res.to_latex(index=False)
    print("Latex table")
    print(latex_table)
def main():
    problema_1("data/1_f1_mannwhitney_algos.csv")
    problema_2("data/2_pir_mannwhitney_marcas.csv")
    problema_3("data/3_salarios_mannwhitney_regiones.csv")
    problema_4("data/4_gpu_mannwhitney_inferencia.csv")
    problema_5("data/5_mae_mannwhitney_limpieza.csv")
    problema_6("data/6_ctr_wilcoxon_prepost.csv")
    problema_7("data/7_dron_wilcoxon_error_prepost.csv")
    problema_8("data/8_energia_wilcoxon_prepost.csv")
    problema_9("data/9_ssim_wilcoxon_prepost.csv")
    problema_10("data/10_api_wilcoxon_latencia_prepost.csv")
main()
