import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from scipy.stats import ttest_1samp,t, ttest_ind ,levene,ttest_rel,pearsonr,shapiro
def cohens_d_1sample(sample,mu0):
    mean=np.mean(sample)
    std=np.std(sample,ddof=1)
    return (mean-mu0)/std 
def cohens_d_2sample(sample1,sample2):
    mean1,mean2=np.mean(sample1),np.mean(sample2)
    var1,var2=np.var(sample1,ddof=1),np.var(sample2,ddof=1)
    n1,n2=len(sample1),len(sample2)
    sp=np.sqrt(((n1-1)*var1+(n2-1)*var2)/(n1+n2-2))
    return (mean1-mean2)/sp
def interpretaciones_cohen(d):
    d_inter=np.abs(d)
    if d_inter>=0 and d_inter<=0.2:
        print("Efecto despreciable")
    elif d_inter>0.2 and d_inter<=0.5:
        print("Efecto pequeño")
    elif d_inter >0.5 and d_inter<=0.8:
        print("Efecto moderado")
    elif d_inter>0.8 and d_inter<=1.2:
        print("Efecto grande")
    elif d_inter>=1.2 and d_inter<=2.0:
        print("Efecto muy grande")
    elif d_inter>2.0:
        print("Efecto enorme")
def correlacion_interpretacion(r):
    mensaje="inverso" if r<0 else "directo"
    print(f"Relacion {mensaje}")
    r=np.abs(r) 
    if r>=0 and r<=0.1:
        print("casi nulo")
    elif r>0.1 and r<=0.3:
        print("Débil")
    elif r>0.3 and r<=0.5:
        print("Moderado")
    elif r>0.5 and r<=0.7:
        print("Fuerte")
    elif r>=0.7 and r<=0.9:
        print("Efecto muy grande")
    elif r>0.9 and r<=1.0:
        print("Muy buena")
def tester_2sample(func,sample1,sample2):
    stats,p=func(sample1,sample2)
    H0=True if p<0.05 else False 
    mensaje="Se rechaza la hipotesis nula" if p <0.05 else "Se acepta la hipotesis nula"
    print(mensaje)
    return {"estadistico":stats,
            "p-value":p,
            "H0":H0}

def tester_1sample(func,sample1,mu=None,uni=False):
    if mu is None:
        stats,p=func(sample1)
    else:    
        stats,p=func(sample1,mu)
    p_tester=p/2 if uni else p
    if p_tester<0.05:
        print("Se rechaza la hipotesis nula")
        H0=False 
    else:
        print("Se acepta la hipotesis nula")
        H0=True 
    return {"estadistico":stats,
            "p-value":p_tester,
            "H0":H0}
def ic_2_samples(A,B,alpha,type_):
    n1,n2=len(A),len(B)
    mean1,mean2=np.mean(A),np.mean(B)
    var1,var2=np.var(A,ddof=1),np.var(B,ddof=1)
    diff=mean1-mean2
    if type_:
        sp=np.sqrt(((n1-1)*var1+(n2-1)*var2)/(n1+n2-2))
        SE=sp*np.sqrt(1/n1+1/n2)
        df=n1+n2-2
    else:
        SE=np.sqrt(var1/n1+var2/n2)
        df=(var1/n1+var2/n2)**2/((var1/n1)**2/(n1-1)+(var2/n2)**2/(n2-1))
    t_critic=t.ppf(1-alpha/2,df)
    margin=t_critic*SE 
    return diff-margin,diff+margin 
def save_box_plot(A,B,labels,title,axis_labels,src):
    plt.figure()
    plt.boxplot([A,B],tick_labels=labels)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.tight_layout() 
    plt.savefig(src,dpi=300)
    plt.close() 
def save_histogram(A, B, labels, title, axis_labels, src, bins=15):
    plt.figure()

    plt.hist(A, bins=bins, alpha=0.6, label=labels[0])
    plt.hist(B, bins=bins, alpha=0.6, label=labels[1])

    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(src, dpi=300)
    plt.close()
def problema_1(src):
    print("-"*20)
    print("Problema 1")
    print("-"*20)
    df=pd.read_csv(src)
    sensor=df["sensor_temp_C"]
    reference=df["reference_temp_C"]
    bias=sensor-reference
    mu0=0 
    results=tester_1sample(ttest_1samp,bias,mu0)
    cohen=cohens_d_1sample(bias,mu0)
    results["cohen_d"]=cohen 
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    print(f"Fuerza de cohen {cohen}")
    interpretaciones_cohen(cohen)
def IC(sample,alpha):
    n=len(sample)
    mean=np.mean(sample)
    std=np.std(sample,ddof=1)
    t_critic=t.ppf(1-alpha/2,df=n-1)
    margin=t_critic*std/np.sqrt(n)
    ci_low=mean-margin
    ci_high=mean+margin
    return ci_low,ci_high 

def problema_2(src):
    print("-"*20)
    print("Problema 2")
    print("-"*20)
    df=pd.read_csv(src)
    api=df["response_time_ms"]
    results=tester_1sample(ttest_1samp,api,120)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    ic=IC(api,0.05)
    results["ic"]=ic 
    print(f"IC 95%=({ic[0]},{ic[1]})")
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)
def problema_3(src):
    print("-"*20)
    print("Problema 3")
    print("-"*20)
    df=pd.read_csv(src)
    dron=df["abs_error_cm"]
    results=tester_1sample(ttest_1samp,dron,3,True)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)
def problema_4(src):
    print("-"*20)
    print("Problema 4")
    print("-"*20)
    df=pd.read_csv(src)
    A=df[df["model"]=="A"]["mae"]
    B=df[df["model"]=="B"]["mae"]
    print("prueba levene")
    print("H0=varianzas iguales")
    print("H1=varianzas diferentes")
    results_2=tester_2sample(levene,A,B) 
    print(f"estadistico t: {results_2['estadistico']}")
    print(f"p value: {results_2['p-value']}")
    print("prueba t student")
    print("H0=mae igual")
    print("H1=mae diferente")
    results=tester_2sample(ttest_ind,A,B)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    results["estadistico_levene"]=results_2["estadistico"]
    results["p_value_levene"]=results_2["p-value"]
    results["H0_levene"]=results_2["H0"]
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)
def problema_5(src):
    print("-"*20)
    print("Problema 5")
    print("-"*20)
    df=pd.read_csv(src)
    A=df[df["station"]=="Norte"]["pm25_ug_m3"]
    B=df[df["station"]=="Sur"]["pm25_ug_m3"]
    print("prueba levene")
    print("H0=varianzas iguales")
    print("H1=varianzas diferentes")
    results_2=tester_2sample(levene,A,B) 
    print(f"estadistico t: {results_2['estadistico']}")
    print(f"p value: {results_2['p-value']}")
    print("Prueba t student")
    results=tester_2sample(ttest_ind,A,B)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    ic=ic_2_samples(A,B,0.05,results_2["H0"])
    print(f"Intervalos de confianza ({ic[0]},{ic[1]})")
    results["estadistico_levene"]=results_2["estadistico"]
    results["p_value_levene"]=results_2["p-value"]
    results["H0_levene"]=results_2["H0"]
    results["ic"]=ic 
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)

def problema_6(src):
    print("-"*20)
    print("Problema 6")
    print("-"*20)
    df=pd.read_csv(src)
    A=df[df["board"]=="Raspberry Pi 5"]["power_W"]
    B=df[df["board"]=="Jetson Nano"]["power_W"]
    print("Prueba t student")
    results=tester_2sample(ttest_ind,A,B)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    d=cohens_d_2sample(A,B)
    print(f"Cohen's d: {d}")
    interpretaciones_cohen(d)
    results["cohen_d"]=d 
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)

def problema_7(src):
    print("-"*20)
    print("Problema 7")
    print("-"*20)
    df=pd.read_csv(src)
    A=df[df["filter"]=="Butterworth"]["residual_error"]
    B=df[df["filter"]=="Kalman"]["residual_error"]
    print("Prueba t student")
    results=tester_2sample(ttest_ind,A,B)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    d=cohens_d_2sample(A,B)
    print(f"Cohen's d: {d}")
    interpretaciones_cohen(d) 
    m_butter,m_kalman=np.mean(A),np.mean(B)
    mensaje="el filtro butter es mayor" if m_butter>m_kalman else "el filtro kalman es mayor"
    print(f"Las medias reflejan que {mensaje}")
    save_box_plot(A,B,["Butterworth","Kalman"],"Comparación de errores",["Filtro","Error residual"],"boxplot_errors.png")
    save_histogram(A,B,["Butterworth","Kalman"],"Distribución de errores",["Error residual","Frecuencias"],"hist_errors.png")
    results["cohen_d"]=d 
    results["mA"]=m_butter
    results["mB"]=m_kalman
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)


def problema_8(src):
    print("-"*20)
    print("Problema 8")
    print("-"*20)
    df=pd.read_csv(src)
    A=df["pre_error"]
    B=df["post_error"]
    print("Prueba t student")
    results=tester_2sample(ttest_rel,A,B)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    print("Prueba de correlacion")
    results_2=tester_2sample(pearsonr,A,B)
    correlacion_interpretacion(results_2["estadistico"])
    print(f"estadistico t: {results_2['estadistico']}")
    print(f"p value: {results_2['p-value']}")
    diferencias=A-B
    ic=IC(diferencias,0.05)
    print(f"Intervalos de confianza ({ic[0]},{ic[1]})")
    d=cohens_d_1sample(diferencias,0)
    print(f"Intensidad de cohen {d}")
    interpretaciones_cohen(d)
    results["estadistico_r"]=results_2["estadistico"]
    results["p_value_pearson"]=results_2["p-value"]
    results["H0_pearson"]=results_2["H0"]
    results["IC"]=ic
    results["d"]=d
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)

def problema_9(src):
    print("-"*20)
    print("Problema 9")
    print("-"*20)
    df=pd.read_csv(src)
    A=df["rmse_before"]
    B=df["rmse_after"]
    print("Prueba t student")
    results_2=tester_2sample(ttest_rel,A,B)
    print(f"estadistico t: {results_2['estadistico']}")
    print(f"p value: {results_2['p-value']}")
    print("Prueba de normalidad")
    diferencias=A-B 
    results=tester_1sample(shapiro,diferencias) 
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    results["estadistico_t"]=results_2["estadistico"]
    results["p_t"]=results_2["p-value"]
    results["H0_t"]=results_2["H0"]
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)


def problema_10(src):
    print("-"*20)
    print("Problema 10")
    print("-"*20)
    df=pd.read_csv(src)
    A=df["error_before_mps2"]
    B=df["error_after_mps2"]
    print("Prueba t student")
    results=tester_2sample(ttest_rel,A,B)
    print(f"estadistico t: {results['estadistico']}")
    print(f"p value: {results['p-value']}")
    diferencias=A-B 
    promedio_diferencias=np.mean(diferencias)
    mensaje="La calirabación mejoro" if promedio_diferencias>0 else "No mejoro"
    print(mensaje)
    d=cohens_d_1sample(diferencias,0)
    print(f"D cohen's: {d}")
    interpretaciones_cohen(d)
    results["cohen_d"]=d 
    results["mean"]=promedio_diferencias
    results_df=pd.DataFrame([results])
    latex_table=results_df.to_latex(index=False)
    print("latex_table")
    print(latex_table)


def main():
    problema_1("data/1_sensor_bias.csv")
    problema_2("data/2_api_response_times.csv")
    problema_3("data/3_drone_positioning_error.csv")
    problema_4("data/4_model_mae_comparison.csv")
    problema_5("data/5_pm25_stations.csv")
    problema_6("data/6_board_power_consumption.csv")
    problema_7("data/7_filter_residuals.csv")
    problema_8("data/8_pre_post_optimizer.csv")
    problema_9("data/9_rmse_before_after_cleaning.csv")
    problema_10("data/10_accelerometer_calibration.csv")
main()
