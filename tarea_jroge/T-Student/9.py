'''
EJERCICIO 9: Efecto de un tratamiento de limpieza de datos

Un dataset tiene valores atipicos que afectan el entrenamiento
de un modelo. Se mide el RMSE antes y despues de aplicar un
metodo de limpieza avanzado. Muestras APAREADAS.

Ejecuta t-test pareado.
Verifica normalidad de diferencias (Shapiro-Wilk).
Interpreta el impacto en el pipeline de datos.

TIPO DE PRUEBA:
  Muestras relacionadas / apareadas (mismo experimento, 2 condiciones)
  Dos colas: buscamos si hay cambio en CUALQUIER direccion
  H0: muD = 0   (la limpieza no cambia el RMSE)
  H1: muD != 0  (la limpieza cambia el RMSE)
  d_i = rmse_antes_i - rmse_despues_i
  alpha = 0.05
  Supuesto: diferencias normalmente distribuidas (verificar con Shapiro)
  gl = n - 1
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_rel, shapiro, t as t_dist

datos = '9_rmse_before_after_cleaning.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

antes   = dataframe['rmse_before']
despues = dataframe['rmse_after']
diffs   = antes - despues

n  = len(diffs)
gl = n - 1

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n (pares) = {n}")
print(f"Media RMSE antes = {antes.mean():.4f}")
print(f"Media RMSE despues = {despues.mean():.4f}")
print(f"Media diferencias = {diffs.mean():.4f}  (antes - despues)")
print(f"Desv. std diferencias = {diffs.std(ddof=1):.4f}")

stat_sw, p_sw = shapiro(diffs)
print("\nPRUEBA DE SHAPIRO-WILK (normalidad de diferencias)")
print(f"W = {stat_sw:.4f}")
print(f"p valor = {p_sw:.4f}")
if p_sw > 0.05:
    print("No se rechaza normalidad => t-test pareado valido.")
else:
    print("Se rechaza normalidad (p < 0.05) => interpretar con cautela; considerar Wilcoxon.")

t_stat, p_valor = ttest_rel(antes, despues)
t_crit = t_dist.ppf(0.975, df=gl)

print("\nPRUEBA t APAREADA (2 colas)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.6f}")
print(f"gl = {gl}")
print(f"t critico = +-{t_crit:.4f}  (gl={gl}, alpha=0.05)")

alpha = 0.05
if p_valor < alpha:
    print("Decision: RECHAZAR H0 => la limpieza reduce significativamente el RMSE.")
else:
    print("Decision: NO RECHAZAR H0 => sin efecto significativo.")

cohens_d = diffs.mean() / diffs.std(ddof=1)
se_d = diffs.std(ddof=1) / np.sqrt(n)
me = t_crit * se_d
ci_inf = diffs.mean() - me
ci_sup = diffs.mean() + me

print(f"\nCohen's d = {cohens_d:.4f}")
if abs(cohens_d) > 0.8:
    print("Tamano del efecto: Grande (> 0.8)")
elif abs(cohens_d) > 0.5:
    print("Tamano del efecto: Mediano (0.5 - 0.8)")
else:
    print("Tamano del efecto: Pequeno (< 0.5)")

print(f"IC 95% diferencias = [{ci_inf:.4f}, {ci_sup:.4f}]")

'''
'''
