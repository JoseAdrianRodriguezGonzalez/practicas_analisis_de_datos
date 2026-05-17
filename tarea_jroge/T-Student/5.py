'''
EJERCICIO 5: Rendimiento entre dos estaciones de sensado

Dos estaciones meteorologicas (Norte y Sur) miden PM2.5
en una misma ciudad. 50 observaciones por estacion.

Prueba si sus medias difieren.
Calcula intervalos de confianza de la diferencia.
Propone razones tecnicas o ambientales para la diferencia.

TIPO DE PRUEBA:
  Dos muestras independientes (estaciones distintas)
  Dos colas: buscamos si son diferentes (sin direccion a priori)
  H0: muN = muS   (ambas estaciones miden igual en promedio)
  H1: muN != muS  (las mediciones difieren)
  alpha = 0.05
  gl = n1 + n2 - 2 = 50 + 50 - 2 = 98
  t critico (bilateral, alpha=0.05, gl=98) = +-1.9845
  Prueba Levene para decidir si usar varianzas iguales o Welch.
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_ind, levene, t as t_dist

datos = '5_pm25_stations.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

norte = dataframe[dataframe['station'] == 'Norte']['pm25_ug_m3']
sur   = dataframe[dataframe['station'] == 'Sur']['pm25_ug_m3']

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n Norte = {len(norte)}")
print(f"Media Norte = {norte.mean():.4f} ug/m3")
print(f"Desv. std Norte = {norte.std(ddof=1):.4f} ug/m3")
print(f"n Sur = {len(sur)}")
print(f"Media Sur = {sur.mean():.4f} ug/m3")
print(f"Desv. std Sur = {sur.std(ddof=1):.4f} ug/m3")

lev_stat, lev_p = levene(norte, sur)
varianzas_iguales = lev_p > 0.05
print(f"\nPRUEBA DE LEVENE => p = {lev_p:.4f}")
if varianzas_iguales:
    print("Varianzas iguales => t-Student estandar.")
else:
    print("Varianzas desiguales => Welch t-test.")

t_stat, p_valor = ttest_ind(norte, sur, equal_var=varianzas_iguales)

print("\nPRUEBA t DE DOS MUESTRAS INDEPENDIENTES (2 colas)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.4f}")
print(f"t critico = +-1.9845  (gl=98, alpha=0.05)")

alpha = 0.05
if p_valor < alpha:
    print("Decision: RECHAZAR H0 => diferencia significativa entre estaciones.")
else:
    print("Decision: NO RECHAZAR H0 => sin diferencia significativa.")

diff_media = norte.mean() - sur.mean()
se_diff = np.sqrt(norte.std(ddof=1)**2/len(norte) + sur.std(ddof=1)**2/len(sur))
gl = len(norte) + len(sur) - 2
t_critico = t_dist.ppf(0.975, df=gl)
me = t_critico * se_diff
ci_inf = diff_media - me
ci_sup = diff_media + me

print("\nIC 95% DE LA DIFERENCIA (Norte - Sur)")
print(f"Diferencia media = {diff_media:.4f} ug/m3")
print(f"IC 95% = [{ci_inf:.4f}, {ci_sup:.4f}] ug/m3")
if ci_inf <= 0 <= ci_sup:
    print("El IC contiene el 0 => diferencia NO significativa.")
else:
    print("El IC NO contiene el 0 => diferencia significativa.")

'''
'''
