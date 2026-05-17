'''
EJERCICIO 4: Comparacion de algoritmos de prediccion

Se comparan dos modelos de ML (Modelo A y Modelo B) en terminos
de sus errores MAE usando muestras independientes (n=25 por modelo).

Determina si existe diferencia significativa entre sus medias.
Evalua la igualdad de varianzas con Levene.
Interpreta implicaciones para seleccion de modelo en produccion.

TIPO DE PRUEBA:
  Dos muestras independientes (grupos A y B separados)
  Dos colas: buscamos diferencia en CUALQUIER direccion
  H0: muA = muB   (los modelos tienen igual MAE promedio)
  H1: muA != muB  (los modelos difieren en MAE promedio)
  alpha = 0.05
  gl = n1 + n2 - 2 = 25 + 25 - 2 = 48
  t critico (bilateral, alpha=0.05, gl=48) = +-2.0106
  Prueba Levene decide si usar varianzas iguales o no.
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_ind, levene

datos = '4_model_mae_comparison.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

mae_A = dataframe[dataframe['model'] == 'A']['mae']
mae_B = dataframe[dataframe['model'] == 'B']['mae']

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n (A) = {len(mae_A)}")
print(f"Media MAE A = {mae_A.mean():.4f}")
print(f"Desv. std A = {mae_A.std(ddof=1):.4f}")
print(f"n (B) = {len(mae_B)}")
print(f"Media MAE B = {mae_B.mean():.4f}")
print(f"Desv. std B = {mae_B.std(ddof=1):.4f}")

lev_stat, lev_p = levene(mae_A, mae_B)
varianzas_iguales = lev_p > 0.05

print("\nPRUEBA DE LEVENE (H0: varianzas iguales)")
print(f"Estadistico F = {lev_stat:.4f}")
print(f"Valor p Levene = {lev_p:.4f}")
if varianzas_iguales:
    print("Conclusion: Varianzas IGUALES => usar t-Student estandar.")
else:
    print("Conclusion: Varianzas DESIGUALES => usar Welch t-test.")

t_stat, p_valor = ttest_ind(mae_A, mae_B, equal_var=varianzas_iguales)

print("\nPRUEBA t DE DOS MUESTRAS INDEPENDIENTES (2 colas)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.4f}")
print(f"t critico = +-2.0106  (gl=48, alpha=0.05)")

alpha = 0.05
if p_valor < alpha:
    print("Decision: RECHAZAR H0 => diferencia significativa entre modelos.")
else:
    print("Decision: NO RECHAZAR H0 => sin diferencia significativa.")

sp = np.sqrt((mae_A.std(ddof=1)**2 + mae_B.std(ddof=1)**2) / 2)
cohens_d = (mae_A.mean() - mae_B.mean()) / sp
print(f"\nCohen's d = {cohens_d:.4f}")
if abs(cohens_d) > 0.8:
    print("Tamano del efecto: Grande (> 0.8)")
elif abs(cohens_d) > 0.5:
    print("Tamano del efecto: Mediano (0.5 - 0.8)")
else:
    print("Tamano del efecto: Pequeno (< 0.5)")

'''
'''
