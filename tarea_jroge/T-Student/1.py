'''
EJERCICIO 1: Validacion de un sensor de temperatura

Un sensor de temperatura afirma tener un sesgo maximo de +-0.5C
respecto a un termometro patron. Se registran 40 mediciones del
sensor mientras el patron marca exactamente 25C.

Prueba t de una muestra para determinar si el sensor presenta sesgo.
Calcula tamano del efecto (Cohen's d).
Interpreta implicaciones para sistema de control termico industrial.

TIPO DE PRUEBA:
  Una muestra (se compara contra un valor de referencia: 25C)
  Dos colas: queremos detectar sesgo en CUALQUIER direccion (+ o -)
  H0: mu = 25   (el sensor NO tiene sesgo)
  H1: mu != 25  (el sensor SI tiene sesgo)
  alpha = 0.05
  gl = n - 1 = 40 - 1 = 39
  t critico (bilateral, alpha=0.05, gl=39) = +-2.0227
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_1samp

datos = '1_sensor_bias.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

n        = len(dataframe['sensor_temp_C'])
media    = dataframe['sensor_temp_C'].mean()
desv_std = dataframe['sensor_temp_C'].std(ddof=1)
mu_ref   = 25.0

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n = {n}")
print(f"Media sensor = {media:.4f} C")
print(f"Desv. estandar = {desv_std:.4f} C")
print(f"Referencia = {mu_ref} C")

t_stat, p_valor = ttest_1samp(dataframe['sensor_temp_C'], mu_ref)

print("\nPRUEBA t DE UNA MUESTRA (2 colas)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.4f}")
print(f"t critico = +-2.0227  (gl=39, alpha=0.05)")

alpha = 0.05
if p_valor < alpha:
    print("Decision: RECHAZAR H0 => el sensor presenta sesgo significativo.")
else:
    print("Decision: NO RECHAZAR H0 => no hay evidencia suficiente de sesgo.")

cohens_d = (media - mu_ref) / desv_std
print(f"\nCohen's d = {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print("Tamano del efecto: Insignificante (< 0.2)")
elif abs(cohens_d) < 0.5:
    print("Tamano del efecto: Pequeno (0.2 - 0.5)")
elif abs(cohens_d) < 0.8:
    print("Tamano del efecto: Mediano (0.5 - 0.8)")
else:
    print("Tamano del efecto: Grande (> 0.8)")

'''
'''
