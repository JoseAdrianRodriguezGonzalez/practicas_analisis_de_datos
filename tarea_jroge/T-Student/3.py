'''
EJERCICIO 3: Inferencia con datos simulados (IA / Robotica)

Simular 150 mediciones de error de posicionamiento (cm) de un
dron siguiendo una trayectoria recta.
Especificacion: error medio menor a 3 cm.

Genera los datos con una distribucion que incluya ruido gaussiano.
Ejecuta una prueba t de una muestra para validar cumplimiento.
Discute como la varianza afecta la confiabilidad.

TIPO DE PRUEBA:
  Una muestra (comparar media contra limite: 3 cm)
  Una cola izquierda: queremos probar si mu < 3 cm
  H0: mu >= 3 cm  (el dron NO cumple especificacion)
  H1: mu < 3 cm   (el dron SI cumple especificacion)
  alpha = 0.05
  gl = n - 1 = 150 - 1 = 149
  t critico (1 cola izquierda, alpha=0.05, gl=149) = -1.6551
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_1samp

datos = '3_drone_positioning_error.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

n        = len(dataframe['abs_error_cm'])
media    = dataframe['abs_error_cm'].mean()
desv_std = dataframe['abs_error_cm'].std(ddof=1)
limite   = 3.0

print("ESTADISTICOS DESCRIPTIVOS (datos simulados del dron)")
print(f"n = {n}")
print(f"Media error = {media:.4f} cm")
print(f"Desv. estandar = {desv_std:.4f} cm")
print(f"Limite especificacion = {limite} cm")

# ttest_1samp da p bilateral; para 1 cola dividir entre 2
# Si t < 0 (media < limite) => cola izquierda => p_unilateral = p_bilateral / 2
t_stat, p_bilateral = ttest_1samp(dataframe['abs_error_cm'], limite)
p_unilateral = p_bilateral / 2 if t_stat < 0 else 1 - p_bilateral / 2

print("\nPRUEBA t DE UNA MUESTRA (1 cola, H1: mu < 3 cm)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"p bilateral = {p_bilateral:.4f}")
print(f"p unilateral = {p_unilateral:.4f}")
print(f"t critico (1 cola) = -1.6551  (gl=149, alpha=0.05)")

alpha = 0.05
if p_unilateral < alpha:
    print("Decision: RECHAZAR H0 => el error medio es significativamente menor a 3 cm. CUMPLE.")
else:
    print("Decision: NO RECHAZAR H0 => no se puede confirmar que mu < 3 cm.")

cohens_d = (media - limite) / desv_std
print(f"\nCohen's d = {cohens_d:.4f}")

'''
'''
