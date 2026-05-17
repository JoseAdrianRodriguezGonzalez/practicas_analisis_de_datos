'''
EJERCICIO 2: Tiempos de respuesta de una API

El tiempo estandar de respuesta de una API critica es 120 ms.
Despues de una actualizacion se toman n=30 mediciones.

Prueba si el nuevo tiempo promedio difiere del estandar.
Estima el intervalo de confianza del 95%.
Determina si los cambios deben revertirse.

TIPO DE PRUEBA:
  Una muestra (se compara contra referencia: 120 ms)
  Dos colas: queremos detectar cambio en CUALQUIER direccion
  H0: mu = 120 ms   (el tiempo NO cambio)
  H1: mu != 120 ms  (el tiempo SI cambio)
  alpha = 0.05
  gl = n - 1 = 30 - 1 = 29
  t critico (bilateral, alpha=0.05, gl=29) = +-2.0452
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_1samp, t as t_dist

datos = '2_api_response_times.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

n        = len(dataframe['response_time_ms'])
media    = dataframe['response_time_ms'].mean()
desv_std = dataframe['response_time_ms'].std(ddof=1)
mu_ref   = 120.0

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n = {n}")
print(f"Media post-actualizacion = {media:.4f} ms")
print(f"Desv. estandar = {desv_std:.4f} ms")
print(f"Referencia = {mu_ref} ms")

t_stat, p_valor = ttest_1samp(dataframe['response_time_ms'], mu_ref)

print("\nPRUEBA t DE UNA MUESTRA (2 colas)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.4f}")
print(f"t critico = +-2.0452  (gl=29, alpha=0.05)")

alpha = 0.05
if p_valor < alpha:
    print("Decision: RECHAZAR H0 => el tiempo de respuesta cambio significativamente.")
else:
    print("Decision: NO RECHAZAR H0 => no hay evidencia de cambio significativo.")

t_critico    = t_dist.ppf(0.975, df=n-1)
error_margen = t_critico * (desv_std / np.sqrt(n))
ci_inferior  = media - error_margen
ci_superior  = media + error_margen

print("\nINTERVALO DE CONFIANZA DEL 95%")
print(f"t critico para IC = {t_critico:.4f}")
print(f"Margen de error = {error_margen:.4f} ms")
print(f"IC 95% = [{ci_inferior:.2f}, {ci_superior:.2f}] ms")
if ci_inferior <= 120 <= ci_superior:
    print("120 ms esta dentro del IC.")
else:
    print("120 ms NO esta dentro del IC.")

'''
'''
