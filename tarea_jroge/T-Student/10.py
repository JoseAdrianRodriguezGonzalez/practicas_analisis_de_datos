'''
EJERCICIO 10: Precision de un sensor antes y despues de calibracion

Un acelerometro se calibra y se toman mediciones apareadas de la
aceleracion real vs medida. 30 pares.

Prueba si la calibracion redujo el error promedio.
Calcula el tamano del efecto.
Evalua si es suficiente para SLAM o control de dron.

TIPO DE PRUEBA:
  Muestras relacionadas / apareadas (mismo sensor, 2 condiciones)
  Una cola derecha: queremos comprobar que la calibracion REDUCE el error
  d_i = error_antes - error_despues  (d_i > 0 si mejora)
  H0: muD = 0   (la calibracion no reduce el error)
  H1: muD > 0   (la calibracion SI reduce el error)
  alpha = 0.05
  gl = n - 1 = 30 - 1 = 29
  t critico (1 cola derecha, alpha=0.05, gl=29) = +1.6991
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_rel, t as t_dist

datos = '10_accelerometer_calibration.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

antes   = dataframe['error_before_mps2']
despues = dataframe['error_after_mps2']
diffs   = antes - despues

n  = len(diffs)
gl = n - 1

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n (pares) = {n}")
print(f"Media error antes = {antes.mean():.4f} m/s2")
print(f"Media error despues = {despues.mean():.4f} m/s2")
print(f"Media diferencias = {diffs.mean():.4f} m/s2  (antes - despues)")
print(f"Desv. std diferencias = {diffs.std(ddof=1):.4f} m/s2")

# ttest_rel da p bilateral; para 1 cola dividir entre 2
# Si t > 0 (antes > despues, mejora) => cola derecha => p_unilateral = p_bilateral / 2
t_stat, p_bilateral = ttest_rel(antes, despues)
p_unilateral = p_bilateral / 2 if t_stat > 0 else 1 - p_bilateral / 2
t_crit_uni = t_dist.ppf(0.95, df=gl)

print("\nPRUEBA t APAREADA (1 cola, H1: muD > 0)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"p bilateral = {p_bilateral:.6f}")
print(f"p unilateral = {p_unilateral:.6f}")
print(f"t critico (1 cola) = +{t_crit_uni:.4f}  (gl={gl}, alpha=0.05)")

alpha = 0.05
if p_unilateral < alpha:
    print("Decision: RECHAZAR H0 => la calibracion reduce significativamente el error.")
else:
    print("Decision: NO RECHAZAR H0 => no hay evidencia de reduccion significativa.")

cohens_d = diffs.mean() / diffs.std(ddof=1)
print(f"\nCohen's d (pareado) = {cohens_d:.4f}")
if abs(cohens_d) > 0.8:
    print("Tamano del efecto: Grande (> 0.8)")
elif abs(cohens_d) > 0.5:
    print("Tamano del efecto: Mediano (0.5 - 0.8)")
else:
    print("Tamano del efecto: Pequeno (< 0.5)")

t_crit_bi = t_dist.ppf(0.975, df=gl)
se_d = diffs.std(ddof=1) / np.sqrt(n)
me = t_crit_bi * se_d
ci_inf = diffs.mean() - me
ci_sup = diffs.mean() + me
reduccion = (1 - despues.mean() / antes.mean()) * 100
print(f"\nIC 95% diferencia = [{ci_inf:.4f}, {ci_sup:.4f}] m/s2")
print(f"Reduccion relativa = {reduccion:.1f}%")

'''
'''
