'''
EJERCICIO 6: Eficiencia energetica en hardware diferente

Se mide el consumo promedio de energia (W) de dos placas
(Raspberry Pi 5 vs Jetson Nano) ejecutando una red neuronal pequena.

Realiza t-test de muestras independientes.
Evalua tamano del efecto.
Discute relevancia en sistemas edge-IA.

TIPO DE PRUEBA:
  Dos muestras independientes (hardware distinto)
  Dos colas: buscamos diferencia en CUALQUIER direccion
  H0: muRPi = muJet   (consumo promedio igual)
  H1: muRPi != muJet  (consumo promedio diferente)
  alpha = 0.05
  Prueba Levene primero para decidir varianzas iguales o Welch.
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_ind, levene, t as t_dist

datos = '6_board_power_consumption.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

rpi = dataframe[dataframe['board'] == 'Raspberry Pi 5']['power_W']
jet = dataframe[dataframe['board'] == 'Jetson Nano']['power_W']

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n RPi 5 = {len(rpi)}")
print(f"Media RPi 5 = {rpi.mean():.4f} W")
print(f"Desv. std RPi 5 = {rpi.std(ddof=1):.4f} W")
print(f"n Jetson Nano = {len(jet)}")
print(f"Media Jetson = {jet.mean():.4f} W")
print(f"Desv. std Jetson = {jet.std(ddof=1):.4f} W")

lev_stat, lev_p = levene(rpi, jet)
varianzas_iguales = lev_p > 0.05
print(f"\nPRUEBA DE LEVENE => F = {lev_stat:.4f}, p = {lev_p:.4f}")
if varianzas_iguales:
    print("Varianzas iguales => t-Student estandar.")
else:
    print("Varianzas desiguales => Welch t-test.")

t_stat, p_valor = ttest_ind(rpi, jet, equal_var=varianzas_iguales)

if not varianzas_iguales:
    s1, s2 = rpi.std(ddof=1), jet.std(ddof=1)
    n1, n2 = len(rpi), len(jet)
    gl = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
else:
    gl = len(rpi) + len(jet) - 2
gl_r = round(gl)
t_crit = t_dist.ppf(0.975, df=gl_r)

print("\nPRUEBA t DE DOS MUESTRAS INDEPENDIENTES (2 colas)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.6f}")
print(f"gl (aprox.) = {gl_r}")
print(f"t critico = +-{t_crit:.4f}  (gl={gl_r}, alpha=0.05)")

alpha = 0.05
if p_valor < alpha:
    print("Decision: RECHAZAR H0 => diferencia significativa en consumo.")
else:
    print("Decision: NO RECHAZAR H0 => sin diferencia significativa.")

sp = np.sqrt((rpi.std(ddof=1)**2 + jet.std(ddof=1)**2) / 2)
cohens_d = (rpi.mean() - jet.mean()) / sp
print(f"\nCohen's d = {cohens_d:.4f}")
if abs(cohens_d) > 0.8:
    print("Tamano del efecto: Grande (> 0.8)")
elif abs(cohens_d) > 0.5:
    print("Tamano del efecto: Mediano (0.5 - 0.8)")
else:
    print("Tamano del efecto: Pequeno (< 0.5)")

'''
'''
