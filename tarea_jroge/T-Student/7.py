'''
EJERCICIO 7: Efecto de un filtro en series de tiempo

Se aplican dos filtros (Butterworth vs Kalman) a un conjunto
de datos ruidoso. Cada filtro produce 40 errores residuales independientes.

Determina si un filtro produce menor error promedio.
Evalua robustez con analisis grafico (boxplots e histogramas).
Discute si la diferencia es estadistica y practicamente relevante.

TIPO DE PRUEBA:
  Dos muestras independientes (filtros distintos, sin emparejamiento)
  Dos colas: buscamos diferencia en CUALQUIER sentido
  H0: muBW = muKL   (ambos filtros tienen igual error promedio)
  H1: muBW != muKL  (los errores difieren)
  alpha = 0.05
  Prueba Levene para verificar igualdad de varianzas.
  gl = n1 + n2 - 2 = 40 + 40 - 2 = 78
  t critico (bilateral, alpha=0.05, gl=78) = +-1.9908
'''

import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, levene

datos = '7_filter_residuals.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

bw = dataframe[dataframe['filter'] == 'Butterworth']['residual_error']
kl = dataframe[dataframe['filter'] == 'Kalman']['residual_error']

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n Butterworth = {len(bw)}")
print(f"Media BW = {bw.mean():.4f}")
print(f"Desv. std BW = {bw.std(ddof=1):.4f}")
print(f"n Kalman = {len(kl)}")
print(f"Media Kalman = {kl.mean():.4f}")
print(f"Desv. std Kalman = {kl.std(ddof=1):.4f}")

lev_stat, lev_p = levene(bw, kl)
varianzas_iguales = lev_p > 0.05
print(f"\nPRUEBA DE LEVENE => F = {lev_stat:.4f}, p = {lev_p:.4f}")
if varianzas_iguales:
    print("Varianzas iguales => t-Student estandar.")
else:
    print("Varianzas desiguales => Welch t-test.")

t_stat, p_valor = ttest_ind(bw, kl, equal_var=varianzas_iguales)

print("\nPRUEBA t DE DOS MUESTRAS INDEPENDIENTES (2 colas)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.4f}")
print(f"t critico = +-1.9908  (gl=78, alpha=0.05)")

alpha = 0.05
if p_valor < alpha:
    print("Decision: RECHAZAR H0 => diferencia significativa entre filtros.")
else:
    print("Decision: NO RECHAZAR H0 => sin diferencia significativa.")

sp = np.sqrt((bw.std(ddof=1)**2 + kl.std(ddof=1)**2) / 2)
cohens_d = (bw.mean() - kl.mean()) / sp
print(f"\nCohen's d = {cohens_d:.4f}")
if abs(cohens_d) > 0.8:
    print("Tamano del efecto: Grande (> 0.8)")
elif abs(cohens_d) > 0.5:
    print("Tamano del efecto: Mediano (0.5 - 0.8)")
else:
    print("Tamano del efecto: Pequeno (< 0.5)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].boxplot([bw, kl], labels=['Butterworth', 'Kalman'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
axes[0].set_title('Boxplot de errores residuales')
axes[0].set_ylabel('Error residual')
axes[0].set_xlabel('Filtro')

axes[1].hist(bw, bins=12, alpha=0.6, label='Butterworth', color='steelblue')
axes[1].hist(kl, bins=12, alpha=0.6, label='Kalman', color='orange')
axes[1].axvline(bw.mean(), color='blue', linestyle='--', label=f'Media BW={bw.mean():.2f}')
axes[1].axvline(kl.mean(), color='red', linestyle='--', label=f'Media KL={kl.mean():.2f}')
axes[1].set_title('Histograma de errores residuales')
axes[1].set_xlabel('Error residual')
axes[1].set_ylabel('Frecuencia')
axes[1].legend()

plt.tight_layout()
plt.savefig('7_filtros_grafico.png', dpi=120)
print("\nGrafico guardado: 7_filtros_grafico.png")
plt.close()

'''
'''
