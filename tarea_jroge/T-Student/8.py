'''
EJERCICIO 8: Evaluacion pre-post de un algoritmo de optimizacion

Se mide el error de un robot movil antes y despues de aplicar
un optimizador metaheuristico (p.ej. Algoritmos Geneticos).
Muestras APAREADAS: mismo robot, mismas instancias.

Realiza una t apareada.
Evalua correlacion entre condiciones pre y post.
Discute el impacto del optimizador en un sistema embebido.

TIPO DE PRUEBA:
  Muestras relacionadas / apareadas (mismas instancias, 2 condiciones)
  Dos colas: buscamos si hay cambio en CUALQUIER direccion
  H0: muD = 0   (la diferencia pre-post es cero, sin efecto)
  H1: muD != 0  (el optimizador cambia el error)
  d_i = pre_i - post_i  (diferencia por instancia)
  alpha = 0.05
  gl = n - 1
  t critico depende de n (ver resultados)
'''

import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_rel, t as t_dist

datos = '8_pre_post_optimizer.csv'
dir_datos = '../'
dataframe = pd.read_csv(os.path.join(dir_datos, datos))

pre   = dataframe['pre_error']
post  = dataframe['post_error']
diffs = pre - post

n  = len(diffs)
gl = n - 1

print("ESTADISTICOS DESCRIPTIVOS")
print(f"n (pares) = {n}")
print(f"Media pre = {pre.mean():.4f}")
print(f"Media post = {post.mean():.4f}")
print(f"Media diferencias = {diffs.mean():.4f}  (pre - post)")
print(f"Desv. std diferencias = {diffs.std(ddof=1):.4f}")

t_stat, p_valor = ttest_rel(pre, post)
t_crit = t_dist.ppf(0.975, df=gl)

print("\nPRUEBA t APAREADA (2 colas)")
print(f"Estadistico t = {t_stat:.4f}")
print(f"Valor p = {p_valor:.6f}")
print(f"gl = {gl}")
print(f"t critico = +-{t_crit:.4f}  (gl={gl}, alpha=0.05)")

alpha = 0.05
if p_valor < alpha:
    print("Decision: RECHAZAR H0 => el optimizador produce cambio significativo.")
else:
    print("Decision: NO RECHAZAR H0 => sin efecto significativo.")

correlacion = pre.corr(post)
print(f"\nCorrelacion pre-post (Pearson r) = {correlacion:.4f}")
if correlacion > 0.7:
    print("Correlacion alta => las instancias dificiles siguen siendo dificiles post-optimizacion.")
elif correlacion > 0.4:
    print("Correlacion moderada.")
else:
    print("Correlacion baja => el comportamiento cambia considerablemente.")

cohens_d = diffs.mean() / diffs.std(ddof=1)
print(f"\nCohen's d (pareado) = {cohens_d:.4f}")
if abs(cohens_d) > 0.8:
    print("Tamano del efecto: Grande (> 0.8)")
elif abs(cohens_d) > 0.5:
    print("Tamano del efecto: Mediano (0.5 - 0.8)")
else:
    print("Tamano del efecto: Pequeno (< 0.5)")

se_d = diffs.std(ddof=1) / np.sqrt(n)
me = t_crit * se_d
ci_inf = diffs.mean() - me
ci_sup = diffs.mean() + me
print(f"\nIC 95% diferencia = [{ci_inf:.4f}, {ci_sup:.4f}]")

'''
'''
