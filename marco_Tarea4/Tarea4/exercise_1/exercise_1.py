"""
RENDIMIENTO DE DOS ALGORITMOS DE CLASIFICACIÓN
Dos algoritmos de clasificación (A y B) se evaluan sobre 25 particiones 
bootstrap de un mismo conjunto de datos. 
Sin embargo, las metricas de F1-score no son normales (alta asimetría).
    - Aplica la prueba U de Mann-Whitney para determinar si el algoritmo B
      supera al algoritmo A.
    - Calcula el tamaño del efecto (r o Cliff's delta).
    - Interpreta que implica la diferencia para un despliegue en tiempo real. 
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, norm

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_1/1_f1_mannwhitney_algos.csv')

#                Separacion de algoritmos                 #
A = df[df['algoritmo'] == 'A']['f1_score']
B = df[df['algoritmo'] == 'B']['f1_score']

# ------------------------------------------------------- #
#                 PRUEBA U DE MANN-WHITNEY                #
# ------------------------------------------------------- #
U, p = mannwhitneyu(A, B, alternative='less')

print(f"Estadistico U: {U:.4f}")
print(f"Valor p: {p:.4f}")

# ------------------------------------------------------- #
#                   TAMAÑO DEL EFECTO R                   #
# ------------------------------------------------------- #
n1 = len(A)
n2 = len(B)

mean_U = n1 * (n2 / 2)
std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

Z = (U - mean_U) / std_U

r = abs(Z) / np.sqrt(n1 + n2)

print(f"\nTamaño del efecto r: {r:.4f}")

#                      Interpretación                     #
if(r < 0.1):
    print("Efecto: Despreciable")
elif(r < 0.3):
    print("Efecto: Pequeño")
elif(r < 0.5):
    print("Efecto: Mediano")
else:
    print("Efecto: Grande")

# ------------------------------------------------------- #
#                     CLIFF'S DELTA                       #
# ------------------------------------------------------- #
def cliffs_delta(x, y):
    nx = len(x)
    ny = len(y)
    gt = 0
    lt = 0

    for xi in x:
        for yi in y:
            if(xi > yi):
                gt += 1
            elif(xi < yi):
                lt += 1

    return (gt - lt) / (nx * ny)

delta = cliffs_delta(B, A)

print(f"\nCliff's delta: {delta:.4f}")

#                      Interpretación                     #
if(delta < 0.147):
    print("Efecto: Despreciable")
elif(delta < 0.33):
    print("Efecto: Pequeño")
elif(delta < 0.474):
    print("Efecto: Mediano")
else:
    print("Efecto: Grande")

# ------------------------------------------------------- #
#                   INTERPRETACIÓN FINAL                  #
# ------------------------------------------------------- #
alpha = 0.05

print("\nInterpretación final:")
if(p < alpha):
    print("Se rechaza la hipótesis nula: El algoritmo B tiene F1-score significativamente mayor que el algoritmo A.")
else:
    print("No se rechaza la hipótesis nula: No hay evidencia suficiente para afirmar que el algoritmo B supera al algoritmo A.")