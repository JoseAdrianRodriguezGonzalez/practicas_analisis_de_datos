"""
COMPARACIÓN DE TIEMPOS DE EJECUCIÓN ENTRE DOS GPUs
Se miden tiempos de inferencia (ms) de un mismo modelo en dos GPUs
diferentes (GPU-A y GPU-B) usando 50 ejecuciones para cada una. Los
tiempos muestran colas pesadas.
    - Determina si GPU-B es significativamente más rápida.
    - Calcula el tamaño del efecto no paramétrico.
    - Concluye sobre la viabilidad de usar GPU-B en aplicaciones de 
      baja latencia.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_4/4_gpu_mannwhitney_inferencia.csv')

#                       Separar GPUs                      #
gpu_A = df[df['gpu'] == 'A']['tiempo_ms']
gpu_B = df[df['gpu'] == 'B']['tiempo_ms']

# ------------------------------------------------------- #
#                 PRUEBA DE MANN-WHITNEY                  #
# ------------------------------------------------------- #
U, p = mannwhitneyu(gpu_B, gpu_A, alternative='less')

print(f"Estadistico U: {U:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: GPU-B es significativamente más rápida que GPU-A.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente para afirmar que GPU-B es más rápida que GPU-A.")

# ------------------------------------------------------- #
#                   TAMAÑO DEL EFECTO R                   #
# ------------------------------------------------------- #
n1 = len(gpu_B)
n2 = len(gpu_A)

mean_U = n1 * (n2 / 2)
std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

Z = (U - mean_U) / std_U
r = abs(Z) / np.sqrt(n1 + n2)

print(f"\nTamaño del efecto r: {r:.4f}")

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

delta = cliffs_delta(gpu_B, gpu_A)

print(f"\nCliff's delta: {delta:.4f}")

# ------------------------------------------------------- #
#                      VISUALIZACIÓN                      #
# ------------------------------------------------------- #
plt.figure()

sns.violinplot(data=df, x='gpu', y='tiempo_ms', inner=None)
sns.boxplot(data=df, x='gpu', y='tiempo_ms', width=0.2)

plt.title('Distribución de Tiempos de Inferencia por GPU')
plt.xlabel('GPU')
plt.ylabel('Tiempo de Inferencia (ms)')
plt.savefig('exercise_4/tiempos_inferencia_violin_boxplot.png')