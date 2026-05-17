"""
LATENCIA DE UNA API ANTES VS DESPUÉS DE MIGRAR INFRACESTRUCTURA
Un servicio migra su backend a una architectura serverless. Se 
recogen 40 pares de mediciones de latencia para llamadas equivalentes
antes/después. Las diferencias tienen clara asimetría.
    - Aplica Wilcoxon para evaluar la mejora de latencia.
    - Reporta la magnitud de la reducción (mediana de diferencias).
    - Discute posibles causas de la variabilidad residual.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_10/10_api_wilcoxon_latencia_prepost.csv')

#                      Separar Métricas                   #
latencia_pre = df['lat_pre_ms']
latencia_post = df['lat_post_ms']

#                        Diferencias                      #
diferencias = latencia_post - latencia_pre
reduccion = latencia_pre - latencia_post

print("Mediana de reducción de latencia (ms):", np.median(reduccion))

# ------------------------------------------------------- #
#                   PRUEBA DE WILCOXON                    #
# ------------------------------------------------------- #
stat, p = wilcoxon(latencia_post, latencia_pre, alternative='less')

print(f"Estadístico W: {stat:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: La migración a serverless reduce significativamente la latencia.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente de reducción en la latencia.")

# ------------------------------------------------------- #
#                   TAMAÑO DEL EFECTO R                   #
# ------------------------------------------------------- #
n = len(df)

mean_W = n * (n + 1) / 4
std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

Z = (stat - mean_W) / std_W
r = abs(Z) / np.sqrt(n)

print(f"\nTamaño del efecto r: {r:.4f}")

if(r < 0.1):
    print("Efecto: Despreciable")
elif(r < 0.3):
    print("Efecto: Pequeño")
elif(r < 0.5):
    print("Efecto: Mediano")
else:
    print("Efecto: Grande")