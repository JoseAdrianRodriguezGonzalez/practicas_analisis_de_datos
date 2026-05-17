"""
ERROR DE LOCALIZACIÓN DE UN DRON ANTES Y DESPUÉS DE CALIBRACIÓN
Un dron realiza 30 vuelos repetidos antes y después de una calibración
avanzada del sistema IMU. Los errores absolutos tienen distribucion 
sesgada.
    - Usa Wilcoxon para determinar si la calibración mejoro la precisión.
    - Evalúa gráficamente las diferencias (scatter + linea de identidad).
    - Menciona posibles fuentes de variabilidad residual.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_7/7_dron_wilcoxon_error_prepost.csv')

#                      Separar Errores                    #
error_pre = df['error_pre_m']
error_post = df['error_post_m']

# ------------------------------------------------------- #
#                   PRUEBA DE WILCOXON                    #
# ------------------------------------------------------- #
stat, p = wilcoxon(error_post, error_pre, alternative='less')

print(f"Estadístico W: {stat:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: La calibración reduce significativamente el error.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente de mejora.")

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

# ------------------------------------------------------- #
#                         GRAFICO                         #
# ------------------------------------------------------- #
plt.figure()

plt.scatter(error_pre, error_post)

min_val = min(error_pre.min(), error_post.min())
max_val = max(error_pre.max(), error_post.max())

plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel('Error Pre-Calibración (m)')
plt.ylabel('Error Post-Calibración (m)')
plt.title('Comparación de Errores Pre y Post Calibración')
plt.savefig('exercise_7/7_drone_error_scatter.png')