"""
COMPARACION DE CONSUMO ENERGETICO ANTES/DESPUES DE OPTIMIZAR CODIGO
Una aplicación edge-IA se ejecuta en el mismo microcontrolador antes
y después de una refactorización. Se miden 20 ciclos de consumo
energético (MJ) en ambas condiciones. Los datos presentan saltos 
discretos por modos de ahorro.
    - Aplica Wilcoxon para comparar ambas condiciones.
    - Indica si la refactorización realmente redujo el consumo.
    - Calcula el tamaño del efecto no paramétrico.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_8/8_energia_wilcoxon_prepost.csv')

#                     Separar Consumos                    #
consumo_pre = df['consumo_pre_mJ']
consumo_post = df['consumo_post_mJ']

# ------------------------------------------------------- #
#                   PRUEBA DE WILCOXON                    #
# ------------------------------------------------------- #
stat, p = wilcoxon(consumo_post, consumo_pre, alternative='less')

print(f"Estadístico W: {stat:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: La refactorización reduce significativamente el consumo energético.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente de reducción en el consumo energético.")

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