"""
EVALUACION PRE-POST DE UN SISTEMA DE RECOMENDACIÓN
Una plataforma actualiza su algoritmo de recomendaciones. Se comparan 25
usuarios, midiendo click-through rate (CTR) antes y después del cambio. 
La diferencia CTR no es normal.
    - Aplica la prueba de Wilcoxon para comparar "antes" vs "después".
    - Determina si la actualización aumenta el CTR.
    - Evalúa si existe un efecto significativo en usuarios de baja actividad.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_6/6_ctr_wilcoxon_prepost.csv')

#                     Separar métodos                     #
ctr_pre = df['ctr_pre']
ctr_post = df['ctr_post']

# ------------------------------------------------------- #
#                    PRUEBA DE WILCOXON                   #
# ------------------------------------------------------- #
stat, p = wilcoxon(ctr_post, ctr_pre, alternative='greater')

print(f"Estadístico W: {stat:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: La actualización aumenta el CTR.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente para afirmar que la actualización aumenta el CTR.")

# ------------------------------------------------------- #
#                   TAMAÑO DEL EFECTO R                   #
# ------------------------------------------------------- #
n = len(df)

#                    Aproximacion de Z                    #
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
#      ANALISIS ADICIONAL: USUARIOS DE BAJA ACTIVIDAD     #
# ------------------------------------------------------- #
threshold = ctr_pre.median()

low_activity = df[df['ctr_pre'] < threshold]

if(len(low_activity) > 0):
    stat_low, p_low = wilcoxon(low_activity['ctr_post'], low_activity['ctr_pre'], alternative='greater')
    
    print(f"\nUsuarios de baja actividad: {len(low_activity)}")
    print(f"Valor p (baja actividad): {p_low:.4f}")

    if(p_low < alpha):
        print("La actualización mejora significativamente el CTR en usuarios de baja actividad.")
    else:
        print("No hay evidencia de mejora significativa en usuarios de baja actividad.")