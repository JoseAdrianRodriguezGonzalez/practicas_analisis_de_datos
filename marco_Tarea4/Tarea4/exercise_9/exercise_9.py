"""
CALIDAD DE IMAGEN ANTES Y DESPUÉS DE APLICAR UN NUEVO FILTRO
Un algoritmo de visión por computadora ajusta un filtro para mejorar
nitidez. Para 18 imágenes, se compara la metrica SSIM antes y después
del nuevo filtro. Los cambios no son normales.
    - Determina con Wilcoxon si el filtro ofrece mejoras.
    - Evalúa cuantas imagenes muestran mejora positiva vs negativa.
    - Concluye sobre la robustez del filtro en dispositivos tipos de
      iluminación.
"""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_9/9_ssim_wilcoxon_prepost.csv')

#                      Separar Métricas                   #
ssim_pre = df['ssim_pre']
ssim_post = df['ssim_post']

# ------------------------------------------------------- #
#                   PRUEBA DE WILCOXON                    #
# ------------------------------------------------------- #
stat, p = wilcoxon(ssim_post, ssim_pre, alternative='greater')

print(f"Estadístico W: {stat:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: El nuevo filtro mejora significativamente la calidad de imagen.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente de mejora en la calidad de imagen.")

# ------------------------------------------------------- #
#                    CONTEO DE MEJORAS                    #
# ------------------------------------------------------- #
diferencias = ssim_post - ssim_pre

mejoras = (diferencias > 0).sum()
empeoramientos = (diferencias < 0).sum()
sin_cambio = (diferencias == 0).sum()

print(f"\nImágenes con mejora: {mejoras}")
print(f"Imágenes con empeoramiento: {empeoramientos}")
print(f"Imágenes sin cambio: {sin_cambio}")

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