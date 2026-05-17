"""
EFECTIVIDAD DE DOS TÉCNICAS DE LIMPIEZA DE DATOS
Dos métodos de limpieza (Método X y Método Y) procesan un conjunto de
datos con mucho ruido. Después de entrenar un modelo, se obtiene el 
error MAE (Mean Absolute Error) para 20 ejecuciones independientes de
cada método. Los errores no son normales.
    - ¿Qué método produce, en general, menor error?
    - Evalúa la hipótesis con Mann-Whitney y reporta la mediana de cada
      grupo.
    - Explica por qué elegir un test no paramétrico fue adecuado.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_5/5_mae_mannwhitney_limpieza.csv')

#                     Separar métodos                     #
X = df[df['metodo'] == 'X']['mae']
Y = df[df['metodo'] == 'Y']['mae']

# ------------------------------------------------------- #
#                         MEDIANAS                        #
# ------------------------------------------------------- #
median_X = X.median()
median_Y = Y.median()

print(f"Mediana MAE Método X: {median_X:.4f}")
print(f"Mediana MAE Método Y: {median_Y:.4f}")

# ------------------------------------------------------- #
#                 PRUEBA DE MANN-WHITNEY                  #
# ------------------------------------------------------- #
U, p = mannwhitneyu(X, Y, alternative='less')

print(f"\nEstadistico U: {U:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: Existe evidencia de que el Método Y produce menor error.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente de diferencia.")