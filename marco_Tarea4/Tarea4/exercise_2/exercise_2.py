"""
DIFERENCIA EN LA ACTIVIDAD DE SENSORES IoT
Se instalan sensores PIR de dos marcas distintas en una fabrica. Se
registran 40 lecturas de activacion diaria para cada marca, pero los 
datos estan sobredispersos y no son gaussianos. 
    - Evalua si una marca detecta mas actividad que la otra.
    - Construye un grafico de distribuciones (violin o boxplot).
    - Discute si la diferencia podria deberse a sensibilidad distinta
      o ruido ambiental.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_2/2_pir_mannwhitney_marcas.csv')

#                       Separar grupos                    #
A = df[df['marca'] == 'A']['activaciones']
B = df[df['marca'] == 'B']['activaciones']

# ------------------------------------------------------- #
#                 PRUEBA DE MANN-WHITNEY                  #
# ------------------------------------------------------- #
U, p = mannwhitneyu(A, B, alternative='two-sided')

print(f"Estadistico U: {U:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: Hay diferencia significativa entre las marcas.")
else:
    print("No rechazamos la hipótesis nula: No hay diferencia significativa entre las marcas.")

# ------------------------------------------------------- #
#                GRAFICO DE DISTRIBUCIONES                #
# ------------------------------------------------------- #
plt.figure()

sns.violinplot(data=df, x='marca', y='activaciones', inner=None)
sns.boxplot(data=df, x='marca', y='activaciones', width=0.2)

plt.title('Distribución de Activaciones por Marca')
plt.xlabel('Marca del Sensor PIR')
plt.ylabel('Número de Activaciones')
plt.savefig('exercise_2/activaciones_violin_boxplot.png')