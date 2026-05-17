"""
SALARIOS DE CIENTIFICOS DE DATOS EN DOS REGIONES
Una empresa compara salarios anuales de ingenieros de datos en dos
regiones (Región Norte: n=30, Región Sur: n=32). La distribución es
fuertemente asimétrica por bonos.
    - Usa Mann-Whitney para comparar salarios. 
    - Evalua la hipótesis de que la region Sur paga más.
    - Pregunta adicional: ¿Que tan robusto seria el test si existieran
      valores extremadamente atípicos?
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ------------------------------------------------------- #
#                       CARGAR DATOS                      #
# ------------------------------------------------------- #
df = pd.read_csv('exercise_3/3_salarios_mannwhitney_regiones.csv')

#                      Separar regiones                   #
norte = df[df['region'] == 'Norte']['salario_anual_mxn']
sur = df[df['region'] == 'Sur']['salario_anual_mxn']

# ------------------------------------------------------- #
#                 PRUEBA DE MANN-WHITNEY                  #
# ------------------------------------------------------- #
U, p = mannwhitneyu(norte, sur, alternative='less')

print(f"Estadistico U: {U:.4f}")
print(f"Valor p: {p:.4f}")

alpha = 0.05

if(p < alpha):
    print("Rechazamos la hipótesis nula: La región Sur paga más que la región Norte.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente para afirmar que la región Sur paga más.")

# ------------------------------------------------------- #
#                GRAFICO DE DISTRIBUCIONES                #
# ------------------------------------------------------- #
plt.figure()

sns.boxplot(data=df, x='region', y='salario_anual_mxn')
sns.stripplot(data=df, x='region', y='salario_anual_mxn')

plt.title('Distribución de Salarios Anuales por Región')
plt.xlabel('Región')
plt.ylabel('Salario Anual (MXN)')
plt.savefig('exercise_3/salarios_boxplot_stripplot.png')