"""
Análisis de preprocesamiento para series temporales
Objetivo: Detectar NaN, outliers, distribuciones antes de generar datos sintéticos
"""

import pandas as pd
import numpy as np


def analizar_valores_nulos(datos):
	"""
	Analiza presencia y porcentaje de valores nulos por columna.
	Parámetros:
		datos: DataFrame a analizar
	"""
	print("ANALISIS DE VALORES NULOS")
	
	nulos_totales = datos.isnull().sum().sum()
	porcentaje_total = (nulos_totales / (datos.shape[0] * datos.shape[1])) * 100
	
	print(f"\nValores nulos totales: {nulos_totales}")
	print(f"Porcentaje del dataset: {porcentaje_total:.2f}%")
	
	print("\nPor columna:")
	for col in datos.columns:
		nulos = datos[col].isnull().sum()
		porcentaje = (nulos / len(datos)) * 100
		if nulos > 0:
			print(f"\t{col}: {nulos} valores nulos ({porcentaje:.2f}%)")
		else:
			print(f"\t{col}: sin valores nulos")


def analizar_outliers(datos):
	"""
	Detecta outliers usando método IQR (rango intercuartílico).
	Parámetros:
		datos: DataFrame con columnas numéricas
	"""
	print("\n\nANALISIS DE OUTLIERS (Método IQR)")
	
	columnas_numericas = datos.select_dtypes(include=[np.number]).columns
	
	for col in columnas_numericas:
		Q1 = datos[col].quantile(0.25)
		Q3 = datos[col].quantile(0.75)
		IQR = Q3 - Q1
		
		limite_inferior = Q1 - 1.5 * IQR
		limite_superior = Q3 + 1.5 * IQR
		
		outliers = datos[(datos[col] < limite_inferior) | (datos[col] > limite_superior)]
		porcentaje_outliers = (len(outliers) / len(datos)) * 100
		
		print(f"\n{col}:")
		print(f"\tRango normal: [{limite_inferior:.2f}, {limite_superior:.2f}]")
		print(f"\tOutliers detectados: {len(outliers)} ({porcentaje_outliers:.2f}%)")
		
		if len(outliers) > 0:
			print(f"\tValores mínimo/máximo: {outliers[col].min():.2f} / {outliers[col].max():.2f}")


def analizar_distribuciones(datos):
	"""
	Analiza distribuciones de columnas numéricas.
	Parámetros:
		datos: DataFrame con columnas numéricas
	"""
	print("\n\nANALISIS DE DISTRIBUCIONES")
	
	columnas_numericas = datos.select_dtypes(include=[np.number]).columns
	
	for col in columnas_numericas:
		valores = datos[col].dropna()
		
		print(f"\n{col}:")
		print(f"\tMedia: {valores.mean():.2f}")
		print(f"\tMediana: {valores.median():.2f}")
		print(f"\tDesv. Estándar: {valores.std():.2f}")
		print(f"\tAsimetría (Skewness): {valores.skew():.4f}")
		print(f"\tCurtosis: {valores.kurtosis():.4f}")
		
		if valores.skew() > 0.5:
			print(f"\t[NOTA] Distribución sesgada positivamente")
		elif valores.skew() < -0.5:
			print(f"\t[NOTA] Distribución sesgada negativamente")
		
		if valores.kurtosis() > 3:
			print(f"\t[NOTA] Distribución con colas pesadas (muchos outliers)")


def diagnostico_series_temporales(datos):
	"""
	Diagnóstico específico para uso en series temporales.
	Parámetros:
		datos: DataFrame con columna 'fecha'
	"""
	print("\n\nDIAGNOSTICO PARA SERIES TEMPORALES")
	
	if 'fecha' not in datos.columns:
		print("Columna 'fecha' no encontrada")
		return
	
	datos['fecha'] = pd.to_datetime(datos['fecha'])
	datos_ordenado = datos.sort_values('fecha')
	fechas = datos_ordenado['fecha'].unique()
	
	print(f"\nRango temporal: {fechas.min()} a {fechas.max()}")
	print(f"Fechas únicas: {len(fechas)}")
	print(f"Observaciones por fecha:")
	
	for fecha in sorted(fechas):
		cantidad = len(datos[datos['fecha'] == fecha])
		print(f"\t{pd.to_datetime(fecha).strftime('%Y-%m')}: {cantidad} registros")
	





def main():
	"""
	Ejecuta todos los análisis de preprocesamiento.
	"""
	print("PREPROCESAMIENTO: ANALISIS DE INTEGRIDAD DE DATOS")
	print("\n")
	
	ruta_csv = 'wine_sales_procesados.csv'
	
	try:
		datos = pd.read_csv(ruta_csv)
		print(f"Archivo cargado: {ruta_csv}")
		print(f"Dimensiones: {datos.shape}")
	except Exception as error:
		print(f"Error al cargar: {error}")
		return
	
	analizar_valores_nulos(datos)
	
	analizar_outliers(datos)
	
	analizar_distribuciones(datos)
	
	diagnostico_series_temporales(datos)
	

if __name__ == "__main__":
	main()
