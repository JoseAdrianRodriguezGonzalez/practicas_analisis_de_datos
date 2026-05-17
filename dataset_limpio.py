"""
Dataset limpio: preprocesamiento de datos para síntesis
Objetivo: Imputar NaN, mantener outliers, agregar por fecha
Salida: Serie temporal univariada de 8 puntos mensuales
"""

import pandas as pd
import numpy as np


def imputar_valores_nulos(datos):
	"""
	Imputa valores nulos en cantidad_resenas con la mediana.
	Parámetros:
		datos: DataFrame con columnas a imputar
	Retorna:
		DataFrame con imputaciones realizadas
	"""
	print("IMPUTACION DE VALORES NULOS")
	
	nulos_antes = datos['cantidad_resenas'].isnull().sum()
	
	if nulos_antes > 0:
		mediana = datos['cantidad_resenas'].median()
		datos['cantidad_resenas'].fillna(mediana, inplace=True)
		
		nulos_despues = datos['cantidad_resenas'].isnull().sum()
		
		print(f"\nColumna: cantidad_resenas")
		print(f"Valores nulos antes: {nulos_antes}")
		print(f"Mediana usada para imputación: {mediana:.2f}")
		print(f"Valores nulos después: {nulos_despues}")
	else:
		print("\nNo hay valores nulos para imputar")
	
	return datos


def validar_outliers(datos):
	print("\n\nVALIDACION DE OUTLIERS")
	
	print("\nDecisión: MANTENER outliers")
	print("Razón: Los valores extremos en ventas_30dias reflejan patrones reales")
	print("       (productos muy populares vs productos con bajo desempeño)")
	
	print("\nEstadísticas de ventas_30dias (potenciales outliers):")
	print(f"Mínimo: {datos['ventas_30dias'].min()}")
	print(f"Máximo: {datos['ventas_30dias'].max()}")
	print(f"Desv. Estándar: {datos['ventas_30dias'].std():.2f}")
	print(f"Coeficiente de variación: {(datos['ventas_30dias'].std() / datos['ventas_30dias'].mean()):.4f}")


def agregar_por_fecha(datos):

	print("\n\nAGREGACION TEMPORAL")
	
	datos['fecha'] = pd.to_datetime(datos['fecha'])
	
	serie_agregada = datos.groupby('fecha')['ventas_30dias'].sum().reset_index()
	serie_agregada.columns = ['fecha', 'ventas']
	serie_agregada = serie_agregada.sort_values('fecha').reset_index(drop=True)
	
	print("\nAgregación completada:")
	print(f"Observaciones originales: {len(datos)}")
	print(f"Observaciones agregadas: {len(serie_agregada)}")
	print(f"Reducción: {((len(datos) - len(serie_agregada)) / len(datos) * 100):.2f}%")
	
	print("\nSerie temporal agregada:")
	for idx, row in serie_agregada.iterrows():
		print(f"\t{row['fecha'].strftime('%Y-%m')}: {row['ventas']:,.0f} botellas")
	
	return serie_agregada


def diagnosticar_huecos(serie_temporal):
	"""
	Identifica huecos en la serie temporal.
	Parámetros:
		serie_temporal: DataFrame con columna 'fecha'
	"""
	print("\n\nDIAGNOSTICO DE HUECOS TEMPORALES")
	
	rango_esperado = pd.date_range(start=serie_temporal['fecha'].min(),
									end=serie_temporal['fecha'].max(),
									freq='MS')
	
	fechas_presentes = set(serie_temporal['fecha'])
	fechas_esperadas = set(rango_esperado)
	
	fechas_faltantes = fechas_esperadas - fechas_presentes
	
	print(f"\nRango temporal completo: {len(rango_esperado)} meses")
	print(f"Meses con datos: {len(fechas_presentes)}")
	print(f"Meses faltantes: {len(fechas_faltantes)}")
	
	if fechas_faltantes:
		print("\nFechas faltantes:")
		for fecha in sorted(fechas_faltantes):
			print(f"\t{fecha.strftime('%Y-%m')}")
		
		print("\nEstas fechas será necesario interpolar en el siguiente paso")
	else:
		print("\nNo hay huecos en la serie temporal")
	
	return fechas_faltantes


def obtener_resumen_final(serie_temporal):
	"""
	Proporciona resumen estadístico de la serie temporal final.
	Parámetros:
		serie_temporal: DataFrame con ventas agregadas
	"""
	print("\n\nRESUMEN DE SERIE TEMPORAL LIMPIA")
	
	ventas = serie_temporal['ventas']
	
	print(f"\nDimensiones: {len(serie_temporal)} observaciones x 2 columnas")
	print(f"\nEstadísticas de ventas_agregadas:")
	print(f"Media: {ventas.mean():,.0f} botellas")
	print(f"Mediana: {ventas.median():,.0f} botellas")
	print(f"Desv. Estándar: {ventas.std():,.0f} botellas")
	print(f"Mínimo: {ventas.min():,.0f} botellas")
	print(f"Máximo: {ventas.max():,.0f} botellas")
	print(f"Rango: {ventas.max() - ventas.min():,.0f} botellas")
	
	print(f"\nTendencia observada:")
	primer_valor = ventas.iloc[0]
	ultimo_valor = ventas.iloc[-1]
	cambio = ((ultimo_valor - primer_valor) / primer_valor) * 100
	
	print(f"Primer mes (mayo 2018): {primer_valor:,.0f}")
	print(f"Último mes (febrero 2019): {ultimo_valor:,.0f}")
	print(f"Cambio porcentual: {cambio:+.2f}%")


def guardar_serie_temporal(serie_temporal, ruta_salida):
	"""
	Guarda la serie temporal limpia en formato CSV.
	Parámetros:
		serie_temporal: DataFrame con serie temporal
		ruta_salida: Ruta donde guardar el archivo
	"""	
	try:
		serie_temporal.to_csv(ruta_salida, index=False, encoding='utf-8')
		print(f"\nArchivo guardado: {ruta_salida}")
		
		import os
		tamaño_kb = os.path.getsize(ruta_salida) / 1024
		print(f"Tamaño: {tamaño_kb:.2f} KB")
	except Exception as error:
		print(f"Error al guardar: {error}")


def main():
	"""
	Función principal que orquesta el preprocesamiento.
	"""
	print("DATASET LIMPIO: PREPROCESAMIENTO PARA SINTESIS DE DATOS")
	
	ruta_entrada = 'wine_sales_procesados.csv'
	ruta_salida = 'serie_temporal_limpia.csv'
	
	try:
		datos = pd.read_csv(ruta_entrada)
		print(f"\nArchivo cargado: {ruta_entrada}")
		print(f"Dimensiones iniciales: {datos.shape}")
	except Exception as error:
		print(f"Error al cargar: {error}")
		return
	
	datos = imputar_valores_nulos(datos)
	
	validar_outliers(datos)
	
	serie_temporal = agregar_por_fecha(datos)
	
	fechas_faltantes = diagnosticar_huecos(serie_temporal)
	
	obtener_resumen_final(serie_temporal)
	
	guardar_serie_temporal(serie_temporal, ruta_salida)
	
	print(f"Próximo paso: Interpolación de {len(fechas_faltantes)} meses faltantes")
	print(f"Serie temporal lista: {ruta_salida}")


if __name__ == "__main__":
	main()
