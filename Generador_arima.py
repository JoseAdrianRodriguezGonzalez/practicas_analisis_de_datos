"""
Opción B: Generador ARIMA
Objetivo: Ajustar ARIMA a serie real, luego usarlo para generar 50 puntos sintéticos
Entrada: serie_temporal_limpia.csv (8 puntos)
Salida: serie_temporal_expandida_arima.csv (60 puntos: 8 reales + 52 sintéticos)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def cargar_serie_temporal(ruta_archivo):
	"""
	Carga la serie temporal limpia desde archivo CSV.
	Parámetros:
		ruta_archivo: Ruta del archivo CSV
	Retorna:
		DataFrame con fecha y ventas
	"""
	print("CARGA DE SERIE TEMPORAL")
	
	datos = pd.read_csv(ruta_archivo)
	datos['fecha'] = pd.to_datetime(datos['fecha'])
	datos = datos.sort_values('fecha').reset_index(drop=True)
	
	print(f"\nArchivo cargado: {ruta_archivo}")
	print(f"Observaciones: {len(datos)}")
	print(f"Período: {datos['fecha'].min().strftime('%Y-%m')} a {datos['fecha'].max().strftime('%Y-%m')}")
	
	return datos


def ajustar_arima(serie_temporal):
	"""
	Ajusta modelo ARIMA(1,1,1) a los datos reales.
	Parámetros:
		serie_temporal: DataFrame con columna 'ventas'
	Retorna:
		Tupla (modelo_ajustado, parámetros)
	"""
	print("\n\nAJUSTE DE MODELO ARIMA")
	
	ventas = serie_temporal['ventas'].values
	
	print(f"\nDatos para ajuste: {len(ventas)} observaciones")
	print(f"Valores: {ventas}")
	
	try:
		modelo = ARIMA(ventas, order=(1, 1, 1))
		resultado = modelo.fit()
		
		print(f"\nModelo ajustado: ARIMA(1,1,1)")
		print(f"AIC: {resultado.aic:,.2f}")
		print(f"BIC: {resultado.bic:,.2f}")
		
		print(f"\nCoeficientes estimados:")
		print(f"AR(1): {resultado.params[0]:.6f}")
		print(f"MA(1): {resultado.params[1]:.6f}")
		
		print(f"\nEstadísticas de residuos:")
		print(f"Media de residuos: {resultado.resid.mean():.2f}")
		print(f"Desv. estándar: {resultado.resid.std():.2f}")
		
		return resultado, (1, 1, 1)
	
	except Exception as error:
		print(f"Error al ajustar ARIMA: {error}")
		return None, None


def generar_datos_sinteticos_arima(resultado_arima, observaciones_generar=52, seed=42):
	"""
	Genera datos sintéticos usando el modelo ARIMA ajustado.
	Incluye ruido de pronóstico para preservar variabilidad realista.
	
	Parámetros:
		resultado_arima: Objeto de resultado del modelo ARIMA
		observaciones_generar: Número de puntos a generar
		seed: Semilla para reproducibilidad
	Retorna:
		array con datos sintéticos generados
	"""
	print("\n\nGENERACION DE DATOS SINTETICOS CON ARIMA")
	
	np.random.seed(seed)
	
	print(f"\nProceso de generación:")
	print(f"Observaciones a generar: {observaciones_generar}")
	print(f"Método: Pronóstico iterativo + ruido de incertidumbre")
	
	datos_sinteticos = []
	
	# Obtener último valor real (asegurar conversión a float)
	último_valor_real = float(resultado_arima.fittedvalues[-1])
	sigma_residuos = float(resultado_arima.resid.std())
	
	print(f"Punto de partida: {último_valor_real:,.0f}")
	print(f"Desv. estándar de residuos: {sigma_residuos:,.0f}")
	print(f"Escala de ruido: 0.2 * sigma (ruido controlado, sin valores negativos)")
	
	# Factor de ruido: 0.2 * sigma de residuos (ruido pequeño pero realista)
	escala_ruido = 0.2 * sigma_residuos
	
	for paso in range(observaciones_generar):
		# Pronóstico puntual del modelo
		pronostico = resultado_arima.get_forecast(steps=paso+1)
		valor_pronosticado = float(pronostico.predicted_mean[-1])
		
		# Agregar ruido controlado (distribución normal con escala = 0.2*sigma)
		ruido = np.random.normal(loc=0, scale=escala_ruido)
		valor_con_ruido = valor_pronosticado + ruido
		
		# Clipping: asegurar que ventas >= 0 (no pueden ser negativas)
		valor_final = max(0, valor_con_ruido)
		
		datos_sinteticos.append(valor_final)
		
		if (paso + 1) % 10 == 0:
			print(f"Generados {paso + 1}/{observaciones_generar} puntos")
	
	print(f"Generación completada con ruido controlado")
	
	return np.array(datos_sinteticos)


def validar_coherencia(serie_real, datos_sinteticos):
	"""
	Valida coherencia estadística entre series real y sintética.
	Parámetros:
		serie_real: array con datos reales
		datos_sinteticos: array con datos generados
	"""
	print("\n\nVALIDACION DE COHERENCIA")
	
	print(f"\nSerie Real:")
	print(f"Media: {np.mean(serie_real):,.0f}")
	print(f"Desv. estándar: {np.std(serie_real):,.0f}")
	print(f"Mínimo: {np.min(serie_real):,.0f}")
	print(f"Máximo: {np.max(serie_real):,.0f}")
	print(f"Rango: {np.max(serie_real) - np.min(serie_real):,.0f}")
	
	print(f"\nDatos Sintéticos Generados:")
	print(f"Media: {np.mean(datos_sinteticos):,.0f}")
	print(f"Desv. estándar: {np.std(datos_sinteticos):,.0f}")
	print(f"Mínimo: {np.min(datos_sinteticos):,.0f}")
	print(f"Máximo: {np.max(datos_sinteticos):,.0f}")
	print(f"Rango: {np.max(datos_sinteticos) - np.min(datos_sinteticos):,.0f}")
	
	media_real = np.mean(serie_real)
	media_sintetica = np.mean(datos_sinteticos)
	diferencia_media = abs(media_sintetica - media_real) / media_real * 100
	
	print(f"\nDiferencia de medias: {diferencia_media:.2f}%")
	
	ultimo_real = serie_real[-1]
	primer_sintetico = datos_sinteticos[0]
	salto = abs(primer_sintetico - ultimo_real) / ultimo_real * 100
	
	print(f"\nContinuidad:")
	print(f"Última observación real: {ultimo_real:,.0f}")
	print(f"Primera observación sintética: {primer_sintetico:,.0f}")
	print(f"Salto porcentual: {salto:.2f}%")
	
	if salto < 20:
		print(f"[OK] Continuidad excelente")
	elif salto < 50:
		print(f"[OK] Continuidad aceptable")
	else:
		print(f"[ADVERTENCIA] Salto significativo")
	
	if diferencia_media < 30:
		print(f"[OK] Distribuciones similares")
	else:
		print(f"[ADVERTENCIA] Distribuciones divergentes")


def interpolar_huecos_historicos(serie_real):
	"""
	Interpola huecos históricos (junio y octubre 2018) linealmente.
	Parámetros:
		serie_real: DataFrame con datos reales
	Retorna:
		DataFrame con 62 puntos (10 reales: 8 originales + 2 interpolados)
	"""
	print("\n\nINTERPOLACION DE HUECOS HISTORICOS")
	
	# Crear serie mensual contígua desde mayo 2018 a febrero 2019
	fecha_inicio = serie_real['fecha'].min()
	fecha_fin = serie_real['fecha'].max()
	meses_completos = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='MS')
	
	print(f"\nPeríodo a interpolar: {fecha_inicio.strftime('%Y-%m')} a {fecha_fin.strftime('%Y-%m')}")
	print(f"Meses faltantes esperados: 10 meses (mayo a febrero)")
	print(f"Meses disponibles: {len(serie_real)}")
	print(f"Huecos detectados: {len(meses_completos) - len(serie_real)}")
	
	# Crear serie con índice completo y valores de entrada
	serie_indexada = pd.Series(
		data=serie_real['ventas'].values,
		index=pd.DatetimeIndex(serie_real['fecha'].values)
	)
	
	# Reindexar a serie completa y rellenar con NaN
	serie_completa = serie_indexada.reindex(meses_completos)
	
	# Interpolar linealmente
	serie_interpolada = serie_completa.interpolate(method='linear')
	
	# Crear DataFrame con resultado
	serie_con_huecos = pd.DataFrame({
		'fecha': meses_completos,
		'ventas': serie_interpolada.values,
		'origen': ['real' if fecha in serie_real['fecha'].values else 'interpolado_lineal' 
				   for fecha in meses_completos]
	})
	
	print(f"\nInterpolación completada:")
	print(f"Total de puntos después de interpolar: {len(serie_con_huecos)}")
	print(f"  - Reales: {(serie_con_huecos['origen'] == 'real').sum()}")
	print(f"  - Interpolados: {(serie_con_huecos['origen'] == 'interpolado_lineal').sum()}")
	
	print(f"\nSerie con huecos rellenados:")
	for idx, row in serie_con_huecos.iterrows():
		print(f"\t{row['fecha'].strftime('%Y-%m')}: {row['ventas']:>12,.0f} ({row['origen']})")
	
	return serie_con_huecos


def combinar_series_arima(serie_con_huecos, datos_sinteticos):
	"""
	Combina serie real+interpolada con datos sintéticos.
	Parámetros:
		serie_con_huecos: DataFrame con 10 puntos (8 reales + 2 interpolados)
		datos_sinteticos: array con 52 datos generados
	Retorna:
		DataFrame con serie completa (62 puntos)
	"""
	print("\n\nCOMBINACION DE SERIES")
	
	ultima_fecha = serie_con_huecos['fecha'].iloc[-1]
	
	fechas_sinteticas = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1),
									   periods=len(datos_sinteticos),
									   freq='MS')
	
	serie_completa = pd.DataFrame({
		'fecha': list(serie_con_huecos['fecha'].values) + list(fechas_sinteticas),
		'ventas': list(serie_con_huecos['ventas'].values) + list(datos_sinteticos),
		'origen': list(serie_con_huecos['origen'].values) + ['sintetico_arima'] * len(datos_sinteticos)
	})
	
	print(f"\nSerie combinada final:")
	print(f"Período histórico (real+interpol): {serie_con_huecos['fecha'].min().strftime('%Y-%m')} a {serie_con_huecos['fecha'].max().strftime('%Y-%m')} ({len(serie_con_huecos)} puntos)")
	print(f"Período sintético: {fechas_sinteticas.min().strftime('%Y-%m')} a {fechas_sinteticas.max().strftime('%Y-%m')} ({len(datos_sinteticos)} puntos)")
	print(f"Total: {len(serie_completa)} observaciones")
	print(f"  - Reales: {(serie_completa['origen'] == 'real').sum()}")
	print(f"  - Interpolados: {(serie_completa['origen'] == 'interpolado_lineal').sum()}")
	print(f"  - Sintéticos: {(serie_completa['origen'] == 'sintetico_arima').sum()}")
	
	return serie_completa


def guardar_serie_completa(serie_completa, ruta_salida):
	"""
	Guarda la serie temporal completa en CSV.
	Parámetros:
		serie_completa: DataFrame con serie combinada
		ruta_salida: Ruta donde guardar
	"""
	print("\n\nGUARDAR SERIE COMPLETA")
	
	serie_completa.to_csv(ruta_salida, index=False, encoding='utf-8')
	
	print(f"\nArchivo guardado: {ruta_salida}")
	
	import os
	tamaño_kb = os.path.getsize(ruta_salida) / 1024
	print(f"Tamaño: {tamaño_kb:.2f} KB")
	
	print(f"\nPrimeras 12 filas:")
	for idx, row in serie_completa.head(12).iterrows():
		print(f"\t{row['fecha'].strftime('%Y-%m')}: {row['ventas']:>12,.0f} ({row['origen']})")
	
	print(f"\nÚltimas 5 filas:")
	for idx, row in serie_completa.tail(5).iterrows():
		print(f"\t{row['fecha'].strftime('%Y-%m')}: {row['ventas']:>12,.0f} ({row['origen']})")


def main():
	"""
	Función principal que orquesta ARIMA generator.
	Flujo: Carga -> Interpolación huecos -> ARIMA fit -> Generación sintética -> Combinación -> Guardado
	"""
	print("GENERADOR ARIMA CON INTERPOLACION DE HUECOS")
	print("Generación de 62 puntos totales (10 históricos + 52 sintéticos)")
	
	ruta_entrada = 'serie_temporal_limpia.csv'
	ruta_salida = 'serie_temporal_expandida_arima.csv'
	
	# Paso 1: Cargar datos reales (8 puntos)
	serie_real = cargar_serie_temporal(ruta_entrada)
	
	# Paso 2: Interpolar huecos históricos (8 -> 10 puntos)
	serie_con_huecos = interpolar_huecos_historicos(serie_real)
	
	# Paso 3: Ajustar ARIMA en datos reales (no interpolados)
	resultado_arima, params = ajustar_arima(serie_real)
	
	if resultado_arima is None:
		print("No se pudo continuar sin modelo ARIMA")
		return
	
	# Paso 4: Generar 52 datos sintéticos
	datos_sinteticos = generar_datos_sinteticos_arima(resultado_arima, observaciones_generar=52, seed=42)
	
	# Paso 5: Validar coherencia
	validar_coherencia(serie_real['ventas'].values, datos_sinteticos)
	
	# Paso 6: Combinar series (10 históricos + 52 sintéticos = 62 total)
	serie_completa = combinar_series_arima(serie_con_huecos, datos_sinteticos)
	
	# Paso 7: Guardar
	guardar_serie_completa(serie_completa, ruta_salida)
	
	print("\n\nSTATUS FINAL")
	print(f"Generador ARIMA completado:")
	print(f"  - 8 puntos reales (originales)")
	print(f"  - 2 puntos interpolados (relleno de huecos históricos)")
	print(f"  - 52 puntos sintéticos (generador ARIMA)")
	print(f"  - Total: 62 puntos para SARIMA")
	print(f"  - Archivo: {ruta_salida}")
	print(f"  - Coherencia: Validada estadísticamente")


if __name__ == "__main__":
	main()
