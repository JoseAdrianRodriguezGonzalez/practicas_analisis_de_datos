"""
ACTIVIDAD 3: Análisis SARIMA Completo
Entrada: serie_temporal_expandida_arima.csv (62 puntos)
Salida: 4 carpetas con análisis, plots y métricas
version 2 mejorada
Pasos:
1. Descomposición clásica (tendencia, estacionalidad, residuo)
2. SARIMA automático (grid search)
3. Pronóstico 12 meses con intervalos de confianza
4. Evaluación RMSE (últimos 12 meses como test)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)


def crear_directorios():
	"""Crea carpetas para resultados."""
	carpetas = ['Paso1_Descomposicion', 'Paso2_SARIMA_Selection',
				'Paso3_Pronostico', 'Paso4_Evaluacion']
	for c in carpetas:
		os.makedirs(c, exist_ok=True)
		print(f" {c}/")
	return carpetas


def cargar_serie(ruta):
	"""Carga serie temporal."""
	print("\nPASO 0: CARGA DE DATOS")
	
	datos = pd.read_csv(ruta)
	datos['fecha'] = pd.to_datetime(datos['fecha'])
	serie = pd.Series(datos['ventas'].values, index=datos['fecha'].values)
	
	print(f"Observaciones: {len(serie)}")
	print(f"Período: {serie.index.min().strftime('%Y-%m')} a {serie.index.max().strftime('%Y-%m')}")
	print(f"Reales: {(datos['origen']=='real').sum()}, "
	      f"Interpolados: {(datos['origen']=='interpolado_lineal').sum()}, "
	      f"Sintéticos: {(datos['origen']=='sintetico_arima').sum()}")
	
	return serie


def paso1_descomposicion(serie):
	"""Descomposición clásica aditiva."""
	print("\nPASO 1: DESCOMPOSICION CLASICA")
	
	descomp = seasonal_decompose(serie, model='additive', period=12)
	
	# Gráfico
	fig, ax = plt.subplots(4, 1, figsize=(14, 10))
	serie.plot(ax=ax[0], color='blue', title='Serie Original')
	descomp.trend.plot(ax=ax[1], color='green', title='Tendencia')
	descomp.seasonal.plot(ax=ax[2], color='orange', title='Estacionalidad')
	descomp.resid.plot(ax=ax[3], color='red', title='Residuos', alpha=0.7)
	
	for i in range(4):
		ax[i].set_ylabel('Ventas')
		ax[i].grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig('Paso1_Descomposicion/descomposicion.png', dpi=300, bbox_inches='tight')
	print(" Gráfico: Paso1_Descomposicion/descomposicion.png")
	plt.close()
	
	# Estadísticas
	with open('Paso1_Descomposicion/estadisticas.txt', 'w') as f:
		f.write("DESCOMPOSICION CLASICA\n" + "="*70 + "\n\n")
		f.write(f"Serie Original:\n  Media: {serie.mean():,.0f}\n  Desv.Est: {serie.std():,.0f}\n")
		f.write(f"  Mín: {serie.min():,.0f}\n  Máx: {serie.max():,.0f}\n\n")
		f.write(f"Tendencia:\n  Mín: {descomp.trend.min():,.0f}\n  Máx: {descomp.trend.max():,.0f}\n\n")
		f.write(f"Estacionalidad:\n  Amplitud: {descomp.seasonal.max() - descomp.seasonal.min():,.0f}\n\n")
		f.write(f"Residuos:\n  Media: {descomp.resid.mean():,.0f}\n  Desv.Est: {descomp.resid.std():,.0f}\n")
	
	print(" Estadísticas: Paso1_Descomposicion/estadisticas.txt")



def paso2_sarima(serie):
	"""Grid search automático de parámetros SARIMA."""
	print("\nPASO 2: SARIMA AUTOMATICO (GRID SEARCH)")
	print("="*70)
	
	print("Búsqueda entre 216 combinaciones (p,d,q ∈ [0,2], P,D,Q ∈ [0,1])...")
	
	mejor_aic = np.inf
	mejor_modelo = None
	mejores_p = None
	resultados = []
	
	for p in range(3):
		for d in range(3):
			for q in range(3):
				for P in range(2):
					for D in range(2):
						for Q in range(2):
							try:
								m = SARIMAX(serie, order=(p,d,q), seasonal_order=(P,D,Q,12),
									   enforce_stationarity=False, enforce_invertibility=False)
								res = m.fit(disp=False, maxiter=200)
								
								if res.aic < mejor_aic:
									mejor_aic = res.aic
									mejor_modelo = res
									mejores_p = (p, d, q, P, D, Q)
									print(f"  Mejor: SARIMA{(p,d,q)}x{(P,D,Q,12)} → AIC={res.aic:.2f}")
								
								resultados.append({'params': (p,d,q,P,D,Q), 'aic': res.aic})
							except:
								pass
	
	p, d, q, P, D, Q = mejores_p
	print(f"\nMejor modelo: SARIMA({p},{d},{q})x({P},{D},{Q},12)")
	print(f"  AIC: {mejor_aic:.2f}")
	print(f"  BIC: {mejor_modelo.bic:.2f}")
	
	# Guardar top 10
	df_res = pd.DataFrame(resultados).sort_values('aic').head(10)
	with open('Paso2_SARIMA_Selection/parametros.txt', 'w') as f:
		f.write("GRID SEARCH SARIMA\n" + "="*70 + "\n\n")
		f.write("TOP 10 MODELOS:\n")
		for i, row in df_res.iterrows():
			p, d, q, P, D, Q = row['params']
			f.write(f"  SARIMA({p},{d},{q})x({P},{D},{Q},12) - AIC: {row['aic']:.2f}\n")
		f.write(f"\nSELECCIONADO:\n  SARIMA({p},{d},{q})x({P},{D},{Q},12)\n")
	
	print(" Parámetros: Paso2_SARIMA_Selection/parametros.txt")
	
	# Diagnósticos
	fig = mejor_modelo.plot_diagnostics(figsize=(14, 10))
	plt.tight_layout()
	plt.savefig('Paso2_SARIMA_Selection/diagnosticos.png', dpi=300, bbox_inches='tight')
	print(" Diagnósticos: Paso2_SARIMA_Selection/diagnosticos.png")
	plt.close()
	
	return mejor_modelo, mejores_p



def paso3_pronostico(modelo, serie):
	"""Pronóstico 12 meses adelante."""
	print("\nPASO 3: PRONOSTICO 12 MESES")
	
	fc = modelo.get_forecast(steps=12)
	fc_mean = fc.predicted_mean
	fc_ci = fc.conf_int(alpha=0.05)
	
	ultima_fecha = serie.index[-1]
	fechas_fut = pd.date_range(start=ultima_fecha + pd.DateOffset(months=1), periods=12, freq='MS')
	
	# Gráfico
	fig, ax = plt.subplots(figsize=(14, 8))
	
	serie.plot(ax=ax, label='Serie Histórica', color='blue', linewidth=2)
	fc_mean.plot(ax=ax, label='Pronóstico', color='red', linewidth=2, marker='o')
	ax.fill_between(fechas_fut, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2, color='red', 
	                 label='IC 95%')
	
	ax.set_xlabel('Fecha')
	ax.set_ylabel('Ventas (botellas)')
	ax.set_title('SARIMA: Pronóstico 12 Meses')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig('Paso3_Pronostico/pronostico.png', dpi=300, bbox_inches='tight')
	print(" Gráfico: Paso3_Pronostico/pronostico.png")
	plt.close()
	
	# Tabla
	with open('Paso3_Pronostico/proyecciones.txt', 'w') as f:
		f.write("PRONOSTICO SARIMA: 12 MESES\n" + "="*70 + "\n\n")
		f.write(f"{'Fecha':<12} {'Pronóstico':>15} {'LI_95%':>15} {'LS_95%':>15}\n")
		f.write("-"*70 + "\n")
		for fecha, media, (li, ls) in zip(fechas_fut, fc_mean, fc_ci.values):
			f.write(f"{fecha.strftime('%Y-%m'):<12} {media:>15,.0f} {li:>15,.0f} {ls:>15,.0f}\n")
		f.write(f"\nEstadísticas:\n  Media: {fc_mean.mean():,.0f}\n  Desv.Est: {fc_mean.std():,.0f}\n")
	
	print(" Proyecciones: Paso3_Pronostico/proyecciones.txt")




def paso4_evaluacion(serie, modelo):
	"""Evaluación: RMSE en últimos 12 meses."""
	print("\nPASO 4: EVALUACION (RMSE)")
	
	# Train/Test split
	train = serie[:-12]
	test = serie[-12:]
	
	print(f"entrenamiento: {len(train)} observaciones")
	print(f"Prueba: {len(test)} observaciones (últimos 12 meses)")
	
	# Pronósticos
	fc = modelo.get_forecast(steps=12)
	fc_mean = fc.predicted_mean
	
	# Métricas
	rmse = np.sqrt(mean_squared_error(test, fc_mean))
	mae = mean_absolute_error(test, fc_mean)
	mape = np.mean(np.abs((test - fc_mean) / test)) * 100
	
	print(f"\nMétricas:")
	print(f"  RMSE: {rmse:,.0f}")
	print(f"  MAE:  {mae:,.0f}")
	print(f"  MAPE: {mape:.2f}%")
	
	# Gráfico
	fig, ax = plt.subplots(figsize=(14, 8))
	
	train.plot(ax=ax, label='Entrenamiento', color='blue', linewidth=2)
	test.plot(ax=ax, label='Prueba (Real)', color='green', linewidth=2, marker='o')
	fc_mean.plot(ax=ax, label='Pronóstico', color='red', linewidth=2, marker='s', linestyle='--')
	
	ax.set_xlabel('Fecha')
	ax.set_ylabel('Ventas (botellas)')
	ax.set_title(f'Evaluación: RMSE={rmse:,.0f}, MAPE={mape:.2f}%')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig('Paso4_Evaluacion/evaluacion.png', dpi=300, bbox_inches='tight')
	print(" Gráfico: Paso4_Evaluacion/evaluacion.png")
	plt.close()
	
	# Tabla detallada
	with open('Paso4_Evaluacion/metricas.txt', 'w') as f:
		f.write("EVALUACION DEL MODELO\n" + "="*70 + "\n\n")
		f.write(f"Conjunto de Entrenamiento: {len(train)} observaciones\n")
		f.write(f"Conjunto de Prueba: {len(test)} observaciones (últimos 12 meses)\n")
		f.write(f"Período test: {test.index[0].strftime('%Y-%m')} a {test.index[-1].strftime('%Y-%m')}\n\n")
		
		f.write("METRICAS DE ERROR:\n")
		f.write(f"  RMSE: {rmse:,.0f} botellas\n")
		f.write(f"  MAE:  {mae:,.0f} botellas\n")
		f.write(f"  MAPE: {mape:.2f}%\n\n")
		
		f.write(f"{'Fecha':<12} {'Real':>15} {'Pronóstico':>15} {'Error':>15} {'% Error':>12}\n")
		f.write("-"*70 + "\n")
		
		errores_porc = []
		for fecha, real, pred in zip(test.index, test.values, fc_mean.values):
			error = real - pred
			pct_err = (error / real * 100) if real != 0 else 0
			errores_porc.append(pct_err)
			f.write(f"{fecha.strftime('%Y-%m'):<12} {real:>15,.0f} {pred:>15,.0f} {error:>15,.0f} {pct_err:>11.2f}%\n")
		
		f.write(f"\nError promedio: {np.mean(errores_porc):.2f}%\n")
		f.write(f"Desv.Est: {np.std(errores_porc):.2f}%\n")
	
	print("Métricas: Paso4_Evaluacion/metricas.txt")



def main():
	print("ACTIVIDAD 3: ANALISIS SARIMA COMPLETO")

	
	print("\nCreando estructura de carpetas...")
	crear_directorios()
	
	# Cargar datos
	serie = cargar_serie('serie_temporal_expandida_arima.csv')
	
	# Paso 1: Descomposición
	paso1_descomposicion(serie)
	
	# Paso 2: SARIMA
	modelo, params = paso2_sarima(serie)
	
	# Paso 3: Pronóstico
	paso3_pronostico(modelo, serie)
	
	# Paso 4: Evaluación
	paso4_evaluacion(serie, modelo)
	

	print("\nResultados en carpetas:")
	print("  • Paso1_Descomposicion/")
	print("  • Paso2_SARIMA_Selection/")
	print("  • Paso3_Pronostico/")
	print("  • Paso4_Evaluacion/\n")


if __name__ == "__main__":
	main()
