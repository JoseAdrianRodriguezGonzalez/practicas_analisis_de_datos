"""
Prototipo: Análisis y transformación de dataset de ventas de vino
Objetivo: Cargar xlsx, explorar estructura, renombrar columnas y exportar a CSV

Columnas originales en chino:
    ID: Identificador único del producto
    日期: Fecha en formato YYYYMM
    商品名称: Nombre del producto
    价格: Precio del producto
    近30天销量（瓶）: Ventas últimos 30 días en botellas
    评论数: Cantidad de reseñas
    品牌: Marca del producto
    被搜索次数: Veces que fue buscado
"""

import pandas as pd
import os


def cargar_datos(ruta_archivo):
    """
    Carga el archivo xlsx desde la ruta especificada.
    Parámetros:
        ruta_archivo: Ruta completa al archivo xlsx
    Retorna:
        DataFrame con los datos cargados
    """
    try:
        datos = pd.read_excel(ruta_archivo, sheet_name=0)
        print("Archivo cargado exitosamente")
        print("Dimensiones:", datos.shape)
        return datos
    except Exception as error:
        print("Error al cargar archivo:", error)
        return None


def explorar_estructura(datos):
    """
    Explora y imprime información sobre la estructura del dataset.
    Parámetros:
        datos: DataFrame a explorar
    """
    print("\nEXPLORACION DE ESTRUCTURA")
    print("\nColumnas originales:")
    for idx, col in enumerate(datos.columns, 1):
        print(f"\t{idx}. {col}")
    
    print("\nTipos de datos:")
    for col, dtype in datos.dtypes.items():
        print(f"\t{col}: {dtype}")
    
    print("\nPrimeras filas:")
    print(datos.head())
    
    print("\nValores nulos por columna:")
    for col in datos.columns:
        nulos = datos[col].isnull().sum()
        print(f"\t{col}: {nulos}")
    
    print("\nEstadísticas descriptivas:")
    print(datos.describe())


def renombrar_columnas(datos):
    """
    Renombra las columnas del dataset de chino a español/inglés legible.
    Parámetros:
        datos: DataFrame con columnas a renombrar
    Retorna:
        DataFrame con columnas renombradas
    """
    mapeo_columnas = {
        'ID': 'id_producto',
        '日期': 'fecha',
        '商品名称': 'nombre_producto',
        '价格': 'precio',
        '近30天销量（瓶）': 'ventas_30dias',
        '评论数': 'cantidad_resenas',
        '品牌': 'marca',
        '被搜索次数': 'busquedas'
    }
    
    datos_renombrado = datos.rename(columns=mapeo_columnas)
    print("\nColumnas renombradas correctamente")
    print("Nuevas columnas:", list(datos_renombrado.columns))
    
    return datos_renombrado


def procesar_fecha(datos):
    """
    Convierte la columna de fecha de formato YYYYMM a formato datetime legible.
    Parámetros:
        datos: DataFrame con columna 'fecha'
    Retorna:
        DataFrame con fecha convertida
    """
    datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y%m')
    print("\nFechas convertidas a formato datetime")
    print("Rango de fechas:", datos['fecha'].min(), "hasta", datos['fecha'].max())
    
    return datos


def validar_integridad(datos):
    """
    Valida la integridad del dataset procesado.
    Parámetros:
        datos: DataFrame a validar
    """
    print("\nVALIDACION DE INTEGRIDAD")
    
    print("\nRango de valores numéricos:")
    print("\tPrecio - Min:", datos['precio'].min(), "Max:", datos['precio'].max())
    print("\tVentas 30 días - Min:", datos['ventas_30dias'].min(), "Max:", datos['ventas_30dias'].max())
    print("\tResenas - Min:", datos['cantidad_resenas'].min(), "Max:", datos['cantidad_resenas'].max())
    print("\tBusquedas - Min:", datos['busquedas'].min(), "Max:", datos['busquedas'].max())
    
    print("\nMarcas únicas:", datos['marca'].nunique())
    print("\tTop 10 marcas:")
    for marca, cantidad in datos['marca'].value_counts().head(10).items():
        print(f"\t\t{marca}: {cantidad} productos")
    
    print("\nFechas únicas:", datos['fecha'].nunique())
    print("\tFechas presentes:")
    for fecha in sorted(datos['fecha'].unique()):
        cantidad = len(datos[datos['fecha'] == fecha])
        print(f"\t\t{fecha.strftime('%Y-%m')}: {cantidad} registros")


def guardar_csv(datos, ruta_salida):
    """
    Guarda el DataFrame procesado en formato CSV.
    Parámetros:
        datos: DataFrame a guardar
        ruta_salida: Ruta donde guardar el archivo CSV
    """
    try:
        datos.to_csv(ruta_salida, index=False, encoding='utf-8')
        print("\nArchivo CSV guardado en:", ruta_salida)
        print("Tamaño del archivo:", os.path.getsize(ruta_salida), "bytes")
    except Exception as error:
        print("Error al guardar CSV:", error)


def mostrar_resumen(datos):
    """
    Muestra un resumen final del dataset procesado.
    Parámetros:
        datos: DataFrame procesado
    """
    print("\nRESUMEN FINAL DEL DATASET")
    print("Filas totales:", len(datos))
    print("Columnas totales:", len(datos.columns))
    print("Memoria usada:", datos.memory_usage(deep=True).sum() / 1024**2, "MB")
    print("\nUltimas 5 filas del dataset:")
    print(datos.tail())


def main():
    """
    Función principal que orquesta todo el proceso.
    """
    print("PROTOTIPO: TRANSFORMACION DE DATASET XLSX A CSV")
    
    ruta_xlsx = 'wine_sales_data_20182019.xlsx'
    ruta_csv = 'wine_sales_procesados.csv'
    
    datos = cargar_datos(ruta_xlsx)
    
    if datos is None:
        return
    
    explorar_estructura(datos)
    
    datos = renombrar_columnas(datos)
    
    datos = procesar_fecha(datos)
    
    validar_integridad(datos)
    
    guardar_csv(datos, ruta_csv)
    
    mostrar_resumen(datos)
    
    print("\nPROCESO COMPLETADO")


if __name__ == "__main__":
    main()
