"""LOGISMART: CLI de generación, preparación, entrenamiento y auditoría.

Sin argumentos genera el dataset, como la versión anterior.
"""
import argparse
from pathlib import Path
import pandas as pd
from src.generador_dataset import generar_dataset

FECHA_INICIO = pd.Timestamp("2025-08-01")
FECHA_FIN = pd.Timestamp("2026-09-30")
COLUMNAS_ESPERADAS = 21


def validar(df: pd.DataFrame, n: int) -> None:
    if len(df) != n:
        raise ValueError(f"Filas inesperadas: {len(df)} (esperadas: {n})")
    if df.shape[1] != COLUMNAS_ESPERADAS:
        raise ValueError(f"Columnas inesperadas: {df.shape[1]}")
    if df["id_entrega"].isna().any() or not df["id_entrega"].is_unique:
        raise ValueError("Identificadores nulos o duplicados")
    if df["fecha"].isna().any() or not df["fecha"].between(FECHA_INICIO, FECHA_FIN).all():
        raise ValueError("Hay fechas inválidas o fuera del período")
    if df["retraso"].isna().any() or not df["retraso"].isin([0, 1]).all():
        raise ValueError("Indicador de retraso inválido o nulo")
    if df["estado_final"].isna().any() or not (
        df["estado_final"] == df["retraso"].map({0: "Entregada a tiempo", 1: "Entregada con retraso"})
    ).all():
        raise ValueError("Estado final incoherente con el objetivo")
    if (df["n_paradas"] < 1).any() or (df["n_paquetes"] < 1).any():
        raise ValueError("Paquetes o paradas fuera de rango")


def generar(registros: int, semilla: int, salida: Path) -> None:
    if registros <= 0:
        raise ValueError("--registros debe ser mayor que cero")
    print(f"Generando {registros:,} operaciones sintéticas (semilla {semilla})...")
    df = generar_dataset(n=registros, semilla=semilla)
    validar(df, registros)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False, date_format="%Y-%m-%d")
    recuperado = pd.read_csv(salida, parse_dates=["fecha"])
    validar(recuperado, registros)
    if list(recuperado.columns) != list(df.columns):
        raise ValueError("Orden o nombres de columnas no coinciden tras exportar")
    resumen = recuperado.groupby(recuperado["fecha"].dt.to_period("M")).agg(
        operaciones=("id_entrega", "size"), retrasos=("retraso", "sum"), tasa_retraso=("retraso", "mean")
    )
    resumen["tasa_retraso_pct"] = (resumen["tasa_retraso"] * 100).round(2)
    nulos = recuperado.isna().sum()
    print(f"\nArchivo: {salida.resolve()}")
    print(f"Filas: {len(recuperado):,} | Columnas: {recuperado.shape[1]}")
    print(f"Meses representados: {len(resumen)} de 14")
    print(f"Promedio por mes representado: {len(recuperado) / len(resumen):,.0f} operaciones")
    print("\nResumen mensual (tasas calculadas, no impuestas):")
    print(resumen[["operaciones", "retrasos", "tasa_retraso_pct"]].to_string())
    print("\nNulos por columna:")
    print(nulos[nulos > 0].to_string() if (nulos > 0).any() else "Ninguno")
    print("\nValidaciones básicas: OK")


def ejecutar() -> None:
    ap = argparse.ArgumentParser(description="LOGISMART: generación, preparación, entrenamiento y auditoría")
    ap.add_argument("--proceso", choices=["generar", "preparar", "entrenar", "evaluar", "explicar", "optimizar", "sesgos", "evaluacion-final", "batch", "equidad", "validacion-negocio", "automatizar", "visualizar", "completo"], default="generar")
    ap.add_argument("--registros", type=int, default=1_050_000)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--salida", type=Path, default=Path("data/raw/logismart_entregas_simuladas.csv"),
                    help="Ruta CSV de generación, conserva la CLI anterior")
    ap.add_argument("--modo-rapido", action="store_true", help="Limitar entrenamiento para pruebas rápidas")
    ap.add_argument("--muestra-shap", type=int, default=500, help="Operaciones de validación usadas para SHAP")
    ap.add_argument("--max-filas-opt", type=int, default=100_000, help="Muestra máxima de entrenamiento para optimización")
    ap.add_argument("--iteraciones-opt", type=int, default=4, help="Configuraciones aleatorias por modelo")
    ap.add_argument("--entrada-batch", default="data/processed/logismart_modelado.csv",
                    help="CSV preparado; utiliza solo predictores y nunca la etiqueta retraso")
    ap.add_argument("--volumenes-batch", default="1000,10000,75000,150000",
                    help="Volúmenes a medir, separados por coma")
    ap.add_argument("--tam-lote-batch", type=int, default=10000,
                    help="Registros por chunk de lectura/inferencia")
    ap.add_argument("--repeticiones-batch", type=int, default=3)
    ap.add_argument("--ventana-minutos-batch", type=float, default=None,
                    help="Ventana operacional objetivo en minutos; opcional")
    ap.add_argument("--capacidad-alertas-dia", type=int, default=None,
                    help="Capacidad diaria supuesta para simular atención de alertas, no un SLA real")
    ap.add_argument("--entrada-automatizacion", default="data/raw/logismart_entregas_simuladas.csv",
                    help="CSV histórico sintético de entrada para H6-T1")
    ap.add_argument("--max-filas-automatizacion", type=int, default=None,
                    help="Limita filas para prueba local; por defecto procesa todo el CSV")
    ap.add_argument("--tam-lote-automatizacion", type=int, default=10_000,
                    help="Filas por bloque de inferencia en H6-T1")
    ap.add_argument("--carpeta-ejecuciones", default="output/ejecuciones",
                    help="Directorio de predicciones y bitácoras H6-T1")
    ap.add_argument("--carpeta-respaldos", default="output/respaldos",
                    help="Directorio de respaldos ZIP locales H6-T1")
    ap.add_argument("--registro-visualizacion", default=None,
                    help="Ruta opcional de registro.json de H6-T1; usa última corrida exitosa por defecto")
    args = ap.parse_args()
    if args.proceso in ("generar", "completo"):
        generar(args.registros, args.semilla, args.salida)
    if args.proceso in ("preparar", "completo"):
        from src.pipeline_datos import preparar_datos
        preparar_datos(entrada=args.salida)
    if args.proceso in ("entrenar", "completo"):
        from src.modelado import entrenar
        entrenar(modo_rapido=args.modo_rapido, semilla=args.semilla)
    if args.proceso == "optimizar":
        from src.optimizacion import optimizar
        optimizar(max_filas=args.max_filas_opt, iteraciones=args.iteraciones_opt, semilla=args.semilla)
    if args.proceso == "automatizar":
        from src.automatizacion import ejecutar_automatizacion
        ejecutar_automatizacion(
            entrada=args.entrada_automatizacion,
            max_filas=args.max_filas_automatizacion,
            tam_lote=args.tam_lote_automatizacion,
            carpeta_ejecuciones=args.carpeta_ejecuciones,
            carpeta_respaldos=args.carpeta_respaldos,
        )
    if args.proceso == "visualizar":
        from src.visualizacion import generar_visualizacion
        generar_visualizacion(registro=args.registro_visualizacion,
                              carpeta_ejecuciones=args.carpeta_ejecuciones)
    if args.proceso == "batch":
        from src.batch import ejecutar_pruebas_batch
        ejecutar_pruebas_batch(
            entrada=args.entrada_batch,
            volumenes=args.volumenes_batch,
            tam_lote=args.tam_lote_batch,
            repeticiones=args.repeticiones_batch,
            ventana_minutos=args.ventana_minutos_batch,
        )
    if args.proceso == "evaluacion-final":
        from src.evaluacion import comparar_modelos_test
        comparar_modelos_test(semilla=args.semilla)
    if args.proceso == "sesgos":
        from src.sesgos import analizar_sesgos
        analizar_sesgos()
    if args.proceso == "equidad":
        from src.equidad_territorial import auditar_equidad
        auditar_equidad()
    if args.proceso == "validacion-negocio":
        from src.validacion_negocio import validar_negocio
        validar_negocio(capacidad_alertas_dia=args.capacidad_alertas_dia)
    if args.proceso == "evaluar" or (args.proceso == "completo" and not args.modo_rapido):
        from src.evaluacion import evaluar
        evaluar()
    if args.proceso == "explicar" or (args.proceso == "completo" and not args.modo_rapido):
        from src.explicabilidad import explicar
        explicar(muestra=args.muestra_shap, semilla=args.semilla)


if __name__ == "__main__":
    ejecutar()
