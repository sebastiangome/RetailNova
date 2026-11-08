"""H5-T2: inferencia batch y medición de escalabilidad del modelo OFICIAL LOGISMART.

La lectura usa exclusivamente los 17 predictores; no consulta etiquetas ni
ajusta el modelo/umbral. Esta tarea no implementa aún respaldos H6-T1.
Ejecutar desde la raíz: python main.py --proceso batch
"""
from __future__ import annotations

import gc
import json
import os
import platform
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from src.modelado import FEATURES


def interpretar_volumenes(texto: str | list[int]) -> list[int]:
    """Convierte una lista o texto a volúmenes válidos, sin duplicados."""
    if isinstance(texto, str):
        partes = [t.strip() for t in texto.split(",")]
        if any(not p for p in partes):
            raise ValueError("--volumenes-batch: separa enteros positivos por comas")
        try:
            valores = [int(p) for p in partes]
        except ValueError as exc:
            raise ValueError("--volumenes-batch debe contener enteros") from exc
    else:
        valores = [int(v) for v in texto]
    if not valores or any(v <= 0 for v in valores):
        raise ValueError("Debe haber al menos un volumen positivo")
    if len(set(valores)) != len(valores):
        raise ValueError("No repitas volúmenes")
    return sorted(valores)


def predecir_lotes(
    modelo,
    entrada: str | Path,
    max_filas: int,
    tam_lote: int,
    umbral: float,
    archivo_salida: str | Path | None = None,
) -> tuple[int, int]:
    """Lee el CSV en chunks, predice con el modelo ajustado y opcionalmente guarda un CSV.

    CSV de salida: fila_csv (posición 1-indexada), probabilidad_retraso y alerta_retraso.
    No lee la columna objetivo ni utiliza identificadores. Escribir es opcional en benchmarks.
    """
    if max_filas <= 0 or tam_lote <= 0:
        raise ValueError("max_filas y tam_lote deben ser positivos")
    if not 0 < umbral < 1:
        raise ValueError("umbral debe pertenecer a (0,1)")
    entrada = Path(entrada)
    if not entrada.is_file():
        raise FileNotFoundError(f"No existe {entrada}; ejecuta antes --proceso preparar")

    columnas = set(pd.read_csv(entrada, nrows=0).columns)
    faltantes = set(FEATURES) - columnas
    if faltantes:
        raise ValueError(f"Predictores faltantes: {sorted(faltantes)}")

    salida = Path(archivo_salida) if archivo_salida is not None else None
    temporal = None
    if salida is not None:
        salida.parent.mkdir(parents=True, exist_ok=True)
        temporal = salida.with_name(salida.name + ".tmp")

    n, alertas = 0, 0
    try:
        for bloque in pd.read_csv(entrada, usecols=FEATURES, nrows=max_filas, chunksize=tam_lote):
            p = np.asarray(modelo.predict_proba(bloque.loc[:, FEATURES])[:, 1], dtype=float)
            if len(p) != len(bloque) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError("El modelo entregó probabilidades inválidas")
            decisiones = (p >= umbral).astype(int)
            if temporal is not None:
                predicciones = pd.DataFrame({
                    "fila_csv": np.arange(n + 1, n + len(bloque) + 1),
                    "probabilidad_retraso": p,
                    "alerta_retraso": decisiones,
                })
                predicciones.to_csv(temporal, mode="a" if n else "w", header=(n == 0), index=False,
                                   float_format="%.8f")
            n += len(p)
            alertas += int(decisiones.sum())
        if n != max_filas:
            raise ValueError(f"Solo hay {n:,} registros para un volumen solicitado de {max_filas:,}")
        if temporal is not None and salida is not None:
            os.replace(temporal, salida)
    except BaseException:
        if temporal is not None:
            temporal.unlink(missing_ok=True)
        raise
    return n, alertas


def medir_batch(modelo, entrada, volumen: int, tam_lote: int, umbral: float) -> dict:
    """Mide lectura+inferencia completa; modelo ya cargado, sin escritura de predicciones.

    RSS máximo *observado por sondeo* (20 ms), no máximo absoluto garantizado.
    CPU se mide como tiempo de proceso, sumado entre hilos.
    """
    gc.collect()
    proceso = psutil.Process(os.getpid())
    rss_inicio = proceso.memory_info().rss
    maximo = [rss_inicio]
    fin = threading.Event()

    def muestrear():
        while not fin.wait(0.02):
            try:
                maximo[0] = max(maximo[0], proceso.memory_info().rss)
            except (psutil.Error, OSError):
                return

    muestreador = threading.Thread(target=muestrear, daemon=True)
    muestreador.start()
    t_inicio = time.perf_counter()
    cpu_inicio = time.process_time()
    try:
        n, alertas = predecir_lotes(modelo, entrada, volumen, tam_lote, umbral)
    finally:
        cpu_seg = time.process_time() - cpu_inicio
        wall_seg = time.perf_counter() - t_inicio
        fin.set()
        muestreador.join()
    rss_final = proceso.memory_info().rss
    pico = max(maximo[0], rss_inicio, rss_final)
    mb = 1024 ** 2
    return {
        "volumen": n,
        "alertas": alertas,
        "tiempo_segundos": wall_seg,
        "filas_por_segundo": n / wall_seg,
        "cpu_segundos_proceso": cpu_seg,
        "rss_inicio_mb": rss_inicio / mb,
        "rss_fin_mb": rss_final / mb,
        "rss_pico_observado_mb": pico / mb,
        "rss_incremento_observado_mb": max(0, pico - rss_inicio) / mb,
    }


def crear_graficos(resumen: pd.DataFrame, carpeta: Path) -> None:
    """Figuras de evidencia, fuera de la medición de tiempo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    carpeta.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(resumen["volumen"], resumen["mediana_segundos"], marker="o")
    ax.set(title="H5-T2: tiempo por volumen (mediana)", xlabel="Operaciones en el batch",
           ylabel="Segundos (lectura + inferencia)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(carpeta / "batch_tiempo.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(resumen["volumen"], resumen["max_rss_pico_mb"], marker="o")
    ax.set(title="H5-T2: RAM de proceso observada", xlabel="Operaciones en el batch",
           ylabel="RSS máximo muestreado (MB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(carpeta / "batch_memoria.png", dpi=140)
    plt.close(fig)


def ejecutar_pruebas_batch(
    entrada="data/processed/logismart_modelado.csv",
    ruta_modelo="models/modelo_logismart.joblib",
    ruta_resumen="output/resultados/modelado/resultados_modelado.json",
    volumenes: str | list[int] = "1000,10000,75000,150000",
    tam_lote=10000,
    repeticiones=3,
    ventana_minutos: float | None = None,
    carpeta_resultados="output/resultados/batch",
    carpeta_graficos="output/graficos",
    salida_ejemplo="output/batch/predicciones_ejemplo.csv",
) -> dict:
    """Ejecuta un benchmark sin reentrenar y conserva resultados agregados para H5-T2."""
    tamanos = interpretar_volumenes(volumenes)
    if tam_lote <= 0 or repeticiones <= 0:
        raise ValueError("tam_lote y repeticiones deben ser positivos")
    if ventana_minutos is not None and ventana_minutos <= 0:
        raise ValueError("ventana_minutos debe ser positiva")
    ruta_resumen, ruta_modelo = Path(ruta_resumen), Path(ruta_modelo)
    for ruta in (ruta_resumen, ruta_modelo):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}; primero entrena el modelo oficial")
    info_modelo = json.loads(ruta_resumen.read_text(encoding="utf-8"))
    if info_modelo.get("modo_rapido"):
        raise ValueError("H5-T2 requiere el modelo oficial entrenado en modo completo")
    if info_modelo.get("modelo_elegido") != "XGBoost":
        raise ValueError("El modelo oficial no coincide con XGBoost; revisa el resumen")
    umbral = float(info_modelo["umbral_elegido_en_validacion"])
    modelo = joblib.load(ruta_modelo)  # Solo cargar artefactos .joblib propios y confiables.

    # Calentamiento excluido de los tiempos. No se usan etiquetas ni se entrenan modelos.
    predecir_lotes(modelo, entrada, min(1000, tamanos[0]), tam_lote, umbral)
    filas = []
    for volumen in tamanos:
        for repeticion in range(1, repeticiones + 1):
            r = medir_batch(modelo, entrada, volumen, tam_lote, umbral)
            filas.append({"repeticion": repeticion, **r})
            print(f"  {volumen:>7,} filas | repetición {repeticion} | {r['tiempo_segundos']:.3f} s "
                  f"| {r['filas_por_segundo']:,.0f} filas/s | RSS pico ~{r['rss_pico_observado_mb']:.1f} MB")

    mediciones = pd.DataFrame(filas)
    resumen = mediciones.groupby("volumen", sort=True).agg(
        mediana_segundos=("tiempo_segundos", "median"),
        min_segundos=("tiempo_segundos", "min"),
        max_segundos=("tiempo_segundos", "max"),
        mediana_filas_por_segundo=("filas_por_segundo", "median"),
        mediana_cpu_segundos=("cpu_segundos_proceso", "median"),
        max_rss_pico_mb=("rss_pico_observado_mb", "max"),
        mediana_rss_incremento_mb=("rss_incremento_observado_mb", "median"),
    ).reset_index()

    carpeta_resultados = Path(carpeta_resultados)
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    mediciones.to_csv(carpeta_resultados / "batch_mediciones.csv", index=False)
    resumen.to_csv(carpeta_resultados / "batch_resumen.csv", index=False)
    crear_graficos(resumen, Path(carpeta_graficos))
    muestra = min(1000, tamanos[0])
    predecir_lotes(modelo, entrada, muestra, tam_lote, umbral, archivo_salida=salida_ejemplo)

    ventana = None
    if ventana_minutos is not None:
        # Evalúa SOLO el tamaño mayor realmente probado, sin extrapolaciones.
        ultimo = resumen.iloc[-1]
        ventana = {
            "volumen_evaluado": int(ultimo["volumen"]),
            "ventana_minutos_requerida": ventana_minutos,
            "mediana_segundos_observada": float(ultimo["mediana_segundos"]),
            "cumple_en_entorno_de_desarrollo": bool(ultimo["mediana_segundos"] <= ventana_minutos * 60),
            "nota": "No equivale a certificar rendimiento productivo ni una ejecución nocturna real",
        }

    informe = {
        "tarea": "H5-T2",
        "datos": "Sintéticos; lecturas únicamente de predictores de CSV preparado",
        "modelo": info_modelo["modelo_elegido"],
        "umbral_congelado_en_validacion": umbral,
        "volumenes": tamanos,
        "tam_lote": tam_lote,
        "repeticiones": repeticiones,
        "medicion": "Tiempo total de lectura CSV y predict_proba por lotes; modelo ya cargado y calentado; sin persistir CSV dentro del benchmark",
        "memoria": "RSS total del proceso muestreado cada ~20 ms (no memoria máxima exacta ni por inferencia individual)",
        "cpu": "time.process_time() mide segundos de CPU de proceso, sumando hilos; no es porcentaje de utilización",
        "entorno": {"sistema": platform.system(), "python": platform.python_version(),
                    "cpu_logicas": psutil.cpu_count(logical=True),
                    "ram_sistema_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2)},
        "resultados": resumen.to_dict(orient="records"),
        "ventana_operacional": ventana,
        "limitaciones": [
            "Datos sintéticos, no carga real de infraestructura productiva",
            "La lectura desde disco depende de la caché, velocidad del dispositivo y procesos concurrentes",
            "RSS incluye Python, librerías, el modelo cargado y asignaciones previas; el sondeo puede omitir picos breves",
            "Benchmark sobre los predictores del CSV preparado, sin medir generación/preparación ETL ni escritura masiva de resultados",
            "Las tasas de alertas no representan desempeño predictivo; no se leyeron etiquetas y no se ajustaron umbrales",
            "Sin ventana objetivo explícita no es posible afirmar cumplimiento de una ventana nocturna",
        ],
        "ejemplo_predicciones": str(salida_ejemplo),
    }
    (carpeta_resultados / "batch_rendimiento.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nH5-T2: resumen de pruebas batch (medianas)")
    print(resumen[["volumen", "mediana_segundos", "mediana_filas_por_segundo", "max_rss_pico_mb"]]
          .round(3).to_string(index=False))
    print("El modelo oficial y su umbral no fueron modificados; no se usaron etiquetas.")
    print(f"Evidencias en {carpeta_resultados}, {carpeta_graficos} y {salida_ejemplo}")
    return informe


if __name__ == "__main__":
    ejecutar_pruebas_batch()
