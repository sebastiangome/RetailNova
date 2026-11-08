"""H6-T1: orquestación local demostrativa de ETL, inferencia y respaldos.

Reutiliza H3-T4 (`preparar_datos`) y H5-T2 (`predecir_lotes`). No entrena modelos,
no modifica el umbral y no implementa conectores a PostgreSQL ni Amazon S3.
La entrada es el CSV histórico sintético del proyecto (incluye la etiqueta para
validación de calidad del ETL); la inferencia solo lee las 17 FEATURES.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.batch import predecir_lotes
from src.pipeline_datos import preparar_datos


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(archivo: Path) -> str:
    digest = hashlib.sha256()
    with archivo.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _escribir_json(archivo: Path, datos: dict) -> None:
    """Evita registros JSON truncados si una escritura falla a mitad de camino."""
    temporal = archivo.with_suffix(archivo.suffix + ".tmp")
    try:
        temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporal, archivo)
    finally:
        temporal.unlink(missing_ok=True)


def verificar_respaldo(archivo: str | Path) -> bool:
    """Valida estructura, CRC y SHA-256 de los archivos contenidos en el ZIP."""
    ruta = Path(archivo)
    with zipfile.ZipFile(ruta, "r") as paquete:
        if paquete.testzip() is not None:
            raise ValueError("El ZIP no superó la verificación CRC")
        if "manifest.json" not in paquete.namelist():
            raise ValueError("El respaldo carece de manifest.json")
        manifest = json.loads(paquete.read("manifest.json"))
        if manifest.get("version") != 1 or set(manifest.get("archivos", {})) != {
            "predicciones.csv", "registro.json"
        }:
            raise ValueError("Manifest de respaldo incompleto o incompatible")
        for nombre, esperado in manifest["archivos"].items():
            if nombre not in paquete.namelist():
                raise ValueError(f"Falta {nombre} en el respaldo")
            digest = hashlib.sha256()
            total = 0
            with paquete.open(nombre) as f:
                for bloque in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(bloque)
                    total += len(bloque)
            if digest.hexdigest() != esperado["sha256"] or total != esperado["bytes"]:
                raise ValueError(f"El respaldo tiene integridad inválida: {nombre}")
    return True


def crear_respaldo(predicciones: Path, registro: Path, destino: Path) -> Path:
    """Copia verificable de resultados; NO equivale a un respaldo externo/off-site."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "tipo": "respaldo_local_resultados_logismart_h6_t1",
        "archivos": {
            archivo.name: {"sha256": _sha256(archivo), "bytes": archivo.stat().st_size}
            for archivo in (predicciones, registro)
        },
    }
    temporal = destino.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporal, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for archivo in (predicciones, registro):
                zipf.write(archivo, arcname=archivo.name)
            zipf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        verificar_respaldo(temporal)
        os.replace(temporal, destino)
    finally:
        temporal.unlink(missing_ok=True)
    return destino


def ejecutar_automatizacion(
    entrada: str | Path = "data/raw/logismart_entregas_simuladas.csv",
    ruta_modelo: str | Path = "models/modelo_logismart.joblib",
    ruta_resumen: str | Path = "output/resultados/modelado/resultados_modelado.json",
    carpeta_ejecuciones: str | Path = "output/ejecuciones",
    carpeta_respaldos: str | Path = "output/respaldos",
    max_filas: int | None = None,
    tam_lote: int = 10_000,
) -> dict:
    """Ejecuta una corrida manual reproducible, con registro de éxito o error.

    Cada corrida tiene un identificador propio: no sobrescribe resultados anteriores.
    El CSV histórico original y el artefacto .joblib no se modifican.
    """
    carpeta_ejecuciones = Path(carpeta_ejecuciones)
    carpeta_respaldos = Path(carpeta_respaldos)
    identificador = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + secrets.token_hex(3)
    carpeta_corrida = carpeta_ejecuciones / identificador
    carpeta_corrida.mkdir(parents=True, exist_ok=False)
    ruta_registro = carpeta_corrida / "registro.json"
    ruta_predicciones = carpeta_corrida / "predicciones.csv"
    ruta_respaldo = carpeta_respaldos / f"{identificador}.zip"

    inicio = time.perf_counter()
    registro: dict = {
        "tarea": "H6-T1",
        "identificador_ejecucion": identificador,
        "modalidad": "demostracion_local_manual",
        "estado": "en_ejecucion",
        "fecha_inicio_utc": _ahora(),
        "entrada": str(Path(entrada)),
        "modelo": str(Path(ruta_modelo)),
        "etapas": {},
        "integraciones": {"postgresql": "no_implementada", "s3": "no_implementada"},
        "limitaciones": [
            "Ejecución local bajo demanda; no hay planificador ni infraestructura productiva",
            "Entrada histórica sintética requiere la etiqueta retraso durante ETL; la inferencia no la lee",
            "Respaldo ZIP en disco local; sin copia externa ni política de retención automatizada",
            "Solo resultados de predicción y bitácora son respaldados; no se incluye dataset ni modelo",
        ],
    }
    _escribir_json(ruta_registro, registro)

    try:
        if max_filas is not None and max_filas <= 0:
            raise ValueError("max_filas debe ser un entero positivo")
        if tam_lote <= 0:
            raise ValueError("tam_lote debe ser positivo")
        if not Path(entrada).is_file():
            raise FileNotFoundError(f"No se encuentra el CSV original: {entrada}")
        if not Path(ruta_modelo).is_file() or not Path(ruta_resumen).is_file():
            raise FileNotFoundError("Falta el modelo oficial o resultados_modelado.json")

        t0 = time.perf_counter()
        resumen = json.loads(Path(ruta_resumen).read_text(encoding="utf-8"))
        if resumen.get("modo_rapido") or resumen.get("modelo_elegido") != "XGBoost":
            raise ValueError("H6-T1 requiere el modelo oficial XGBoost entrenado en modo completo")
        umbral = float(resumen["umbral_elegido_en_validacion"])
        if not 0 < umbral < 1:
            raise ValueError("Umbral oficial inválido")
        # Solo cargar archivos joblib propios y confiables.
        modelo = joblib.load(ruta_modelo)
        registro["etapas"]["carga_modelo"] = {"segundos": round(time.perf_counter() - t0, 4)}

        # El CSV intermedio se produce en carpeta temporal y se elimina al finalizar.
        with tempfile.TemporaryDirectory(prefix="etl_", dir=carpeta_corrida) as directorio_temporal:
            t0 = time.perf_counter()
            preparado, modelado = preparar_datos(
                entrada=entrada, salida=directorio_temporal, muestra=max_filas
            )
            filas = len(modelado)
            if filas <= 0 or len(preparado) != filas:
                raise ValueError("El ETL no produjo una cantidad válida de operaciones")
            registro["etapas"]["preparacion"] = {
                "segundos": round(time.perf_counter() - t0, 4), "filas": filas
            }
            del preparado, modelado

            t0 = time.perf_counter()
            filas_predichas, alertas = predecir_lotes(
                modelo=modelo,
                entrada=Path(directorio_temporal) / "logismart_modelado.csv",
                max_filas=filas,
                tam_lote=tam_lote,
                umbral=umbral,
                archivo_salida=ruta_predicciones,
            )
            if filas_predichas != filas:
                raise ValueError("La inferencia no procesó todas las filas preparadas")
            registro["etapas"]["inferencia"] = {
                "segundos": round(time.perf_counter() - t0, 4), "filas": filas_predichas
            }

        registro["resultados"] = {
            "filas_procesadas": filas_predichas,
            "alertas_emitidas": alertas,
            "tasa_alertas": alertas / filas_predichas,
            "umbral_oficial": umbral,
            "archivo_predicciones": str(ruta_predicciones),
            "sha256_predicciones": _sha256(ruta_predicciones),
        }
        registro["estado"] = "completado"
        registro["fecha_fin_utc"] = _ahora()
        registro["duracion_segundos_sin_respaldo"] = round(time.perf_counter() - inicio, 4)
        registro["respaldo"] = {"ruta": str(ruta_respaldo), "verificado": True}
        _escribir_json(ruta_registro, registro)

        crear_respaldo(ruta_predicciones, ruta_registro, ruta_respaldo)
        # Se conserva exactamente el mismo registro en el ZIP y en el directorio local.
        # La duración registrada excluye el tiempo de compresión/verificación del ZIP.
        print(f"H6-T1 completada: {filas_predichas:,} filas; {alertas:,} alertas; umbral {umbral:.2f}")
        print(f"Bitácora: {ruta_registro}\nPredicciones: {ruta_predicciones}")
        print(f"Respaldo local verificado: {ruta_respaldo}")
        return registro
    except Exception as exc:
        registro["estado"] = "error"
        registro["fecha_fin_utc"] = _ahora()
        registro["duracion_segundos"] = round(time.perf_counter() - inicio, 4)
        registro["error"] = {"tipo": type(exc).__name__, "mensaje": str(exc)}
        registro["respaldo"] = {"ruta": str(ruta_respaldo), "verificado": False}
        _escribir_json(ruta_registro, registro)
        ruta_respaldo.unlink(missing_ok=True)
        print(f"H6-T1 falló ({type(exc).__name__}): {exc}. Consulta {ruta_registro}")
        raise
