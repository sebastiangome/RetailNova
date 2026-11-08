"""Pruebas H6-T1: integración lógica, bitácoras y respaldo local verificable.

El modelo es un doble de prueba; no se carga ni se modifica el XGBoost oficial.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import automatizacion as auto
from src.modelado import FEATURES


class ModeloPrueba:
    def predict_proba(self, datos):
        probas = np.where(datos["distancia_km"].to_numpy() > 1.5, .8, .1)
        return np.column_stack((1 - probas, probas))


def _etl_simulado(entrada, salida, muestra):
    """Produce los mismos archivos y firma que preparar_datos, para aislar la orquestación."""
    total = min(12, muestra) if muestra is not None else 12
    datos = {}
    categoricas = {"comuna_origen", "comuna_destino", "tipo_vehiculo", "franja_horaria"}
    for columna in FEATURES:
        datos[columna] = ["Santiago"] * total if columna in categoricas else [1.0] * total
    datos["distancia_km"] = [2.0 if i % 2 else 1.0 for i in range(total)]
    preparado = pd.DataFrame(datos)
    preparado["retraso"] = 1  # La etiqueta nunca debe aparecer en el CSV de predicciones.
    Path(salida).mkdir(parents=True, exist_ok=True)
    preparado.to_csv(Path(salida) / "logismart_modelado.csv", index=False)
    preparado.to_csv(Path(salida) / "logismart_preparado.csv", index=False)
    return preparado, preparado.copy()


def _archivos(tmp_path, monkeypatch):
    entrada = tmp_path / "raw.csv"
    entrada.write_text("datos sinteticos de prueba\n", encoding="utf-8")
    modelo = tmp_path / "modelo.joblib"
    modelo.write_bytes(b"doble de prueba; no cargar en produccion")
    resumen = tmp_path / "resultados_modelado.json"
    resumen.write_text(json.dumps({
        "modo_rapido": False, "modelo_elegido": "XGBoost",
        "umbral_elegido_en_validacion": .15,
    }), encoding="utf-8")
    monkeypatch.setattr(auto.joblib, "load", lambda ruta: ModeloPrueba())
    monkeypatch.setattr(auto, "preparar_datos", _etl_simulado)
    return entrada, modelo, resumen


def test_ejecucion_local_completa_y_respaldo_verificado(tmp_path, monkeypatch):
    entrada, modelo, resumen = _archivos(tmp_path, monkeypatch)
    resultado = auto.ejecutar_automatizacion(
        entrada=entrada, ruta_modelo=modelo, ruta_resumen=resumen,
        carpeta_ejecuciones=tmp_path / "ejecuciones",
        carpeta_respaldos=tmp_path / "respaldos",
        max_filas=12, tam_lote=5,
    )
    assert resultado["estado"] == "completado"
    assert resultado["resultados"]["filas_procesadas"] == 12
    assert resultado["resultados"]["alertas_emitidas"] == 6
    assert resultado["resultados"]["umbral_oficial"] == pytest.approx(.15)
    assert set(resultado["etapas"]) == {"carga_modelo", "preparacion", "inferencia"}
    corrida = tmp_path / "ejecuciones" / resultado["identificador_ejecucion"]
    tabla = pd.read_csv(corrida / "predicciones.csv")
    assert list(tabla.columns) == ["fila_csv", "probabilidad_retraso", "alerta_retraso"]
    assert "retraso" not in tabla.columns
    assert tabla["fila_csv"].tolist() == list(range(1, 13))
    assert list(corrida.glob("etl_*")) == []
    zip_path = Path(resultado["respaldo"]["ruta"])
    assert auto.verificar_respaldo(zip_path)
    with zipfile.ZipFile(zip_path) as zipf:
        assert set(zipf.namelist()) == {"predicciones.csv", "registro.json", "manifest.json"}
        assert zipf.read("registro.json") == (corrida / "registro.json").read_bytes()
        assert zipf.read("predicciones.csv") == (corrida / "predicciones.csv").read_bytes()
    assert modelo.read_bytes() == b"doble de prueba; no cargar en produccion"


def test_falla_deja_registro_de_error_y_no_respaldo(tmp_path, monkeypatch):
    _, modelo, resumen = _archivos(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="CSV original"):
        auto.ejecutar_automatizacion(
            entrada=tmp_path / "no-existe.csv", ruta_modelo=modelo, ruta_resumen=resumen,
            carpeta_ejecuciones=tmp_path / "ejecuciones",
            carpeta_respaldos=tmp_path / "respaldos",
        )
    registros = list((tmp_path / "ejecuciones").glob("*/registro.json"))
    assert len(registros) == 1
    registro = json.loads(registros[0].read_text(encoding="utf-8"))
    assert registro["estado"] == "error"
    assert registro["error"]["tipo"] == "FileNotFoundError"
    assert list((tmp_path / "respaldos").glob("*.zip")) == [] if (tmp_path / "respaldos").exists() else True


def test_valida_parametros_y_artefacto_oficial(tmp_path, monkeypatch):
    entrada, modelo, resumen = _archivos(tmp_path, monkeypatch)
    for cambio in ({"max_filas": 0}, {"tam_lote": 0}):
        with pytest.raises(ValueError):
            auto.ejecutar_automatizacion(
                entrada=entrada, ruta_modelo=modelo, ruta_resumen=resumen,
                carpeta_ejecuciones=tmp_path / "ejecuciones",
                carpeta_respaldos=tmp_path / "respaldos", **cambio,
            )
    resumen.write_text(json.dumps({
        "modo_rapido": True, "modelo_elegido": "XGBoost",
        "umbral_elegido_en_validacion": .15,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="modelo oficial"):
        auto.ejecutar_automatizacion(
            entrada=entrada, ruta_modelo=modelo, ruta_resumen=resumen,
            carpeta_ejecuciones=tmp_path / "ejecuciones",
            carpeta_respaldos=tmp_path / "respaldos",
        )


def test_detecta_respaldo_alterado(tmp_path):
    predicciones = tmp_path / "predicciones.csv"
    registro = tmp_path / "registro.json"
    predicciones.write_text("fila_csv,alerta_retraso\n1,0\n", encoding="utf-8")
    registro.write_text('{"estado":"completado"}', encoding="utf-8")
    zip_original = auto.crear_respaldo(predicciones, registro, tmp_path / "ok.zip")
    assert auto.verificar_respaldo(zip_original)
    zip_malicioso = tmp_path / "alterado.zip"
    with zipfile.ZipFile(zip_original) as src, zipfile.ZipFile(zip_malicioso, "w") as dst:
        for nombre in src.namelist():
            contenido = src.read(nombre)
            if nombre == "predicciones.csv":
                contenido = contenido.replace(b"1,0", b"1,1")
            dst.writestr(nombre, contenido)
    with pytest.raises(ValueError, match="integridad inválida"):
        auto.verificar_respaldo(zip_malicioso)
