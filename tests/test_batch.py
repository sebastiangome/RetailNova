"""Pruebas aisladas de H5-T2: no requieren el modelo de 825k filas ni archivos reales."""
import json

import numpy as np
import pandas as pd
import pytest

from src.batch import interpretar_volumenes, predecir_lotes, ejecutar_pruebas_batch
from src.modelado import FEATURES
from src.pipeline_datos import CATEGORICAS


class ModeloPrueba:
    def predict_proba(self, X):
        assert list(X.columns) == FEATURES
        prob = np.where(X["es_hora_punta"].astype(int).to_numpy() == 1, 0.8, 0.2)
        return np.column_stack([1 - prob, prob])


def datos_prueba(filas=50):
    d = {}
    for c in FEATURES:
        d[c] = ["Santiago"] * filas if c in CATEGORICAS else [2.0] * filas
    d["es_hora_punta"] = [i % 2 for i in range(filas)]
    d["retraso"] = [0] * filas  # La variable objetivo no debe leerse para inferencia.
    return pd.DataFrame(d)


def test_parsear_volumenes():
    assert interpretar_volumenes("100,10,50") == [10, 50, 100]
    for texto in ("", "1,1", "0,1", "a,5", "1,,5"):
        with pytest.raises(ValueError):
            interpretar_volumenes(texto)


def test_batch_guarda_predicciones_sin_etiqueta(tmp_path):
    entrada = tmp_path / "entrada.csv"
    salida = tmp_path / "predicciones.csv"
    datos_prueba(12).to_csv(entrada, index=False)
    n, alertas = predecir_lotes(ModeloPrueba(), entrada, max_filas=11, tam_lote=4,
                                umbral=0.5, archivo_salida=salida)
    assert (n, alertas) == (11, 5)
    pred = pd.read_csv(salida)
    assert list(pred.columns) == ["fila_csv", "probabilidad_retraso", "alerta_retraso"]
    assert pred["fila_csv"].tolist() == list(range(1, 12))
    assert pred["alerta_retraso"].tolist() == [i % 2 for i in range(11)]
    assert "retraso" not in pred.columns


def test_no_producir_salida_incompleta(tmp_path):
    entrada = tmp_path / "corto.csv"
    salida = tmp_path / "predicciones.csv"
    datos_prueba(8).to_csv(entrada, index=False)
    with pytest.raises(ValueError, match="Solo hay"):
        predecir_lotes(ModeloPrueba(), entrada, max_filas=12, tam_lote=4,
                       umbral=0.5, archivo_salida=salida)
    assert not salida.exists()
    with pytest.raises(ValueError, match="Predictores faltantes"):
        pd.DataFrame({"es_hora_punta": [0]}).to_csv(entrada, index=False)
        predecir_lotes(ModeloPrueba(), entrada, max_filas=1, tam_lote=1, umbral=0.5)


def test_benchmark_pequeno_y_ventana_opcional(tmp_path, monkeypatch):
    entrada = tmp_path / "entrada.csv"
    modelo = tmp_path / "modelo.joblib"
    resumen = tmp_path / "resultados_modelado.json"
    datos_prueba(50).to_csv(entrada, index=False)
    modelo.write_bytes(b"archivo simulado")
    resumen.write_text(json.dumps({"modo_rapido": False, "modelo_elegido": "XGBoost",
                                     "umbral_elegido_en_validacion": 0.5}))
    monkeypatch.setattr("src.batch.joblib.load", lambda path: ModeloPrueba())
    informe = ejecutar_pruebas_batch(
        entrada=entrada, ruta_modelo=modelo, ruta_resumen=resumen,
        volumenes=[10, 30], tam_lote=7, repeticiones=2, ventana_minutos=1,
        carpeta_resultados=tmp_path / "resultados", carpeta_graficos=tmp_path / "graficos",
        salida_ejemplo=tmp_path / "batch" / "ejemplo.csv",
    )
    assert [x["volumen"] for x in informe["resultados"]] == [10, 30]
    assert informe["ventana_operacional"]["volumen_evaluado"] == 30
    assert len(pd.read_csv(tmp_path / "resultados" / "batch_mediciones.csv")) == 4
    assert len(pd.read_csv(tmp_path / "batch" / "ejemplo.csv")) == 10
    assert (tmp_path / "graficos" / "batch_tiempo.png").exists()
    assert (tmp_path / "resultados" / "batch_rendimiento.json").exists()
