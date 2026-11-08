"""Pruebas simples del pipeline. Ejecutar desde la raíz: py -m pytest -q"""
import pytest

from src.generador_dataset import generar_dataset
from src import pipeline_datos as p


@pytest.fixture(scope="module")
def raw():
    return generar_dataset(n=30_000, semilla=42)


def test_seudonimos_estables_y_distintos(raw):
    a = p.seudonimizar_conductores(raw)
    b = p.seudonimizar_conductores(raw)
    assert a["id_conductor"].equals(b["id_conductor"])                    # mismo conductor, mismo código
    assert a["id_conductor"].nunique() == raw["id_conductor"].nunique()   # conductores distintos, códigos distintos
    assert len(a) == len(raw)


def test_validacion_sin_incumplimientos(raw):
    assert p.validar_calidad(raw)["filas_afectadas"].sum() == 0


def test_modelado_sin_identificadores_ni_posteriores(raw):
    prep = p.generar_caracteristicas(p.seudonimizar_conductores(p.aplicar_limpieza(raw)))
    modelo = p.minimizar_para_modelado(prep)
    assert len(modelo) == len(raw)
    assert not set(p.PROHIBIDAS) & set(modelo.columns)
    assert {"fecha", "retraso"} <= set(modelo.columns)


def test_imputacion_usa_solo_train(raw):
    modelo = p.minimizar_para_modelado(p.generar_caracteristicas(raw)).sort_values("fecha")
    corte = int(len(modelo) * 0.7)
    train = modelo.iloc[:corte].drop(columns=["fecha", "retraso"])
    pre = p.construir_preprocesador().fit(train)
    mediana_aprendida = pre.named_transformers_["num"].statistics_[p.NUMERICAS.index("peso_kg")]
    assert mediana_aprendida == pytest.approx(train["peso_kg"].median())


def test_pipeline_completo(tmp_path, raw):
    entrada = tmp_path / "raw.csv"
    raw.to_csv(entrada, index=False)
    preparado, modelo = p.preparar_datos(entrada, tmp_path / "processed")
    assert (tmp_path / "processed" / "logismart_modelado.csv").exists()
    assert len(preparado) == len(modelo) == len(raw)
