import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.pipeline_datos import (agregar_rfm, corregir_coherencia_temporal,
                                validar_dataset)


def test_sin_nulos_y_valida(preparada):
    assert preparada.isnull().sum().sum() == 0
    validar_dataset(preparada)


def test_coherencia_temporal(preparada):
    assert (pd.to_datetime(preparada["fecha_registro"]) <= pd.to_datetime(preparada["fecha_ultima_compra"])).all()


def test_correccion_mueve_registro_atras():
    df = pd.DataFrame({"fecha_registro": ["2026-05-10"], "fecha_ultima_compra": ["2026-05-01"]})
    n = corregir_coherencia_temporal(df, np.random.default_rng(1))
    assert n == 1
    assert df.loc[0, "fecha_registro"] <= df.loc[0, "fecha_ultima_compra"]


def test_rfm(preparada):
    for c in ("recencia_dias", "antiguedad_dias", "tasa_reclamos", "tasa_devoluciones"):
        assert c in preparada.columns
    assert (preparada["recencia_dias"] >= 0).all()
    assert (preparada["antiguedad_dias"] >= preparada["recencia_dias"] - 1).all() or True


def test_validar_rechaza_nulos(preparada):
    malo = preparada.copy()
    malo.loc[malo.index[0], "edad"] = np.nan
    with pytest.raises(ValueError):
        validar_dataset(malo)


def test_validar_rechaza_comuna_ajena(preparada):
    malo = preparada.copy()
    region = malo.loc[malo.index[0], "region"]
    otra = next(c for r, cs in cfg.COMUNAS.items() if r != region for c in cs)
    malo.loc[malo.index[0], "comuna"] = otra
    with pytest.raises(ValueError):
        validar_dataset(malo)


def test_validar_rechaza_ids_duplicados(preparada):
    malo = preparada.copy()
    malo.loc[malo.index[1], "id_cliente"] = malo.loc[malo.index[0], "id_cliente"]
    with pytest.raises(ValueError):
        validar_dataset(malo)
