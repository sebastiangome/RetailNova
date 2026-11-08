import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.modelado import (dividir_datos, elegir_modelo, seleccionar_umbral_oof)


def test_recencia_no_es_feature():
    assert cfg.VARIABLE_FUGA not in cfg.FEATURES_NUMERICAS + cfg.FEATURES_CATEGORICAS


def test_dividir_estratificado(preparada):
    X_tr, X_te, y_tr, y_te, r_tr, r_te = dividir_datos(preparada)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    assert cfg.VARIABLE_FUGA not in X_tr.columns
    assert len(X_tr) + len(X_te) == len(preparada)


def test_desempate_prefiere_modelo_simple():
    comp = pd.DataFrame({"Modelo": ["Random Forest", cfg.MODELO_SIMPLE], "ROC_AUC_CV": [0.909, 0.908]})
    assert elegir_modelo(comp) == cfg.MODELO_SIMPLE


def test_gana_el_mejor_si_la_brecha_es_grande():
    comp = pd.DataFrame({"Modelo": ["Random Forest", cfg.MODELO_SIMPLE], "ROC_AUC_CV": [0.95, 0.90]})
    assert elegir_modelo(comp) == "Random Forest"


def test_umbral_oof_cumple_recall():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 4000)
    p = np.clip(y * 0.35 + rng.normal(0.35, 0.2, 4000), 0, 1)
    u = seleccionar_umbral_oof(p, y, 0.75)
    recall = ((p >= u) & (y == 1)).sum() / y.sum()
    assert recall >= 0.75
    # es el umbral más alto que lo cumple
    u2 = u + 0.02
    assert ((p >= u2) & (y == 1)).sum() / y.sum() < 0.75 + 0.02
