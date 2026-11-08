import numpy as np
import pytest

from src.equidad_regional import auditar_regiones


def test_regiones_iguales_cumplen():
    y = np.array([1, 0, 1, 0] * 30)
    reg = np.array(["A", "A", "B", "B"] * 30)
    tabla = auditar_regiones(y, y, reg, 10.0)
    assert (tabla["¿Cumple RNF6 (<10%)?"] == "Sí").all()
    assert len(tabla) == 2


def test_region_mala_incumple():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.4, 600)
    reg = np.array(["A"] * 300 + ["B"] * 300)
    pred = y.copy()
    pred[300:] = 1 - pred[300:]  # región B totalmente errada
    tabla = auditar_regiones(y, pred, reg, 10.0)
    assert (tabla["¿Cumple RNF6 (<10%)?"] == "No").any()


def test_largos_distintos():
    with pytest.raises(ValueError):
        auditar_regiones([1, 0], [1], ["A", "B"])
