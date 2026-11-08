"""Comprobaciones rápidas de H4-T3 (no entrenan modelos)."""
import pandas as pd
import pytest

from src.optimizacion import ESPACIOS, construir_folds_temporales, preparar_muestra


def test_cv_cronologica_sin_fuga():
    fechas = pd.date_range("2025-08-01", "2026-06-30", freq="D")
    folds = construir_folds_temporales(fechas)
    assert len(folds) == 2
    for tr, va in folds:
        assert len(set(tr) & set(va)) == 0
        assert fechas[tr].max() < fechas[va].min()


def test_muestra_determinista_y_ordenada():
    df = pd.DataFrame({"fecha": pd.date_range("2025-08-01", periods=1200),
                       "valor": range(1200)})
    a = preparar_muestra(df, 1000, 42)
    b = preparar_muestra(df, 1000, 42)
    pd.testing.assert_frame_equal(a, b)
    assert a["fecha"].is_monotonic_increasing
    assert len(a) == 1000


def test_parametros_de_ambos_modelos():
    assert "Random Forest" in ESPACIOS
    assert "XGBoost" in ESPACIOS
    assert "modelo__max_depth" in ESPACIOS["Random Forest"]
    assert "modelo__learning_rate" in ESPACIOS["XGBoost"]


def test_muestra_demasiado_pequena():
    with pytest.raises(ValueError, match="max-filas"):
        preparar_muestra(pd.DataFrame(), 50, 42)
