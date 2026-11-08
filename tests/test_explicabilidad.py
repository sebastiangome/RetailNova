"""Pruebas rápidas de estructuras SHAP, sin depender de un modelo entrenado."""
import numpy as np
import pytest

from src.explicabilidad import valores_clase_positiva


def test_shap_bidimensional():
    a = np.arange(12, dtype=float).reshape(3, 4)
    np.testing.assert_array_equal(valores_clase_positiva(a, 3, 4), a)


def test_shap_dos_clases():
    a = np.arange(24, dtype=float).reshape(3, 4, 2)
    np.testing.assert_array_equal(valores_clase_positiva(a, 3, 4), a[:, :, 1])


def test_shap_formato_invalido():
    with pytest.raises(ValueError, match="Formato SHAP"):
        valores_clase_positiva(np.zeros((3, 5)), 3, 4)
