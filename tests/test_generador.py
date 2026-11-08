import numpy as np
import pytest

from src import config as cfg
from src.generador_dataset import (cargar_estado_rng, generar_base_historica,
                                   guardar_estado_rng, validar_base)


def test_reproducible():
    a, _ = generar_base_historica(500, 42)
    b, _ = generar_base_historica(500, 42)
    assert a.equals(b)


def test_semilla_distinta_cambia_datos():
    a, _ = generar_base_historica(500, 42)
    b, _ = generar_base_historica(500, 7)
    assert not a.equals(b)


def test_base_valida_y_tasa(base_chica):
    hist, muestra, _ = base_chica
    validar_base(hist)
    assert 0.20 <= hist["abandono"].mean() <= 0.28
    assert hist["id_cliente"].is_unique
    assert len(muestra) < len(hist)


def test_estado_rng_continua_secuencia(tmp_path):
    _, rng = generar_base_historica(300, 42)
    ruta = tmp_path / "rng.json"
    guardar_estado_rng(rng, ruta)
    esperado = rng.integers(0, 1000, 10)
    restaurado = cargar_estado_rng(ruta)
    assert np.array_equal(restaurado.integers(0, 1000, 10), esperado)


def test_comunas_pertenecen_a_su_region(base_chica):
    hist, _, _ = base_chica
    for region, comunas in cfg.COMUNAS.items():
        usadas = set(hist.loc[hist["region"] == region, "comuna"].dropna())
        assert usadas <= set(comunas)
