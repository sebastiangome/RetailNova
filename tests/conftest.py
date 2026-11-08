import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src import config as cfg
from src.generador_dataset import generar_base_historica, recortar_12_meses
from src.pipeline_datos import (agregar_rfm, corregir_coherencia_temporal,
                                imputar_nulos)


@pytest.fixture(scope="session")
def base_chica():
    """Base sintética pequeña (3.000 clientes) + muestra 12m + rng posterior."""
    hist, rng = generar_base_historica(3000, cfg.SEMILLA)
    muestra = recortar_12_meses(hist, rng)
    return hist, muestra, rng


@pytest.fixture(scope="session")
def preparada(base_chica):
    _, muestra, rng = base_chica
    df = muestra.copy()
    corregir_coherencia_temporal(df, rng)
    imputar_nulos(df, rng)
    return agregar_rfm(df)
