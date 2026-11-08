import pandas as pd
import pytest

from src import config as cfg
from src.visualizacion import construir_datos, inyectar_datos


@pytest.fixture(scope="module")
def scored(preparada):
    d = preparada.copy()
    d["prob_abandono"] = (d["reclamos"] * 0.1 + d["abandono"] * 0.5).clip(0, 1)
    d["riesgo_abandono"] = pd.qcut(d["prob_abandono"].rank(method="first"), [0, .66, .9, 1],
                                   labels=["Bajo", "Medio", "Alto"]).astype(str)
    return d[cfg.COLUMNAS_EXPORTACION]


def test_cubo_suma_total(scored):
    datos = construir_datos(scored)
    cubo = next(v for v in datos.values() if isinstance(v, list) and v and "ab" in v[0])
    assert sum(f["n"] for f in cubo) == len(scored)
    assert sum(f["ab"] for f in cubo) == int(scored["abandono"].sum())


def test_columnas_faltantes(scored):
    with pytest.raises(ValueError):
        construir_datos(scored.drop(columns=["monto_total_acumulado"]))


def test_inyeccion_reemplaza_marcadores():
    plantilla = "a/*DATOS_START*/const DATOS = {};/*DATOS_END*/b"
    out = inyectar_datos(plantilla, {"x": "</script>"})
    assert "</script>" not in out.split("DATOS_START")[1]
    assert out.startswith("a") and out.endswith("b")


def test_plantilla_tiene_marcadores():
    t = (cfg.ROOT / "src" / "plantilla_dashboard.html").read_text(encoding="utf-8") if hasattr(cfg, "ROOT") else open("src/plantilla_dashboard.html", encoding="utf-8").read()
    assert "/*DATOS_START*/" in t and "/*DATOS_END*/" in t
