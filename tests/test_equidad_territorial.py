"""H5-T3: no ajustar modelo y validar agregaciones de tasas desde conteos."""
import pandas as pd
import pytest

from src.equidad_territorial import calcular_equidad, verificar_oficial


def filas():
    # Todos los conteos son coherentes; dos comunas por región.
    return pd.DataFrame([
        {"comuna_destino": "Santiago", "operaciones": 200, "retrasos": 40, "alertas": 50, "tp": 20, "fp": 30},
        {"comuna_destino": "Ñuñoa", "operaciones": 100, "retrasos": 20, "alertas": 8, "tp": 2, "fp": 6},
        {"comuna_destino": "Valparaíso", "operaciones": 150, "retrasos": 30, "alertas": 35, "tp": 15, "fp": 20},
        {"comuna_destino": "Quilpué", "operaciones": 60, "retrasos": 10, "alertas": 7, "tp": 3, "fp": 4},
    ])


def test_calculo_regional_y_falsos_negativos():
    comunas, zonas, segmentos, informe = calcular_equidad(filas())
    rm = zonas.set_index("zona").loc["Región Metropolitana"]
    assert rm["operaciones"] == 300
    assert rm["retrasos"] == 60
    assert rm["tp"] == 22
    assert rm["fp"] == 36
    assert rm["recall"] == pytest.approx(22 / 60)
    assert rm["fpr"] == pytest.approx(36 / 240)
    assert comunas["fn"].sum() == 60
    assert zonas["fn"].sum() == 60
    assert segmentos["fn"].sum() == 60
    assert informe["rebalanceo"]["aplicado_previamente"] is False


def test_intervalos_y_representacion_test():
    comunas, zonas, _, informe = calcular_equidad(filas())
    assert len(zonas) == 2
    assert comunas["participacion_operaciones"].sum() == pytest.approx(1)
    assert comunas["participacion_falsos_negativos"].sum() == pytest.approx(1)
    a = comunas.set_index("comuna_destino").loc["Santiago"]
    assert a["recall_ic95_inf"] < a["recall"] < a["recall_ic95_sup"]
    assert a["fpr_ic95_inf"] < a["fpr"] < a["fpr_ic95_sup"]
    assert informe["operaciones"] == 510


def test_comuna_desconocida_rechazada():
    casos = filas()
    casos.loc[0, "comuna_destino"] = "Comuna sin clasificación"
    with pytest.raises(ValueError, match="sin región asignada"):
        calcular_equidad(casos)


def test_verificacion_contra_modelo_oficial():
    casos = filas()
    tp = int(casos["tp"].sum())
    fp = int(casos["fp"].sum())
    fn = int((casos["retrasos"] - casos["tp"]).sum())
    tn = int((casos["operaciones"] - casos["retrasos"] - casos["fp"]).sum())
    oficial = {"modelo_elegido": "XGBoost", "modo_rapido": False,
               "resultado_test": {"matriz_confusion": [[tn, fp], [fn, tp]]}}
    verificar_oficial(casos, oficial)
    casos.loc[0, "fp"] += 1
    with pytest.raises(ValueError, match="no corresponde"):
        verificar_oficial(casos, oficial)


def test_division_sin_retrasos_no_inventa_recall():
    casos = filas()
    casos.loc[3, ["retrasos", "alertas", "tp", "fp"]] = [0, 0, 0, 0]
    comunas, _, _, _ = calcular_equidad(casos)
    q = comunas.set_index("comuna_destino").loc["Quilpué"]
    assert pd.isna(q["recall"])
    assert pd.isna(q["recall_ic95_inf"])
