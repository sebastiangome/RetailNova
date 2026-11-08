import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.validacion_negocio import comparar_escenarios, validar_auditoria, validar_negocio


def auditoria():
    return pd.DataFrame([
        {"umbral":.10,"tp":7,"fp":13,"fn":3,"tn":17, "precision":7/20,"recall":7/10,"f1":14/30,"tasa_alertas":.5},
        {"umbral":.15,"tp":5,"fp":8,"fn":5,"tn":22, "precision":5/13,"recall":.5,"f1":10/23,"tasa_alertas":13/40},
        {"umbral":.20,"tp":3,"fp":4,"fn":7,"tn":26, "precision":3/7,"recall":.3,"f1":6/17,"tasa_alertas":7/40},
    ])


def test_conteos_y_niveles_correctos():
    t=validar_auditoria(auditoria(),40)
    escenarios,niveles=comparar_escenarios(t,0.15,40)
    assert escenarios["alertas_validacion"].tolist()==[20,13,7]
    assert niveles["bajo"]["operaciones_validacion"]==20
    assert niveles["observacion"]["operaciones_validacion"]==7
    assert niveles["prioritario"]["operaciones_validacion"]==13
    assert escenarios["cabe_en_capacidad_simulada"].isna().all()


def test_capacidad_es_opcional_y_no_reoptimiza():
    escenarios,_=comparar_escenarios(auditoria(),0.15,40,capacidad_alertas_dia=1)
    assert escenarios["cabe_en_capacidad_simulada"].tolist()==[True,True,True]
    with pytest.raises(ValueError): comparar_escenarios(auditoria(),0.15,40,capacidad_alertas_dia=0)


def test_rechaza_incoherencias_y_faltantes():
    t=auditoria()
    t.loc[0, "fp"] = cast(int, t.loc[0, "fp"]) + 1
    with pytest.raises(ValueError, match="matrices"):
        validar_auditoria(t,40)
    with pytest.raises(ValueError, match="umbral"):
        comparar_escenarios(auditoria().iloc[1:],0.15,40)


def test_integracion_genera_reporte_sin_test(tmp_path: Path):
    resultados=tmp_path/"resultados"; graficos=tmp_path/"graficos"
    resultados.mkdir()
    (resultados/"auditoria_umbrales_validacion.json").write_text(auditoria().to_json(orient="records"),encoding="utf-8")
    modelo={"modo_rapido":False,"modelo_elegido":"XGBoost","umbral_elegido_en_validacion":0.15,"filas_utilizadas":{"validacion":40}}
    (resultados/"resultados_modelado.json").write_text(json.dumps(modelo),encoding="utf-8")
    (resultados/"equidad_territorial.json").write_text(json.dumps({"tarea":"H5-T3"}),encoding="utf-8")
    r=validar_negocio(ruta_umbrales=resultados/"auditoria_umbrales_validacion.json",
        ruta_modelado=resultados/"resultados_modelado.json",ruta_equidad=resultados/"equidad_territorial.json",
        carpeta_resultados=resultados,carpeta_graficos=graficos)
    assert r["umbral_modificado"] is False
    assert r["validacion_gerencia"].startswith("No realizada")
    assert (resultados/"reporte_validacion_negocio.md").is_file()
    assert "20 %" in (resultados/"reporte_validacion_negocio.md").read_text(encoding="utf-8")
    assert (graficos/"validacion_negocio_umbrales.png").is_file()
