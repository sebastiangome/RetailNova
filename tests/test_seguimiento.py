import json

import pytest

from src.seguimiento import cargar_seguimiento


def test_resumen_real():
    doc = cargar_seguimiento()
    assert doc["resumen"]["total"] == 11
    assert doc["resumen"]["completadas"] == 10
    assert doc["resumen"]["desviaciones"] == 3


def test_estado_invalido(tmp_path):
    f = tmp_path / "t.json"
    base = cargar_seguimiento()
    base["tareas"][0]["estado"] = "Inventado"
    base.pop("resumen")
    f.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_seguimiento(f)


def test_ids_duplicados(tmp_path):
    f = tmp_path / "t.json"
    base = cargar_seguimiento()
    base["tareas"][1]["id"] = base["tareas"][0]["id"]
    base.pop("resumen")
    f.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_seguimiento(f)
