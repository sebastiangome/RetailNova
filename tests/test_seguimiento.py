import importlib.util
import json
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gen", RAIZ / "seguimiento" / "generar_seguimiento.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

CORTE = date(2026, 11, 4)
BACKLOG = json.loads((RAIZ / "seguimiento/backlog_github.json").read_text(encoding="utf-8"))


def test_estados_solo_completado_o_en_proceso():
    doc = gen.construir(BACKLOG, CORTE)
    assert {t["estado"] for t in doc["tareas"]} <= {"Completado", "En proceso"}
    r = doc["resumen"]
    assert r["total"] == len(BACKLOG["items"]) == 24
    assert r["completadas"] + r["en_proceso"] == r["total"]
    assert len({t["id"] for t in doc["tareas"]}) == 24


def test_estado_pendiente_pasa_a_en_proceso():
    copia = json.loads(json.dumps(BACKLOG))
    copia["items"][0]["status"] = "Pendiente"
    assert gen.construir(copia, CORTE)["resumen"]["en_proceso"] >= 1


def test_json_publicado_esta_actualizado():
    publicado = json.loads((RAIZ / "dashboard/ejemplos/03_seguimiento_backlog.json").read_text(encoding="utf-8"))
    esperado = gen.construir(BACKLOG, date.fromisoformat(publicado["fecha_corte"]))
    assert publicado == esperado
    assert json.dumps(esperado, ensure_ascii=False) in (RAIZ / "dashboard/index.html").read_text(encoding="utf-8")
