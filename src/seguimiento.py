"""Seguimiento del proyecto (Indicador 2 de la pauta): tareas planificadas vs. ejecutadas
y desviaciones con acción correctiva. Lee `seguimiento/tareas.json`."""
from __future__ import annotations

import json
from pathlib import Path

from src import config as cfg

ESTADOS_VALIDOS = {"Completado", "En proceso", "Pendiente"}
CAMPOS_TAREA = {"id", "titulo", "responsable", "planificado", "ejecutado", "estado"}


def cargar_seguimiento(ruta: Path = cfg.SEGUIMIENTO / "tareas.json") -> dict:
    """Carga y valida el seguimiento; agrega un `resumen` calculado (no escrito a mano)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}")
    doc = json.loads(ruta.read_text(encoding="utf-8"))
    ids = [t.get("id") for t in doc.get("tareas", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Las tareas deben existir y tener identificadores únicos")
    for t in doc["tareas"]:
        faltan = CAMPOS_TAREA - set(t)
        if faltan:
            raise ValueError(f"Tarea {t.get('id')} sin campos: {sorted(faltan)}")
        if t["estado"] not in ESTADOS_VALIDOS:
            raise ValueError(f"Estado inválido en {t['id']}: {t['estado']}")
    total = len(doc["tareas"])
    completadas = sum(t["estado"] == "Completado" for t in doc["tareas"])
    doc["resumen"] = {
        "total": total,
        "completadas": completadas,
        "en_proceso": sum(t["estado"] == "En proceso" for t in doc["tareas"]),
        "pendientes": sum(t["estado"] == "Pendiente" for t in doc["tareas"]),
        "avance_pct": round(completadas / total * 100, 1),
        "desviaciones": len(doc.get("desviaciones", [])),
    }
    return doc
