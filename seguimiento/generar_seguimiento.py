"""Convierte el export del tablero GitHub Projects en seguimiento.json para el dashboard.

Uso (desde la raíz del repo):
    gh project item-list 1 --owner carlos31255 --limit 100 --format json > seguimiento/backlog_github.json
    python seguimiento/generar_seguimiento.py

Escribe dashboard/ejemplos/03_seguimiento_backlog.json y actualiza los datos
incrustados en dashboard/index.html (entre los marcadores SEGUIMIENTO).
Estado: "Completado" si el tablero dice Completado; cualquier otro estado = "En proceso".
Las fechas son el cronograma planificado (simulado, 16 semanas).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MARCA = re.compile(r"(/\*SEGUIMIENTO_START\*/).*?(/\*SEGUIMIENTO_END\*/)", re.S)


def construir(backlog: dict, corte: date) -> dict:
    tareas = []
    for item in backlog["items"]:
        m = re.match(r"(H\d-T\d)\s*[—-]\s*(.*)", item["title"])
        if not m:
            raise ValueError(f"Título sin código HX-TY: {item['title']}")
        tid, titulo = m.groups()
        tareas.append({
            "id": tid, "titulo": titulo, "issue": item["content"]["number"],
            "url": item["content"]["url"],
            "fase": item.get("fase CRISP-DM"), "prioridad": item.get("prioridad MoSCoW"),
            "estado": "Completado" if item["status"] == "Completado" else "En proceso",
            "avance": item.get("avance (%)", 0),
            "inicio": item["fecha Inicio"][:10], "termino": item["fecha Término"][:10],
        })
    tareas.sort(key=lambda t: (int(t["id"][1]), int(t["id"][4])))
    n = len(tareas)
    hechas = sum(t["estado"] == "Completado" for t in tareas)
    return {
        "tipo": "seguimiento_backlog", "proyecto": "LOGISMART",
        "fuente": "GitHub Projects · Planificación · Backlog LOGISMART",
        "fecha_corte": corte.isoformat(),
        "resumen": {
            "total": n, "completadas": hechas, "en_proceso": n - hechas,
            "avance_promedio": round(sum(t["avance"] for t in tareas) / n, 1),
        },
        "tareas": tareas,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backlog", default=RAIZ / "seguimiento/backlog_github.json")
    ap.add_argument("--salida", default=RAIZ / "dashboard/ejemplos/03_seguimiento_backlog.json")
    ap.add_argument("--html", default=RAIZ / "dashboard/index.html")
    ap.add_argument("--corte", default=date.today().isoformat())
    a = ap.parse_args()
    doc = construir(json.loads(Path(a.backlog).read_text(encoding="utf-8")), date.fromisoformat(a.corte))
    Path(a.salida).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html = Path(a.html)
    nuevo, n = MARCA.subn(
        lambda m: m.group(1) + "const SEGUIMIENTO_INICIAL = " + json.dumps(doc, ensure_ascii=False) + ";" + m.group(2),
        html.read_text(encoding="utf-8"))
    if n != 1:
        raise SystemExit("No se encontraron los marcadores SEGUIMIENTO en el HTML")
    html.write_text(nuevo, encoding="utf-8")
    print(f"{doc['resumen']['completadas']}/{doc['resumen']['total']} completadas · JSON y dashboard actualizados")


if __name__ == "__main__":
    main()
