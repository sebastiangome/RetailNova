"""Dashboard de RetailNova: página HTML autocontenida (sin servidor ni librerías externas).

Lee el CSV puntuado y los resultados del modelo, arma un "cubo" de datos agregados
(región × canal × riesgo) más muestras acotadas, los incrusta en la plantilla y
escribe `dashboard/index.html`. Los filtros funcionan en el navegador sobre esos
datos incrustados; no se envía nada a Internet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src.seguimiento import cargar_seguimiento

PLANTILLA = Path(__file__).resolve().parent / "plantilla_dashboard.html"
MARCA = ("/*DATOS_START*/", "/*DATOS_END*/")
N_BINS = 20
TOP_POR_GRUPO = 50
N_DISPERSION = 2500


def construir_datos(scored: pd.DataFrame, umbral: float | None = None, modelo: dict | None = None,
                    regional: list | None = None, shap: list | None = None,
                    seguimiento: dict | None = None, semilla: int = cfg.SEMILLA) -> dict:
    """Agrega los datos que consume el dashboard. No incluye la tabla completa de clientes."""
    faltan = set(cfg.COLUMNAS_EXPORTACION) - set(scored.columns)
    if faltan:
        raise ValueError(f"Al CSV puntuado le faltan columnas: {sorted(faltan)}")
    d = scored.copy()
    claves = ["region", "canal_compra", "riesgo_abandono"]

    cubo = (d.groupby(claves)
              .agg(n=("id_cliente", "size"), ab=("abandono", "sum"), monto=("monto_total_acumulado", "sum"))
              .reset_index())
    cubo_json = [{"r": r.region, "c": r.canal_compra, "k": r.riesgo_abandono,
                  "n": int(r.n), "ab": int(r.ab), "monto": float(r.monto)} for r in cubo.itertuples()]

    d["_bin"] = np.minimum((d["prob_abandono"] * N_BINS).astype(int), N_BINS - 1)
    hist_json = []
    for (r, c, k), g in d.groupby(claves):
        conteo = np.bincount(g["_bin"], minlength=N_BINS)
        hist_json.append({"r": r, "c": c, "k": k, "h": [int(x) for x in conteo]})

    muestra = d.sample(n=min(N_DISPERSION, len(d)), random_state=semilla)
    dispersion = [{"x": int(x), "y": round(float(y), 2), "k": k, "r": r, "c": c}
                  for x, y, k, r, c in zip(muestra["recencia_dias"], muestra["frecuencia_compra"],
                                           muestra["riesgo_abandono"], muestra["region"],
                                           muestra["canal_compra"])]

    top = (d.sort_values("prob_abandono", ascending=False)
             .groupby(claves, sort=False).head(TOP_POR_GRUPO))
    top_json = [{"id": t.id_cliente, "r": t.region, "c": t.canal_compra, "k": t.riesgo_abandono,
                 "p": round(float(t.prob_abandono), 4), "m": float(t.monto_total_acumulado),
                 "rec": int(t.recencia_dias)} for t in top.itertuples()]

    return {
        "meta": {
            "total": int(len(d)),
            "tasa_abandono": float(d["abandono"].mean()),
            "umbral": umbral,
            "regiones": sorted(d["region"].unique()),
            "canales": sorted(d["canal_compra"].unique()),
            "n_dispersion": len(dispersion),
            "n_bins": N_BINS,
            "corte": cfg.FECHA_CORTE.strftime("%d-%m-%Y"),
        },
        "cubo": cubo_json,
        "hist": hist_json,
        "dispersion": dispersion,
        "top": top_json,
        "modelo": modelo,
        "regional": regional,
        "shap": shap,
        "seguimiento": seguimiento,
    }


def _leer_json(ruta: Path):
    ruta = Path(ruta)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def inyectar_datos(plantilla: str, datos: dict) -> str:
    """Reemplaza el bloque entre marcadores por `const DATOS = {...};`."""
    ini, fin = MARCA
    if plantilla.count(ini) != 1 or plantilla.count(fin) != 1:
        raise ValueError("La plantilla debe tener exactamente un par de marcadores DATOS")
    a = plantilla.index(ini) + len(ini)
    b = plantilla.index(fin)
    # `</` dentro de <script> podría cerrar la etiqueta antes de tiempo
    cuerpo = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return plantilla[:a] + "const DATOS = " + cuerpo + ";" + plantilla[b:]


def generar_visualizacion(scored: pd.DataFrame | None = None, salida: Path = cfg.DASHBOARD / "index.html",
                          carpeta_resultados: Path = cfg.RESULTADOS,
                          ruta_seguimiento: Path = cfg.SEGUIMIENTO / "tareas.json") -> dict:
    """Proceso `visualizar`: genera dashboard/index.html y el JSON de indicadores."""
    if scored is None:
        if not cfg.CSV_SCORED.is_file():
            raise FileNotFoundError(f"No existe {cfg.CSV_SCORED}; ejecuta primero --proceso puntuar")
        scored = pd.read_csv(cfg.CSV_SCORED)
    carpeta_resultados = Path(carpeta_resultados)
    seleccion = _leer_json(carpeta_resultados / "modelado" / "seleccion_modelo.json")
    evaluacion = _leer_json(carpeta_resultados / "evaluacion" / "evaluacion_final.json")
    shap = _leer_json(carpeta_resultados / "explicabilidad" / "explicabilidad_shap.json")

    modelo = None
    if seleccion and evaluacion:
        oof = next((m for m in evaluacion["metricas"] if "OOF" in m["Umbral"]), evaluacion["metricas"][-1])
        modelo = {"nombre": seleccion["modelo"], "auc_cv": seleccion["auc_cv"],
                  "recall_oof": seleccion["recall_oof"], "test": oof,
                  "meta_recall": evaluacion["meta_recall"], "cumple": evaluacion["cumple_meta_recall"]}
    datos = construir_datos(
        scored, umbral=seleccion["umbral_decision"] if seleccion else None, modelo=modelo,
        regional=evaluacion["regional"] if evaluacion else None,
        shap=shap["importancia_global"][:8] if shap else None,
        seguimiento=cargar_seguimiento(ruta_seguimiento) if Path(ruta_seguimiento).is_file() else None)

    html = inyectar_datos(PLANTILLA.read_text(encoding="utf-8"), datos)
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(html, encoding="utf-8")

    carpeta_viz = carpeta_resultados / "visualizacion"
    carpeta_viz.mkdir(parents=True, exist_ok=True)
    indicadores = {
        "clientes": datos["meta"]["total"],
        "tasa_abandono": datos["meta"]["tasa_abandono"],
        "riesgo_alto": int((scored["riesgo_abandono"] == "Alto").sum()),
        "monto_riesgo_alto": float(scored.loc[scored["riesgo_abandono"] == "Alto", "monto_total_acumulado"].sum()),
        "umbral_decision": datos["meta"]["umbral"],
    }
    (carpeta_viz / "indicadores.json").write_text(
        json.dumps(indicadores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dashboard generado: {salida} ({salida.stat().st_size / 1024:.0f} KB)")
    return indicadores
