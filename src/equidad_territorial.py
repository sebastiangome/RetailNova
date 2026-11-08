"""H5-T3: auditoría de equidad territorial del XGBoost oficial de LOGISMART.

Reutiliza los conteos territoriales y el diagnóstico de H4-T4; no entrena,
no realiza inferencias nuevas, ni cambia el umbral elegido en validación.
Los resultados descriptivos en test NO constituyen demostración de discriminación.

Uso: python main.py --proceso equidad
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.sesgos import calcular_diagnostico, intervalo_wilson

# Agrupación administrativa de las 19 comunas representadas en LOGISMART.
ZONAS: dict[str, str] = {
    "Santiago": "Región Metropolitana", "Las Condes": "Región Metropolitana",
    "Maipú": "Región Metropolitana", "Providencia": "Región Metropolitana",
    "La Florida": "Región Metropolitana", "Puente Alto": "Región Metropolitana",
    "Ñuñoa": "Región Metropolitana", "Quilicura": "Región Metropolitana",
    "San Bernardo": "Región Metropolitana", "Pudahuel": "Región Metropolitana",
    "Colina": "Región Metropolitana", "Lampa": "Región Metropolitana",
    "Valparaíso": "Región de Valparaíso", "Viña del Mar": "Región de Valparaíso",
    "Quilpué": "Región de Valparaíso", "Villa Alemana": "Región de Valparaíso",
    "Concón": "Región de Valparaíso", "Casablanca": "Región de Valparaíso",
    "Quintero": "Región de Valparaíso",
}
CONTEOS = ["operaciones", "retrasos", "alertas", "tp", "fp", "fn", "tn"]


def _intervalos(tabla: pd.DataFrame, exitos: str, total: str, prefijo: str) -> None:
    """Agrega límites Wilson cuando existe denominador; NaN si no existe."""
    pares = [intervalo_wilson(int(a), int(b)) for a, b in zip(tabla[exitos], tabla[total])]
    tabla[f"{prefijo}_ic95_inf"] = [a for a, _ in pares]
    tabla[f"{prefijo}_ic95_sup"] = [b for _, b in pares]


def _registros_json(tabla: pd.DataFrame, columnas: list[str]) -> list[dict]:
    """Serializa tablas agregadas con None cuando alguna tasa es indefinida."""
    return json.loads(tabla[columnas].to_json(orient="records", force_ascii=False))


def _tasas(tabla: pd.DataFrame) -> pd.DataFrame:
    """Calcula tasas a partir de conteos sumables, nunca promedios de tasas."""
    t = tabla.copy()
    t["no_retrasos"] = t["operaciones"] - t["retrasos"]
    t["precision"] = t["tp"] / t["alertas"].replace(0, np.nan)
    t["recall"] = t["tp"] / t["retrasos"].replace(0, np.nan)
    t["fpr"] = t["fp"] / t["no_retrasos"].replace(0, np.nan)
    t["fnr"] = t["fn"] / t["retrasos"].replace(0, np.nan)
    t["tasa_retrasos"] = t["retrasos"] / t["operaciones"]
    t["tasa_alertas"] = t["alertas"] / t["operaciones"]
    for pref, ex, den in (("recall", "tp", "retrasos"),
                          ("precision", "tp", "alertas"),
                          ("fpr", "fp", "no_retrasos")):
        _intervalos(t, ex, den, pref)
    return t


def calcular_equidad(comunas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Audita brechas descriptivas sobre agregados de predicciones preexistentes.

    Devuelve tablas por comuna, región, representación y un reporte JSON.
    No elige umbrales ni hiperparámetros; no usa datos de train para inferir
    cómo se distribuye el entrenamiento entre comunas.
    """
    base, diagnostico = calcular_diagnostico(comunas)
    desconocidas = sorted(set(base["comuna_destino"]) - set(ZONAS))
    if desconocidas:
        raise ValueError(f"Comunas sin región asignada: {desconocidas}")

    base["zona"] = base["comuna_destino"].map(ZONAS)
    mediana_operaciones = float(base["operaciones"].median())
    base["representacion_test"] = np.where(
        base["operaciones"] < mediana_operaciones,
        "Bajo mediana", "Igual o sobre mediana",
    )
    base = _tasas(base)
    n = int(base["operaciones"].sum())
    retrasos = int(base["retrasos"].sum())
    fn = int(base["fn"].sum())
    fp = int(base["fp"].sum())
    global_recall = float(base["tp"].sum()) / retrasos if retrasos else None
    global_precision = float(base["tp"].sum()) / int(base["alertas"].sum()) if base["alertas"].sum() else None
    no_retrasos = n - retrasos
    global_fpr = fp / no_retrasos if no_retrasos else None

    base["participacion_operaciones"] = base["operaciones"] / n
    base["participacion_falsos_negativos"] = base["fn"] / fn if fn else np.nan
    base["participacion_falsos_positivos"] = base["fp"] / fp if fp else np.nan
    base["brecha_recall_global_pp"] = (base["recall"] - global_recall) * 100 if global_recall is not None else np.nan
    base["brecha_fpr_global_pp"] = (base["fpr"] - global_fpr) * 100 if global_fpr is not None else np.nan
    # Banderas exploratorias, NO pruebas de significancia ni veredictos de sesgo.
    base["revision_recall"] = (
        (base["retrasos"] >= 30)
        & base["recall_ic95_sup"].lt(global_recall) if global_recall is not None else False
    )
    base["revision_fpr"] = (
        (base["no_retrasos"] >= 30)
        & base["fpr_ic95_inf"].gt(global_fpr) if global_fpr is not None else False
    )

    zonas = _tasas(base.groupby("zona", as_index=False)[CONTEOS].sum())
    representacion = _tasas(base.groupby("representacion_test", as_index=False)[CONTEOS].sum())
    zonas["participacion_operaciones"] = zonas["operaciones"] / n
    representacion["participacion_operaciones"] = representacion["operaciones"] / n

    casos_revisar = base.loc[base["revision_recall"] | base["revision_fpr"],
        ["comuna_destino", "recall", "fpr", "revision_recall", "revision_fpr"]]
    informe = {
        "tarea": "H5-T3",
        "fuente": "output/resultados/sesgos_equidad/auditoria_comunas_test.json (XGBoost oficial H4-T1)",
        "conjunto": "test sintético, agosto-septiembre 2026; descripción, no ajuste",
        "numero_comunas": int(len(base)),
        "numero_zonas": int(len(zonas)),
        "operaciones": n, "retrasos": retrasos,
        "falsos_negativos": fn, "falsos_positivos": fp,
        "recall_global": global_recall,
        "precision_global": global_precision,
        "fpr_global": global_fpr,
        "mediana_operaciones_por_comuna_test": mediana_operaciones,
        "spearman_operaciones_recall": diagnostico["correlacion_spearman_operaciones_recall"],
        "brecha_recall_comunas_pp": (
            float((base["recall"].max() - base["recall"].min()) * 100)
            if base["recall"].notna().sum() >= 2 else None
        ),
        "zonas": _registros_json(zonas, ["zona", "operaciones", "retrasos", "recall", "precision", "fpr", "fn", "fp"]),
        "segmentos_representacion_test": _registros_json(representacion, ["representacion_test", "operaciones", "retrasos", "recall", "precision", "fpr", "fn", "fp"]),
        "comunas_revision_exploratoria": _registros_json(casos_revisar, ["comuna_destino", "recall", "fpr", "revision_recall", "revision_fpr"]),
        "rebalanceo": {
            "aplicado_previamente": False,
            "efectividad": "No evaluable: no se implementó un modelo reentrenado con rebalanceo",
            "plan_validacion": "Probar únicamente con train y validar temporalmente en julio; no elegir estrategias mediante test",
        },
        "medidas_correctivas_propuestas_no_implementadas": [
            "Revisar cobertura histórica y calidad de etiquetas por comuna y por mes (especialmente minoritarias)",
            "Examinar diferencias por distancia, horarios, vehículo y prevalencia de retrasos antes de atribuir la brecha a representación",
            "Ensayar ponderación o rebalanceo solo en train y medir recall, precision y FPR por comuna en validación temporal",
            "Monitorear tasas y errores por comuna con intervalos e investigar alertas incorrectas y retrasos no detectados",
            "Usar las alertas como apoyo logístico, nunca como mecanismo automático de sanción de conductores",
        ],
        "limitaciones": [
            "Datos sintéticos; no constituyen evidencia de equidad o discriminación en operaciones reales",
            "El dataset de test ya fue examinado en H4-T1/H4-T4/H5-T1; esta auditoría es descriptiva, no una validación nueva",
            "Se dispone de la representación de test, no de una auditoría equivalente de la distribución territorial del train",
            "Intervalos Wilson aproximados suponen independencia; pueden subestimar incertidumbre por agrupación de rutas/conductores",
            "Comparación de intervalos con tasa global no es prueba formal de significancia; existe comparación múltiple",
            "Diferencias pueden confundirse por distinta prevalencia, recorridos, tipos de entrega y horario",
            "No se modificó el modelo oficial ni se escogieron umbrales usando test",
        ],
    }
    return base.sort_values("operaciones", ascending=False), zonas, representacion, informe


def verificar_oficial(tabla: pd.DataFrame, oficial: dict) -> None:
    """Evita mezclar agregados de otra corrida con el modelo oficial."""
    if oficial.get("modo_rapido") or oficial.get("modelo_elegido") != "XGBoost":
        raise ValueError("Se requiere el modelo XGBoost oficial en modo completo")
    met = oficial["resultado_test"]
    matriz = met["matriz_confusion"]
    (tn, fp), (fn, tp) = matriz
    esperados = {
        "operaciones": tn + fp + fn + tp,
        "tp": tp, "fp": fp,
        "retrasos": fn + tp,
    }
    reales = {key: int(tabla[key].sum()) for key in esperados}
    if reales != esperados:
        raise ValueError(f"Auditoría territorial no corresponde con métricas oficiales. {reales} != {esperados}")


def _graficar(comunas: pd.DataFrame, zonas: pd.DataFrame, destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destino.mkdir(parents=True, exist_ok=True)
    # Nuevo respecto de H4-T4: trade-off entre detección y falsas alarmas.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(comunas["fpr"] * 100, comunas["recall"] * 100,
               s=np.maximum(25, comunas["operaciones"] / 180))
    for _, fila in comunas.iterrows():
        ax.annotate(str(fila["comuna_destino"]),
                    (fila["fpr"] * 100, fila["recall"] * 100),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Falsos positivos / entregas sin retraso (FPR, %)",
           ylabel="Retrasos detectados (recall, %)",
           title="H5-T3: detección vs falsas alarmas por comuna — datos sintéticos")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destino / "equidad_recall_vs_fpr.png", dpi=155)
    plt.close(fig)

    # Conteos absolutos: visibiliza dónde se acumulan retrasos no detectados.
    top = comunas.sort_values("fn", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["comuna_destino"], top["fn"])
    ax.set(xlabel="Retrasos reales no alertados (FN)",
           title="H5-T3: falsos negativos por comuna — datos sintéticos")
    fig.tight_layout()
    fig.savefig(destino / "equidad_falsos_negativos.png", dpi=155)
    plt.close(fig)

    # Tasas calculadas desde conteos agregados de dos regiones (no promedio simple de comunas).
    graf = zonas.set_index("zona")[["precision", "recall", "fpr"]] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    graf.plot.bar(ax=ax, rot=0)
    ax.set(ylabel="Porcentaje (%)", xlabel="", title="H5-T3: métricas por región — test sintético")
    ax.legend(["Precision", "Recall", "FPR"])
    fig.tight_layout()
    fig.savefig(destino / "equidad_comparacion_zonas.png", dpi=155)
    plt.close(fig)


def _reporte_md(informe: dict) -> str:
    fmt = lambda a: "no calculable" if a is None else f"{a * 100:.2f} %"
    lineas = [
        "# LOGISMART — H5-T3 Auditoría de equidad territorial",
        "",
        "**Alcance:** modelo XGBoost oficial; test sintético de agosto-septiembre 2026. Auditoría descriptiva; no se reentrenó.",
        "",
        "## Resultados globales",
        f"- Operaciones: {informe['operaciones']:,}; retrasos: {informe['retrasos']:,}.",
        f"- Recall: {fmt(informe['recall_global'])}; precision: {fmt(informe['precision_global'])}; FPR: {fmt(informe['fpr_global'])}.",
        f"- FN: {informe['falsos_negativos']:,}; FP: {informe['falsos_positivos']:,}.",
        f"- Brecha descriptiva entre comunas (recall): {informe['brecha_recall_comunas_pp']:.2f} puntos porcentuales." if informe["brecha_recall_comunas_pp"] is not None else "- Brecha no calculable.",
        f"- Spearman entre representación en test y recall: {informe['spearman_operaciones_recall']:.2f} (asociación no causal)." if informe["spearman_operaciones_recall"] is not None else "- Spearman no calculable.",
        "",
        "## Comparación regional",
        "| Región | Operaciones | Recall | Precision | FPR | Falsos negativos |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for z in informe["zonas"]:
        lineas.append(f"| {z['zona']} | {z['operaciones']:,} | {fmt(z['recall'])} | {fmt(z['precision'])} | {fmt(z['fpr'])} | {z['fn']:,} |")
    lineas.extend([
        "", "## Diferencias y riesgos", "",
        "Las tasas de detección y falsas alarmas varían por comuna. Esto amerita seguimiento y análisis de las condiciones logísticas; no demuestra discriminación.",
        "Las regiones agrupan operaciones con distintas tasas base; sus resultados tampoco determinan causalidad.",
        "Las banderas de revisión de la tabla CSV son descriptivas y no equivalen a una prueba estadística ajustada por comparaciones múltiples.",
        "", "## Rebalanceo", "",
        "**No se aplicó rebalanceo previamente**, por lo que no es posible evaluar su efectividad. Ningún método se considera validado por este informe.",
        "", "## Acciones propuestas (pendientes de implementar y validar)", "",
    ])
    lineas.extend(f"- {accion}" for accion in informe["medidas_correctivas_propuestas_no_implementadas"])
    lineas.extend(["", "## Limitaciones", ""])
    lineas.extend(f"- {limite}" for limite in informe["limitaciones"])
    return "\n".join(lineas) + "\n"


def auditar_equidad(
    entrada="output/resultados/sesgos_equidad/auditoria_comunas_test.json",
    oficial="output/resultados/modelado/resultados_modelado.json",
    carpeta_resultados="output/resultados/sesgos_equidad",
    carpeta_graficos="output/graficos",
) -> dict:
    """Genera tablas, gráficos y reporte desde conteos de H4-T4; no requiere .joblib."""
    ruta_entrada, ruta_oficial = Path(entrada), Path(oficial)
    for ruta in (ruta_entrada, ruta_oficial):
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero: python main.py --proceso evaluar")
    datos = pd.read_json(ruta_entrada)
    info = json.loads(ruta_oficial.read_text(encoding="utf-8"))
    verificar_oficial(datos, info)
    comunas, zonas, representacion, informe = calcular_equidad(datos)
    informe["umbral_oficial_fijado_en_validacion"] = float(info["umbral_elegido_en_validacion"])
    destino = Path(carpeta_resultados)
    destino.mkdir(parents=True, exist_ok=True)
    dg = Path(carpeta_graficos)
    comunas.to_csv(destino / "equidad_comunas.csv", index=False)
    zonas.to_csv(destino / "equidad_zonas.csv", index=False)
    representacion.to_csv(destino / "equidad_representacion_test.csv", index=False)
    (destino / "equidad_territorial.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (destino / "reporte_equidad_territorial.md").write_text(
        _reporte_md(informe), encoding="utf-8"
    )
    _graficar(comunas, zonas, dg)
    print(f"H5-T3: {informe['numero_comunas']} comunas; {informe['numero_zonas']} regiones; {informe['operaciones']:,} operaciones")
    print(f"Recall global {informe['recall_global']:.3f} | Precision {informe['precision_global']:.3f} | FPR {informe['fpr_global']:.3f}")
    print(zonas[["zona", "operaciones", "recall", "precision", "fpr", "fn"]].round(4).to_string(index=False))
    print(f"Archivos: {destino / 'equidad_territorial.json'}, tablas CSV, reporte Markdown y gráficos en {dg}")
    print("No se modificó el modelo ni su umbral; rebalanceo no aplicado.")
    return informe


if __name__ == "__main__":
    auditar_equidad()
