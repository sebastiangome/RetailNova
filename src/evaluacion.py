"""Evaluación final: el conjunto de prueba se usa UNA sola vez, con modelo y umbral ya fijados."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from src import config as cfg
from src.equidad_regional import auditar_regiones
from src.modelado import cargar_modelo, cargar_seleccion, dividir_datos


def fila_metricas(y_true, proba, pred, etiqueta: str) -> dict:
    return {
        "Umbral": etiqueta,
        "ROC_AUC": float(roc_auc_score(y_true, proba)),
        "PR_AUC": float(average_precision_score(y_true, proba)),
        "Recall": float(recall_score(y_true, pred)),
        "Precisión": float(precision_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Alertas generadas": int(np.sum(pred)),
    }


def evaluar(df: pd.DataFrame, pipeline=None, seleccion: dict | None = None,
            carpeta: Path = cfg.RESULTADOS / "evaluacion", carpeta_graficos: Path = cfg.GRAFICOS) -> dict:
    """Proceso `evaluar`: métricas globales (umbral 0,5 y OOF) y auditoría regional RNF6."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    pipeline = pipeline or cargar_modelo()
    seleccion = seleccion or cargar_seleccion()
    umbral = seleccion["umbral_decision"]

    _, X_test, _, y_test, _, region_test = dividir_datos(df)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred_05 = (proba >= 0.5).astype(int)
    pred = (proba >= umbral).astype(int)

    resumen = pd.DataFrame([fila_metricas(y_test, proba, pred_05, "0.50 (referencia)"),
                            fila_metricas(y_test, proba, pred, f"{umbral:.4f} (OOF)")])
    resumen.to_csv(carpeta / "metricas_test.csv", index=False)
    print(resumen.round(3).to_string(index=False))

    cumple = bool(recall_score(y_test, pred) >= cfg.META_RECALL)
    print(f"\nMeta de recall del plan (>= 70 %, umbral OOF): {'CUMPLE' if cumple else 'NO CUMPLE'}")
    print(classification_report(y_test, pred, target_names=["Activo", "Abandona"]))

    regional = auditar_regiones(y_test, pred, region_test)
    regional.to_csv(carpeta / "evaluacion_regional.csv", index=False)
    print(regional.to_string(index=False))

    salida = {
        "modelo": seleccion["modelo"], "umbral_decision": umbral,
        "n_prueba": int(len(y_test)), "cumple_meta_recall": cumple,
        "meta_recall": cfg.META_RECALL,
        "metricas": resumen.to_dict(orient="records"),
        "regional": regional.to_dict(orient="records"),
        "nota": "Escenario sintético: no son resultados reales de RetailNova.",
    }
    (carpeta / "evaluacion_final.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    _graficos(y_test, proba, pred, seleccion["modelo"], carpeta_graficos)
    return salida


def _graficos(y_test, proba, pred, nombre: str, carpeta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axs[0].text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    axs[0].set_xticks([0, 1], ["Activo", "Abandona"])
    axs[0].set_yticks([0, 1], ["Activo", "Abandona"])
    axs[0].set_xlabel("Predicho")
    axs[0].set_ylabel("Real")
    axs[0].set_title(f"Matriz de confusión (umbral OOF) — {nombre}")
    axs[1].plot(fpr, tpr, color="#F4A300", linewidth=2, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
    axs[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axs[1].set_xlabel("Tasa de falsos positivos")
    axs[1].set_ylabel("Recall")
    axs[1].set_title(f"Curva ROC — {nombre}")
    axs[1].legend()
    plt.tight_layout()
    plt.savefig(Path(carpeta) / "evaluacion_test.png", dpi=130)
    plt.close()
