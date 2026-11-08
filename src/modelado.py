"""Modelado supervisado LOGISMART (H4-T1).

Usa data/processed/logismart_modelado.csv creado por src.pipeline_datos.
La partición es temporal: train antes de julio de 2026, validación julio,
prueba agosto y septiembre. Se seleccionan modelo y umbral SOLO con validación.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.pipeline_datos import (
    CATEGORICAS, FECHA, NUMERICAS, OBJETIVO, PROHIBIDAS,
    construir_preprocesador,
)

FEATURES = NUMERICAS + CATEGORICAS


def dividir_temporalmente(df: pd.DataFrame):
    """Devuelve train/val/test cronológicos sin mezclar registros de otros períodos."""
    df = df.copy()
    df[FECHA] = pd.to_datetime(df[FECHA], errors="raise")
    train = df.loc[df[FECHA] < "2026-07-01"].copy()
    val = df.loc[df[FECHA].between("2026-07-01", "2026-07-31")].copy()
    test = df.loc[df[FECHA].between("2026-08-01", "2026-09-30")].copy()
    if not all(len(s) for s in (train, val, test)):
        raise ValueError("Uno de los períodos train/val/test está vacío")
    if len(train) + len(val) + len(test) != len(df):
        raise ValueError("Existen fechas fuera de los períodos definidos")
    return train, val, test


def separar_xy(df: pd.DataFrame):
    """Separa predictores X de la variable objetivo y, preservando índices."""
    if set(PROHIBIDAS) & set(FEATURES):
        raise AssertionError("Se ha incluido una variable prohibida en FEATURES")
    faltan = set(FEATURES + [OBJETIVO]) - set(df.columns)
    if faltan:
        raise ValueError(f"Columnas no presentes: {sorted(faltan)}")
    X = df.loc[:, FEATURES].copy()
    y = df[OBJETIVO].astype(int).copy()
    if not y.isin([0, 1]).all():
        raise ValueError("La variable objetivo debe ser binaria")
    return X, y


def metricas(y, probabilidades, umbral=0.5):
    pred = (probabilidades >= umbral).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probabilidades)),
        "pr_auc": float(average_precision_score(y, probabilidades)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "matriz_confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "tasa_retraso_real": float(np.mean(y)),
        "tasa_alertas": float(np.mean(pred)),
        "umbral": float(umbral),
    }


def elegir_umbral(y_val, prob_val):
    """Elige umbral por F1 de VALIDACIÓN; nunca consulta el conjunto test."""
    candidatos = np.arange(0.10, 0.91, 0.05)
    f1 = [f1_score(y_val, prob_val >= u, zero_division=0) for u in candidatos]
    return float(candidatos[int(np.argmax(f1))])


def muestrear(df, limite, semilla=42):
    """Muestreo por período, exclusivamente para pruebas rápidas."""
    if limite is None or len(df) <= limite:
        return df
    return df.sample(n=limite, random_state=semilla).sort_values(FECHA)


def entrenar(
    entrada="data/processed/logismart_modelado.csv",
    salida="models",
    modo_rapido=False,
    semilla=42,
):
    """Compara dos modelos con validación, evalúa ganador una vez en test y guarda evidencia."""
    entrada = Path(entrada)
    if not entrada.is_file():
        raise FileNotFoundError(f"No existe {entrada}. Ejecuta primero el pipeline de preparación.")
    df = pd.read_csv(entrada, parse_dates=[FECHA])
    train, val, test = dividir_temporalmente(df)
    conteos_completos = {"train": len(train), "validacion": len(val), "test": len(test)}

    if modo_rapido:
        train = muestrear(train, 40_000, semilla)
        val = muestrear(val, 10_000, semilla)
        test = muestrear(test, 15_000, semilla)

    X_train, y_train = separar_xy(train)
    X_val, y_val = separar_xy(val)
    X_test, y_test = separar_xy(test)
    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=120 if not modo_rapido else 60,
            max_depth=14, min_samples_leaf=8, max_samples=0.40,
            n_jobs=-1, random_state=semilla,
        ),
    }
    try:
        from xgboost import XGBClassifier
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=200 if not modo_rapido else 80,
            max_depth=5, learning_rate=0.06, subsample=0.8,
            colsample_bytree=0.85, tree_method="hist", n_jobs=4,
            random_state=semilla, eval_metric="logloss",
        )
    except ImportError as exc:
        raise ImportError("Instala xgboost desde requirements.txt para comparar ambos modelos") from exc

    evaluaciones = {}
    pipelines = {}
    for nombre, clasificador in modelos.items():
        pipeline = Pipeline([
            ("preprocesador", clone(construir_preprocesador())),
            ("modelo", clasificador),
        ])
        pipeline.fit(X_train, y_train)  # imputer y one-hot aprendidos SOLO con train
        p_val = pipeline.predict_proba(X_val)[:, 1]
        evaluaciones[nombre] = metricas(y_val, p_val)
        pipelines[nombre] = pipeline
        print(f"Validación {nombre}: ROC-AUC {evaluaciones[nombre]['roc_auc']:.4f} "
              f"| PR-AUC {evaluaciones[nombre]['pr_auc']:.4f}")

    # PR-AUC desempata en caso de ROC-AUC igual.
    ganador = max(evaluaciones, key=lambda k: (evaluaciones[k]["roc_auc"], evaluaciones[k]["pr_auc"]))
    elegido = pipelines[ganador]
    umbral = elegir_umbral(y_val, elegido.predict_proba(X_val)[:, 1])
    p_test = elegido.predict_proba(X_test)[:, 1]
    resultado_test = metricas(y_test, p_test, umbral=umbral)

    # En H4-T1 no se reentrena sobre validación: guardamos exactamente el pipeline evaluado.
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    archivo_modelo = salida / "modelo_logismart.joblib"
    joblib.dump(elegido, archivo_modelo)
    resumen = {
        "nota": "Datos sintéticos: estas métricas no son rendimiento operativo real.",
        "modo_rapido": bool(modo_rapido),
        "periodos": {"train": "2025-08 a 2026-06", "validacion": "2026-07", "test": "2026-08 a 2026-09"},
        "filas_por_periodo_dataset": conteos_completos,
        "filas_utilizadas": {"train": len(train), "validacion": len(val), "test": len(test)},
        "resultados_validacion_umbral_0_5": evaluaciones,
        "modelo_elegido": ganador,
        "umbral_elegido_en_validacion": umbral,
        "resultado_test": resultado_test,
        "archivo_modelo": str(archivo_modelo),
    }
    resultados_dir = Path("output/resultados/modelado")
    resultados_dir.mkdir(parents=True, exist_ok=True)
    archivo_resultados = resultados_dir / "resultados_modelado.json"
    archivo_resultados.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ganador por validación: {ganador} | umbral: {umbral:.2f}")
    print(f"Test final: ROC-AUC {resultado_test['roc_auc']:.4f} "
          f"| PR-AUC {resultado_test['pr_auc']:.4f} "
          f"| recall {resultado_test['recall']:.4f} | F1 {resultado_test['f1']:.4f}")
    print(f"Guardado: {archivo_modelo} y {archivo_resultados}")
    return resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Modelado predictivo de LOGISMART")
    ap.add_argument("--entrada", default="data/processed/logismart_modelado.csv")
    ap.add_argument("--salida", default="models")
    ap.add_argument("--modo-rapido", action="store_true")
    args = ap.parse_args()
    entrenar(args.entrada, args.salida, args.modo_rapido)
