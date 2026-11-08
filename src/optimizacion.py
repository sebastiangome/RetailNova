"""H4-T3: búsqueda de hiperparámetros, validación temporal y costos computacionales.

La optimización usa solamente TRAIN (hasta junio 2026). RandomizedSearchCV
trabaja con folds cronológicos por mes; julio evalúa las configuraciones
resultantes. TEST (agosto-septiembre) no se lee ni usa en esta etapa.

Se preserva el modelo oficial: no se escribe en models/.
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.modelado import separar_xy
from src.pipeline_datos import FECHA, construir_preprocesador


ESPACIOS = {
    "Random Forest": {
        "modelo__n_estimators": [80, 120, 180],
        "modelo__max_depth": [8, 12, 16, None],
        "modelo__min_samples_leaf": [4, 8, 16],
        "modelo__max_features": ["sqrt", 0.75],
    },
    "XGBoost": {
        "modelo__n_estimators": [100, 160, 240],
        "modelo__max_depth": [3, 4, 5, 6],
        "modelo__learning_rate": [0.03, 0.06, 0.10],
        "modelo__subsample": [0.7, 0.85, 1.0],
        "modelo__colsample_bytree": [0.7, 0.85, 1.0],
        "modelo__min_child_weight": [1, 3, 6],
    },
}


def construir_folds_temporales(fechas):
    """Folds expansivos: hasta abril -> mayo; hasta mayo -> junio 2026.

    Los cortes por mes evitan que operaciones del mismo día caigan a
    ambos lados de un límite de entrenamiento/validación.
    """
    fechas = pd.to_datetime(pd.Series(fechas).reset_index(drop=True))
    if fechas.isna().any():
        raise ValueError("Fechas nulas en la muestra de optimización")
    folds = []
    for corte, inicio, fin in (
        ("2026-05-01", "2026-05-01", "2026-06-01"),
        ("2026-06-01", "2026-06-01", "2026-07-01"),
    ):
        tr = np.flatnonzero((fechas < pd.Timestamp(corte)).to_numpy())
        va = np.flatnonzero(((fechas >= pd.Timestamp(inicio)) & (fechas < pd.Timestamp(fin))).to_numpy())
        if not len(tr) or not len(va):
            raise ValueError(f"Faltan filas para fold {inicio}: train={len(tr)}, val={len(va)}")
        if fechas.iloc[tr].max() >= fechas.iloc[va].min():
            raise AssertionError("Fuga temporal detectada en los folds")
        folds.append((tr, va))
    return folds


def preparar_muestra(train, max_filas, semilla):
    """Muestra determinista del período de train; ordenada por fecha para auditoría."""
    if max_filas is not None and max_filas < 1000:
        raise ValueError("--max-filas debe ser >= 1000")
    if max_filas is not None and len(train) > max_filas:
        train = train.sample(n=max_filas, random_state=semilla)
    return train.sort_values(FECHA, kind="stable").reset_index(drop=True)


def modelos_base(semilla):
    from xgboost import XGBClassifier
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=8,
            max_samples=0.40, n_jobs=2, random_state=semilla,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.85,
            tree_method="hist", n_jobs=2, random_state=semilla,
            eval_metric="logloss",
        ),
    }


def _pipeline(clf):
    return Pipeline([
        ("preprocesador", clone(construir_preprocesador())),
        ("modelo", clf),
    ])


def _metricas_probabilidades(y, proba):
    return {"roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba))}


def _cronometrar_fit(estimador, X, y):
    """Tiempo de pared y pico de asignaciones Python (no RAM total del proceso)."""
    tracemalloc.start()
    inicio = time.perf_counter()
    try:
        estimador.fit(X, y)
        segundos = time.perf_counter() - inicio
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return float(segundos), float(pico / (1024 * 1024))


def optimizar(
    entrada="data/processed/logismart_modelado.csv",
    salida="output/resultados/modelado",
    max_filas=100_000,
    iteraciones=4,
    semilla=42,
):
    """Compara base vs optimización con train; verifica resultado en julio.

    No lee predictores/etiquetas de TEST para seleccionar ninguna configuración,
    no reentrena en test ni reemplaza `models/modelo_logismart.joblib`.
    """
    if iteraciones < 1:
        raise ValueError("--iteraciones debe ser >= 1")
    ruta = Path(entrada)
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta {ruta}. Ejecuta primero la preparación.")
    df = pd.read_csv(ruta, parse_dates=[FECHA])
    # Los límites dependen solo de fecha: no se utilizan filas de test.
    train = df.loc[df[FECHA] < pd.Timestamp("2026-07-01")].copy()
    val = df.loc[(df[FECHA] >= pd.Timestamp("2026-07-01")) &
                 (df[FECHA] < pd.Timestamp("2026-08-01"))].copy()
    if train.empty or val.empty:
        raise ValueError("Entrenamiento o validación vacíos")
    muestra = preparar_muestra(train, max_filas, semilla)
    X_tr, y_tr = separar_xy(muestra)
    X_val, y_val = separar_xy(val)
    cv = construir_folds_temporales(muestra[FECHA])

    filas = []
    resultados = {}
    for nombre, estimador in modelos_base(semilla).items():
        print(f"\n{nombre}: base y RandomizedSearchCV ({iteraciones} candidatos, 2 folds)")
        base = _pipeline(clone(estimador))
        t_base, mem_base = _cronometrar_fit(base, X_tr, y_tr)
        met_base = _metricas_probabilidades(y_val, base.predict_proba(X_val)[:, 1])

        busqueda = RandomizedSearchCV(
            estimator=_pipeline(clone(estimador)),
            param_distributions=ESPACIOS[nombre],
            n_iter=iteraciones,
            scoring="average_precision",
            cv=cv,
            refit=False,  # se mide por separado el refit sobre toda la muestra
            random_state=semilla,
            n_jobs=1,
            verbose=0,
            return_train_score=False,
            error_score="raise",
        )
        inicio = time.perf_counter()
        busqueda.fit(X_tr, y_tr)
        t_busqueda = float(time.perf_counter() - inicio)
        mejor = _pipeline(clone(estimador)).set_params(**busqueda.best_params_)
        t_tuned, mem_tuned = _cronometrar_fit(mejor, X_tr, y_tr)
        met_tuned = _metricas_probabilidades(y_val, mejor.predict_proba(X_val)[:, 1])

        cv_resultados = pd.DataFrame(busqueda.cv_results_)
        registros_cv = []
        for _, fila in cv_resultados.iterrows():
            registros_cv.append({
                "parametros": fila["params"],
                "pr_auc_cv_media": float(fila["mean_test_score"]),
                "pr_auc_cv_desv": float(fila["std_test_score"]),
                "segundos_fit_cv_medio": float(fila["mean_fit_time"]),
                "ranking": int(fila["rank_test_score"]),
            })
        registros_cv.sort(key=lambda r: r["ranking"])
        resultados[nombre] = {
            "parametros_base": base.named_steps["modelo"].get_params(),
            "hiperparametros_elegidos": busqueda.best_params_,
            "pr_auc_cv_mejor": float(busqueda.best_score_),
            "configuraciones_cv": registros_cv,
            "validacion_julio": {"base": met_base, "optimizado": met_tuned},
            "eficiencia": {
                "base_segundos_fit": t_base,
                "optimizado_segundos_fit": t_tuned,
                "busqueda_segundos": t_busqueda,
                "base_pico_memoria_python_mb": mem_base,
                "optimizado_pico_memoria_python_mb": mem_tuned,
            },
        }
        for version, metric, tiempo, memoria in (
            ("base", met_base, t_base, mem_base),
            ("optimizado", met_tuned, t_tuned, mem_tuned),
        ):
            filas.append({"modelo": nombre, "version": version,
                          "roc_auc_validacion": metric["roc_auc"],
                          "pr_auc_validacion": metric["pr_auc"],
                          "segundos_ajuste": tiempo,
                          "pico_memoria_python_mb": memoria,
                          "filas_train": len(X_tr), "filas_validacion": len(X_val)})
        print(f"  base: PR-AUC julio {met_base['pr_auc']:.4f} | {t_base:.1f}s")
        print(f"  optimizado: PR-AUC julio {met_tuned['pr_auc']:.4f} | {t_tuned:.1f}s")
        print(f"  mejor PR-AUC CV {busqueda.best_score_:.4f} | búsqueda {t_busqueda:.1f}s")

    tabla = pd.DataFrame(filas)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    archivo_json = salida / "optimizacion_hiperparametros.json"
    archivo_csv = salida / "comparacion_optimizacion.csv"
    evidencia = {
        "nota": "Dataset sintético. H4-T3 NO consulta el conjunto test, ni reemplaza el modelo oficial.",
        "estrategia": "RandomizedSearchCV con dos folds mensuales expansivos dentro de train",
        "metrica_busqueda": "PR-AUC (average_precision) de folds en train",
        "validacion_externa": "julio 2026: solo evaluación comparativa, sin selección de parámetros",
        "periodos_cv": ["hasta abril -> mayo 2026", "hasta mayo -> junio 2026"],
        "registros_train_disponibles": int(len(train)),
        "registros_train_usados": int(len(muestra)),
        "registros_validacion": int(len(val)),
        "iteraciones_por_modelo": int(iteraciones),
        "semilla": int(semilla),
        "memoria": "tracemalloc captura asignaciones Python, no RAM total ni memoria nativa de XGBoost",
        "resultados": resultados,
    }
    # Conversión segura de tipos numpy en los hiperparámetros del estimador.
    archivo_json.write_text(json.dumps(evidencia, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tabla.to_csv(archivo_csv, index=False)
    print(f"\nEvidencias: {archivo_json} y {archivo_csv}")
    print("Modelo oficial conservado. No se realizaron predicciones sobre test.")
    return evidencia, tabla


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LOGISMART H4-T3: optimización de hiperparámetros")
    ap.add_argument("--entrada", default="data/processed/logismart_modelado.csv")
    ap.add_argument("--salida", default="output/resultados/modelado")
    ap.add_argument("--max-filas", type=int, default=100_000)
    ap.add_argument("--iteraciones", type=int, default=4)
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()
    optimizar(args.entrada, args.salida, args.max_filas, args.iteraciones, args.semilla)
