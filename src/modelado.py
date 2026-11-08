"""Modelamiento: selección por validación cruzada, umbral OOF y entrenamiento final.

Metodología (evita sobreestimar el desempeño):
1. El modelo se elige con validación cruzada estratificada de 5 folds SOLO sobre el
   conjunto de entrenamiento.
2. Ante un empate técnico (< 0,01 de AUC) se prefiere la Regresión Logística, más
   simple y explicable (RF4/RNF4 del plan).
3. El umbral de decisión se fija con predicciones fuera de muestra (OOF) del
   entrenamiento: el umbral más alto que aún logra recall >= 75 %.
4. El conjunto de prueba no se toca aquí; se evalúa una sola vez en `evaluacion.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as cfg


def construir_preprocesador() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), cfg.FEATURES_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg.FEATURES_CATEGORICAS),
    ])


def dividir_datos(df: pd.DataFrame, semilla: int = cfg.SEMILLA):
    """Partición estratificada 80/20. Devuelve X_train, X_test, y_train, y_test, region_train, region_test."""
    if cfg.VARIABLE_FUGA in cfg.FEATURES_NUMERICAS:
        raise ValueError(f"{cfg.VARIABLE_FUGA} provoca fuga de datos y no puede ser predictora")
    X = df[cfg.FEATURES_NUMERICAS + cfg.FEATURES_CATEGORICAS].copy()
    return train_test_split(X, df["abandono"].copy(), df["region"].copy(),
                            test_size=0.2, random_state=semilla, stratify=df["abandono"])


def modelos_candidatos(semilla: int = cfg.SEMILLA) -> dict:
    return {
        cfg.MODELO_SIMPLE: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=semilla),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=semilla, n_jobs=-1),
    }


def validacion_cruzada(semilla: int = cfg.SEMILLA) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)


def comparar_modelos_cv(X_train, y_train, modelos: dict | None = None, cv=None):
    """Compara los modelos con CV sobre entrenamiento. Devuelve (folds_df, comparacion_df)."""
    modelos = modelos or modelos_candidatos()
    cv = cv or validacion_cruzada()
    filas_folds, filas_cv = [], []
    for nombre, modelo in modelos.items():
        pipe = Pipeline([("prep", construir_preprocesador()), ("modelo", modelo)])
        s = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=1,
                           scoring={"auc": "roc_auc", "f1": "f1", "recall": "recall"})
        for fold in range(cv.get_n_splits()):
            filas_folds.append({"Modelo": nombre, "Fold": fold + 1, "ROC_AUC": s["test_auc"][fold],
                                "F1_umbral_0.5": s["test_f1"][fold],
                                "Recall_umbral_0.5": s["test_recall"][fold]})
        filas_cv.append({"Modelo": nombre, "ROC_AUC_CV": s["test_auc"].mean(),
                         "ROC_AUC_std": s["test_auc"].std(), "F1_CV_0.5": s["test_f1"].mean(),
                         "Recall_CV_0.5": s["test_recall"].mean()})
    comparacion = pd.DataFrame(filas_cv).sort_values("ROC_AUC_CV", ascending=False).reset_index(drop=True)
    return pd.DataFrame(filas_folds), comparacion


def elegir_modelo(comparacion: pd.DataFrame) -> str:
    """Mejor AUC de CV; si la diferencia con el segundo es < 0,01, el modelo simple."""
    mejor = comparacion.iloc[0]["Modelo"]
    if len(comparacion) > 1:
        brecha = comparacion.iloc[0]["ROC_AUC_CV"] - comparacion.iloc[1]["ROC_AUC_CV"]
        if brecha < cfg.DIFERENCIA_AUC_EMPATE and cfg.MODELO_SIMPLE in set(comparacion["Modelo"]):
            return cfg.MODELO_SIMPLE
    return mejor


def seleccionar_umbral_oof(proba_oof: np.ndarray, y_train, recall_objetivo: float = cfg.RECALL_OOF_OBJETIVO):
    """Umbral más alto con recall OOF >= objetivo. Si ninguno alcanza, usa 0,5."""
    precision, recall, umbrales = precision_recall_curve(y_train, proba_oof)
    candidatos = np.where(recall[:-1] >= recall_objetivo)[0]
    return float(umbrales[candidatos[-1]]) if len(candidatos) else 0.5


def entrenar(df: pd.DataFrame, carpeta_resultados: Path = cfg.RESULTADOS / "modelado",
             ruta_modelo: Path = cfg.MODELO_JOBLIB, carpeta_graficos: Path = cfg.GRAFICOS) -> dict:
    """Proceso `entrenar`: CV → modelo elegido → umbral OOF → ajuste final → guardado."""
    carpeta_resultados, carpeta_graficos = Path(carpeta_resultados), Path(carpeta_graficos)
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    X_train, _, y_train, _, _, _ = dividir_datos(df)

    folds, comparacion = comparar_modelos_cv(X_train, y_train)
    folds.to_csv(carpeta_resultados / "validacion_cruzada_5_folds.csv", index=False)
    comparacion.to_csv(carpeta_resultados / "comparacion_cv.csv", index=False)
    print(comparacion.round(4).to_string(index=False))

    nombre = elegir_modelo(comparacion)
    print(f"\nModelo elegido ANTES de tocar el conjunto de prueba: {nombre}")

    modelo = modelos_candidatos()[nombre]
    pipe = Pipeline([("prep", construir_preprocesador()), ("modelo", modelo)])
    proba_oof = cross_val_predict(pipe, X_train, y_train, cv=validacion_cruzada(),
                                  method="predict_proba", n_jobs=1)[:, 1]
    umbral = seleccionar_umbral_oof(proba_oof, y_train)
    pred_oof = proba_oof >= umbral
    seleccion = {
        "modelo": nombre,
        "umbral_decision": umbral,
        "recall_oof": float(recall_score(y_train, pred_oof)),
        "precision_oof": float(precision_score(y_train, pred_oof, zero_division=0)),
        "auc_cv": {r["Modelo"]: [float(r["ROC_AUC_CV"]), float(r["ROC_AUC_std"])]
                   for _, r in comparacion.iterrows()},
        "n_entrenamiento": int(len(X_train)),
    }
    print(f"Umbral OOF = {umbral:.4f} · recall OOF {seleccion['recall_oof']:.3f} · "
          f"precisión OOF {seleccion['precision_oof']:.3f}")

    pipe.fit(X_train, y_train)
    ruta_modelo = Path(ruta_modelo)
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ruta_modelo)
    (carpeta_resultados / "seleccion_modelo.json").write_text(
        json.dumps(seleccion, ensure_ascii=False, indent=2), encoding="utf-8")

    _grafico_umbral(y_train, proba_oof, carpeta_graficos)
    return seleccion


def _grafico_umbral(y_train, proba_oof, carpeta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    precision, recall, _ = precision_recall_curve(y_train, proba_oof)
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(recall, precision, color="#F4A300", label="Precisión–Recall (OOF, entrenamiento)")
    plt.axvline(cfg.META_RECALL, linestyle="--", color="gray", label="Meta de recall del plan (70%)")
    plt.xlabel("Recall")
    plt.ylabel("Precisión")
    plt.title("Selección de umbral sobre datos de entrenamiento (OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(carpeta) / "curva_pr_entrenamiento.png", dpi=130)
    plt.close()


def cargar_modelo(ruta: Path = cfg.MODELO_JOBLIB):
    if not Path(ruta).is_file():
        raise FileNotFoundError(f"No existe {ruta}; ejecuta primero --proceso entrenar")
    return joblib.load(ruta)


def cargar_seleccion(carpeta: Path = cfg.RESULTADOS / "modelado") -> dict:
    ruta = Path(carpeta) / "seleccion_modelo.json"
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}; ejecuta primero --proceso entrenar")
    return json.loads(ruta.read_text(encoding="utf-8"))
