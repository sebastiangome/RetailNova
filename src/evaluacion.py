"""Auditoría reproducible de LOGISMART: umbrales en validación y comunas en test.

No reentrena, no cambia el modelo ni el umbral seleccionado en H4-T1.
Ejecutar desde raíz: python -m src.evaluacion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modelado import dividir_temporalmente, separar_xy, metricas
from src.pipeline_datos import FECHA


def tabla_umbrales(y_val, probabilidades, umbrales=None):
    """Compara decisiones usando únicamente VALIDACIÓN, nunca test."""
    if umbrales is None:
        umbrales = np.round(np.arange(0.05, 0.51, 0.05), 2)
    filas = []
    for u in umbrales:
        met = metricas(y_val, probabilidades, umbral=float(u))
        tn, fp = met["matriz_confusion"][0]
        fn, tp = met["matriz_confusion"][1]
        filas.append({"umbral": round(float(u), 4), "precision": met["precision"],
                      "recall": met["recall"], "f1": met["f1"],
                      "tasa_alertas": met["tasa_alertas"],
                      "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(filas)


def tabla_comunas(test, probabilidades, umbral, minimo_positivos=30):
    """Audita el destino en TEST: métricas descriptivas, no selecciona umbral."""
    if len(test) != len(probabilidades):
        raise ValueError("Test y probabilidades tienen longitudes distintas")
    datos = test[["comuna_destino", "retraso"]].copy().reset_index(drop=True)
    datos["p"] = np.asarray(probabilidades)
    datos["alerta"] = datos["p"].ge(float(umbral)).astype(int)
    datos["tp"] = ((datos["retraso"] == 1) & (datos["alerta"] == 1)).astype(int)
    datos["fp"] = ((datos["retraso"] == 0) & (datos["alerta"] == 1)).astype(int)
    grupos = datos.groupby("comuna_destino", observed=True).agg(
        operaciones=("retraso", "size"), retrasos=("retraso", "sum"),
        alertas=("alerta", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
    )
    grupos["tasa_retrasos"] = grupos["retrasos"] / grupos["operaciones"]
    grupos["tasa_alertas"] = grupos["alertas"] / grupos["operaciones"]
    grupos["precision"] = grupos["tp"] / grupos["alertas"].replace(0, np.nan)
    grupos["recall"] = grupos["tp"] / grupos["retrasos"].replace(0, np.nan)
    grupos["retrasos_sin_alerta"] = grupos["retrasos"] - grupos["tp"]
    grupos["muestra_suficiente_recall"] = grupos["retrasos"] >= minimo_positivos
    return grupos.reset_index().sort_values("operaciones", ascending=False)


def evaluar(entrada="data/processed/logismart_modelado.csv", carpeta_modelos="models"):
    """Regenera auditoría a partir de pipeline guardado y división temporal fija."""
    carpeta = Path(carpeta_modelos)
    resultados_dir = Path("output/resultados/modelado")
    ruta_json = resultados_dir / "resultados_modelado.json"
    ruta_modelo = carpeta / "modelo_logismart.joblib"
    if not ruta_json.exists() or not ruta_modelo.exists():
        raise FileNotFoundError("Ejecuta primero: python main.py --proceso entrenar")
    resumen = json.loads(ruta_json.read_text(encoding="utf-8"))
    if resumen.get("modo_rapido"):
        raise ValueError("La auditoría oficial requiere entrenamiento completo, no modo rápido")
    modelo = joblib.load(ruta_modelo)  # archivo local de confianza; nunca cargar joblib ajenos
    df = pd.read_csv(entrada, parse_dates=[FECHA])
    _, val, test = dividir_temporalmente(df)
    if (len(val) != resumen["filas_utilizadas"]["validacion"] or
            len(test) != resumen["filas_utilizadas"]["test"]):
        raise ValueError("Los períodos del CSV no coinciden con los del entrenamiento guardado")
    X_val, y_val = separar_xy(val)
    X_test, y_test = separar_xy(test)
    proba_val = modelo.predict_proba(X_val)[:, 1]
    proba_test = modelo.predict_proba(X_test)[:, 1]
    umbral = float(resumen["umbral_elegido_en_validacion"])
    # Protege contra desalineación accidental modelo/JSON o CSV sustituido.
    auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)
    if not (np.isclose(auc, resumen["resultado_test"]["roc_auc"], atol=1e-8) and
            np.isclose(pr_auc, resumen["resultado_test"]["pr_auc"], atol=1e-8)):
        raise ValueError("Modelo/CSV no corresponden con las métricas oficiales del JSON")
    umbrales = tabla_umbrales(y_val, proba_val, umbrales=sorted(set(
        np.round(np.arange(0.05, 0.51, 0.05), 2).tolist() + [round(umbral, 4)])))
    comunas = tabla_comunas(test, proba_test, umbral)
    carpeta.mkdir(parents=True, exist_ok=True)
    resultados_dir.mkdir(parents=True, exist_ok=True)
    dir_comunas = Path("output/resultados/sesgos_equidad")
    dir_comunas.mkdir(parents=True, exist_ok=True)
    ruta_u = resultados_dir / "auditoria_umbrales_validacion.json"
    ruta_c = dir_comunas / "auditoria_comunas_test.json"
    ruta_u.write_text(umbrales.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    ruta_c.write_text(comunas.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    print("Auditoría LOGISMART (modelo sin reentrenamiento)")
    print(f"Umbral oficial: {umbral:.2f} | AUC test: {auc:.4f} | PR-AUC test: {pr_auc:.4f}")
    print("\nUmbrales en VALIDACIÓN:")
    print(umbrales[["umbral", "precision", "recall", "f1", "tasa_alertas"]].round(3).to_string(index=False))
    print("\nComunas en TEST (primeras 10 por operaciones):")
    print(comunas[["comuna_destino", "operaciones", "retrasos", "precision", "recall", "tasa_alertas"]].head(10).round(3).to_string(index=False))
    print(f"\nGuardado: {ruta_u} y {ruta_c}")
    return {"umbrales_validacion": umbrales, "comunas_test": comunas}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoría de umbrales y comunas de LOGISMART")
    parser.add_argument("--entrada", default="data/processed/logismart_modelado.csv")
    parser.add_argument("--carpeta-modelos", default="models")
    args = parser.parse_args()
    evaluar(entrada=args.entrada, carpeta_modelos=args.carpeta_modelos)


# --- H5-T1: evaluación comparativa final, sin reajustar el modelo oficial ---
def reporte_errores(met):
    """Traduce la matriz de confusión en errores de interés operacional."""
    (tn, fp), (fn, tp) = met["matriz_confusion"]
    return {
        "verdaderos_negativos": int(tn), "falsos_positivos": int(fp),
        "falsos_negativos": int(fn), "verdaderos_positivos": int(tp),
        "retrasos_reales": int(fn + tp), "alertas_emitidas": int(fp + tp),
        "retrasos_detectados": int(tp),
        "retrasos_no_detectados": int(fn),
    }


def comparar_modelos_test(
    entrada="data/processed/logismart_modelado.csv",
    carpeta_modelos="models",
    salida="output",
    semilla=42,
):
    """H5-T1: RF original vs XGBoost oficial con el mismo test cronológico.

    RF se ajusta exclusivamente en train y su umbral se define en julio.
    XGBoost y su umbral se cargan sin cambios; test nunca selecciona parámetros.
    """
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from src.modelado import elegir_umbral
    from src.pipeline_datos import construir_preprocesador

    ruta_modelo = Path(carpeta_modelos) / "modelo_logismart.joblib"
    ruta_resultados = Path("output/resultados/modelado/resultados_modelado.json")
    if not (ruta_modelo.exists() and ruta_resultados.exists()):
        raise FileNotFoundError("Falta modelo oficial o resultados_modelado.json. Ejecuta primero el entrenamiento completo.")
    oficial = json.loads(ruta_resultados.read_text(encoding="utf-8"))
    if oficial.get("modo_rapido") or oficial.get("modelo_elegido") != "XGBoost":
        raise ValueError("H5-T1 requiere resultados oficiales completos y XGBoost ganador de H4-T1")
    df = pd.read_csv(entrada, parse_dates=[FECHA])
    train, val, test = dividir_temporalmente(df)
    esperados = oficial["filas_utilizadas"]
    if any(len(parte) != esperados[clave] for parte, clave in
           [(train, "train"), (val, "validacion"), (test, "test")]):
        raise ValueError("El CSV no coincide con el número de registros del modelo oficial")
    X_train, y_train = separar_xy(train)
    X_val, y_val = separar_xy(val)
    X_test, y_test = separar_xy(test)

    # Comparación con RANDOM FOREST ORIGINAL, no el optimizado de 100k de H4-T3.
    rf = Pipeline([
        ("preprocesador", clone(construir_preprocesador())),
        ("modelo", RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=8,
            max_samples=0.40, n_jobs=-1, random_state=semilla,
        )),
    ])
    rf.fit(X_train, y_train)
    proba_rf_val = rf.predict_proba(X_val)[:, 1]
    umbral_rf = elegir_umbral(y_val, proba_rf_val)
    # El umbral se fija antes de calcular resultados de test.
    resultados = {"Random Forest": metricas(y_test, rf.predict_proba(X_test)[:, 1], umbral=umbral_rf)}

    xgb = joblib.load(ruta_modelo)  # solo artefacto local de confianza
    proba_xgb = xgb.predict_proba(X_test)[:, 1]
    umbral_xgb = float(oficial["umbral_elegido_en_validacion"])
    resultado_xgb = metricas(y_test, proba_xgb, umbral=umbral_xgb)
    # Validación robusta contra la huella de métricas guardadas en H4-T1.
    for clave in ("roc_auc", "pr_auc", "recall", "precision", "f1"):
        if not np.isclose(resultado_xgb[clave], oficial["resultado_test"][clave], atol=1e-8):
            raise ValueError(f"Inconsistencia del modelo oficial/CSV: {clave}")
    if resultado_xgb["matriz_confusion"] != oficial["resultado_test"]["matriz_confusion"]:
        raise ValueError("Inconsistencia de matriz de confusión del XGBoost oficial")
    resultados["XGBoost"] = resultado_xgb

    for met in resultados.values():
        met["conteos"] = reporte_errores(met)
    # Criterio original H4-T1 (ROC-AUC y PR-AUC solo de julio), no el test.
    seleccionado = oficial["modelo_elegido"]
    resumen = {
        "tarea": "H5-T1", "tipo_datos": "sintéticos", "conjunto_test": "agosto-septiembre 2026",
        "filas_train": len(train), "filas_validacion": len(val), "filas_test": len(test),
        "nota": "Comparación descriptiva en test; ningún ajuste o selección a partir de test.",
        "random_forest": "Reentrenado con parámetros originales de H4-T1 y train completo; umbral elegido en julio",
        "xgboost": "Pipeline oficial cargado sin reentrenar, umbral de julio preservado",
        "modelo_oficial_seleccionado_en_validacion": seleccionado,
        "resultados_test": resultados,
        "limitaciones": [
            "Los datos son sintéticos y no cuantifican impacto real de negocio",
            "El test ya fue consultado en H4-T1 y H4-T4; no es una evaluación prospectiva nueva",
            "Precision y Recall dependen del umbral fijado en validación",
            "Una alerta no garantiza una reducción de los retrasos; requiere intervención y evaluación operacional",
            "Los umbrales pueden diferir entre modelos y se fijaron exclusivamente con validación",
        ],
    }
    carpeta = Path(salida)
    destino = carpeta / "resultados" / "modelado"
    destino.mkdir(parents=True, exist_ok=True)
    graficos = carpeta / "graficos"
    graficos.mkdir(parents=True, exist_ok=True)
    archivo_json = destino / "evaluacion_final_modelos.json"
    archivo_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    tabla = pd.DataFrame([
        {"modelo": nombre, **{k: met[k] for k in
         ("umbral", "roc_auc", "pr_auc", "precision", "recall", "f1", "tasa_alertas")}, **met["conteos"]}
        for nombre, met in resultados.items()
    ])
    archivo_csv = destino / "comparacion_modelos_test.csv"
    tabla.to_csv(archivo_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, (nombre, met) in zip(axs, resultados.items()):
        matriz = np.array(met["matriz_confusion"])
        ax.imshow(matriz, cmap="Blues")
        ax.set(title=f"{nombre} | umbral {met['umbral']:.2f}", xlabel="Predicción", ylabel="Real")
        ax.set_xticks([0, 1], ["A tiempo", "Retraso"])
        ax.set_yticks([0, 1], ["A tiempo", "Retraso"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matriz[i, j]:,}".replace(",", "."),
                        ha="center", va="center", color="white" if matriz[i, j] > matriz.max() / 2 else "black")
    fig.suptitle("LOGISMART H5-T1 — matrices de confusión en test sintético")
    archivo_png = graficos / "matrices_confusion_modelos_test.png"
    fig.savefig(archivo_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("H5-T1: comparación final sin cambios al modelo oficial")
    print(tabla[["modelo", "precision", "recall", "roc_auc", "pr_auc", "f1", "falsos_positivos", "falsos_negativos"]].round(4).to_string(index=False))
    print(f"Archivos: {archivo_json} | {archivo_csv} | {archivo_png}")
    return resumen
