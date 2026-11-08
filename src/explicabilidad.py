"""H4-T2: explicabilidad SHAP del modelo LOGISMART ya entrenado.

Complementa el análisis global con:
- importancia global agregada por predictor original,
- beeswarm SHAP agrupado por predictor original,
- waterfall SHAP de una observación local de alto riesgo.

No entrena, no selecciona umbral y no utiliza test para decisiones.
Los artefactos son reproducibles; el notebook solo los presenta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modelado import dividir_temporalmente, separar_xy
from src.pipeline_datos import FECHA, NUMERICAS, CATEGORICAS


def agrupar_columnas(preprocesador):
    """Devuelve el predictor original de cada columna transformada.

    Incluye las columnas indicadoras de nulos creadas por SimpleImputer y
    agrupa los dummies one-hot según su variable original.
    """
    nombres = list(preprocesador.get_feature_names_out())
    resultado = [None] * len(nombres)
    nums = preprocesador.named_transformers_["num"]
    cats = preprocesador.named_transformers_["cat"]
    num_slice = preprocesador.output_indices_["num"]
    cat_slice = preprocesador.output_indices_["cat"]

    n_num = list(nums.get_feature_names_out(NUMERICAS))
    if len(n_num) != num_slice.stop - num_slice.start:
        raise ValueError("Dimensión numérica no coincide")
    for j, nombre in enumerate(n_num, num_slice.start):
        prefijo = "missingindicator_"
        original = nombre[len(prefijo):] if nombre.startswith(prefijo) else nombre
        if original not in NUMERICAS:
            raise ValueError(f"Variable numérica desconocida: {nombre}")
        resultado[j] = original

    cat_names = list(cats.get_feature_names_out(CATEGORICAS))
    if len(cat_names) != cat_slice.stop - cat_slice.start:
        raise ValueError("Dimensión categórica no coincide")
    for j, nombre in enumerate(cat_names, cat_slice.start):
        matches = [c for c in CATEGORICAS if nombre.startswith(c + "_")]
        if len(matches) != 1:
            raise ValueError(f"No se puede asociar categoría: {nombre}")
        resultado[j] = matches[0]
    if any(x is None for x in resultado):
        raise ValueError("Existen columnas transformadas sin correspondencia")
    return resultado


def valores_clase_positiva(valores, n_filas, n_features):
    """Normaliza formatos SHAP (2D, 3D clases, o lista por clase)."""
    if isinstance(valores, list):
        valores = valores[1] if len(valores) == 2 else valores[0]
    valores = np.asarray(valores)
    if valores.ndim == 3 and valores.shape[-1] == 2:
        valores = valores[:, :, 1]
    if valores.shape != (n_filas, n_features):
        raise ValueError(f"Formato SHAP inesperado: {valores.shape}")
    if not np.isfinite(valores).all():
        raise ValueError("SHAP produjo valores no finitos")
    return valores


def base_value_positivo(explicador):
    """Obtiene el valor base asociado a la clase positiva."""
    base = getattr(explicador, "expected_value", 0.0)
    if isinstance(base, list):
        base = base[1] if len(base) == 2 else base[0]
    base = np.asarray(base).squeeze()
    if base.ndim > 0:
        if base.size == 2:
            base = base[1]
        else:
            base = base.flat[0]
    return float(base)


def _explanation_agrupada(shap_values_agrupados: pd.DataFrame, X_original: pd.DataFrame, base_value: float):
    """Crea un objeto shap.Explanation agrupado por predictor original."""
    import shap

    columnas = list(shap_values_agrupados.columns)
    data = X_original[columnas].copy()
    # SHAP beeswarm colorea variables numéricas; categorías de texto se dejan sin color
    # para evitar códigos ordinales artificiales que no tienen significado.
    for columna in data.columns:
        data[columna] = pd.to_numeric(data[columna], errors="coerce")
    return shap.Explanation(
        values=shap_values_agrupados[columnas].to_numpy(),
        base_values=np.full(len(shap_values_agrupados), base_value, dtype=float),
        data=data.to_numpy(dtype=float),
        feature_names=columnas,
    )


def explicar(
    entrada="data/processed/logismart_modelado.csv",
    carpeta_modelos="models",
    muestra=500,
    semilla=42,
):
    """Explica una muestra de validación mediante el pipeline guardado."""
    if muestra < 1:
        raise ValueError("--muestra debe ser un entero positivo")
    carpeta = Path(carpeta_modelos)
    ruta_modelo = carpeta / "modelo_logismart.joblib"
    resultados_dir = Path("output/resultados/explicabilidad")
    graficos_dir = Path("output/graficos")
    ruta_resumen = Path("output/resultados/modelado/resultados_modelado.json")
    if not ruta_modelo.is_file() or not ruta_resumen.is_file():
        raise FileNotFoundError("Ejecuta primero: python main.py --proceso entrenar")
    resumen = json.loads(ruta_resumen.read_text(encoding="utf-8"))
    if resumen.get("modo_rapido"):
        raise ValueError("Se requiere el entrenamiento completo, no modo rápido")
    if resumen.get("modelo_elegido") != "XGBoost":
        raise ValueError("Esta implementación de SHAP requiere que el modelo elegido sea XGBoost")

    # Solo modelos locales confiables: joblib puede ejecutar código al cargar.
    pipeline = joblib.load(ruta_modelo)
    pre = pipeline.named_steps["preprocesador"]
    clf = pipeline.named_steps["modelo"]
    if clf.__class__.__name__ != "XGBClassifier":
        raise ValueError("El artefacto guardado no contiene XGBClassifier")

    df = pd.read_csv(entrada, parse_dates=[FECHA])
    _, validacion, _ = dividir_temporalmente(df)
    if len(validacion) != resumen["filas_utilizadas"]["validacion"]:
        raise ValueError("El CSV no coincide en tamaño con la validación del entrenamiento")
    seleccion = validacion.sample(n=min(muestra, len(validacion)), random_state=semilla).copy()
    X, _ = separar_xy(seleccion)
    proba = pipeline.predict_proba(X)[:, 1]

    transformado = pre.transform(X)
    if hasattr(transformado, "toarray"):
        transformado = transformado.toarray()
    transformado = np.asarray(transformado, dtype=np.float32)
    nombres = list(pre.get_feature_names_out())
    grupos = agrupar_columnas(pre)
    if transformado.shape[1] != len(grupos):
        raise ValueError("Desalineación entre columnas y valores transformados")

    try:
        import shap
    except ImportError as exc:
        raise ImportError("Instala SHAP: python -m pip install shap>=0.45") from exc

    explicador = shap.TreeExplainer(clf)
    base_value = base_value_positivo(explicador)
    valores = valores_clase_positiva(explicador.shap_values(transformado), len(X), len(nombres))

    # Sumar dummies y missing indicators por observación ANTES de sacar |SHAP| medio.
    shap_agrupados = pd.DataFrame(valores, columns=range(len(nombres))).T.groupby(grupos, sort=False).sum().T
    shap_agrupados.columns = list(dict.fromkeys(grupos))
    global_importancias = shap_agrupados.abs().mean().sort_values(ascending=False)

    # Elegir ejemplo local: la operación con mayor probabilidad predicha en la muestra.
    fila_ejemplo = int(np.argmax(proba))
    local = shap_agrupados.iloc[fila_ejemplo].sort_values(key=np.abs, ascending=False)
    riesgo = float(proba[fila_ejemplo])

    salida = {
        "nota": "Datos sintéticos. SHAP explica el modelo, no demuestra causalidad ni impacto real.",
        "modelo": resumen["modelo_elegido"],
        "conjunto_explicado": "validacion (julio 2026)",
        "numero_operaciones": int(len(X)),
        "semilla_muestreo": int(semilla),
        "escala_shap": "log-odds / margen bruto de XGBoost; NO puntos porcentuales de probabilidad",
        "agrupacion": "SHAP de indicadores one-hot e indicadores de nulos sumados por predictor original",
        "importancia_global_media_abs_shap": [
            {"variable": str(c), "valor": float(v)} for c, v in global_importancias.items()
        ],
        "ejemplo_local": {
            "posicion_en_muestra": fila_ejemplo,
            "probabilidad_predicha": riesgo,
            "base_value": base_value,
            "top_atribuciones_por_variable": [
                {"variable": str(c), "shap": float(v)} for c, v in local.head(12).items()
            ],
            "valores_originales": X.iloc[fila_ejemplo][list(global_importancias.index[:12])].to_dict(),
        },
        "artefactos": {
            "barras_globales": "output/graficos/importancia_shap.png",
            "beeswarm": "output/graficos/beeswarm_shap.png",
            "waterfall": "output/graficos/waterfall_shap.png",
        }
    }
    carpeta.mkdir(parents=True, exist_ok=True)
    resultados_dir.mkdir(parents=True, exist_ok=True)
    graficos_dir.mkdir(parents=True, exist_ok=True)
    ruta_json = resultados_dir / "explicabilidad_shap.json"
    ruta_json.write_text(json.dumps(salida, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) barras globales
    top = global_importancias.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values)
    ax.set_xlabel("Media de |SHAP| (margen bruto / log-odds)")
    ax.set_title("LOGISMART · Importancia global SHAP (validación)")
    fig.tight_layout()
    ruta_png = graficos_dir / "importancia_shap.png"
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) beeswarm agrupado
    exp_agrupada = _explanation_agrupada(shap_agrupados, X, base_value)
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(exp_agrupada, max_display=15, show=False)
    plt.title("LOGISMART · SHAP beeswarm (validación)")
    plt.tight_layout()
    ruta_bee = graficos_dir / "beeswarm_shap.png"
    plt.savefig(ruta_bee, dpi=150, bbox_inches="tight")
    plt.close()

    # 3) waterfall local
    columnas_local = list(global_importancias.index)
    exp_local = shap.Explanation(
        values=shap_agrupados.iloc[fila_ejemplo][columnas_local].to_numpy(),
        base_values=base_value,
        data=X.iloc[fila_ejemplo][columnas_local].to_numpy(dtype=object),
        feature_names=columnas_local,
    )
    plt.figure(figsize=(9, 7))
    shap.plots.waterfall(exp_local, max_display=12, show=False)
    plt.title("LOGISMART · SHAP waterfall (operación local de alto riesgo)")
    plt.tight_layout()
    ruta_wf = graficos_dir / "waterfall_shap.png"
    plt.savefig(ruta_wf, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Explicabilidad SHAP: {len(X):,} operaciones de validación | {len(global_importancias)} predictores")
    print("Mayores importancias (media |SHAP|, no causal):")
    print(global_importancias.head(10).round(4).to_string())
    print(f"Ejemplo local: fila {fila_ejemplo} | probabilidad predicha {riesgo:.4f}")
    print(f"Guardado: {ruta_json}, {ruta_png}, {ruta_bee} y {ruta_wf}")
    return salida


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explicabilidad SHAP de LOGISMART")
    parser.add_argument("--entrada", default="data/processed/logismart_modelado.csv")
    parser.add_argument("--carpeta-modelos", default="models")
    parser.add_argument("--muestra", type=int, default=500)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()
    explicar(args.entrada, args.carpeta_modelos, args.muestra, args.semilla)
