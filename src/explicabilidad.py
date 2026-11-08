"""Explicabilidad con SHAP (RF4/RNF4): importancia global y ficha por cliente."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config as cfg
from src.modelado import cargar_modelo, dividir_datos


def calcular_shap(pipeline, X: pd.DataFrame, muestra: int = 2000, semilla: int = cfg.SEMILLA):
    """Devuelve (shap_df con una columna por variable, índice de la muestra)."""
    import shap

    prep = pipeline.named_steps["prep"]
    modelo = pipeline.named_steps["modelo"]
    nombres = cfg.FEATURES_NUMERICAS + list(
        prep.named_transformers_["cat"].get_feature_names_out(cfg.FEATURES_CATEGORICAS))

    idx = X.sample(n=min(muestra, len(X)), random_state=semilla).index
    Xt = prep.transform(X.loc[idx])
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    if hasattr(modelo, "feature_importances_"):          # modelos de árboles
        valores = shap.TreeExplainer(modelo).shap_values(Xt)
        if isinstance(valores, list):
            valores = valores[1]
        elif valores.ndim == 3:
            valores = valores[:, :, 1]
    else:                                                # modelos lineales
        valores = shap.LinearExplainer(modelo, Xt).shap_values(Xt)
    return pd.DataFrame(valores, columns=nombres), idx


def explicar(df: pd.DataFrame, pipeline=None, muestra: int = 2000,
             carpeta: Path = cfg.RESULTADOS / "explicabilidad", carpeta_graficos: Path = cfg.GRAFICOS) -> dict:
    """Proceso `explicar`: importancia global + ficha de un cliente de ejemplo."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    pipeline = pipeline or cargar_modelo()
    _, X_test, _, _, _, _ = dividir_datos(df)

    shap_df, idx = calcular_shap(pipeline, X_test, muestra)
    importancia = shap_df.abs().mean().sort_values(ascending=False)

    id_ejemplo = df.loc[idx[0], "id_cliente"]
    proba = float(pipeline.predict_proba(X_test.loc[[idx[0]]])[0, 1])
    razones = shap_df.iloc[0].abs().sort_values(ascending=False).head(3)
    ficha = {
        "id_cliente": str(id_ejemplo),
        "probabilidad_abandono": proba,
        "factores": [{"variable": v, "efecto": "aumenta" if shap_df.iloc[0][v] > 0 else "reduce",
                      "impacto": float(i)} for v, i in razones.items()],
    }
    salida = {
        "n_muestra": int(len(shap_df)),
        "importancia_global": [{"variable": v, "impacto_medio_abs": float(i)}
                               for v, i in importancia.head(10).items()],
        "ficha_ejemplo": ficha,
    }
    (carpeta / "explicabilidad_shap.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    _grafico(importancia, carpeta_graficos)
    print("Top 5 variables que más explican el riesgo:")
    print(importancia.head(5).round(3).to_string())
    return salida


def _grafico(importancia: pd.Series, carpeta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    importancia.head(10).sort_values().plot(kind="barh", color="#F4A300")
    plt.title("Importancia de variables (SHAP)")
    plt.xlabel("Impacto medio |SHAP| sobre la predicción")
    plt.tight_layout()
    plt.savefig(Path(carpeta) / "importancia_shap.png", dpi=130)
    plt.close()
