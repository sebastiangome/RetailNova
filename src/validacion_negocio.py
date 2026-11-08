"""H5-T4 — Validación conceptual de negocio y niveles de alerta de LOGISMART.

Lee artefactos de VALIDACIÓN preexistentes, sin acceder a etiquetas del test,
entrenar modelos ni cambiar umbrales. Los criterios operativos son simulados,
no aprobaciones reales de la Gerencia de Operaciones.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

DIAS_JULIO_2026 = 31
CANDIDATOS = (0.10, 0.20)


def validar_auditoria(tabla: pd.DataFrame, filas_validacion: int) -> pd.DataFrame:
    """Comprueba que cada fila describa una matriz de confusión coherente."""
    requeridas = {"umbral", "precision", "recall", "f1", "tasa_alertas", "tp", "fp", "fn", "tn"}
    faltantes = requeridas - set(tabla.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en auditoría: {sorted(faltantes)}")
    if filas_validacion <= 0 or tabla.empty:
        raise ValueError("La auditoría de validación está vacía o no tiene registros")
    t = tabla.copy()
    for columna in sorted(requeridas):
        t[columna] = pd.to_numeric(t[columna], errors="raise")
    if not np.isfinite(t[list(requeridas)].to_numpy(dtype=float)).all():
        raise ValueError("La auditoría contiene valores no finitos")
    if not t["umbral"].between(0, 1, inclusive="neither").all():
        raise ValueError("Umbral fuera del intervalo (0,1)")
    if (t["umbral"].round(7).duplicated()).any():
        raise ValueError("Existen umbrales duplicados")
    if not t[["precision", "recall", "f1", "tasa_alertas"]].ge(0).all().all() or not t[["precision", "recall", "f1", "tasa_alertas"]].le(1).all().all():
        raise ValueError("Hay métricas fuera del intervalo [0,1]")
    conteos = t[["tp", "fp", "fn", "tn"]].to_numpy(dtype=float)
    if (conteos < 0).any() or not np.all(np.equal(conteos, np.floor(conteos))):
        raise ValueError("Conteos negativos o no enteros")
    t[["tp", "fp", "fn", "tn"]] = t[["tp", "fp", "fn", "tn"]].astype(int)
    if not (t[["tp", "fp", "fn", "tn"]].sum(axis=1) == filas_validacion).all():
        raise ValueError("Las matrices de confusión no coinciden con las filas de validación")
    if (t["tp"] + t["fn"]).nunique() != 1:
        raise ValueError("El número de retrasos reales varía entre umbrales")
    alertas = t["tp"] + t["fp"]
    precision = np.divide(t["tp"], alertas, out=np.zeros(len(t),dtype=float), where=alertas.ne(0))
    recall = np.divide(t["tp"], t["tp"] + t["fn"], out=np.zeros(len(t),dtype=float), where=(t["tp"] + t["fn"]).ne(0))
    if not np.allclose(t["precision"],precision,atol=1e-7) or not np.allclose(t["recall"],recall,atol=1e-7) or not np.allclose(t["tasa_alertas"],alertas/filas_validacion,atol=1e-7):
        raise ValueError("Métricas incoherentes con los conteos registrados")
    return t.sort_values("umbral").reset_index(drop=True)


def fila_umbral(tabla: pd.DataFrame, valor: float) -> pd.Series:
    coincidencias = tabla.loc[np.isclose(tabla["umbral"].to_numpy(dtype=float),valor,atol=1e-7,rtol=0)]
    if len(coincidencias) != 1:
        raise ValueError(f"No existe una fila única para el umbral {valor:.2f}")
    return coincidencias.iloc[0]


def comparar_escenarios(tabla: pd.DataFrame, umbral_oficial: float, filas_val: int,
                        capacidad_alertas_dia: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Compara escenarios en julio sin optimizar umbrales ni cambiar el modelo."""
    t = validar_auditoria(tabla, filas_val)
    if not 0 < umbral_oficial < 1:
        raise ValueError("El umbral oficial debe estar entre cero y uno")
    if capacidad_alertas_dia is not None and capacidad_alertas_dia <= 0:
        raise ValueError("La capacidad diaria debe ser un entero positivo")
    if not (CANDIDATOS[0] < umbral_oficial < CANDIDATOS[1]):
        raise ValueError("La clasificación propuesta requiere un umbral oficial entre 0.10 y 0.20")
    escenarios = [(CANDIDATOS[0],"Mayor cobertura"), (umbral_oficial,"Oficial: equilibrio de F1"), (CANDIDATOS[1],"Menos alertas")]
    filas=[]
    for u, nombre in escenarios:
        dato = fila_umbral(t,u)
        total_alertas = int(dato["tp"]+dato["fp"])
        por_dia = total_alertas / DIAS_JULIO_2026
        filas.append({
            "escenario":nombre,"umbral":float(u),"precision":float(dato["precision"]),
            "recall":float(dato["recall"]),"f1":float(dato["f1"]),
            "tasa_alertas":float(dato["tasa_alertas"]),
            "alertas_validacion":total_alertas,"alertas_diarias_promedio":round(por_dia,2),
            "verdaderos_positivos":int(dato["tp"]),"falsos_positivos":int(dato["fp"]),
            "falsos_negativos":int(dato["fn"]),
            "cabe_en_capacidad_simulada":None if capacidad_alertas_dia is None else bool(por_dia<=capacidad_alertas_dia),
        })
    r = pd.DataFrame(filas)
    u_bajo = int(r.iloc[0]["alertas_validacion"])
    u_oficial = int(r.iloc[1]["alertas_validacion"])
    if not filas_val >= u_bajo >= u_oficial:
        raise ValueError("Las alertas no son monótonas entre umbrales")
    niveles = {
        "bajo": {"regla": "p < 0.10", "operaciones_validacion":filas_val-u_bajo,
                 "accion": "No priorizar automáticamente; seguimiento normal"},
        "observacion": {"regla": "0.10 <= p < umbral_oficial", "operaciones_validacion":u_bajo-u_oficial,
                 "accion": "Revisión contextual si hay disponibilidad"},
        "prioritario": {"regla": "p >= umbral_oficial", "operaciones_validacion":u_oficial,
                 "accion": "Revisión humana preventiva de la entrega"},
    }
    return r, niveles


def _informe_markdown(reporte: dict, escenarios: pd.DataFrame) -> str:
    def entero(x: int) -> str: return f"{int(x):,}".replace(",",".")
    lineas = [
        "# LOGISMART — H5-T4: validación conceptual de negocio y alertas", "",
        "**Alcance:** simulación académica basada en datos sintéticos de validación (julio de 2026); no equivale a aprobación de la Gerencia de Operaciones.",
        "**Modelo oficial:** XGBoost, umbral seleccionado previamente por F1 en julio; este proceso no cambia el modelo ni el umbral.", "",
        "## Escenarios de decisión", "",
        "| Escenario | Umbral | Recall | Precision | Alertas en julio | Alertas/día (promedio) | Falsos positivos | Falsos negativos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _,row in escenarios.iterrows():
        lineas.append(f"| {row['escenario']} | {row['umbral']:.2f} | {row['recall']:.1%} | {row['precision']:.1%} | {entero(row['alertas_validacion'])} | {row['alertas_diarias_promedio']:.1f} | {entero(row['falsos_positivos'])} | {entero(row['falsos_negativos'])} |")
    niveles = reporte['niveles_riesgo']
    lineas += [
        "", "## Niveles propuestos para revisión humana", "",
        f"- **Bajo (p < 0.10):** {entero(niveles['bajo']['operaciones_validacion'])} operaciones de julio; sin alerta prioritaria.",
        f"- **Observación (0.10 ≤ p < {reporte['umbral_oficial']:.2f}):** {entero(niveles['observacion']['operaciones_validacion'])}; revisar según contexto.",
        f"- **Prioritario (p ≥ {reporte['umbral_oficial']:.2f}):** {entero(niveles['prioritario']['operaciones_validacion'])}; revisión preventiva por personal de operaciones.",
        "", "Las categorías son prioridades relativas de revisión; **no** representan probabilidades altas de retraso en términos absolutos."
    ]
    capacidad = reporte['capacidad_alertas_dia_simulada']
    lineas += ["", "## Capacidad operativa y objetivo de negocio", ""]
    if capacidad is None:
        lineas += ["No se proporcionó una capacidad diaria de revisión; **no puede certificarse viabilidad operativa ni cumplimiento de SLA**."]
    else:
        lineas += [f"Se simuló una capacidad de **{entero(capacidad)} alertas revisables por día** (supuesto académico, no capacidad verificada). Comparar promedios no asegura que ningún día supere la capacidad."]
    lineas += [
        "", "La meta de **reducir retrasos en 20 %** es un objetivo del negocio, **no evaluable** mediante estas métricas. Una alerta no demuestra prevención; se requieren acciones registradas, seguimiento de resultados y comparación operacional posterior.",
        "", "## Riesgos y recomendaciones", "",
        "- El umbral 0.10 detecta más retrasos, pero incrementa la cantidad de falsos positivos y revisiones.",
        "- El umbral 0.20 reduce la carga, pero deja más retrasos sin alerta. No se propone reemplazar el umbral oficial.",
        "- Estimar el personal y el costo real de atender alertas, y validar el volumen diario y picos horarios antes de uso operacional.",
        "- Usar un humano en el circuito de decisión; prohibir sanciones automáticas a conductores basadas en las alertas.",
        "- Considerar las diferencias territoriales señaladas en H5-T3 al supervisar errores y tasas de alerta.",
        "- Antes de una validación productiva, definir con operaciones quién revisa alertas, los tiempos de reacción y qué acciones preventivas se permiten.",
        "", "## Limitaciones", "",
        "- Escenarios evaluados en julio de 2026; cantidades y precisión pueden variar en meses posteriores.",
        "- Son 31 días calendario, no jornadas hábiles; las alertas/día representan un promedio, no carga pico.",
        "- El test no se empleó para fijar nuevos umbrales ni seleccionar una política distinta.",
        "- No hay estimaciones verificadas de costos, dotación, capacidad operacional ni validación formal con gerencia.",
        "- Los datos son sintéticos y no acreditan una mejora del negocio ni reducción efectiva de retrasos.",
        "", "**Conclusión:** conservar provisionalmente 0.15 como umbral oficial, clasificar el riesgo para apoyar priorización humana y solicitar validación de capacidad/acciones al área operacional antes de desplegar.", ""
    ]
    return "\n".join(lineas)


def validar_negocio(
    ruta_umbrales: str | Path = "output/resultados/modelado/auditoria_umbrales_validacion.json",
    ruta_modelado: str | Path = "output/resultados/modelado/resultados_modelado.json",
    ruta_equidad: str | Path = "output/resultados/sesgos_equidad/equidad_territorial.json",
    carpeta_resultados: str | Path = "output/resultados/negocio",
    carpeta_graficos: str | Path = "output/graficos",
    capacidad_alertas_dia: int | None = None,
) -> dict:
    """Genera evidencias reutilizando resultados, sin cargar el modelo ni el CSV de test."""
    ruta_umbrales, ruta_modelado, ruta_equidad = map(Path,(ruta_umbrales,ruta_modelado,ruta_equidad))
    for ruta in (ruta_umbrales,ruta_modelado,ruta_equidad):
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta {ruta}; ejecuta previamente evaluar/equidad")
    oficial = json.loads(ruta_modelado.read_text(encoding="utf-8"))
    equidad = json.loads(ruta_equidad.read_text(encoding="utf-8"))
    if oficial.get("modo_rapido") or oficial.get("modelo_elegido") != "XGBoost":
        raise ValueError("Se requiere el XGBoost oficial y entrenamiento completo")
    if equidad.get("tarea") != "H5-T3":
        raise ValueError("El informe de equidad debe corresponder a H5-T3")
    umbral = float(oficial["umbral_elegido_en_validacion"])
    n_val = int(oficial["filas_utilizadas"]["validacion"])
    t = pd.read_json(ruta_umbrales)
    escenarios, niveles = comparar_escenarios(t,umbral,n_val,capacidad_alertas_dia)
    if sum(int(x["operaciones_validacion"]) for x in niveles.values()) != n_val:
        raise ValueError("El total de niveles de riesgo no coincide con validación")
    reporte = {
        "tarea":"H5-T4", "modelo":"XGBoost", "datos":"sintéticos",
        "periodo":"validación: julio 2026", "filas_validacion":n_val,
        "umbral_oficial":umbral, "umbral_modificado":False,
        "validacion_gerencia":"No realizada; únicamente simulación conceptual fundamentada en los requerimientos",
        "capacidad_alertas_dia_simulada":capacidad_alertas_dia,
        "escenarios":escenarios.replace({np.nan:None}).to_dict(orient="records"),
        "niveles_riesgo":niveles,
        "objetivo_reduccion_retrasos_20_pct":"No demostrable; requiere intervenciones y medición de impacto real",
        "equidad_territorial":"H5-T3 identifica disparidades descriptivas; se exige seguimiento por comuna, sin cambio automático de umbrales",
        "conclusion":"Conservar XGBoost y umbral 0.15 hasta disponer de capacidad/aceptación operacional verificadas",
    }
    resultados = Path(carpeta_resultados); resultados.mkdir(parents=True,exist_ok=True)
    graficos = Path(carpeta_graficos); graficos.mkdir(parents=True,exist_ok=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5,4.8))
    p=escenarios
    ax.plot(p["umbral"],p["recall"],marker="o",label="Recall")
    ax.plot(p["umbral"],p["precision"],marker="o",label="Precision")
    ax.plot(p["umbral"],p["tasa_alertas"],marker="o",label="Proporción con alerta")
    ax.axvline(umbral,linestyle="--",alpha=0.65,label=f"Umbral oficial {umbral:.2f}")
    ax.set(ylim=(0,1),xlabel="Umbral de probabilidad",ylabel="Proporción",title="LOGISMART H5-T4: escenarios en validación (julio 2026)")
    ax.grid(alpha=0.2);ax.legend()
    fig.tight_layout()
    fig.savefig(graficos/"validacion_negocio_umbrales.png",dpi=160)
    plt.close(fig)
    (resultados/"validacion_negocio.json").write_text(json.dumps(reporte,ensure_ascii=False,indent=2),encoding="utf-8")
    escenarios.to_csv(resultados/"validacion_negocio_umbrales.csv",index=False)
    (resultados/"reporte_validacion_negocio.md").write_text(_informe_markdown(reporte,escenarios),encoding="utf-8")
    print("H5-T4: validación conceptual de negocio (julio 2026)")
    print(escenarios[["escenario","umbral","precision","recall","alertas_validacion","falsos_positivos","falsos_negativos","alertas_diarias_promedio"]].round(3).to_string(index=False))
    print("Umbral oficial preservado. No se utilizó test ni se acreditó reducción operativa de retrasos.")
    if capacidad_alertas_dia is None:
        print("Sin capacidad operacional informada: cumplimiento de ventana/atención NO VERIFICADO.")
    print(f"Evidencias: {resultados} y {graficos}")
    return reporte


if __name__ == "__main__":
    validar_negocio()
