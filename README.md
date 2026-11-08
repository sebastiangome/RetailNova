# RetailNova — Predicción de abandono de clientes

SCY1102 Gestión de Proyectos de Datos · Sección 001D · DuocUC
Equipo: Luis Hurtubia, Rafael Pacheco, Sebastián Curi, Sebastián Gómez.

Caso 1: RetailNova pierde ~24 % de sus clientes frecuentes. Este repositorio genera datos sintéticos, los limpia, entrena un modelo de clasificación, audita su equidad regional, lo explica (SHAP) y produce un **dashboard HTML autocontenido** desde código.

> Datos 100 % sintéticos. Las probabilidades no son riesgo real calibrado.

## Uso rápido

```bash
pip install -r requirements.txt
python main.py --proceso completo      # ~30 s
# abrir dashboard/index.html en el navegador (no necesita servidor)
python -m pytest                        # pruebas
```

## Procesos (`python main.py --proceso <nombre>`)

| Proceso | Qué hace | Salida |
|---|---|---|
| `generar` | Base sintética de 3 años (185.000) y muestra de 12 meses; guarda el estado del RNG | `data/raw/` |
| `preparar` | Coherencia temporal, imputación, validaciones, variables RFM | `data/processed/retailnova_clientes_12m_prepared.csv` |
| `entrenar` | CV 5 folds (solo train), elección de modelo, umbral OOF, ajuste final | `models/`, `output/resultados/modelado/` |
| `evaluar` | Evaluación única en test + auditoría regional RNF6 | `output/resultados/` |
| `explicar` | SHAP global | `explicabilidad_shap.json`, gráfico |
| `puntuar` | Probabilidad y nivel de riesgo (Bajo/Medio/Alto) por cliente | `retailnova_clientes_12m_scored.csv` |
| `visualizar` | Genera el dashboard con los datos embebidos | `dashboard/index.html` |
| `seguimiento` | Valida y resume `seguimiento/tareas.json` | consola |
| `completo` | Todo lo anterior en orden | — |

Opciones: `--registros`, `--semilla`, `--muestra-shap`.

## Decisiones metodológicas

- **Fuga de datos:** `recencia_dias` define el abandono (>180 días), por lo que no entra como predictora (hay una guarda en el código y una prueba).
- **Validación cruzada** de 5 folds estratificados solo sobre entrenamiento; si la diferencia de AUC entre modelos es < 0,01 se prefiere Regresión Logística (más simple y explicable).
- **Umbral** elegido por predicciones out-of-fold: el más alto con recall ≥ 0,75. El test se evalúa una sola vez.
- **RNF6:** brecha de F1 por región vs. F1 global < 10 %.
- **Riesgo:** percentiles 66 y 90 de la probabilidad → Bajo / Medio / Alto.
- **Reproducibilidad:** semilla 42 y estado del generador aleatorio serializado entre `generar` y `preparar`.

## Resultados (semilla 42)

| Métrica | Valor |
|---|---|
| Clientes (12 meses) | 51.346 · abandono 24,0 % |
| AUC CV: Random Forest / Reg. Logística | 0,909 / 0,908 → se elige Reg. Logística |
| Test: ROC-AUC · PR-AUC | 0,907 · 0,745 |
| Test: recall · precisión · F1 (umbral 0,64) | 0,740 · 0,646 · 0,690 |
| Brecha regional (Biobío / RM / Valparaíso) | 2,6 % / 0,1 % / 1,4 % — cumple |
| Riesgo Alto / Medio / Bajo | 5.135 / 12.323 / 33.888 |
| Variables SHAP principales | antigüedad, frecuencia de compra, cantidad de compras, reclamos |

## Estructura

```
main.py            CLI de procesos
src/               módulos (config, generador, pipeline, modelado, equidad, evaluación, SHAP, scoring, visualización)
src/plantilla_dashboard.html   plantilla del dashboard
dashboard/index.html           dashboard generado (sin librerías externas)
seguimiento/tareas.json        tareas y desviaciones del proyecto
tests/             pruebas pytest
.github/workflows/ pruebas y publicación en GitHub Pages
```

## Limitaciones

- Datos sintéticos: el AUC alto refleja las reglas del generador, no el comportamiento real.
- Probabilidades no calibradas; el nivel de riesgo es relativo (percentiles).
- El dashboard es una vista estática: para datos nuevos hay que ejecutar `--proceso completo` y volver a generarlo.
