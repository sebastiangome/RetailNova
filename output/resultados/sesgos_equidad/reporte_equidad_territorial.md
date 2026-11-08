# LOGISMART — H5-T3 Auditoría de equidad territorial

**Alcance:** modelo XGBoost oficial; test sintético de agosto-septiembre 2026. Auditoría descriptiva; no se reentrenó.

## Resultados globales
- Operaciones: 149,935; retrasos: 21,795.
- Recall: 30.95 %; precision: 34.56 %; FPR: 9.97 %.
- FN: 15,049; FP: 12,776.
- Brecha descriptiva entre comunas (recall): 30.09 puntos porcentuales.
- Spearman entre representación en test y recall: 0.70 (asociación no causal).

## Comparación regional
| Región | Operaciones | Recall | Precision | FPR | Falsos negativos |
|---|---:|---:|---:|---:|---:|
| Región Metropolitana | 126,092 | 31.57 % | 34.44 % | 10.38 % | 12,711 |
| Región de Valparaíso | 23,843 | 27.37 % | 35.32 % | 7.82 % | 2,338 |

## Diferencias y riesgos

Las tasas de detección y falsas alarmas varían por comuna. Esto amerita seguimiento y análisis de las condiciones logísticas; no demuestra discriminación.
Las regiones agrupan operaciones con distintas tasas base; sus resultados tampoco determinan causalidad.
Las banderas de revisión de la tabla CSV son descriptivas y no equivalen a una prueba estadística ajustada por comparaciones múltiples.

## Rebalanceo

**No se aplicó rebalanceo previamente**, por lo que no es posible evaluar su efectividad. Ningún método se considera validado por este informe.

## Acciones propuestas (pendientes de implementar y validar)

- Revisar cobertura histórica y calidad de etiquetas por comuna y por mes (especialmente minoritarias)
- Examinar diferencias por distancia, horarios, vehículo y prevalencia de retrasos antes de atribuir la brecha a representación
- Ensayar ponderación o rebalanceo solo en train y medir recall, precision y FPR por comuna en validación temporal
- Monitorear tasas y errores por comuna con intervalos e investigar alertas incorrectas y retrasos no detectados
- Usar las alertas como apoyo logístico, nunca como mecanismo automático de sanción de conductores

## Limitaciones

- Datos sintéticos; no constituyen evidencia de equidad o discriminación en operaciones reales
- El dataset de test ya fue examinado en H4-T1/H4-T4/H5-T1; esta auditoría es descriptiva, no una validación nueva
- Se dispone de la representación de test, no de una auditoría equivalente de la distribución territorial del train
- Intervalos Wilson aproximados suponen independencia; pueden subestimar incertidumbre por agrupación de rutas/conductores
- Comparación de intervalos con tasa global no es prueba formal de significancia; existe comparación múltiple
- Diferencias pueden confundirse por distinta prevalencia, recorridos, tipos de entrega y horario
- No se modificó el modelo oficial ni se escogieron umbrales usando test
