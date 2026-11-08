# LOGISMART — H5-T4: validación conceptual de negocio y alertas

**Alcance:** simulación académica basada en datos sintéticos de validación (julio de 2026); no equivale a aprobación de la Gerencia de Operaciones.
**Modelo oficial:** XGBoost, umbral seleccionado previamente por F1 en julio; este proceso no cambia el modelo ni el umbral.

## Escenarios de decisión

| Escenario | Umbral | Recall | Precision | Alertas en julio | Alertas/día (promedio) | Falsos positivos | Falsos negativos |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mayor cobertura | 0.10 | 71.2% | 21.5% | 34.070 | 1099.0 | 26.737 | 2.960 |
| Oficial: equilibrio de F1 | 0.15 | 52.3% | 26.2% | 20.581 | 663.9 | 15.196 | 4.908 |
| Menos alertas | 0.20 | 34.4% | 31.1% | 11.409 | 368.0 | 7.864 | 6.748 |

## Niveles propuestos para revisión humana

- **Bajo (p < 0.10):** 40.696 operaciones de julio; sin alerta prioritaria.
- **Observación (0.10 ≤ p < 0.15):** 13.489; revisar según contexto.
- **Prioritario (p ≥ 0.15):** 20.581; revisión preventiva por personal de operaciones.

Las categorías son prioridades relativas de revisión; **no** representan probabilidades altas de retraso en términos absolutos.

## Capacidad operativa y objetivo de negocio

No se proporcionó una capacidad diaria de revisión; **no puede certificarse viabilidad operativa ni cumplimiento de SLA**.

La meta de **reducir retrasos en 20 %** es un objetivo del negocio, **no evaluable** mediante estas métricas. Una alerta no demuestra prevención; se requieren acciones registradas, seguimiento de resultados y comparación operacional posterior.

## Riesgos y recomendaciones

- El umbral 0.10 detecta más retrasos, pero incrementa la cantidad de falsos positivos y revisiones.
- El umbral 0.20 reduce la carga, pero deja más retrasos sin alerta. No se propone reemplazar el umbral oficial.
- Estimar el personal y el costo real de atender alertas, y validar el volumen diario y picos horarios antes de uso operacional.
- Usar un humano en el circuito de decisión; prohibir sanciones automáticas a conductores basadas en las alertas.
- Considerar las diferencias territoriales señaladas en H5-T3 al supervisar errores y tasas de alerta.
- Antes de una validación productiva, definir con operaciones quién revisa alertas, los tiempos de reacción y qué acciones preventivas se permiten.

## Limitaciones

- Escenarios evaluados en julio de 2026; cantidades y precisión pueden variar en meses posteriores.
- Son 31 días calendario, no jornadas hábiles; las alertas/día representan un promedio, no carga pico.
- El test no se empleó para fijar nuevos umbrales ni seleccionar una política distinta.
- No hay estimaciones verificadas de costos, dotación, capacidad operacional ni validación formal con gerencia.
- Los datos son sintéticos y no acreditan una mejora del negocio ni reducción efectiva de retrasos.

**Conclusión:** conservar provisionalmente 0.15 como umbral oficial, clasificar el riesgo para apoyar priorización humana y solicitar validación de capacidad/acciones al área operacional antes de desplegar.
