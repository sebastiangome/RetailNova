# LOGISMART — Evidencia de cierre local H6-T1

Ejecución: `20261104T170000050763Z_a51e87` (4 de noviembre de 2026, hora registrada en UTC).

- **Entrada:** histórico sintético de 1.050.000 operaciones; preparación y modelado ejecutados localmente.
- **Inferencia:** XGBoost oficial, umbral 0,15; 161.875 alertas de 1.050.000 filas (**15,4167 %**).
- **Duración sin respaldo:** 19,8873 segundos; preparación 13,4479 segundos e inferencia 6,2576 segundos.
- **Persistencia:** CSV de predicciones, JSON de bitácora y ZIP de respaldo local, según registro de ejecución.
- **Comprobación externa del CSV:** 1.050.000 filas, sin nulos, orden consecutivo, todas las alertas consistentes con el umbral, SHA-256 coincidente con la bitácora.
- **Respaldo ZIP:** `verificado: true` según bitácora; ZIP no adjuntado a esta revisión, por lo que no se comprobó independientemente.
- **Limitaciones:** datos sintéticos y retrospectivos; no producción, no planificador, no validación de capacidad de negocio, no reducción de retrasos acreditada. PostgreSQL y S3 no se requieren para la EV3.

Este documento registra una **demostración local funcional** dentro del alcance definido para la EV3; no constituye una implantación productiva. Las predicciones CSV completas y los respaldos ZIP quedan en directorios locales ignorados por Git para evitar versionar archivos masivos.
