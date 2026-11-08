"""RetailNova: CLI de generación, preparación, modelamiento, evaluación y dashboard.

Uso:  python main.py --proceso <nombre>      (sin argumentos: `completo`)
"""
import argparse

from src import config as cfg

PROCESOS = ["generar", "preparar", "entrenar", "evaluar", "explicar", "puntuar",
            "visualizar", "seguimiento", "completo"]


def ejecutar() -> None:
    ap = argparse.ArgumentParser(description="RetailNova: predicción de riesgo de abandono de clientes")
    ap.add_argument("--proceso", choices=PROCESOS, default="completo")
    ap.add_argument("--registros", type=int, default=cfg.N_HISTORICO,
                    help="Clientes de la base histórica (3 años) a simular")
    ap.add_argument("--semilla", type=int, default=cfg.SEMILLA)
    ap.add_argument("--muestra-shap", type=int, default=2000,
                    help="Clientes de prueba usados para calcular SHAP")
    args = ap.parse_args()
    p = args.proceso

    if p in ("generar", "completo"):
        from src.generador_dataset import generar
        generar(args.registros, args.semilla)
    if p in ("preparar", "completo"):
        from src.pipeline_datos import preparar_datos
        preparar_datos()

    necesita_datos = p in ("entrenar", "evaluar", "explicar", "puntuar", "completo")
    if necesita_datos:
        import pandas as pd
        if not cfg.CSV_PREPARADO.is_file():
            raise SystemExit(f"Falta {cfg.CSV_PREPARADO}; ejecuta --proceso preparar")
        df = pd.read_csv(cfg.CSV_PREPARADO)
        if p in ("entrenar", "completo"):
            from src.modelado import entrenar
            entrenar(df)
        if p in ("evaluar", "completo"):
            from src.evaluacion import evaluar
            evaluar(df)
        if p in ("explicar", "completo"):
            from src.explicabilidad import explicar
            explicar(df, muestra=args.muestra_shap)
        if p in ("puntuar", "completo"):
            from src.scoring import puntuar
            puntuar(df)

    if p in ("visualizar", "completo"):
        from src.visualizacion import generar_visualizacion
        generar_visualizacion()
    if p == "seguimiento":
        from src.seguimiento import cargar_seguimiento
        r = cargar_seguimiento()["resumen"]
        print(f"{r['completadas']}/{r['total']} tareas completadas ({r['avance_pct']} %) · "
              f"{r['en_proceso']} en proceso · {r['desviaciones']} desviaciones registradas")


if __name__ == "__main__":
    ejecutar()
