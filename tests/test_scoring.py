import numpy as np
import pandas as pd

from src.scoring import categorizar_riesgo


def test_percentiles_66_90():
    p = pd.Series(np.random.default_rng(0).random(10000))
    r = categorizar_riesgo(p)
    frac = r.value_counts(normalize=True)
    assert abs(frac["Bajo"] - 0.66) < 0.02
    assert abs(frac["Medio"] - 0.24) < 0.02
    assert abs(frac["Alto"] - 0.10) < 0.02


def test_mayor_prob_mayor_riesgo():
    p = pd.Series(np.linspace(0, 1, 1000))
    r = categorizar_riesgo(p)
    assert r.iloc[0] == "Bajo" and r.iloc[-1] == "Alto"
