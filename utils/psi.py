import numpy as np
import pandas as pd

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index.
    - expected: 기준 분포(예: 과거/학습 시점)
    - actual: 현재 분포
    """
    e = expected.dropna().astype(float).values
    a = actual.dropna().astype(float).values
    if len(e) == 0 or len(a) == 0:
        return np.nan

    quantiles = np.quantile(e, np.linspace(0, 1, bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    def _hist(x):
        h, _ = np.histogram(x, bins=quantiles)
        h = h / max(h.sum(), 1)
        return np.clip(h, 1e-6, 1)

    e_hist = _hist(e)
    a_hist = _hist(a)
    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))
