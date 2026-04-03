# functions/paired_ttest_one_sided_greater.py

import numpy as np
from scipy.stats import ttest_rel

def paired_ttest_one_sided_greater(metric_base: dict, metric_adapted: dict):
    """Einseitig: H1 mean(new - base) > 0"""
    qids = sorted(set(metric_base) & set(metric_adapted))
    x = np.array([metric_base[q] for q in qids], dtype=float)
    y = np.array([metric_adapted[q]  for q in qids], dtype=float)

    t, p = ttest_rel(y, x, alternative="greater")  # einseitiger t-Test
    return {
        "n": len(qids),
        "mean_base": float(x.mean()),
        "mean_new": float(y.mean()),
        "mean_diff": float((y - x).mean()),
        "t": float(t),
        "p": float(p),
    }