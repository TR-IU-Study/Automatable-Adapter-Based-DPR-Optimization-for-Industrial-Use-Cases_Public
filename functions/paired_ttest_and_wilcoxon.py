# functions/paired_ttest_and_wilcoxon.py

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def paired_ttest_and_wilcoxon(metric_base: dict, metric_adapted: dict):
    """Einseitig: H1 mean(new - base) > 0 bzw. adapted > base"""
    qids = sorted(set(metric_base) & set(metric_adapted))

    x = np.array([metric_base[q] for q in qids], dtype=float)      # Backbone / Baseline
    y = np.array([metric_adapted[q] for q in qids], dtype=float)   # Adapter / neues Modell

    diff = y - x

    # Einseitiger gepaarter t-Test
    t_stat, t_p = ttest_rel(y, x, alternative="greater")

    # Einseitiger Wilcoxon Signed-Rank Test
    # Prüft, ob y systematisch größer ist als x.
    try:
        w_stat, w_p = wilcoxon(y, x, alternative="greater")
    except ValueError:
        # Falls alle Differenzen 0 sind, kann der Wilcoxon-Test nicht sinnvoll berechnet werden.
        w_stat, w_p = np.nan, np.nan

    return {
        "n": len(qids),
        "mean_base": float(x.mean()),
        "mean_new": float(y.mean()),
        "mean_diff": float(diff.mean()),

        # Gepaarter t-Test
        "t_statistic": float(t_stat),
        "t_p": float(t_p),
        "t_significant_0_05": bool(t_p < 0.05),

        # Wilcoxon Signed-Rank Test
        "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_p": float(w_p) if not np.isnan(w_p) else np.nan,
        "wilcoxon_significant_0_05": bool(w_p < 0.05) if not np.isnan(w_p) else False,
    }