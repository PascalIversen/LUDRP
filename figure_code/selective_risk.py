"""Workstream B — selective-risk / risk-coverage metrics that separate uncertainty
*ranking quality* from a model's baseline point-prediction accuracy (Reviewer 2, point 2).

R2: uncertainty-based filtering "partly combines two effects: better uncertainty ranking
and better baseline predictive performance." We disentangle them with the risk-coverage
framework (Geifman & El-Yaniv, 2017):

  - selective risk at coverage c = MSE over the fraction c of pairs with LOWEST uncertainty;
  - AURC = area under the risk-coverage curve (lower is better) — confounds accuracy + ranking;
  - oracle AURC = same curve when ranking by the TRUE error (best achievable);
  - random AURC = baseline MSE (ranking carries no information);
  - excess-AURC (E-AURC) = AURC - oracle AURC;
  - ranking skill = 1 - (AURC - oracleAURC)/(baselineMSE - oracleAURC) in [0,1],
        1 = oracle ordering, 0 = random ordering. This is PURE ranking quality, independent
        of the model's absolute MSE.

So baseline MSE measures accuracy; ranking skill (and Spearman(|err|,unc)) measures
uncertainty quality. A model can have low MSE yet poor ranking skill (e.g. Random Forest).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def risk_coverage_curve(uncertainty, squared_error):
    """Return (coverage, selective_risk, aurc) ranking by ascending uncertainty."""
    unc = np.asarray(uncertainty, float)
    se = np.asarray(squared_error, float)
    order = np.argsort(unc, kind="mergesort")
    se_ord = se[order]
    n = len(se_ord)
    coverage = np.arange(1, n + 1) / n
    selective_risk = np.cumsum(se_ord) / np.arange(1, n + 1)
    aurc = float(np.trapz(selective_risk, coverage))
    return coverage, selective_risk, aurc


def selective_risk_metrics(y_pred, y_true, uncertainty):
    """Disentangle accuracy (baseline MSE) from ranking quality (skill, E-AURC, Spearman)."""
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    se = (y_pred - y_true) ** 2
    abs_err = np.abs(y_pred - y_true)
    baseline_mse = float(se.mean())

    _, _, aurc = risk_coverage_curve(uncertainty, se)
    _, _, aurc_oracle = risk_coverage_curve(abs_err, se)  # oracle: rank by true error ascending
    eaurc = aurc - aurc_oracle
    denom = baseline_mse - aurc_oracle
    ranking_skill = float(1.0 - eaurc / denom) if denom > 1e-12 else np.nan

    rho = float(spearmanr(abs_err, np.asarray(uncertainty, float)).statistic)
    return {
        "baseline_mse": round(baseline_mse, 5),
        "aurc": round(aurc, 5),
        "aurc_oracle": round(aurc_oracle, 5),
        "eaurc": round(eaurc, 5),
        "ranking_skill": round(ranking_skill, 4),
        "spearman_abserr_unc": round(rho, 4),
        "n": int(len(se)),
    }
