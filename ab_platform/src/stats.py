"""src/stats.py — Statistical analysis for daily A/B campaign metrics."""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ── Frequentist t-test (daily metric data) ──────────────────────────────────

@dataclass
class TTestResult:
    metric: str
    ctrl_name: str
    test_name: str
    ctrl_mean: float
    test_mean: float
    ctrl_std: float
    test_std: float
    ctrl_n: int
    test_n: int
    t_stat: float
    p_value: float
    ci_diff_low: float
    ci_diff_high: float
    relative_uplift: float
    cohens_d: float
    significant: bool
    alpha: float
    direction: str      # "test_wins" | "ctrl_wins" | "no_diff"


def run_ttest(ctrl_vals: np.ndarray, test_vals: np.ndarray,
              metric: str, ctrl_name: str, test_name: str,
              alpha: float = 0.05) -> TTestResult:
    ctrl_vals = ctrl_vals[~np.isnan(ctrl_vals)]
    test_vals = test_vals[~np.isnan(test_vals)]

    t, p = stats.ttest_ind(test_vals, ctrl_vals, equal_var=False)  # Welch's

    # 95 % CI on the difference in means
    se = np.sqrt(ctrl_vals.var(ddof=1) / len(ctrl_vals) +
                 test_vals.var(ddof=1) / len(test_vals))
    df_welch = (
        (ctrl_vals.var(ddof=1) / len(ctrl_vals) + test_vals.var(ddof=1) / len(test_vals)) ** 2 /
        ((ctrl_vals.var(ddof=1) / len(ctrl_vals)) ** 2 / (len(ctrl_vals) - 1) +
         (test_vals.var(ddof=1) / len(test_vals)) ** 2 / (len(test_vals) - 1))
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df_welch)
    diff = test_vals.mean() - ctrl_vals.mean()
    ci_low  = diff - t_crit * se
    ci_high = diff + t_crit * se

    pooled_std = np.sqrt((ctrl_vals.var(ddof=1) + test_vals.var(ddof=1)) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

    rel_uplift = (diff / ctrl_vals.mean() * 100) if ctrl_vals.mean() != 0 else 0.0
    significant = p < alpha

    if significant and test_vals.mean() > ctrl_vals.mean():
        direction = "test_wins"
    elif significant and test_vals.mean() < ctrl_vals.mean():
        direction = "ctrl_wins"
    else:
        direction = "no_diff"

    return TTestResult(
        metric=metric, ctrl_name=ctrl_name, test_name=test_name,
        ctrl_mean=float(ctrl_vals.mean()), test_mean=float(test_vals.mean()),
        ctrl_std=float(ctrl_vals.std(ddof=1)), test_std=float(test_vals.std(ddof=1)),
        ctrl_n=len(ctrl_vals), test_n=len(test_vals),
        t_stat=float(t), p_value=float(p),
        ci_diff_low=float(ci_low), ci_diff_high=float(ci_high),
        relative_uplift=float(rel_uplift),
        cohens_d=float(cohens_d),
        significant=significant, alpha=alpha,
        direction=direction,
    )


def run_all_metrics(df: pd.DataFrame, ctrl_name: str, test_name: str,
                    metrics: list, alpha: float = 0.05) -> List[TTestResult]:
    from src.data_utils import get_group_series
    results = []
    for m in metrics:
        c = get_group_series(df, ctrl_name, m)
        t = get_group_series(df, test_name, m)
        if len(c) >= 3 and len(t) >= 3:
            results.append(run_ttest(c, t, m, ctrl_name, test_name, alpha))
    return results


def results_to_df(results: List[TTestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Metric": r.metric,
            "Control Mean": r.ctrl_mean,
            "Test Mean": r.test_mean,
            "Uplift %": r.relative_uplift,
            "p-value": r.p_value,
            "Cohen's d": r.cohens_d,
            "Significant": "✅" if r.significant else "❌",
            "Winner": r.test_name if r.direction == "test_wins"
                      else (r.ctrl_name if r.direction == "ctrl_wins" else "—"),
        })
    return pd.DataFrame(rows)


# ── Bayesian Beta-Binomial (purchase / add-to-cart totals) ──────────────────

@dataclass
class BayesianResult:
    ctrl_name: str
    test_name: str
    metric: str
    prob_test_better: float
    expected_loss_if_test: float
    expected_loss_if_ctrl: float
    ctrl_posterior_mean: float
    test_posterior_mean: float
    samples_ctrl: np.ndarray = field(repr=False)
    samples_test: np.ndarray = field(repr=False)
    recommendation: str
    rec_color: str


def run_bayesian_conversion(
    ctrl_successes: int, ctrl_trials: int,
    test_successes: int, test_trials: int,
    ctrl_name: str, test_name: str,
    metric: str = "Purchase",
    prior_a: float = 1.0, prior_b: float = 1.0,
    n_samples: int = 100_000,
) -> BayesianResult:
    rng = np.random.default_rng(42)
    s_ctrl = rng.beta(prior_a + ctrl_successes, prior_b + ctrl_trials - ctrl_successes, n_samples)
    s_test = rng.beta(prior_a + test_successes, prior_b + test_trials - test_successes, n_samples)

    prob = float(np.mean(s_test > s_ctrl))
    loss_test  = float(np.mean(np.maximum(s_ctrl - s_test, 0)))
    loss_ctrl  = float(np.mean(np.maximum(s_test - s_ctrl, 0)))

    ctrl_mean = (prior_a + ctrl_successes) / (prior_a + prior_b + ctrl_trials)
    test_mean = (prior_a + test_successes) / (prior_a + prior_b + test_trials)

    if prob >= 0.95:
        rec = f"Strong evidence for Test Campaign ({prob:.1%} probability of superiority)"
        col = "green"
    elif prob >= 0.80:
        rec = f"Moderate evidence for Test Campaign ({prob:.1%}) — collect more data"
        col = "orange"
    elif prob <= 0.05:
        rec = f"Strong evidence for Control Campaign ({1-prob:.1%} probability of superiority)"
        col = "red"
    elif prob <= 0.20:
        rec = f"Moderate evidence for Control ({1-prob:.1%}) — collect more data"
        col = "orange"
    else:
        rec = f"Inconclusive result (Test: {prob:.1%} vs Control: {1-prob:.1%})"
        col = "orange"

    return BayesianResult(
        ctrl_name=ctrl_name, test_name=test_name, metric=metric,
        prob_test_better=prob,
        expected_loss_if_test=loss_test,
        expected_loss_if_ctrl=loss_ctrl,
        ctrl_posterior_mean=ctrl_mean,
        test_posterior_mean=test_mean,
        samples_ctrl=s_ctrl,
        samples_test=s_test,
        recommendation=rec, rec_color=col,
    )


# ── Sequential monitoring (daily cumulative z-scores) ───────────────────────

def run_sequential_monitoring(
    df: pd.DataFrame, ctrl_name: str, test_name: str,
    metric: str, alpha: float = 0.05,
) -> Dict[str, Any]:
    """Track cumulative means over time with O'Brien-Fleming boundaries."""
    ctrl_df = df[df["Campaign"] == ctrl_name].sort_values("Date").reset_index(drop=True)
    test_df = df[df["Campaign"] == test_name].sort_values("Date").reset_index(drop=True)

    n = min(len(ctrl_df), len(test_df))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    dates, z_scores, obf_upper, obf_lower = [], [], [], []
    ctrl_cumulative, test_cumulative = [], []

    for i in range(2, n + 1):
        c_vals = ctrl_df[metric].iloc[:i].dropna()
        t_vals = test_df[metric].iloc[:i].dropna()
        if len(c_vals) < 2 or len(t_vals) < 2:
            continue

        t_info = i / n
        boundary = z_crit / np.sqrt(t_info)

        diff = t_vals.mean() - c_vals.mean()
        se = np.sqrt(c_vals.var(ddof=1) / len(c_vals) + t_vals.var(ddof=1) / len(t_vals))
        z = diff / se if se > 0 else 0.0

        dates.append(ctrl_df["Date"].iloc[i - 1])
        z_scores.append(float(z))
        obf_upper.append(float(boundary))
        obf_lower.append(float(-boundary))
        ctrl_cumulative.append(float(c_vals.mean()))
        test_cumulative.append(float(t_vals.mean()))

    early_stop = None
    for i, (z, b) in enumerate(zip(z_scores, obf_upper)):
        if abs(z) >= b:
            early_stop = {"index": i, "date": dates[i], "z": z,
                          "direction": "test" if z > 0 else "control"}
            break

    return {
        "dates": dates,
        "z_scores": z_scores,
        "obf_upper": obf_upper,
        "obf_lower": obf_lower,
        "ctrl_cumulative": ctrl_cumulative,
        "test_cumulative": test_cumulative,
        "early_stop": early_stop,
        "metric": metric,
    }


# ── Sample size / power ──────────────────────────────────────────────────────

def required_sample_size(baseline_mean: float, baseline_std: float,
                         mde_pct: float, alpha: float = 0.05,
                         power: float = 0.80) -> int:
    delta = baseline_mean * mde_pct / 100
    if delta == 0 or baseline_std == 0:
        return -1
    n = 2 * ((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) ** 2
             * baseline_std ** 2 / delta ** 2)
    return int(np.ceil(n))
