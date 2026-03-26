"""
analysis.py
Full A/B testing pipeline:
  1. Load & validate data
  2. Exploratory Data Analysis (EDA)
  3. Statistical significance testing (z-test + chi-square)
  4. Effect size (Cohen's h) & confidence intervals
  5. Visualisations  →  outputs/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/ab_test_data.csv"
PLOTS_DIR   = "outputs/plots"
ALPHA       = 0.05          # significance level
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {"control": "#5B8DB8", "treatment": "#E07B54"}
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── 1. Load & Validate ───────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"user_id", "group", "converted", "session_duration",
                "pages_viewed", "device"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    assert df["user_id"].nunique() == len(df), "Duplicate user IDs found!"
    assert set(df["group"].unique()) == {"control", "treatment"}, \
        "Expected exactly two groups: control & treatment"
    print("✅  Data loaded successfully")
    print(f"    Shape  : {df.shape}")
    print(f"    Groups : {df['group'].value_counts().to_dict()}\n")
    return df


# ── 2. EDA ───────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> dict:
    summary = (
        df.groupby("group")
        .agg(
            users=("user_id", "count"),
            conversions=("converted", "sum"),
            conversion_rate=("converted", "mean"),
            avg_session_sec=("session_duration", "mean"),
            avg_pages=("pages_viewed", "mean"),
        )
        .round(4)
    )
    print("── EDA Summary ──────────────────────────────")
    print(summary.to_string())
    print()
    return summary.to_dict()


# ── 3. Statistical Tests ─────────────────────────────────────────────────────
def run_stats(df: pd.DataFrame) -> dict:
    groups = {g: d for g, d in df.groupby("group")}
    ctrl, trt = groups["control"], groups["treatment"]

    n_ctrl, n_trt = len(ctrl), len(trt)
    conv_ctrl = ctrl["converted"].sum()
    conv_trt  = trt["converted"].sum()
    rate_ctrl = conv_ctrl / n_ctrl
    rate_trt  = conv_trt  / n_trt

    # Two-proportion z-test
    z_stat, p_val = proportions_ztest(
        [conv_trt, conv_ctrl], [n_trt, n_ctrl], alternative="two-sided"
    )

    # 95 % confidence intervals
    ci_ctrl = proportion_confint(conv_ctrl, n_ctrl, alpha=ALPHA, method="wilson")
    ci_trt  = proportion_confint(conv_trt,  n_trt,  alpha=ALPHA, method="wilson")

    # Cohen's h (effect size for proportions)
    cohens_h = 2 * (np.arcsin(np.sqrt(rate_trt)) - np.arcsin(np.sqrt(rate_ctrl)))

    # Chi-square test (sanity check)
    contingency = pd.crosstab(df["group"], df["converted"])
    chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)

    # Relative uplift
    uplift = (rate_trt - rate_ctrl) / rate_ctrl * 100

    results = dict(
        n_control=n_ctrl, n_treatment=n_trt,
        conv_control=conv_ctrl, conv_treatment=conv_trt,
        rate_control=rate_ctrl, rate_treatment=rate_trt,
        ci_control=ci_ctrl, ci_treatment=ci_trt,
        z_stat=z_stat, p_value=p_val,
        chi2=chi2, p_chi2=p_chi2,
        cohens_h=cohens_h, uplift_pct=uplift,
        significant=(p_val < ALPHA),
    )

    print("── Statistical Results ──────────────────────")
    print(f"  Control   rate : {rate_ctrl:.2%}  95% CI [{ci_ctrl[0]:.2%}, {ci_ctrl[1]:.2%}]")
    print(f"  Treatment rate : {rate_trt:.2%}  95% CI [{ci_trt[0]:.2%}, {ci_trt[1]:.2%}]")
    print(f"  Relative uplift: {uplift:+.2f}%")
    print(f"  Z-statistic    : {z_stat:.4f}")
    print(f"  p-value        : {p_val:.4f}")
    print(f"  Chi-square p   : {p_chi2:.4f}")
    print(f"  Cohen's h      : {cohens_h:.4f}")
    sig = "✅ SIGNIFICANT" if p_val < ALPHA else "❌ NOT significant"
    print(f"  Result (α={ALPHA}): {sig}\n")
    return results


# ── 4. Visualisations ────────────────────────────────────────────────────────
def plot_conversion_rates(results: dict):
    groups  = ["Control", "Treatment"]
    rates   = [results["rate_control"], results["rate_treatment"]]
    ci_low  = [results["ci_control"][0],  results["ci_treatment"][0]]
    ci_high = [results["ci_control"][1],  results["ci_treatment"][1]]
    errors  = [[r - l for r, l in zip(rates, ci_low)],
               [h - r for r, h in zip(rates, ci_high)]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(groups, rates, color=list(PALETTE.values()),
                  width=0.45, zorder=3, edgecolor="white", linewidth=1.5)
    ax.errorbar(groups, rates, yerr=errors, fmt="none",
                color="#333", capsize=8, linewidth=2, capthick=2, zorder=4)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.003,
                f"{rate:.2%}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Conversion Rate")
    ax.set_title("Conversion Rate by Group\n(with 95% Wilson Confidence Intervals)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0, max(rates) * 1.3)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_conversion_rates.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/01_conversion_rates.png")


def plot_session_and_pages(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, label in zip(
        axes,
        ["session_duration", "pages_viewed"],
        ["Session Duration (seconds)", "Pages Viewed"],
    ):
        for group, color in PALETTE.items():
            data = df[df["group"] == group][col]
            ax.hist(data, bins=30, alpha=0.55, color=color,
                    label=group.capitalize(), edgecolor="white")
            ax.axvline(data.mean(), color=color, linewidth=2, linestyle="--")

        ax.set_xlabel(label)
        ax.set_ylabel("Users")
        ax.set_title(f"Distribution of {label}", fontweight="bold")
        ax.legend()

    plt.suptitle("Behavioural Metrics by Group", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_behavioral_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/02_behavioral_metrics.png")


def plot_device_breakdown(df: pd.DataFrame):
    device_conv = (
        df.groupby(["device", "group"])["converted"]
        .mean()
        .reset_index()
        .rename(columns={"converted": "conversion_rate"})
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    devices = ["desktop", "mobile", "tablet"]
    x = np.arange(len(devices))
    w = 0.35

    for i, (group, color) in enumerate(PALETTE.items()):
        rates = [device_conv.query("group==@group and device==@d")
                 ["conversion_rate"].values[0] for d in devices]
        bars = ax.bar(x + i * w, rates, w, label=group.capitalize(),
                      color=color, edgecolor="white", linewidth=1.2)
        for bar, r in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.002,
                    f"{r:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([d.capitalize() for d in devices])
    ax.set_ylabel("Conversion Rate")
    ax.set_title("Conversion Rate by Device & Group", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_device_breakdown.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/03_device_breakdown.png")


def plot_summary_dashboard(df: pd.DataFrame, results: dict):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("A/B Test Results Dashboard", fontsize=16,
                 fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── KPI cards (text axes) ──
    kpis = [
        ("Uplift",     f"{results['uplift_pct']:+.2f}%",  "#2ecc71" if results["uplift_pct"] > 0 else "#e74c3c"),
        ("p-value",    f"{results['p_value']:.4f}",       "#2ecc71" if results["significant"] else "#e74c3c"),
        ("Cohen's h",  f"{results['cohens_h']:.4f}",      "#5B8DB8"),
    ]
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color + "22")
        ax.text(0.5, 0.6, value, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.2, label, ha="center", va="center",
                fontsize=11, color="#555", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

    # ── Conversion rate bar ──
    ax2 = fig.add_subplot(gs[1, :2])
    groups  = ["Control", "Treatment"]
    rates   = [results["rate_control"], results["rate_treatment"]]
    bars    = ax2.barh(groups, rates, color=list(PALETTE.values()),
                       height=0.4, edgecolor="white")
    for bar, r in zip(bars, rates):
        ax2.text(r + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{r:.2%}", va="center", fontweight="bold")
    ax2.set_xlabel("Conversion Rate")
    ax2.set_title("Conversion Rates", fontweight="bold")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # ── Significance verdict ──
    ax3 = fig.add_subplot(gs[1, 2])
    verdict = "SIGNIFICANT ✓" if results["significant"] else "NOT SIGNIFICANT ✗"
    color   = "#2ecc71" if results["significant"] else "#e74c3c"
    ax3.text(0.5, 0.55, verdict, ha="center", va="center",
             fontsize=13, fontweight="bold", color=color,
             transform=ax3.transAxes)
    ax3.text(0.5, 0.3, f"α = {ALPHA}", ha="center", va="center",
             fontsize=11, color="#777", transform=ax3.transAxes)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_facecolor(color + "15")
    for spine in ax3.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)
    ax3.set_title("Verdict", fontweight="bold")

    plt.savefig(f"{PLOTS_DIR}/04_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/04_dashboard.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  A/B TESTING PIPELINE")
    print("="*50 + "\n")

    df      = load_data(DATA_PATH)
    eda     = run_eda(df)
    results = run_stats(df)

    print("── Generating Plots ─────────────────────────")
    plot_conversion_rates(results)
    plot_session_and_pages(df)
    plot_device_breakdown(df)
    plot_summary_dashboard(df, results)

    print("\n✅  Pipeline complete. Plots saved to outputs/plots/\n")
