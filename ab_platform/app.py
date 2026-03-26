"""
app.py — A/B Testing Intelligence Platform
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

from src.data_utils import (
    load_and_merge, compute_derived, summary_by_group,
    get_group_series, campaign_names,
    METRIC_COLS, FUNNEL_STAGES, COLORS, SPEND_COL,
)
from src.stats import (
    run_ttest, run_all_metrics, results_to_df,
    run_bayesian_conversion, run_sequential_monitoring,
    required_sample_size,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Testing Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stMetric { background: #1e2130; border-radius: 10px; padding: 12px; }
    .stMetric label { color: #9da5b4 !important; font-size: 13px !important; }
    .block-container { padding-top: 1.5rem; }
    .verdict-box {
        padding: 16px 20px; border-radius: 10px;
        font-size: 15px; font-weight: 500; margin: 8px 0;
    }
    .verdict-green  { background: #0a2e1a; border-left: 4px solid #22c55e; color: #4ade80; }
    .verdict-red    { background: #2e0a0a; border-left: 4px solid #ef4444; color: #f87171; }
    .verdict-orange { background: #2e1f0a; border-left: 4px solid #f97316; color: #fb923c; }
    .section-title  { font-size: 22px; font-weight: 700; margin: 20px 0 10px; color: #e2e8f0; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data
def cached_load(ctrl_path, test_path):
    df = load_and_merge(ctrl_path, test_path)
    df = compute_derived(df)
    return df

def verdict_html(text: str, color: str) -> str:
    return f'<div class="verdict-box verdict-{color}">💡 {text}</div>'

def fmt_num(v, decimals=0):
    if pd.isna(v): return "—"
    if decimals == 0: return f"{v:,.0f}"
    return f"{v:,.{decimals}f}"

def fmt_pct(v):
    if pd.isna(v): return "—"
    return f"{v:.2%}"

def color_p(val):
    if isinstance(val, str): return ""
    return "color: #22c55e" if val < 0.05 else "color: #f97316"

CTRL_COLOR = "#4C72B0"
TEST_COLOR = "#DD8452"


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/test-tube.png", width=60)
    st.title("A/B Platform")
    st.markdown("---")

    st.subheader("📂 Data Source")
    data_mode = st.radio("", ["Use Kaggle Dataset", "Upload Your Own CSVs"], label_visibility="collapsed")

    if data_mode == "Use Kaggle Dataset":
        ctrl_path = "data/control_group.csv"
        test_path = "data/test_group.csv"
        data_ready = os.path.exists(ctrl_path) and os.path.exists(test_path)
        if not data_ready:
            st.error("Place `control_group.csv` and `test_group.csv` in the `data/` folder.")
            st.stop()
    else:
        ctrl_file = st.file_uploader("Control Group CSV", type="csv")
        test_file = st.file_uploader("Test Group CSV", type="csv")
        if ctrl_file and test_file:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                f.write(ctrl_file.read()); ctrl_path = f.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                f.write(test_file.read()); test_path = f.name
        else:
            st.info("Upload both CSV files to continue.")
            st.stop()

    st.markdown("---")
    st.subheader("⚙️ Test Parameters")
    alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01,
                      help="Probability threshold for rejecting H₀")
    st.caption(f"Confidence level: **{(1-alpha):.0%}**")

    st.markdown("---")
    st.subheader("🎯 Primary Metric")
    primary_metric = st.selectbox("", METRIC_COLS, index=METRIC_COLS.index("Purchase"))

    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · SciPy")


# ── Load data ────────────────────────────────────────────────────────────────
df = cached_load(ctrl_path, test_path)
ctrl_name, test_name = campaign_names(df)
summary = summary_by_group(df)

ctrl_row = summary[summary["Campaign"] == ctrl_name].iloc[0]
test_row  = summary[summary["Campaign"] == test_name].iloc[0]

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🧪 A/B Testing Intelligence Platform")
st.markdown(f"**Dataset:** Kaggle Marketing A/B Test &nbsp;|&nbsp; **Period:** {df['Date'].min().date()} → {df['Date'].max().date()} &nbsp;|&nbsp; **α = {alpha}**")
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "📈 Metric Deep Dive",
    "🔬 Statistical Tests",
    "🧠 Bayesian Analysis",
    "📡 Sequential Monitor",
    "🔄 Funnel Analysis",
    "📐 Sample Size Planner",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Campaign Snapshot</div>', unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics_kpi = [
        ("💰 Total Spend",   ctrl_row.Total_Spend,     test_row.Total_Spend,     "${:,.0f}"),
        ("👁️ Impressions",   ctrl_row.Total_Impressions,test_row.Total_Impressions,"{:,.0f}"),
        ("🖱️ Clicks",        ctrl_row.Total_Clicks,    test_row.Total_Clicks,    "{:,.0f}"),
        ("🛒 Add to Cart",   ctrl_row.Total_AddToCart, test_row.Total_AddToCart, "{:,.0f}"),
        ("✅ Purchases",     ctrl_row.Total_Purchases, test_row.Total_Purchases, "{:,.0f}"),
    ]
    for col, (label, cv, tv, fmt) in zip([col1,col2,col3,col4,col5], metrics_kpi):
        delta_pct = ((tv - cv) / cv * 100) if cv else 0
        col.metric(label,
                   fmt.format(tv),
                   f"{delta_pct:+.1f}% vs Control")

    st.markdown("---")

    # Side-by-side rate comparison
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📋 Head-to-Head Summary")
        compare = pd.DataFrame({
            "Metric": ["Spend ($)", "Impressions", "Reach", "Clicks", "Add to Cart", "Purchases",
                        "CTR", "Purchase Rate", "Add to Cart Rate", "Cost/Click ($)", "Cost/Purchase ($)"],
            ctrl_name: [
                fmt_num(ctrl_row.Total_Spend), fmt_num(ctrl_row.Total_Impressions),
                fmt_num(ctrl_row.Total_Reach), fmt_num(ctrl_row.Total_Clicks),
                fmt_num(ctrl_row.Total_AddToCart), fmt_num(ctrl_row.Total_Purchases),
                fmt_pct(ctrl_row.CTR), fmt_pct(ctrl_row.Purchase_Rate),
                fmt_pct(ctrl_row.Add_to_Cart_Rate),
                fmt_num(ctrl_row.Cost_per_Click,2), fmt_num(ctrl_row.Cost_per_Purchase,2),
            ],
            test_name: [
                fmt_num(test_row.Total_Spend), fmt_num(test_row.Total_Impressions),
                fmt_num(test_row.Total_Reach), fmt_num(test_row.Total_Clicks),
                fmt_num(test_row.Total_AddToCart), fmt_num(test_row.Total_Purchases),
                fmt_pct(test_row.CTR), fmt_pct(test_row.Purchase_Rate),
                fmt_pct(test_row.Add_to_Cart_Rate),
                fmt_num(test_row.Cost_per_Click,2), fmt_num(test_row.Cost_per_Purchase,2),
            ],
        })
        st.dataframe(compare.set_index("Metric"), use_container_width=True)

    with col_r:
        st.markdown("#### 🏆 Relative Performance (Test vs Control)")
        metrics_bar = ["Impressions","Reach","Website Clicks","Searches",
                       "View Content","Add to Cart","Purchase"]
        uplifts = []
        for m in metrics_bar:
            cv = ctrl_row[f"Total_{m.replace(' ','').replace('[','').replace(']','')}"] if f"Total_{m.replace(' ','').replace('[','').replace(']','')}" in ctrl_row else None
            tv = test_row.get(f"Total_{m.replace(' ','').replace('[','').replace(']','')}", None)
            # fallback: manual
            c_ser = get_group_series(df, ctrl_name, m)
            t_ser = get_group_series(df, test_name, m)
            cm, tm = np.nanmean(c_ser), np.nanmean(t_ser)
            uplift = (tm - cm) / cm * 100 if cm else 0
            uplifts.append(uplift)

        fig = go.Figure(go.Bar(
            x=uplifts, y=metrics_bar, orientation="h",
            marker_color=[TEST_COLOR if u >= 0 else "#ef4444" for u in uplifts],
            text=[f"{u:+.1f}%" for u in uplifts],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", height=350,
            xaxis_title="Relative Uplift %",
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(zeroline=True, zerolinecolor="#555"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Time series of primary metric
    st.markdown(f"#### 📅 Daily {primary_metric} Over Time")
    ctrl_ts = df[df["Campaign"] == ctrl_name].sort_values("Date")
    test_ts  = df[df["Campaign"] == test_name].sort_values("Date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctrl_ts["Date"], y=ctrl_ts[primary_metric],
                             mode="lines+markers", name=ctrl_name,
                             line=dict(color=CTRL_COLOR, width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=test_ts["Date"], y=test_ts[primary_metric],
                             mode="lines+markers", name=test_name,
                             line=dict(color=TEST_COLOR, width=2), marker=dict(size=6)))
    fig.update_layout(template="plotly_dark", height=300,
                      yaxis_title=primary_metric,
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — METRIC DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Metric Deep Dive</div>', unsafe_allow_html=True)

    all_metrics = METRIC_COLS + ["CTR","Purchase Rate","Add to Cart Rate",
                                  "Cost per Click","Cost per Purchase","ROAS"]
    sel_metric = st.selectbox("Select metric to explore", all_metrics, index=all_metrics.index(primary_metric))

    ctrl_vals = get_group_series(df, ctrl_name, sel_metric)
    test_vals = get_group_series(df, test_name, sel_metric)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Control Mean",  fmt_num(np.nanmean(ctrl_vals), 1))
    col2.metric(f"Test Mean",     fmt_num(np.nanmean(test_vals), 1),
                f"{(np.nanmean(test_vals)-np.nanmean(ctrl_vals))/np.nanmean(ctrl_vals)*100:+.1f}%")
    col3.metric("Control Std",    fmt_num(np.nanstd(ctrl_vals, ddof=1), 1))
    col4.metric("Test Std",       fmt_num(np.nanstd(test_vals, ddof=1), 1))

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Daily Distribution", "Box Plot", "Cumulative Average"))

    # Histogram
    for vals, name, color in [(ctrl_vals, ctrl_name, CTRL_COLOR), (test_vals, test_name, TEST_COLOR)]:
        fig.add_trace(go.Histogram(x=vals, name=name, marker_color=color,
                                   opacity=0.6, nbinsx=15), row=1, col=1)

    # Box plot
    for vals, name, color in [(ctrl_vals, ctrl_name, CTRL_COLOR), (test_vals, test_name, TEST_COLOR)]:
        fig.add_trace(go.Box(y=vals, name=name, marker_color=color,
                             boxmean="sd", showlegend=False), row=1, col=2)

    # Cumulative average
    ctrl_ts2 = df[df["Campaign"] == ctrl_name].sort_values("Date")[[" Date", sel_metric]].dropna() if " Date" in df.columns else df[df["Campaign"] == ctrl_name].sort_values("Date")[["Date", sel_metric]].dropna()
    test_ts2  = df[df["Campaign"] == test_name].sort_values("Date")[["Date", sel_metric]].dropna()
    ctrl_cum = ctrl_ts2[sel_metric].expanding().mean()
    test_cum  = test_ts2[sel_metric].expanding().mean()
    fig.add_trace(go.Scatter(x=ctrl_ts2["Date"], y=ctrl_cum, name=ctrl_name,
                             line=dict(color=CTRL_COLOR), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=test_ts2["Date"], y=test_cum, name=test_name,
                             line=dict(color=TEST_COLOR), showlegend=False), row=1, col=3)

    fig.update_layout(template="plotly_dark", height=380,
                      barmode="overlay", margin=dict(t=40, b=10),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### 🔗 Metric Correlation Heatmap")
    corr_df = df[df["Campaign"] == test_name][METRIC_COLS].corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
        colorscale="RdBu", zmid=0,
        text=corr_df.round(2).values, texttemplate="%{text}",
    ))
    fig_heat.update_layout(template="plotly_dark", height=400,
                           title="Test Campaign — Pairwise Correlations",
                           margin=dict(t=40, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Frequentist Statistical Tests (Welch\'s t-test)</div>',
                unsafe_allow_html=True)

    test_metrics = st.multiselect(
        "Select metrics to test",
        METRIC_COLS + ["CTR","Purchase Rate","Add to Cart Rate","Cost per Click","Cost per Purchase"],
        default=METRIC_COLS,
    )

    if test_metrics:
        results = run_all_metrics(df, ctrl_name, test_name, test_metrics, alpha)
        res_df  = results_to_df(results)

        # Style the table
        styled = (
            res_df.style
            .format({
                "Control Mean": "{:,.2f}", "Test Mean": "{:,.2f}",
                "Uplift %": "{:+.2f}%", "p-value": "{:.4f}",
                "Cohen's d": "{:.3f}",
            })
            .applymap(lambda v: "color: #22c55e" if v == "✅" else ("color: #f97316" if v == "❌" else ""),
                      subset=["Significant"])
            .background_gradient(subset=["Uplift %"], cmap="RdYlGn", vmin=-30, vmax=30)
        )
        st.dataframe(styled, use_container_width=True, height=360)

        # Waterfall of uplifts
        st.markdown("#### 📊 Uplift Waterfall")
        uplifts = [r.relative_uplift for r in results]
        labels  = [r.metric for r in results]
        sigs    = [r.significant for r in results]

        fig = go.Figure(go.Bar(
            x=labels, y=uplifts,
            marker_color=["#22c55e" if (u >= 0 and s) else ("#ef4444" if (u < 0 and s) else "#6b7280")
                          for u, s in zip(uplifts, sigs)],
            text=[f"{u:+.1f}%" for u in uplifts],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="#555")
        fig.update_layout(template="plotly_dark", height=350,
                          yaxis_title="Relative Uplift %",
                          margin=dict(t=10, b=10),
                          xaxis_tickangle=-20)
        fig.add_annotation(x=0.01, y=0.97, xref="paper", yref="paper",
                           text="Green = sig. positive · Red = sig. negative · Grey = not significant",
                           showarrow=False, font=dict(size=11, color="#9da5b4"),
                           align="left")
        st.plotly_chart(fig, use_container_width=True)

        # Individual metric verdict
        st.markdown("#### 🔎 Per-Metric Verdicts")
        for r in results:
            p_str = f"p = {r.p_value:.4f}"
            if r.direction == "test_wins":
                st.markdown(verdict_html(
                    f"**{r.metric}**: Test wins (+{r.relative_uplift:.1f}%) | {p_str} | Cohen's d = {r.cohens_d:.3f}", "green"),
                    unsafe_allow_html=True)
            elif r.direction == "ctrl_wins":
                st.markdown(verdict_html(
                    f"**{r.metric}**: Control wins ({r.relative_uplift:.1f}%) | {p_str} | Cohen's d = {r.cohens_d:.3f}", "red"),
                    unsafe_allow_html=True)
            else:
                st.markdown(verdict_html(
                    f"**{r.metric}**: No significant difference | {p_str}", "orange"),
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BAYESIAN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Bayesian A/B Analysis — Beta-Binomial Model</div>',
                unsafe_allow_html=True)
    st.info("Bayesian analysis treats conversions (Purchase / Add to Cart) as success rates with a Beta posterior. "
            "No p-value cutoffs — we quantify the *probability* one campaign is better.")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        bayes_metric = st.selectbox("Conversion Metric", ["Purchase", "Add to Cart"])
        prior_a = st.slider("Prior α (prior successes)", 0.1, 10.0, 1.0, 0.1)
        prior_b = st.slider("Prior β (prior failures)",  0.1, 10.0, 1.0, 0.1)
        n_samples = st.select_slider("Monte Carlo samples", [10_000, 50_000, 100_000], value=100_000)

    ctrl_clicks = int(get_group_series(df, ctrl_name, "Website Clicks").sum())
    test_clicks  = int(get_group_series(df, test_name, "Website Clicks").sum())
    ctrl_conv   = int(get_group_series(df, ctrl_name, bayes_metric).sum())
    test_conv   = int(get_group_series(df, test_name, bayes_metric).sum())

    br = run_bayesian_conversion(
        ctrl_conv, ctrl_clicks, test_conv, test_clicks,
        ctrl_name, test_name, bayes_metric,
        prior_a, prior_b, n_samples,
    )

    with col_r:
        st.markdown(verdict_html(br.recommendation, br.rec_color), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("P(Test > Control)", f"{br.prob_test_better:.1%}")
        c2.metric("Expected Loss (Test)", f"{br.expected_loss_if_test:.4f}")
        c3.metric("Expected Loss (Control)", f"{br.expected_loss_if_ctrl:.4f}")

    # Posterior distributions
    x_range = np.linspace(0, 0.3, 500)
    from scipy.stats import beta as beta_dist
    pdf_ctrl = beta_dist.pdf(x_range, prior_a + ctrl_conv, prior_b + ctrl_clicks - ctrl_conv)
    pdf_test  = beta_dist.pdf(x_range, prior_a + test_conv,  prior_b + test_clicks  - test_conv)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_range, y=pdf_ctrl, fill="tozeroy", name=ctrl_name,
                             line=dict(color=CTRL_COLOR), fillcolor="rgba(76,114,176,0.25)"))
    fig.add_trace(go.Scatter(x=x_range, y=pdf_test,  fill="tozeroy", name=test_name,
                             line=dict(color=TEST_COLOR), fillcolor="rgba(221,132,82,0.25)"))
    fig.add_vline(x=br.ctrl_posterior_mean, line_dash="dash", line_color=CTRL_COLOR,
                  annotation_text=f"μ={br.ctrl_posterior_mean:.3f}")
    fig.add_vline(x=br.test_posterior_mean,  line_dash="dash", line_color=TEST_COLOR,
                  annotation_text=f"μ={br.test_posterior_mean:.3f}")
    fig.update_layout(template="plotly_dark", height=350,
                      title=f"Posterior Distributions — {bayes_metric} Rate",
                      xaxis_title="Conversion Rate", yaxis_title="Density",
                      margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Difference distribution
    n_plot = 50_000
    rng = np.random.default_rng(99)
    diff_samples = rng.beta(prior_a + test_conv,  prior_b + test_clicks - test_conv,  n_plot) - \
                   rng.beta(prior_a + ctrl_conv, prior_b + ctrl_clicks - ctrl_conv, n_plot)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=diff_samples, nbinsx=80, name="P(Test − Control)",
                                marker_color=TEST_COLOR, opacity=0.75))
    fig2.add_vline(x=0, line_color="white", line_dash="dash")
    prob_pos = float(np.mean(diff_samples > 0))
    fig2.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper",
                        text=f"P(Test > Control) = {prob_pos:.1%}",
                        showarrow=False, font=dict(size=14, color="#22c55e" if prob_pos > 0.5 else "#ef4444"))
    fig2.update_layout(template="plotly_dark", height=300,
                       title="Distribution of the Difference (Test − Control)",
                       xaxis_title="Δ Conversion Rate",
                       margin=dict(t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)

    # ROPE analysis
    st.markdown("#### 🎯 ROPE Analysis (Region Of Practical Equivalence)")
    rope_val = st.slider("ROPE width (±)", 0.001, 0.05, 0.01, 0.001,
                         help="Differences smaller than this are 'practically equivalent'")
    p_rope  = float(np.mean(np.abs(diff_samples) < rope_val))
    p_test  = float(np.mean(diff_samples >  rope_val))
    p_ctrl  = float(np.mean(diff_samples < -rope_val))
    r1, r2, r3 = st.columns(3)
    r1.metric("P(Practically Equivalent)", f"{p_rope:.1%}")
    r2.metric("P(Test Meaningfully Better)", f"{p_test:.1%}")
    r3.metric("P(Control Meaningfully Better)", f"{p_ctrl:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SEQUENTIAL MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Sequential Monitoring — O\'Brien-Fleming Boundaries</div>',
                unsafe_allow_html=True)
    st.info("Sequential testing lets you monitor results as data arrives without inflating false-positive rates. "
            "The O'Brien-Fleming boundary starts strict and relaxes over time.")

    seq_metric = st.selectbox("Metric to monitor", METRIC_COLS,
                              index=METRIC_COLS.index("Purchase"))
    n_looks    = st.slider("Number of monitoring checkpoints", 5, 28, 20)

    seq = run_sequential_monitoring(df, ctrl_name, test_name, seq_metric, alpha)

    # Main sequential chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=seq["dates"], y=seq["obf_upper"],
                             name="OBF Upper Boundary", line=dict(color="#ef4444", dash="dash", width=2),
                             fill=None))
    fig.add_trace(go.Scatter(x=seq["dates"], y=seq["obf_lower"],
                             name="OBF Lower Boundary", line=dict(color="#ef4444", dash="dash", width=2),
                             fill="tonexty", fillcolor="rgba(239,68,68,0.07)"))
    fig.add_trace(go.Scatter(x=seq["dates"], y=seq["z_scores"],
                             name="Z-score", line=dict(color="#a78bfa", width=2.5),
                             mode="lines+markers", marker=dict(size=5)))
    fig.add_hline(y=0, line_color="#555", line_dash="dot")

    if seq["early_stop"]:
        es = seq["early_stop"]
        fig.add_vline(x=es["date"], line_color="#fbbf24", line_dash="dash")
        fig.add_annotation(x=es["date"], y=es["z"],
                           text=f"⚡ Early stop\n{es['direction']} wins",
                           showarrow=True, arrowhead=2, arrowcolor="#fbbf24",
                           font=dict(color="#fbbf24", size=12))

    fig.update_layout(template="plotly_dark", height=380,
                      title="Sequential Z-scores vs O'Brien-Fleming Boundaries",
                      yaxis_title="Z-score", xaxis_title="Date",
                      margin=dict(t=50, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative means
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=seq["dates"], y=seq["ctrl_cumulative"],
                              name=ctrl_name, line=dict(color=CTRL_COLOR, width=2)))
    fig2.add_trace(go.Scatter(x=seq["dates"], y=seq["test_cumulative"],
                              name=test_name, line=dict(color=TEST_COLOR, width=2)))
    fig2.update_layout(template="plotly_dark", height=280,
                       title=f"Cumulative Daily Mean — {seq_metric}",
                       yaxis_title=seq_metric, margin=dict(t=50, b=10),
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

    if seq["early_stop"]:
        es = seq["early_stop"]
        winner = test_name if es["direction"] == "test" else ctrl_name
        st.markdown(verdict_html(
            f"⚡ Early stopping triggered on {es['date'].date()} — **{winner}** shows significant superiority",
            "green" if es["direction"] == "test" else "red"
        ), unsafe_allow_html=True)
    else:
        st.markdown(verdict_html(
            "No early stopping triggered — boundaries not crossed during the observation window.", "orange"
        ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FUNNEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">Conversion Funnel Analysis</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    for col, camp, color in [(col_l, ctrl_name, CTRL_COLOR), (col_r, test_name, TEST_COLOR)]:
        vals = [get_group_series(df, camp, m).sum() for m in FUNNEL_STAGES]
        pcts = [v / vals[0] * 100 if vals[0] > 0 else 0 for v in vals]
        with col:
            fig = go.Figure(go.Funnel(
                y=FUNNEL_STAGES,
                x=vals,
                textinfo="value+percent initial",
                marker=dict(color=[color] * len(FUNNEL_STAGES)),
                connector=dict(line=dict(color="#555", width=1)),
            ))
            fig.update_layout(template="plotly_dark", height=450,
                              title=camp, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Stage-by-stage drop-off comparison
    st.markdown("#### 📉 Stage Drop-off Rates")
    stages_pairs = list(zip(FUNNEL_STAGES[:-1], FUNNEL_STAGES[1:]))
    rows = []
    for s1, s2 in stages_pairs:
        c1 = get_group_series(df, ctrl_name, s1).sum()
        c2 = get_group_series(df, ctrl_name, s2).sum()
        t1 = get_group_series(df, test_name, s1).sum()
        t2 = get_group_series(df, test_name, s2).sum()
        ctrl_rate = c2/c1 if c1 else 0
        test_rate  = t2/t1 if t1 else 0
        rows.append({
            "Stage": f"{s1} → {s2}",
            f"{ctrl_name} Conversion": ctrl_rate,
            f"{test_name} Conversion": test_rate,
            "Δ (Test − Control)": test_rate - ctrl_rate,
        })
    drop_df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=drop_df["Stage"], y=drop_df[f"{ctrl_name} Conversion"],
                         name=ctrl_name, marker_color=CTRL_COLOR))
    fig.add_trace(go.Bar(x=drop_df["Stage"], y=drop_df[f"{test_name} Conversion"],
                         name=test_name, marker_color=TEST_COLOR))
    fig.update_layout(template="plotly_dark", height=350, barmode="group",
                      yaxis_tickformat=".0%", yaxis_title="Conversion Rate",
                      margin=dict(t=10, b=10), xaxis_tickangle=-15,
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    styled_drop = (
        drop_df.style
        .format({f"{ctrl_name} Conversion": "{:.2%}", f"{test_name} Conversion": "{:.2%}",
                 "Δ (Test − Control)": "{:+.2%}"})
        .applymap(lambda v: "color: #22c55e" if isinstance(v, float) and v > 0
                  else ("color: #ef4444" if isinstance(v, float) and v < 0 else ""),
                  subset=["Δ (Test − Control)"])
    )
    st.dataframe(styled_drop, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SAMPLE SIZE PLANNER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">Sample Size & Power Planner</div>', unsafe_allow_html=True)
    st.info("Use this planner before running your next test to ensure you collect enough data.")

    col_l, col_r = st.columns(2)
    with col_l:
        plan_metric  = st.selectbox("Metric to plan for", METRIC_COLS, index=METRIC_COLS.index("Purchase"))
        baseline_arr = get_group_series(df, ctrl_name, plan_metric)
        baseline_mean = float(np.nanmean(baseline_arr))
        baseline_std  = float(np.nanstd(baseline_arr, ddof=1))
        st.metric("Baseline Mean (from data)", fmt_num(baseline_mean, 1))
        st.metric("Baseline Std Dev",          fmt_num(baseline_std, 1))

        mde_pct  = st.slider("Minimum Detectable Effect (%)", 1, 50, 10,
                             help="Smallest relative improvement worth detecting")
        pwr      = st.slider("Target Power (1−β)", 0.70, 0.95, 0.80, 0.05,
                             help="Probability of detecting a real effect")

        n_needed = required_sample_size(baseline_mean, baseline_std, mde_pct, alpha, pwr)
        st.markdown("---")
        st.metric("📋 Required Sample Size (per group)", f"{n_needed:,}" if n_needed > 0 else "N/A")
        st.metric("📋 Total Participants Needed", f"{n_needed*2:,}" if n_needed > 0 else "N/A")
        days_to_run = n_needed / max(baseline_mean, 1) if n_needed > 0 else None
        if days_to_run:
            st.metric("⏱ Estimated Days (at current rate)", f"{days_to_run:.0f}")

    with col_r:
        st.markdown("#### Power Curve")
        mde_range = np.arange(1, 51)
        n_vals = [required_sample_size(baseline_mean, baseline_std, m, alpha, pwr) for m in mde_range]
        n_vals_clean = [n if n > 0 else np.nan for n in n_vals]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mde_range, y=n_vals_clean,
                                 mode="lines", fill="tozeroy",
                                 line=dict(color="#a78bfa", width=2),
                                 fillcolor="rgba(167,139,250,0.15)"))
        fig.add_vline(x=mde_pct, line_color="#fbbf24", line_dash="dash",
                      annotation_text=f"Selected MDE: {mde_pct}%",
                      annotation_font_color="#fbbf24")
        fig.update_layout(template="plotly_dark", height=300,
                          xaxis_title="MDE (%)", yaxis_title="Required n (per group)",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Power vs Sample Size")
        n_range2 = np.linspace(50, max(n_needed * 2, 1000), 200)
        if baseline_mean > 0 and baseline_std > 0:
            delta = baseline_mean * mde_pct / 100
            effect_sizes = delta / baseline_std
            from scipy.stats import norm
            powers = [min(norm.cdf(effect_sizes * np.sqrt(n / 2) - norm.ppf(1 - alpha / 2)) +
                          norm.cdf(-effect_sizes * np.sqrt(n / 2) - norm.ppf(1 - alpha / 2)), 1.0)
                      for n in n_range2]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=n_range2, y=powers,
                                      mode="lines", line=dict(color=TEST_COLOR, width=2),
                                      fill="tozeroy", fillcolor="rgba(221,132,82,0.125)"))
            fig2.add_hline(y=pwr, line_dash="dash", line_color="#fbbf24",
                           annotation_text=f"Target power: {pwr:.0%}",
                           annotation_font_color="#fbbf24")
            if n_needed > 0:
                fig2.add_vline(x=n_needed, line_dash="dash", line_color="#22c55e",
                               annotation_text=f"n={n_needed:,}",
                               annotation_font_color="#22c55e")
            fig2.update_layout(template="plotly_dark", height=280,
                               xaxis_title="Sample Size (per group)", yaxis_title="Power",
                               yaxis_tickformat=".0%", margin=dict(t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    # Multi-MDE table
    st.markdown("#### 📊 Quick Reference — Required n for Different Scenarios")
    mde_options = [5, 10, 15, 20, 30]
    pwr_options = [0.70, 0.80, 0.90]
    table_data = {}
    for p in pwr_options:
        table_data[f"Power={p:.0%}"] = [
            required_sample_size(baseline_mean, baseline_std, m, alpha, p)
            for m in mde_options
        ]
    ref_df = pd.DataFrame(table_data, index=[f"MDE={m}%" for m in mde_options])
    st.dataframe(ref_df.style.format("{:,}").background_gradient(cmap="YlOrRd_r", axis=None),
                 use_container_width=True)
