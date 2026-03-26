# 🧪 A/B Testing Intelligence Platform

An enterprise-grade, interactive A/B testing platform built with **Streamlit + Plotly + SciPy**.
Powered by the [Kaggle Marketing A/B Testing Dataset](https://www.kaggle.com/datasets/amirmotefaker/ab-testing-dataset).

---

## 🚀 Quick Start (3 steps)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📁 Project Structure

```
ab_platform/
├── app.py                    # Main Streamlit app (7 analysis tabs)
├── src/
│   ├── data_utils.py         # Data loading, cleaning, feature engineering
│   └── stats.py              # Frequentist, Bayesian, Sequential stats
├── data/
│   ├── control_group.csv     # Kaggle dataset — Control Campaign
│   └── test_group.csv        # Kaggle dataset — Test Campaign
└── requirements.txt
```

---

## 📊 Platform Features

| Tab | What it does |
|-----|-------------|
| **Overview** | KPI dashboard, head-to-head comparison, daily time series |
| **Metric Deep Dive** | Per-metric distribution, box plots, cumulative averages, correlation heatmap |
| **Statistical Tests** | Welch's t-test for all metrics, uplift waterfall, per-metric verdicts |
| **Bayesian Analysis** | Beta-Binomial posterior distributions, P(Test > Control), ROPE analysis, expected loss |
| **Sequential Monitor** | O'Brien-Fleming boundaries, early stopping detection, cumulative mean tracking |
| **Funnel Analysis** | Full conversion funnel, stage drop-off comparison |
| **Sample Size Planner** | Power curves, required n calculator, multi-scenario reference table |

---


---

## 🧠 Statistical Methods

- **Welch's t-test** — compares daily metric means without assuming equal variance
- **Cohen's d** — standardised effect size
- **Bayesian Beta-Binomial** — posterior inference on conversion rates; ROPE analysis
- **O'Brien-Fleming sequential testing** — controls Type I error across multiple looks
- **Power analysis** — sample size planning with customisable α, power, and MDE

---

## VS Code Tips

- Press **F5** and select `Run Streamlit App` to launch from the debugger
- Install the **Pylance** extension for full IntelliSense
