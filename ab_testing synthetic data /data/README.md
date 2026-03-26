# A/B Testing – Website Optimisation Pipeline

A full end-to-end A/B testing pipeline in Python covering data generation,
EDA, statistical significance testing, and visualisation.

---

## Project Structure

```
ab_testing_project/
├── data/                    # Raw & processed datasets
├── src/
│   ├── generate_data.py     # Synthetic dataset generator
│   └── analysis.py          # Full analysis pipeline
├── outputs/
│   └── plots/               # All generated charts
├── .vscode/
│   ├── settings.json
│   └── launch.json
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Generate the dataset
```bash
python src/generate_data.py
```

### 4 — Run the full analysis
```bash
python src/analysis.py
```

Charts are saved to `outputs/plots/`.

---

## What the Pipeline Does

| Step | Description |
|------|-------------|
| **Data Generation** | Simulates 10 000 users split across control & treatment groups with realistic conversion, session, and device data |
| **EDA** | Group-level summary stats — users, conversions, rates, avg session, avg pages |
| **Stats Testing** | Two-proportion z-test + chi-square test, 95 % Wilson confidence intervals, Cohen's h effect size, relative uplift % |
| **Visualisations** | Conversion rate bar chart, behavioural metric histograms, device breakdown, summary dashboard |

---

## Key Parameters (edit in `src/`)

| File | Variable | Default |
|------|----------|---------|
| `generate_data.py` | `control_conversion_rate` | `0.10` |
| `generate_data.py` | `treatment_conversion_rate` | `0.13` |
| `generate_data.py` | `n_control` / `n_treatment` | `5000` each |
| `analysis.py` | `ALPHA` | `0.05` |

---


