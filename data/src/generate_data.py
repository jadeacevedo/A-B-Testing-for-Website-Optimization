"""
generate_data.py
Generates a synthetic A/B test dataset simulating a website
conversion experiment (e.g. button colour change, new landing page).
"""

import numpy as np
import pandas as pd

def generate_ab_data(
    n_control: int = 5000,
    n_treatment: int = 5000,
    control_conversion_rate: float = 0.10,
    treatment_conversion_rate: float = 0.13,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    control = pd.DataFrame({
        "user_id":    range(1, n_control + 1),
        "group":      "control",
        "converted":  rng.binomial(1, control_conversion_rate, n_control),
        "session_duration": rng.normal(180, 60, n_control).clip(10).round(1),
        "pages_viewed":     rng.poisson(4, n_control).clip(1),
        "device": rng.choice(["desktop", "mobile", "tablet"],
                             n_control, p=[0.55, 0.35, 0.10]),
    })

    treatment = pd.DataFrame({
        "user_id":    range(n_control + 1, n_control + n_treatment + 1),
        "group":      "treatment",
        "converted":  rng.binomial(1, treatment_conversion_rate, n_treatment),
        "session_duration": rng.normal(200, 65, n_treatment).clip(10).round(1),
        "pages_viewed":     rng.poisson(5, n_treatment).clip(1),
        "device": rng.choice(["desktop", "mobile", "tablet"],
                             n_treatment, p=[0.55, 0.35, 0.10]),
    })

    df = pd.concat([control, treatment], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_ab_data()
    out = "data/ab_test_data.csv"
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}  ({len(df):,} rows)")
    print(df.head())
