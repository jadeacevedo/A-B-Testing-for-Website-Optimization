"""src/data_utils.py — Load, clean, and merge the A/B campaign CSVs."""

import pandas as pd
import numpy as np
from typing import Tuple

METRIC_COLS = [
    "Impressions", "Reach", "Website Clicks",
    "Searches", "View Content", "Add to Cart", "Purchase",
]
SPEND_COL = "Spend [USD]"

RENAME = {
    "# of Impressions":    "Impressions",
    "# of Website Clicks": "Website Clicks",
    "# of Searches":       "Searches",
    "# of View Content":   "View Content",
    "# of Add to Cart":    "Add to Cart",
    "# of Purchase":       "Purchase",
    "Spend [USD]":         "Spend [USD]",
    "Reach":               "Reach",
    "Campaign Name":       "Campaign",
    "Date":                "Date",
}

FUNNEL_STAGES = ["Impressions", "Reach", "Website Clicks",
                 "Searches", "View Content", "Add to Cart", "Purchase"]

COLORS = {"Control Campaign": "#4C72B0", "Test Campaign": "#DD8452"}


def load_and_merge(ctrl_path: str, test_path: str) -> pd.DataFrame:
    ctrl = _load_one(ctrl_path)
    test = _load_one(test_path)
    df = pd.concat([ctrl, test], ignore_index=True).sort_values("Date").reset_index(drop=True)
    return df


def _load_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=RENAME)
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")
    for col in METRIC_COLS + [SPEND_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CTR"]               = df["Website Clicks"] / df["Impressions"]
    df["Search Rate"]       = df["Searches"]        / df["Website Clicks"]
    df["View Rate"]         = df["View Content"]    / df["Website Clicks"]
    df["Add to Cart Rate"]  = df["Add to Cart"]     / df["Website Clicks"]
    df["Purchase Rate"]     = df["Purchase"]        / df["Website Clicks"]
    df["Cost per Click"]    = df["Spend [USD]"]     / df["Website Clicks"]
    df["Cost per Purchase"] = df["Spend [USD]"]     / df["Purchase"]
    df["ROAS"]              = df["Purchase"]        / df["Spend [USD]"]
    return df


def summary_by_group(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("Campaign")
        .agg(
            Days=("Date", "count"),
            Total_Spend=("Spend [USD]", "sum"),
            Total_Impressions=("Impressions", "sum"),
            Total_Reach=("Reach", "sum"),
            Total_Clicks=("Website Clicks", "sum"),
            Total_Searches=("Searches", "sum"),
            Total_ViewContent=("View Content", "sum"),
            Total_AddToCart=("Add to Cart", "sum"),
            Total_Purchases=("Purchase", "sum"),
        ).reset_index()
    )
    agg["CTR"]               = agg["Total_Clicks"]    / agg["Total_Impressions"]
    agg["Purchase_Rate"]     = agg["Total_Purchases"] / agg["Total_Clicks"]
    agg["Add_to_Cart_Rate"]  = agg["Total_AddToCart"] / agg["Total_Clicks"]
    agg["Cost_per_Purchase"] = agg["Total_Spend"]     / agg["Total_Purchases"]
    agg["Cost_per_Click"]    = agg["Total_Spend"]     / agg["Total_Clicks"]
    return agg


def get_group_series(df: pd.DataFrame, campaign: str, metric: str) -> np.ndarray:
    return df[df["Campaign"] == campaign][metric].dropna().values


def campaign_names(df: pd.DataFrame) -> Tuple[str, str]:
    names = sorted(df["Campaign"].unique())
    ctrl = next((n for n in names if "control" in n.lower()), names[0])
    test = next((n for n in names if n != ctrl), names[-1])
    return ctrl, test
