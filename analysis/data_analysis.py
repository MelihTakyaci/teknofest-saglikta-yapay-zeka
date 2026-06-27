"""
Teknofest 2026 - Data Analysis Script
Generates summary statistics, missing value reports, and class distribution tables.
Does NOT modify source data. Saves outputs to reports/assets/.

Usage:
    python analysis/data_analysis.py
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "EĞİTİM (TRAIN) SETLERİ 2")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

PANEL_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}


def load_panels():
    panels = {}
    for name, fname in PANEL_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        panels[name] = pd.read_csv(path, low_memory=False)
        print(f"Loaded {name}: {panels[name].shape}")
    return panels


def dataset_inventory(panels):
    rows = []
    for name, df in panels.items():
        total_cells = df.shape[0] * df.shape[1]
        missing = df.isnull().sum().sum()
        rows.append({
            "Panel": name,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Total_Cells": total_cells,
            "Missing_Cells": missing,
            "Missing_Pct": round(missing / total_cells * 100, 2),
            "Constant_Cols": int((df.nunique() <= 1).sum()),
            "Duplicate_IDs": int(df["Variant_ID"].duplicated().sum()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_DIR, "dataset_inventory.csv"), index=False)
    print("\n=== Dataset Inventory ===")
    print(result.to_string(index=False))
    return result


def class_distribution(panels):
    rows = []
    for name, df in panels.items():
        vc = df["Label"].value_counts()
        rows.append({
            "Panel": name,
            "Pathogenic_1": int(vc.get(1, 0)),
            "Benign_0": int(vc.get(0, 0)),
            "Total": len(df),
            "Pathogenic_Pct": round(vc.get(1, 0) / len(df) * 100, 1),
            "Imbalance_Ratio": round(vc.max() / vc.min(), 2),
        })
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"), index=False)
    print("\n=== Class Distribution ===")
    print(result.to_string(index=False))
    return result


def missing_value_analysis(panels):
    for name, df in panels.items():
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        missing_count = df.isnull().sum()
        summary = pd.DataFrame({
            "Column": missing_count.index,
            "Missing_Count": missing_count.values,
            "Missing_Pct": missing_pct.values,
        }).sort_values("Missing_Pct", ascending=False)
        fname = f"missing_values_{name.lower()}.csv"
        summary.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

    master = panels["MASTER"]
    print("\n=== Missingness Bands (MASTER) ===")
    pct = master.isnull().mean() * 100
    bands = [
        ("0%", 0, 0.001),
        ("0-10%", 0.001, 10),
        ("10-25%", 10, 25),
        ("25-50%", 25, 50),
        ("50-75%", 50, 75),
        ("75-90%", 75, 90),
        ("90-100%", 90, 100.001),
    ]
    band_rows = []
    for bname, lo, hi in bands:
        count = int(((pct >= lo) & (pct < hi)).sum())
        band_rows.append({"Band": bname, "Column_Count": count})
        print(f"  {bname}: {count} columns")
    pd.DataFrame(band_rows).to_csv(
        os.path.join(OUTPUT_DIR, "missingness_bands_master.csv"), index=False
    )

    print("\n=== Missingness by Label (MASTER) ===")
    for label in [0, 1]:
        subset = master[master["Label"] == label]
        pct_val = subset.isnull().sum().sum() / (subset.shape[0] * subset.shape[1]) * 100
        print(f"  Label={label} (n={len(subset)}): {pct_val:.1f}% missing")

    print("\n=== Missingness by Feature Group ===")
    group_rows = []
    for name, df in panels.items():
        row = {"Panel": name}
        for prefix, label in [("AL_", "AL"), ("EK_", "EK"), ("CAT_", "CAT"), ("AA_", "AA")]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            row[f"{label}_Missing_Pct"] = round(df[cols].isnull().mean().mean() * 100, 1)
        group_rows.append(row)
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(os.path.join(OUTPUT_DIR, "missingness_by_feature_group.csv"), index=False)
    print(group_df.to_string(index=False))


def feature_label_correlation(master):
    numeric_df = master.select_dtypes(include=[np.number])
    corrs = numeric_df.corr()["Label"].drop("Label").abs().sort_values(ascending=False)
    top30 = corrs.head(30).reset_index()
    top30.columns = ["Feature", "Abs_Correlation"]
    top30.to_csv(os.path.join(OUTPUT_DIR, "top30_feature_correlations.csv"), index=False)
    print("\n=== Top 15 Features by |Correlation with Label| ===")
    print(top30.head(15).to_string(index=False))


def ek_feature_summary(master):
    ek_cols = [c for c in master.columns if c.startswith("EK_")]
    rows = []
    for c in ek_cols:
        for label in [0, 1]:
            subset = master[master["Label"] == label][c]
            rows.append({
                "Feature": c,
                "Label": label,
                "Mean": round(subset.mean(), 4),
                "Std": round(subset.std(), 4),
                "Median": round(subset.median(), 4),
                "Missing_Pct": round(subset.isnull().mean() * 100, 1),
            })
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_DIR, "ek_features_by_label.csv"), index=False)
    print("\n=== EK Features by Label ===")
    print(result.to_string(index=False))


def variant_overlap(panels):
    names = list(panels.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            overlap = len(set(panels[n1]["Variant_ID"]) & set(panels[n2]["Variant_ID"]))
            rows.append({"Panel_1": n1, "Panel_2": n2, "Overlap": overlap})
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_DIR, "variant_id_overlap.csv"), index=False)
    print("\n=== Variant ID Overlap ===")
    print(result.to_string(index=False))


def main():
    print("=" * 60)
    print("TEKNOFEST 2026 — Data Analysis Script")
    print("=" * 60)

    panels = load_panels()
    dataset_inventory(panels)
    class_distribution(panels)
    missing_value_analysis(panels)
    feature_label_correlation(panels["MASTER"])
    ek_feature_summary(panels["MASTER"])
    variant_overlap(panels)

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
