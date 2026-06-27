#!/usr/bin/env python3
"""
TEKNOFEST Healthcare AI - Phase 01: Comprehensive Exploratory Data Analysis
Missense Variant Pathogenicity Classification

Generates: Markdown reports, CSV tables, Jupyter Notebook, Word document
"""

import os
import sys
import warnings
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "EĞİTİM (TRAIN) SETLERİ 2"
REPORT_DIR = BASE_DIR / "reports" / "phase_01_data_understanding"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 01: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ============================================================
# A. DATASET LOADING
# ============================================================
print("\n[A] Loading datasets...")

datasets = OrderedDict()
file_info = []

for fname in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(fname)
    key = fname.stem.replace("YARISMA_TRAIN_", "")
    datasets[key] = df
    size_mb = fname.stat().st_size / (1024 * 1024)
    role = "General training set" if key == "MASTER" else f"Panel dataset ({key})"
    file_info.append({
        "File": fname.name,
        "Type": "CSV",
        "Size (MB)": round(size_mb, 2),
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Role": role
    })
    print(f"  Loaded {key}: {df.shape[0]} rows x {df.shape[1]} cols")

file_info_df = pd.DataFrame(file_info)
file_info_df.to_csv(REPORT_DIR / "dataset_inventory.csv", index=False)

# ============================================================
# B. SCHEMA INSPECTION
# ============================================================
print("\n[B] Schema inspection...")

master = datasets["MASTER"]
all_cols = list(master.columns)

al_cols = [c for c in all_cols if c.startswith("AL_")]
cat_cols = [c for c in all_cols if c.startswith("CAT_")]
ek_cols = [c for c in all_cols if c.startswith("EK_")]
aa_cols = [c for c in all_cols if c.startswith("AA_")]
id_cols = ["Variant_ID"]
target_col = "Label"

col_groups = {
    "AL (numeric features)": len(al_cols),
    "CAT (categorical features)": len(cat_cols),
    "EK (additional features)": len(ek_cols),
    "AA (amino acid features)": len(aa_cols),
    "Variant_ID": 1,
    "Label": 1
}

schema_info = []
for key, df in datasets.items():
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    n_const = sum(df[c].nunique(dropna=True) <= 1 for c in df.columns)
    n_dup_rows = df.duplicated().sum()
    schema_info.append({
        "Dataset": key,
        "Shape": f"{df.shape[0]}x{df.shape[1]}",
        "Numeric Cols": len(numeric_cols),
        "Object Cols": len(object_cols),
        "Constant Cols": n_const,
        "Duplicate Rows": n_dup_rows
    })
    print(f"  {key}: {len(numeric_cols)} numeric, {len(object_cols)} object, {n_const} constant, {n_dup_rows} duplicate rows")

schema_df = pd.DataFrame(schema_info)
schema_df.to_csv(REPORT_DIR / "schema_summary.csv", index=False)

# Identify constant columns per dataset
const_cols_per_ds = {}
for key, df in datasets.items():
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    const_cols_per_ds[key] = const
    if const:
        print(f"  {key} constant columns: {const[:10]}{'...' if len(const) > 10 else ''}")

# ============================================================
# C. TARGET LABEL ANALYSIS
# ============================================================
print("\n[C] Target label analysis...")

class_dist = []
for key, df in datasets.items():
    vc = df[target_col].value_counts()
    total = len(df)
    n_path = int(vc.get(1, 0))
    n_ben = int(vc.get(0, 0))
    n_other = total - n_path - n_ben
    pos_ratio = n_path / total if total > 0 else 0
    imb_ratio = max(n_path, n_ben) / min(n_path, n_ben) if min(n_path, n_ben) > 0 else float('inf')
    class_dist.append({
        "Dataset": key,
        "Total N": total,
        "Pathogenic (1)": n_path,
        "Benign (0)": n_ben,
        "Other/Unknown": n_other,
        "Positive Ratio": round(pos_ratio, 4),
        "Imbalance Ratio": round(imb_ratio, 2)
    })
    print(f"  {key}: Pathogenic={n_path}, Benign={n_ben}, Ratio={pos_ratio:.3f}, Imbalance={imb_ratio:.2f}")

class_dist_df = pd.DataFrame(class_dist)
class_dist_df.to_csv(REPORT_DIR / "class_distribution.csv", index=False)

# ============================================================
# D. MISSING VALUE ANALYSIS
# ============================================================
print("\n[D] Missing value analysis...")

miss_summary_all = []
miss_by_target_all = []

for key, df in datasets.items():
    print(f"\n  --- {key} ---")
    # Per-column missingness
    miss_pct = df.isnull().mean() * 100
    miss_count = df.isnull().sum()

    # Group by missingness bands
    bands = {
        "0%": (miss_pct == 0).sum(),
        "0-5%": ((miss_pct > 0) & (miss_pct <= 5)).sum(),
        "5-20%": ((miss_pct > 5) & (miss_pct <= 20)).sum(),
        "20-50%": ((miss_pct > 20) & (miss_pct <= 50)).sum(),
        "50-80%": ((miss_pct > 50) & (miss_pct <= 80)).sum(),
        "80-95%": ((miss_pct > 80) & (miss_pct <= 95)).sum(),
        ">95%": (miss_pct > 95).sum()
    }
    for band, cnt in bands.items():
        print(f"    {band}: {cnt} columns")

    # Per-row missingness
    row_miss = df.isnull().mean(axis=1) * 100
    print(f"    Row missingness: mean={row_miss.mean():.1f}%, median={row_miss.median():.1f}%, max={row_miss.max():.1f}%")

    # Feature-level missingness table
    for col in df.columns:
        mp = miss_pct[col]
        miss_summary_all.append({
            "Dataset": key,
            "Column": col,
            "Missing Count": int(miss_count[col]),
            "Missing Pct": round(mp, 2),
            "Band": (
                "0%" if mp == 0 else
                "0-5%" if mp <= 5 else
                "5-20%" if mp <= 20 else
                "20-50%" if mp <= 50 else
                "50-80%" if mp <= 80 else
                "80-95%" if mp <= 95 else
                ">95%"
            )
        })

    # Missingness association with target
    if target_col in df.columns:
        path_mask = df[target_col] == 1
        ben_mask = df[target_col] == 0
        for col in al_cols + ek_cols:
            if col not in df.columns:
                continue
            miss_path = df.loc[path_mask, col].isnull().mean()
            miss_ben = df.loc[ben_mask, col].isnull().mean()
            diff = abs(miss_path - miss_ben)
            miss_by_target_all.append({
                "Dataset": key,
                "Column": col,
                "Miss% Pathogenic": round(miss_path * 100, 2),
                "Miss% Benign": round(miss_ben * 100, 2),
                "Abs Difference": round(diff * 100, 2)
            })

miss_summary_df = pd.DataFrame(miss_summary_all)
miss_summary_df.to_csv(REPORT_DIR / "missing_value_per_column.csv", index=False)

miss_target_df = pd.DataFrame(miss_by_target_all)
miss_target_df.to_csv(REPORT_DIR / "missing_value_by_target.csv", index=False)

# Top differential missingness
top_miss_diff = miss_target_df.sort_values("Abs Difference", ascending=False).head(30)
print("\n  Top 15 features with differential missingness by target:")
print(top_miss_diff.head(15).to_string(index=False))

# ============================================================
# E. DISTRIBUTION AND OUTLIER ANALYSIS
# ============================================================
print("\n[E] Distribution and outlier analysis (MASTER dataset)...")

numeric_feats = [c for c in al_cols + ek_cols if c in master.columns and master[c].dtype in [np.float64, np.int64, float, int]]

desc_stats = []
for col in numeric_feats:
    s = master[col].dropna()
    if len(s) == 0:
        continue
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()
    try:
        skew = float(s.skew())
    except:
        skew = np.nan

    # Detect feature type heuristic
    feat_type = "numeric"
    if s.min() >= 0 and s.max() <= 1.001:
        feat_type = "probability/score [0,1]"
    elif s.nunique() == 2:
        feat_type = "binary"
    elif s.min() >= 0 and s.max() > 1000:
        feat_type = "count/position-like"

    desc_stats.append({
        "Column": col,
        "Count": int(len(s)),
        "Missing%": round(master[col].isnull().mean() * 100, 2),
        "Mean": round(float(s.mean()), 6),
        "Median": round(float(s.median()), 6),
        "Std": round(float(s.std()), 6),
        "Min": round(float(s.min()), 6),
        "Q1": round(float(q1), 6),
        "Q3": round(float(q3), 6),
        "Max": round(float(s.max()), 6),
        "Skewness": round(skew, 4),
        "Unique": int(s.nunique()),
        "N Outliers (3xIQR)": int(n_outliers),
        "Inferred Type": feat_type
    })

desc_stats_df = pd.DataFrame(desc_stats)
desc_stats_df.to_csv(REPORT_DIR / "feature_statistics.csv", index=False)

# Summary counts
type_counts = desc_stats_df["Inferred Type"].value_counts()
print(f"  Feature type breakdown:")
for t, c in type_counts.items():
    print(f"    {t}: {c}")

highly_skewed = desc_stats_df[desc_stats_df["Skewness"].abs() > 3]
print(f"  Highly skewed features (|skew|>3): {len(highly_skewed)}")

# ============================================================
# F. FEATURE-TARGET RELATIONSHIP ANALYSIS
# ============================================================
print("\n[F] Feature-target association analysis (MASTER)...")

path_idx = master[target_col] == 1
ben_idx = master[target_col] == 0

assoc_results = []
for col in numeric_feats:
    s_path = master.loc[path_idx, col].dropna()
    s_ben = master.loc[ben_idx, col].dropna()
    if len(s_path) < 5 or len(s_ben) < 5:
        continue
    med_diff = s_path.median() - s_ben.median()
    try:
        u_stat, p_val = mannwhitneyu(s_path, s_ben, alternative='two-sided')
        # Rank-biserial correlation as effect size
        n1, n2 = len(s_path), len(s_ben)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
    except:
        p_val = np.nan
        effect_size = np.nan

    assoc_results.append({
        "Column": col,
        "Median Pathogenic": round(float(s_path.median()), 6),
        "Median Benign": round(float(s_ben.median()), 6),
        "Median Diff (P-B)": round(float(med_diff), 6),
        "Mann-Whitney p-value": float(p_val) if not np.isnan(p_val) else None,
        "Effect Size (r)": round(float(effect_size), 4) if not np.isnan(effect_size) else None,
        "Abs Effect Size": round(abs(float(effect_size)), 4) if not np.isnan(effect_size) else None
    })

assoc_df = pd.DataFrame(assoc_results)
assoc_df = assoc_df.sort_values("Abs Effect Size", ascending=False)
assoc_df.to_csv(REPORT_DIR / "feature_target_association.csv", index=False)

top20_path = assoc_df[assoc_df["Median Diff (P-B)"] > 0].head(20)
top20_ben = assoc_df[assoc_df["Median Diff (P-B)"] < 0].head(20)
weak_feats = assoc_df[assoc_df["Abs Effect Size"] < 0.05].tail(20)

print(f"  Total features tested: {len(assoc_df)}")
print(f"  Significant (p<0.001): {(assoc_df['Mann-Whitney p-value'] < 0.001).sum()}")
print(f"\n  Top 10 features (strongest class separation):")
print(assoc_df.head(10)[["Column", "Median Diff (P-B)", "Abs Effect Size", "Mann-Whitney p-value"]].to_string(index=False))

# ============================================================
# G. CORRELATION AND REDUNDANCY ANALYSIS
# ============================================================
print("\n[G] Correlation and redundancy analysis (MASTER)...")

# Use a sample for speed if large
feat_for_corr = [c for c in numeric_feats if master[c].notna().sum() > 100]
if len(feat_for_corr) > 200:
    # Take top 200 by variance
    variances = master[feat_for_corr].var().sort_values(ascending=False)
    feat_for_corr = variances.head(200).index.tolist()

corr_matrix = master[feat_for_corr].corr(method='spearman')

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(feat_for_corr)):
    for j in range(i + 1, len(feat_for_corr)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.85:
            ca = feat_for_corr[i]
            cb = feat_for_corr[j]
            miss_a = master[ca].isnull().mean() * 100
            miss_b = master[cb].isnull().mean() * 100
            severity = "|r|>0.95" if abs(r) > 0.95 else "|r|>0.85"
            action = "Consider dropping one" if abs(r) > 0.95 else "Monitor, may be redundant"
            high_corr_pairs.append({
                "Feature A": ca,
                "Feature B": cb,
                "Spearman r": round(float(r), 4),
                "Severity": severity,
                "Missing% A": round(miss_a, 2),
                "Missing% B": round(miss_b, 2),
                "Suggested Action": action
            })

corr_pairs_df = pd.DataFrame(high_corr_pairs)
if len(corr_pairs_df) > 0:
    corr_pairs_df = corr_pairs_df.sort_values("Spearman r", key=abs, ascending=False)
corr_pairs_df.to_csv(REPORT_DIR / "correlation_pairs.csv", index=False)

print(f"  Pairs with |r|>0.85: {len(corr_pairs_df)}")
n95 = (corr_pairs_df['Spearman r'].abs() > 0.95).sum() if len(corr_pairs_df) > 0 else 0
print(f"  Pairs with |r|>0.95: {n95}")

# ============================================================
# H. PANEL-LEVEL COMPARISON
# ============================================================
print("\n[H] Panel-level comparison...")

panel_comparison = []
for key, df in datasets.items():
    n_path = (df[target_col] == 1).sum()
    n_ben = (df[target_col] == 0).sum()
    total = len(df)
    miss_overall = df[al_cols].isnull().mean().mean() * 100
    n_const = sum(df[c].nunique(dropna=True) <= 1 for c in df.columns)
    pos_ratio = n_path / total if total > 0 else 0

    panel_comparison.append({
        "Dataset": key,
        "Rows": total,
        "Pathogenic": n_path,
        "Benign": n_ben,
        "Pos Ratio": round(pos_ratio, 4),
        "Mean Missing% (AL features)": round(miss_overall, 2),
        "Constant Cols": n_const,
        "Difficulty Estimate": ""
    })

panel_df = pd.DataFrame(panel_comparison)

# Estimate difficulty
for i, row in panel_df.iterrows():
    difficulty_factors = []
    if row["Rows"] < 200:
        difficulty_factors.append("very small sample")
    elif row["Rows"] < 500:
        difficulty_factors.append("small sample")
    if row["Pos Ratio"] < 0.3 or row["Pos Ratio"] > 0.7:
        difficulty_factors.append("imbalanced")
    if row["Mean Missing% (AL features)"] > 30:
        difficulty_factors.append("high missingness")
    if row["Constant Cols"] > 5:
        difficulty_factors.append("many constant cols")
    panel_df.at[i, "Difficulty Estimate"] = "; ".join(difficulty_factors) if difficulty_factors else "moderate"

panel_df.to_csv(REPORT_DIR / "panel_comparison.csv", index=False)

print(panel_df.to_string(index=False))

# Feature availability comparison across panels
print("\n  Checking feature availability across panels...")
feat_avail = {}
for key, df in datasets.items():
    for col in al_cols:
        if col not in feat_avail:
            feat_avail[col] = {}
        feat_avail[col][key] = round(df[col].notna().mean() * 100, 1)

feat_avail_df = pd.DataFrame(feat_avail).T
feat_avail_df.to_csv(REPORT_DIR / "feature_availability_across_panels.csv")

# Distribution shift detection
print("  Detecting distribution shifts between MASTER and panels...")
shift_results = []
for panel_key in ["CFTR", "PAH", "KANSER"]:
    if panel_key not in datasets:
        continue
    panel = datasets[panel_key]
    for col in numeric_feats[:100]:  # top 100 for speed
        sm = master[col].dropna()
        sp = panel[col].dropna()
        if len(sm) < 10 or len(sp) < 10:
            continue
        try:
            ks_stat, ks_p = stats.ks_2samp(sm, sp)
            shift_results.append({
                "Panel": panel_key,
                "Feature": col,
                "KS Statistic": round(ks_stat, 4),
                "KS p-value": ks_p,
                "Significant Shift": ks_p < 0.001
            })
        except:
            pass

shift_df = pd.DataFrame(shift_results)
shift_df.to_csv(REPORT_DIR / "distribution_shift_panel_vs_master.csv", index=False)
n_sig_shifts = shift_df.groupby("Panel")["Significant Shift"].sum()
print("  Significant distribution shifts per panel:")
for p, n in n_sig_shifts.items():
    print(f"    {p}: {int(n)} features shifted")

# ============================================================
# I. LEAKAGE AND VALIDITY RISK AUDIT
# ============================================================
print("\n[I] Leakage and validity risk audit...")

leakage_risks = []

# 1. Check if any feature perfectly predicts label
for col in numeric_feats[:50]:
    s = master[[col, target_col]].dropna()
    if len(s) < 100:
        continue
    try:
        auc_proxy = abs(assoc_df.loc[assoc_df["Column"] == col, "Abs Effect Size"].values[0]) if col in assoc_df["Column"].values else 0
        if auc_proxy > 0.9:
            leakage_risks.append({
                "Risk": f"Feature {col} has very high effect size ({auc_proxy:.3f})",
                "Severity": "HIGH",
                "Evidence": f"Effect size = {auc_proxy:.3f}, near-perfect class separation",
                "Mitigation": "Investigate if this feature encodes the label directly"
            })
    except:
        pass

# 2. EK features (extra features) - check for leakage
for col in ek_cols:
    if col in master.columns:
        nuniq = master[col].nunique(dropna=True)
        dtype = str(master[col].dtype)
        leakage_risks.append({
            "Risk": f"EK feature {col} ({dtype}, {nuniq} unique values) - unclear provenance",
            "Severity": "MEDIUM",
            "Evidence": f"EK prefix suggests external/extra annotation; dtype={dtype}, nunique={nuniq}",
            "Mitigation": "Verify EK features are not derived from clinical labels"
        })

# 3. CAT features
for col in cat_cols:
    if col in master.columns:
        vals = master[col].dropna().unique()[:10]
        leakage_risks.append({
            "Risk": f"Categorical feature {col} - check for population/ancestry encoding",
            "Severity": "LOW",
            "Evidence": f"Values include: {list(vals)[:5]}",
            "Mitigation": "Ensure CAT features don't encode test-set-identifiable metadata"
        })

# 4. AA features (amino acids)
leakage_risks.append({
    "Risk": "AA_1, AA_2 encode amino acid identity (single-letter codes)",
    "Severity": "LOW",
    "Evidence": "Values are single amino acid letters (R, Q, N, S, etc.)",
    "Mitigation": "These are legitimate variant features, but check for very rare AAs that could identify specific variants"
})

# 5. Variant_ID
leakage_risks.append({
    "Risk": "Variant_ID could be used to look up external annotations",
    "Severity": "MEDIUM",
    "Evidence": "IDs like VAR_004572 suggest an indexed database",
    "Mitigation": "Do not use Variant_ID as a feature; used only for identification"
})

# 6. Duplicate variants across panels and master
for panel_key in ["CFTR", "PAH", "KANSER"]:
    if panel_key not in datasets:
        continue
    overlap = set(master["Variant_ID"]) & set(datasets[panel_key]["Variant_ID"])
    if overlap:
        leakage_risks.append({
            "Risk": f"{len(overlap)} shared Variant_IDs between MASTER and {panel_key}",
            "Severity": "HIGH" if len(overlap) > 10 else "MEDIUM",
            "Evidence": f"{len(overlap)} overlapping variant identifiers",
            "Mitigation": "Ensure train/test splits do not leak through shared variants"
        })

# 7. EK_1, EK_2 look like scores
for col in ["EK_1", "EK_2"]:
    if col in master.columns:
        s = master[col].dropna()
        if len(s) > 0:
            leakage_risks.append({
                "Risk": f"{col} appears to be a continuous score (range {s.min():.2f}-{s.max():.2f})",
                "Severity": "MEDIUM",
                "Evidence": f"Mean={s.mean():.2f}, Std={s.std():.2f}, possibly a pathogenicity predictor score",
                "Mitigation": "If this is a pre-computed pathogenicity score, it could be a strong but circular predictor"
            })

leakage_df = pd.DataFrame(leakage_risks)
leakage_df.to_csv(REPORT_DIR / "leakage_risk_register.csv", index=False)
print(f"  Total leakage risks identified: {len(leakage_df)}")
print(f"  HIGH severity: {(leakage_df['Severity'] == 'HIGH').sum()}")
print(f"  MEDIUM severity: {(leakage_df['Severity'] == 'MEDIUM').sum()}")

# ============================================================
# FIGURES
# ============================================================
print("\n[FIGS] Generating figures...")

# 1. Class distribution barplot
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, (key, df) in enumerate(datasets.items()):
    vc = df[target_col].value_counts().sort_index()
    axes[idx].bar(["Benign (0)", "Pathogenic (1)"], [vc.get(0, 0), vc.get(1, 0)],
                  color=["#2196F3", "#F44336"])
    axes[idx].set_title(f"{key}\n(n={len(df)})")
    axes[idx].set_ylabel("Count")
    for j, v in enumerate([vc.get(0, 0), vc.get(1, 0)]):
        axes[idx].text(j, v + 2, str(v), ha='center', fontsize=10)
plt.suptitle("Class Distribution Across Datasets", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# 2. Missing value heatmap (top 50 most-missing features in MASTER)
miss_pct_master = master[al_cols + ek_cols].isnull().mean().sort_values(ascending=False)
top50_miss = miss_pct_master.head(50)
fig, ax = plt.subplots(figsize=(14, 6))
ax.barh(range(len(top50_miss)), top50_miss.values * 100, color='#FF9800')
ax.set_yticks(range(len(top50_miss)))
ax.set_yticklabels(top50_miss.index, fontsize=7)
ax.set_xlabel("Missing %")
ax.set_title("Top 50 Features by Missing Percentage (MASTER)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / "missing_values_top50.png", dpi=150, bbox_inches='tight')
plt.close()

# 3. Top 20 feature-target associations
top20 = assoc_df.head(20)
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#F44336' if d > 0 else '#2196F3' for d in top20["Effect Size (r)"].values]
ax.barh(range(len(top20)), top20["Abs Effect Size"].values, color=colors)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["Column"].values, fontsize=8)
ax.set_xlabel("Absolute Effect Size (rank-biserial r)")
ax.set_title("Top 20 Features by Class Separation (MASTER)")
ax.invert_yaxis()
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#F44336', label='Higher in Pathogenic'),
                   Patch(facecolor='#2196F3', label='Higher in Benign')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig(FIG_DIR / "top20_feature_target_association.png", dpi=150, bbox_inches='tight')
plt.close()

# 4. Feature type distribution pie
type_counts = desc_stats_df["Inferred Type"].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
       colors=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
ax.set_title("Feature Type Distribution (MASTER)")
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_type_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# 5. Correlation heatmap (top 30 most associated features)
top30_feats = assoc_df.head(30)["Column"].tolist()
top30_feats_avail = [f for f in top30_feats if f in master.columns]
if len(top30_feats_avail) > 5:
    corr_sub = master[top30_feats_avail].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_sub, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=True, yticklabels=True, ax=ax,
                fmt='.1f', square=True, linewidths=0.5)
    ax.set_title("Spearman Correlation: Top 30 Predictive Features")
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap_top30.png", dpi=150, bbox_inches='tight')
    plt.close()

# 6. Panel comparison: missingness profiles
fig, ax = plt.subplots(figsize=(10, 5))
for key, df in datasets.items():
    miss_profile = df[al_cols].isnull().mean().sort_values()
    ax.plot(range(len(miss_profile)), miss_profile.values * 100, label=key, alpha=0.8)
ax.set_xlabel("Feature Index (sorted by missingness)")
ax.set_ylabel("Missing %")
ax.set_title("Missingness Profile Across Datasets")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "panel_missingness_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# 7. EK features distribution by class
ek_numeric = [c for c in ek_cols if c in master.columns and master[c].dtype in [np.float64, np.int64, float]]
if ek_numeric:
    n_ek = len(ek_numeric)
    fig, axes = plt.subplots(1, min(n_ek, 6), figsize=(4 * min(n_ek, 6), 4))
    if n_ek == 1:
        axes = [axes]
    for idx, col in enumerate(ek_numeric[:6]):
        ax = axes[idx]
        master.groupby(target_col)[col].plot(kind='kde', ax=ax, legend=True)
        ax.set_title(col)
        ax.legend(["Benign", "Pathogenic"])
    plt.suptitle("EK Feature Distributions by Class", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ek_features_by_class.png", dpi=150, bbox_inches='tight')
    plt.close()

# 8. Duplicate Variant_ID analysis across datasets
print("\n  Checking Variant_ID overlaps...")
overlap_matrix = {}
for k1, d1 in datasets.items():
    overlap_matrix[k1] = {}
    for k2, d2 in datasets.items():
        overlap = len(set(d1["Variant_ID"]) & set(d2["Variant_ID"]))
        overlap_matrix[k1][k2] = overlap
overlap_df = pd.DataFrame(overlap_matrix)
print(overlap_df.to_string())
overlap_df.to_csv(REPORT_DIR / "variant_id_overlap.csv")

# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================
print("\n[CAT] Categorical feature analysis...")

cat_analysis = []
for key, df in datasets.items():
    for col in cat_cols + aa_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        n_unique = df[col].nunique(dropna=True)
        miss_pct = df[col].isnull().mean() * 100
        cat_analysis.append({
            "Dataset": key,
            "Column": col,
            "N Unique": n_unique,
            "Missing%": round(miss_pct, 2),
            "Top Value": str(vc.index[0]) if len(vc) > 0 else "",
            "Top Count": int(vc.values[0]) if len(vc) > 0 else 0,
            "Sample Values": str(list(df[col].dropna().unique()[:8]))
        })

cat_analysis_df = pd.DataFrame(cat_analysis)
cat_analysis_df.to_csv(REPORT_DIR / "categorical_analysis.csv", index=False)

# ============================================================
# CONFLICTING LABELS CHECK
# ============================================================
print("\n[CONFLICT] Checking for duplicate feature vectors with conflicting labels...")

for key, df in datasets.items():
    feat_cols = [c for c in df.columns if c not in ["Variant_ID", "Label"]]
    dup_feat = df[df.duplicated(subset=feat_cols, keep=False)]
    if len(dup_feat) > 0:
        conflicting = dup_feat.groupby(feat_cols, dropna=False)[target_col].nunique()
        n_conflict = (conflicting > 1).sum()
        print(f"  {key}: {len(dup_feat)} duplicate feature rows, {n_conflict} with conflicting labels")
    else:
        print(f"  {key}: No duplicate feature vectors found")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE. Generating reports...")
print("=" * 60)

# ============================================================
# MARKDOWN REPORTS
# ============================================================

def write_md(filename, content):
    with open(REPORT_DIR / filename, 'w', encoding='utf-8') as f:
        f.write(content)

# 1. Dataset Inventory
write_md("dataset_inventory.md", f"""# Dataset Inventory

## Project Files

| File | Type | Size (MB) | Rows | Columns | Role |
|------|------|-----------|------|---------|------|
{chr(10).join(f"| {r['File']} | {r['Type']} | {r['Size (MB)']} | {r['Rows']} | {r['Columns']} | {r['Role']} |" for _, r in file_info_df.iterrows())}

## Key Observations
- **Total datasets**: 4 (1 master + 3 panels)
- **Master dataset**: YARISMA_TRAIN_MASTER.csv ({datasets['MASTER'].shape[0]} variants, {datasets['MASTER'].shape[1]} features)
- **Panel datasets**: CFTR ({datasets['CFTR'].shape[0]} variants), PAH ({datasets['PAH'].shape[0]} variants), KANSER ({datasets['KANSER'].shape[0]} variants)
- **All datasets share the same column schema** ({datasets['MASTER'].shape[1]} columns)
- **Feature names are ANONYMIZED** (AL_1 to AL_334, CAT_1 to CAT_6, EK_1 to EK_9)

## Column Groups
| Group | Count | Description |
|-------|-------|-------------|
| AL_* | {len(al_cols)} | Anonymized numeric features |
| CAT_* | {len(cat_cols)} | Categorical features |
| EK_* | {len(ek_cols)} | Extra/additional features |
| AA_* | {len(aa_cols)} | Amino acid features |
| Variant_ID | 1 | Variant identifier |
| Label | 1 | Target (0=Benign, 1=Pathogenic) |

## DOCX Report Summary
The file `2026_Saglıkta_Yapay_Zeka_V01.docx` is the competition preparation report.
Key points from dataset context:
- Task: Binary classification of missense variants (Pathogenic vs Benign)
- Features are anonymized to prevent direct database lookup
- Panel datasets represent gene-specific subsets
""")

# 2. Schema Summary
write_md("dataset_schema_summary.md", f"""# Dataset Schema Summary

## Schema Overview

{schema_df.to_markdown(index=False)}

## Column Groups Detail

### AL Features (AL_1 to AL_{len(al_cols)})
- **Count**: {len(al_cols)} features
- **Type**: Mostly float64 (numeric)
- **Nature**: Anonymized. Many appear to be probability scores [0,1], others are continuous.
- **Missing**: Variable, some features have >80% missing

### CAT Features (CAT_1 to CAT_6)
- **Count**: {len(cat_cols)} features
- **Type**: Object/string
- **CAT_1**: Population/ancestry labels (gnomADe_EAS, gnomADe_FIN, gnomADg_AMR, SAS, etc.)
- **CAT_2**: AllofUs population labels (AllofUs_EAS, AllofUs_EUR, AllofUs_OTH, etc.)
- **CAT_3, CAT_4**: Genotype-like strings (C/C, T/T, G/G)
- **CAT_5**: Population/data source labels
- **CAT_6**: Similar to CAT_3/CAT_4

### EK Features (EK_1 to EK_9)
- **Count**: {len(ek_cols)} features
- **EK_1, EK_2**: Continuous scores (likely pathogenicity prediction scores)
- **EK_3**: Continuous, available for fewer samples
- **EK_4 to EK_9**: Various, some numeric, some with high missingness

### AA Features (AA_1, AA_2)
- **AA_1**: Reference amino acid (single letter code)
- **AA_2**: Alternate amino acid (single letter code)
- These encode the missense substitution

### Variant_ID
- Format: VAR_XXXXXX
- Unique identifier per variant
- NOT a feature for modeling

### Label
- Binary: 0 = Benign, 1 = Pathogenic
- Already encoded as integers
- No VUS/uncertain significance present

## Constant Columns
{chr(10).join(f"- **{key}**: {len(const_cols_per_ds[key])} constant columns" + (f" ({', '.join(const_cols_per_ds[key][:5])}{'...' if len(const_cols_per_ds[key]) > 5 else ''})" if const_cols_per_ds[key] else "") for key in const_cols_per_ds)}
""")

# 3. Class Distribution
write_md("class_distribution.md", f"""# Class Distribution Analysis

## Distribution Table

{class_dist_df.to_markdown(index=False)}

## Key Findings

1. **Binary labels**: The target is already binary (0=Benign, 1=Pathogenic). No VUS or uncertain classes exist.
2. **MASTER dataset**: {class_dist_df.iloc[0]['Positive Ratio']:.1%} pathogenic rate — {'balanced' if 0.4 < class_dist_df.iloc[0]['Positive Ratio'] < 0.6 else 'imbalanced'}
3. **CFTR panel**: {class_dist_df[class_dist_df['Dataset']=='CFTR']['Positive Ratio'].values[0]:.1%} pathogenic — {'high imbalance' if class_dist_df[class_dist_df['Dataset']=='CFTR']['Imbalance Ratio'].values[0] > 2 else 'relatively balanced'}
4. **PAH panel**: {class_dist_df[class_dist_df['Dataset']=='PAH']['Positive Ratio'].values[0]:.1%} pathogenic
5. **KANSER panel**: {class_dist_df[class_dist_df['Dataset']=='KANSER']['Positive Ratio'].values[0]:.1%} pathogenic

## Mapping Decision
No mapping was required. The labels are already:
- `0` = Benign
- `1` = Pathogenic

This was verified by inspecting the Label column across all datasets. No text-based labels, no multi-class structure.

## Implication for Modeling
- {'Class imbalance is moderate. Consider stratified sampling and class-weight adjustments.' if class_dist_df.iloc[0]['Imbalance Ratio'] > 1.5 else 'Classes are roughly balanced. Standard approaches should work.'}
- Panel-specific models may face severe imbalance in small panels.
- PR-AUC (Precision-Recall AUC) should be used alongside ROC-AUC.

![Class Distribution](figures/class_distribution.png)
""")

# 4. Missing Value Analysis
miss_bands_master = miss_summary_df[miss_summary_df["Dataset"] == "MASTER"]
band_counts = miss_bands_master["Band"].value_counts()
write_md("missing_value_analysis.md", f"""# Missing Value Analysis

## Missingness Bands (MASTER dataset)

| Band | Column Count |
|------|-------------|
| 0% | {band_counts.get('0%', 0)} |
| 0-5% | {band_counts.get('0-5%', 0)} |
| 5-20% | {band_counts.get('5-20%', 0)} |
| 20-50% | {band_counts.get('20-50%', 0)} |
| 50-80% | {band_counts.get('50-80%', 0)} |
| 80-95% | {band_counts.get('80-95%', 0)} |
| >95% | {band_counts.get('>95%', 0)} |

## Row-Level Missingness (MASTER)
- Mean: {master.isnull().mean(axis=1).mean()*100:.1f}%
- Median: {master.isnull().mean(axis=1).median()*100:.1f}%
- Max: {master.isnull().mean(axis=1).max()*100:.1f}%

## Differential Missingness by Target (Top 15)

Features where missingness rate differs most between Pathogenic and Benign variants:

{top_miss_diff.head(15).to_markdown(index=False)}

## Missingness Pattern Assessment

- **Random (MAR/MCAR)**: Many features show similar missingness across both classes, suggesting data availability issues.
- **Systematic (MNAR)**: Features with high differential missingness may have biologically-driven missing patterns (e.g., certain annotations only available for well-studied variants).
- **Panel-specific**: Panel datasets show different missingness profiles, especially in AL_1 to AL_26 region.

## Recommendations
- Features with >80% missing: candidates for removal unless they carry strong signal for available samples
- Features with 5-50% missing: impute carefully (median for numeric, mode for categorical; or use model-native handling)
- Features with <5% missing: safe to impute
- **Tree-based models** (XGBoost, LightGBM) can handle missing values natively — preferred approach

![Missing Values](figures/missing_values_top50.png)
![Panel Missingness](figures/panel_missingness_comparison.png)
""")

# 5. Distribution and Outlier Analysis
n_prob = len(desc_stats_df[desc_stats_df["Inferred Type"] == "probability/score [0,1]"])
n_binary = len(desc_stats_df[desc_stats_df["Inferred Type"] == "binary"])
n_generic = len(desc_stats_df[desc_stats_df["Inferred Type"] == "numeric"])
n_count = len(desc_stats_df[desc_stats_df["Inferred Type"] == "count/position-like"])

write_md("distribution_and_outlier_analysis.md", f"""# Distribution and Outlier Analysis

## Feature Type Summary (MASTER)

| Type | Count |
|------|-------|
| Probability/Score [0,1] | {n_prob} |
| Binary (0/1) | {n_binary} |
| Generic Numeric | {n_generic} |
| Count/Position-like | {n_count} |

## Key Statistics
- **Total numeric features analyzed**: {len(desc_stats_df)}
- **Highly skewed (|skew|>3)**: {len(highly_skewed)}
- **Features with extreme outliers**: {len(desc_stats_df[desc_stats_df['N Outliers (3xIQR)'] > 10])}

## Transformation Recommendations

| Feature Type | Recommended Transform | Notes |
|-------------|----------------------|-------|
| Probability scores [0,1] | None (already bounded) | May benefit from logit transform for linear models |
| Binary features | None | Already 0/1 |
| Count/position-like | log1p | Right-skewed distributions |
| Generic numeric | Standard scaling or quantile transform | For linear models; tree models don't need it |

## For Tree-Based Models (Recommended)
- **No transformation needed** — tree models are invariant to monotonic transforms
- Focus on handling missing values (native support in XGBoost/LightGBM)
- Binary features act as implicit splits

## Detailed Statistics
See `feature_statistics.csv` for per-feature statistics.

![Feature Type Distribution](figures/feature_type_distribution.png)
""")

# 6. Feature-Target Association
write_md("feature_target_association.md", f"""# Feature-Target Association Analysis

## Method
- **Test**: Mann-Whitney U (non-parametric, robust to non-normality)
- **Effect Size**: Rank-biserial correlation (r)
- **Interpretation**: |r| > 0.3 = medium, |r| > 0.5 = large

## Top 20 Features Associated with Pathogenic Class (higher median in pathogenic)

{top20_path[['Column', 'Median Pathogenic', 'Median Benign', 'Median Diff (P-B)', 'Abs Effect Size', 'Mann-Whitney p-value']].to_markdown(index=False)}

## Top 20 Features Associated with Benign Class (higher median in benign)

{top20_ben[['Column', 'Median Pathogenic', 'Median Benign', 'Median Diff (P-B)', 'Abs Effect Size', 'Mann-Whitney p-value']].to_markdown(index=False)}

## Nearly Useless Features (lowest effect size)

{weak_feats[['Column', 'Abs Effect Size', 'Mann-Whitney p-value']].head(10).to_markdown(index=False)}

## Summary Statistics
- **Total features tested**: {len(assoc_df)}
- **Significant at p<0.001**: {(assoc_df['Mann-Whitney p-value'] < 0.001).sum()}
- **Medium+ effect size (|r|>0.3)**: {(assoc_df['Abs Effect Size'] > 0.3).sum()}
- **Large effect size (|r|>0.5)**: {(assoc_df['Abs Effect Size'] > 0.5).sum()}

## Caution
These are **associations only**, not causal relationships. Feature importance in a trained model may differ substantially from univariate associations.

![Top 20 Features](figures/top20_feature_target_association.png)
""")

# 7. Correlation Redundancy
n_95 = (corr_pairs_df['Spearman r'].abs() > 0.95).sum() if len(corr_pairs_df) > 0 else 0
n_85 = len(corr_pairs_df)

write_md("correlation_redundancy_analysis.md", f"""# Correlation and Redundancy Analysis

## Summary
- **Pairs with |r| > 0.85**: {n_85}
- **Pairs with |r| > 0.95**: {n_95}

## Top Redundant Pairs

{corr_pairs_df.head(30).to_markdown(index=False) if len(corr_pairs_df) > 0 else "No pairs found with |r| > 0.85"}

## Recommendations
- Pairs with |r| > 0.95 are nearly perfectly correlated — one can likely be removed without information loss
- Pairs with 0.85 < |r| < 0.95 should be monitored but may carry complementary signal
- **Do not automatically drop** — tree models can handle correlated features
- For linear models or neural networks, consider PCA or feature selection

![Correlation Heatmap](figures/correlation_heatmap_top30.png)
""")

# 8. Panel Comparison
write_md("panel_comparison.md", f"""# Panel-Level Comparison

## Panel Overview

{panel_df.to_markdown(index=False)}

## Variant ID Overlaps

{overlap_df.to_markdown()}

## Distribution Shifts

Significant distribution shifts detected (KS test, p<0.001) between MASTER and panels:

{n_sig_shifts.to_frame('N Features Shifted').to_markdown()}

## Panel Difficulty Assessment

### CFTR Panel
- **Size**: {datasets['CFTR'].shape[0]} variants — **very small**
- **Challenge**: Limited sample size makes reliable model training difficult
- Cross-validation with few folds (e.g., 5-fold) may have high variance

### PAH Panel
- **Size**: {datasets['PAH'].shape[0]} variants — small
- **Challenge**: Moderate sample, distribution may differ from MASTER

### KANSER Panel
- **Size**: {datasets['KANSER'].shape[0]} variants — small
- **Challenge**: Cancer gene panel, potentially different feature distributions

## Panel Strategy Recommendations
1. **Global-first approach**: Train on MASTER, evaluate on panels
2. **Fine-tuning**: Optionally fine-tune or calibrate on panel-specific data
3. **Panel-specific models**: Only if global model fails significantly on panels
4. **Feature selection may differ** across panels due to missingness patterns

![Panel Missingness](figures/panel_missingness_comparison.png)
""")

# 9. Leakage Risk Register
write_md("leakage_risk_register.md", f"""# Leakage and Validity Risk Register

## Risk Summary
- **HIGH severity**: {(leakage_df['Severity'] == 'HIGH').sum()}
- **MEDIUM severity**: {(leakage_df['Severity'] == 'MEDIUM').sum()}
- **LOW severity**: {(leakage_df['Severity'] == 'LOW').sum()}

## Full Risk Register

{leakage_df.to_markdown(index=False)}

## Critical Warnings

1. **EK_1, EK_2 are suspicious**: These features have a continuous range that resembles pathogenicity prediction scores. If they are pre-computed from the same label we're predicting, they are **circular predictors** and will inflate accuracy unrealistically.

2. **Variant_ID overlap between MASTER and panels**: Shared variants could cause data leakage if the same variant appears in both training and test data during evaluation.

3. **Feature anonymization**: We cannot verify whether any AL_* feature directly encodes clinical significance databases (ClinVar, HGMD) that would make the prediction trivially easy.

## Mitigation Strategy
- Build models **with and without EK features** to quantify their impact
- Ensure strict Variant_ID-based splits to prevent leakage
- Report both EK-included and EK-excluded performance metrics
""")

# 10. Executive Summary
write_md("phase_01_executive_summary.md", f"""# Phase 01: Executive Summary

## 1. What Data We Have
- **4 datasets**: 1 master ({datasets['MASTER'].shape[0]} variants) + 3 gene panels (CFTR: {datasets['CFTR'].shape[0]}, PAH: {datasets['PAH'].shape[0]}, KANSER: {datasets['KANSER'].shape[0]})
- **{datasets['MASTER'].shape[1]} columns** per dataset (shared schema)
- **{len(al_cols)} anonymized numeric features** (AL_*), 6 categorical, 9 extra features, 2 amino acid features
- All feature names are **anonymized** — true biological meaning is hidden

## 2. Target Label
- Binary: 0 = Benign, 1 = Pathogenic
- Already integer-encoded, no mapping needed
- No VUS or uncertain significance labels

## 3. Class Imbalance
{class_dist_df.to_markdown(index=False)}

## 4. Main Missing Value Risks
- Many features have substantial missingness (20-80%)
- Missingness patterns differ between panels
- Some features show differential missingness between classes (potential MNAR)
- Tree-based models with native missing value handling are strongly preferred

## 5. Feature Groups
| Group | Count | Nature |
|-------|-------|--------|
| Probability/Score [0,1] | {n_prob} | Likely pathogenicity prediction scores |
| Binary (0/1) | {n_binary} | Threshold-based indicators |
| Continuous | {n_generic} | Various numeric features |
| Count/Position-like | {n_count} | Possibly allele counts or genomic positions |
| Categorical | {len(cat_cols)} | Population/ancestry, genotype labels |
| Amino Acid | 2 | Reference and alternate amino acids |

## 6. Strongest Early Signals
Top 5 features by effect size:
{assoc_df.head(5)[['Column', 'Abs Effect Size', 'Mann-Whitney p-value']].to_markdown(index=False)}

## 7. Biggest Leakage Risks
1. **EK_1, EK_2** may be pre-computed pathogenicity scores (circular prediction risk)
2. **Variant_ID overlap** between MASTER and panels
3. **Anonymized features** prevent verification of feature provenance

## 8. Panel-Specific Challenges
- **CFTR**: Very small (n={datasets['CFTR'].shape[0]}), high variance in cross-validation
- **All panels**: Distribution shifts detected from MASTER
- **Missingness patterns differ** across panels, especially for early AL features

## 9. Recommended Phase 2 Strategy

### Baseline Model
- **LightGBM or XGBoost** with default hyperparameters
- Handles missing values natively
- Fast training, good for iteration

### Strongest Model Family
- **Gradient Boosted Trees** (LightGBM, XGBoost, CatBoost)
- Consider **stacking ensemble** with multiple tree-based models

### Validation Strategy
- **Stratified K-Fold** (5-fold) on MASTER
- **Variant_ID-aware splitting** to prevent leakage
- Panel-specific evaluation separate from MASTER

### Metrics
- **Primary**: ROC-AUC
- **Secondary**: PR-AUC, F1, Sensitivity, Specificity
- **Clinical**: Optimize sensitivity (minimize false negatives for pathogenic variants)

### Thresholding
- Default threshold 0.5 for baseline
- Optimize threshold using PR curve for sensitivity-focused classification
- Consider asymmetric cost matrix (FN for pathogenic is worse than FP)

### Calibration
- Platt scaling or isotonic regression post-training
- Important if probability outputs are used for clinical decision-making

### Explainability
- SHAP values for global and local feature importance
- Partial dependence plots for top features

### Panel Strategy
- Train global model on MASTER
- Evaluate per-panel performance
- If panel performance is poor: fine-tune with panel data or train panel-specific models

## 10. B-Plan (If Dataset Is Too Noisy)
1. Aggressive feature selection (top 50-100 features by effect size)
2. Reduce to features with <30% missingness
3. Train separate models per panel
4. Ensemble: blend global + panel-specific predictions
5. If performance remains poor: focus on well-covered features and accept lower accuracy on high-missingness variants
""")

print("\n[REPORTS] All Markdown reports generated.")

# ============================================================
# WORD DOCUMENT
# ============================================================
print("\n[WORD] Generating Word document...")

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

doc = Document()

# Styles
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    return h

def add_table_from_df(doc, df, max_rows=None):
    if max_rows:
        df = df.head(max_rows)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
    return table

# Title
title = doc.add_heading('TEKNOFEST Healthcare AI\nPhase 01: Data Understanding Report', 0)
doc.add_paragraph(f'Generated: 2026-05-24')
doc.add_paragraph('Missense Variant Pathogenicity Classification — Exploratory Data Analysis')
doc.add_page_break()

# Table of Contents placeholder
add_heading(doc, 'Table of Contents', 1)
doc.add_paragraph('1. Dataset Inventory')
doc.add_paragraph('2. Schema Summary')
doc.add_paragraph('3. Class Distribution')
doc.add_paragraph('4. Missing Value Analysis')
doc.add_paragraph('5. Distribution and Outlier Analysis')
doc.add_paragraph('6. Feature-Target Association')
doc.add_paragraph('7. Correlation and Redundancy')
doc.add_paragraph('8. Panel Comparison')
doc.add_paragraph('9. Leakage Risk Register')
doc.add_paragraph('10. Executive Summary and Recommendations')
doc.add_page_break()

# 1. Dataset Inventory
add_heading(doc, '1. Dataset Inventory', 1)
doc.add_paragraph('The project contains 4 CSV datasets: 1 master training set and 3 gene-specific panel datasets.')
add_table_from_df(doc, file_info_df)
doc.add_paragraph('')
doc.add_paragraph(f'All datasets share the same column schema with {datasets["MASTER"].shape[1]} columns.')
doc.add_paragraph(f'Feature names are ANONYMIZED (AL_1 to AL_{len(al_cols)}, CAT_1 to CAT_6, EK_1 to EK_9).')

# Column groups
add_heading(doc, 'Column Groups', 2)
cg_df = pd.DataFrame([
    {"Group": "AL_*", "Count": len(al_cols), "Description": "Anonymized numeric features"},
    {"Group": "CAT_*", "Count": len(cat_cols), "Description": "Categorical features"},
    {"Group": "EK_*", "Count": len(ek_cols), "Description": "Extra/additional features"},
    {"Group": "AA_*", "Count": len(aa_cols), "Description": "Amino acid features"},
    {"Group": "Variant_ID", "Count": 1, "Description": "Variant identifier"},
    {"Group": "Label", "Count": 1, "Description": "Target (0=Benign, 1=Pathogenic)"},
])
add_table_from_df(doc, cg_df)
doc.add_page_break()

# 2. Schema Summary
add_heading(doc, '2. Schema Summary', 1)
add_table_from_df(doc, schema_df)
doc.add_paragraph('')
doc.add_paragraph('CAT_1: Population/ancestry (gnomAD, AllofUs labels)')
doc.add_paragraph('CAT_2: AllofUs population annotations')
doc.add_paragraph('CAT_3, CAT_4: Genotype strings (e.g., C/C, T/T)')
doc.add_paragraph('AA_1, AA_2: Reference and alternate amino acids (single-letter codes)')
doc.add_page_break()

# 3. Class Distribution
add_heading(doc, '3. Class Distribution', 1)
add_table_from_df(doc, class_dist_df)
doc.add_paragraph('')
doc.add_paragraph('Labels are binary (0=Benign, 1=Pathogenic). No VUS or uncertain labels.')
try:
    doc.add_picture(str(FIG_DIR / "class_distribution.png"), width=Inches(6))
except:
    doc.add_paragraph('[Figure: class_distribution.png]')
doc.add_page_break()

# 4. Missing Value Analysis
add_heading(doc, '4. Missing Value Analysis', 1)
band_summary = pd.DataFrame({
    "Band": ["0%", "0-5%", "5-20%", "20-50%", "50-80%", "80-95%", ">95%"],
    "Columns": [band_counts.get(b, 0) for b in ["0%", "0-5%", "5-20%", "20-50%", "50-80%", "80-95%", ">95%"]]
})
add_table_from_df(doc, band_summary)
doc.add_paragraph('')

add_heading(doc, 'Differential Missingness by Target (Top 15)', 2)
add_table_from_df(doc, top_miss_diff.head(15))

try:
    doc.add_picture(str(FIG_DIR / "missing_values_top50.png"), width=Inches(6))
except:
    doc.add_paragraph('[Figure: missing_values_top50.png]')
try:
    doc.add_picture(str(FIG_DIR / "panel_missingness_comparison.png"), width=Inches(6))
except:
    doc.add_paragraph('[Figure: panel_missingness_comparison.png]')
doc.add_page_break()

# 5. Distribution and Outlier Analysis
add_heading(doc, '5. Distribution and Outlier Analysis', 1)
type_summary = pd.DataFrame({
    "Feature Type": ["Probability/Score [0,1]", "Binary (0/1)", "Generic Numeric", "Count/Position-like"],
    "Count": [n_prob, n_binary, n_generic, n_count]
})
add_table_from_df(doc, type_summary)
doc.add_paragraph(f'Highly skewed features (|skew|>3): {len(highly_skewed)}')

try:
    doc.add_picture(str(FIG_DIR / "feature_type_distribution.png"), width=Inches(4.5))
except:
    doc.add_paragraph('[Figure: feature_type_distribution.png]')
doc.add_page_break()

# 6. Feature-Target Association
add_heading(doc, '6. Feature-Target Association', 1)
doc.add_paragraph('Method: Mann-Whitney U test with rank-biserial effect size.')
doc.add_paragraph(f'Total features tested: {len(assoc_df)}')
doc.add_paragraph(f'Significant at p<0.001: {(assoc_df["Mann-Whitney p-value"] < 0.001).sum()}')
doc.add_paragraph(f'Large effect size (|r|>0.5): {(assoc_df["Abs Effect Size"] > 0.5).sum()}')

add_heading(doc, 'Top 20 Features by Effect Size', 2)
top20_display = assoc_df.head(20)[['Column', 'Median Diff (P-B)', 'Abs Effect Size', 'Mann-Whitney p-value']].copy()
top20_display['Mann-Whitney p-value'] = top20_display['Mann-Whitney p-value'].apply(lambda x: f"{x:.2e}" if x is not None else "")
add_table_from_df(doc, top20_display)

try:
    doc.add_picture(str(FIG_DIR / "top20_feature_target_association.png"), width=Inches(6))
except:
    doc.add_paragraph('[Figure: top20_feature_target_association.png]')
doc.add_page_break()

# 7. Correlation and Redundancy
add_heading(doc, '7. Correlation and Redundancy', 1)
doc.add_paragraph(f'Pairs with |r| > 0.85: {n_85}')
doc.add_paragraph(f'Pairs with |r| > 0.95: {n_95}')

if len(corr_pairs_df) > 0:
    add_heading(doc, 'Top Redundant Pairs', 2)
    add_table_from_df(doc, corr_pairs_df, max_rows=20)

try:
    doc.add_picture(str(FIG_DIR / "correlation_heatmap_top30.png"), width=Inches(5.5))
except:
    doc.add_paragraph('[Figure: correlation_heatmap_top30.png]')
doc.add_page_break()

# 8. Panel Comparison
add_heading(doc, '8. Panel Comparison', 1)
add_table_from_df(doc, panel_df)
doc.add_paragraph('')

add_heading(doc, 'Variant ID Overlap Matrix', 2)
overlap_display = overlap_df.reset_index().rename(columns={"index": "Dataset"})
add_table_from_df(doc, overlap_display)
doc.add_paragraph('')

add_heading(doc, 'Distribution Shifts', 2)
for p, n in n_sig_shifts.items():
    doc.add_paragraph(f'{p}: {int(n)} features with significant distribution shift from MASTER')
doc.add_page_break()

# 9. Leakage Risk Register
add_heading(doc, '9. Leakage Risk Register', 1)
doc.add_paragraph(f'HIGH severity risks: {(leakage_df["Severity"] == "HIGH").sum()}')
doc.add_paragraph(f'MEDIUM severity risks: {(leakage_df["Severity"] == "MEDIUM").sum()}')
doc.add_paragraph(f'LOW severity risks: {(leakage_df["Severity"] == "LOW").sum()}')
add_table_from_df(doc, leakage_df[leakage_df["Severity"].isin(["HIGH", "MEDIUM"])][["Risk", "Severity", "Mitigation"]])
doc.add_page_break()

# 10. Executive Summary
add_heading(doc, '10. Executive Summary and Recommendations', 1)

add_heading(doc, 'Key Findings', 2)
doc.add_paragraph(f'1. We have {datasets["MASTER"].shape[0]} master variants + 3 panels ({datasets["CFTR"].shape[0]} + {datasets["PAH"].shape[0]} + {datasets["KANSER"].shape[0]}) with {len(al_cols)} anonymized features.')
doc.add_paragraph(f'2. Target is binary (Pathogenic/Benign) with positive ratio ~{class_dist_df.iloc[0]["Positive Ratio"]:.1%} in MASTER.')
doc.add_paragraph(f'3. {(assoc_df["Abs Effect Size"] > 0.3).sum()} features show medium-to-large effect sizes.')
doc.add_paragraph('4. EK_1, EK_2 are suspicious (potential pre-computed pathogenicity scores).')
doc.add_paragraph('5. Panel datasets have distribution shifts and different missingness patterns.')

add_heading(doc, 'Recommended Phase 2 Strategy', 2)
doc.add_paragraph('Baseline: LightGBM/XGBoost with native missing value handling')
doc.add_paragraph('Validation: Stratified 5-fold CV with Variant_ID-aware splits')
doc.add_paragraph('Primary metric: ROC-AUC; Secondary: PR-AUC, Sensitivity')
doc.add_paragraph('Panel strategy: Global model first, panel-specific fine-tuning if needed')
doc.add_paragraph('Explainability: SHAP values')
doc.add_paragraph('Run models with AND without EK features to assess leakage risk')

add_heading(doc, 'Top 5 Risks', 2)
doc.add_paragraph('1. EK features may cause circular prediction (label leakage)')
doc.add_paragraph('2. Small panel sizes lead to high variance estimates')
doc.add_paragraph('3. Anonymized features prevent feature validation')
doc.add_paragraph('4. Distribution shift between MASTER and panels')
doc.add_paragraph('5. High missingness in some features may correlate with label')

# Save Word
word_path = REPORT_DIR / "Phase_01_EDA_Report.docx"
doc.save(str(word_path))
print(f"  Word document saved: {word_path}")

# ============================================================
# JUPYTER NOTEBOOK
# ============================================================
print("\n[NOTEBOOK] Generating Jupyter Notebook...")

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

cells = []

# Title
cells.append(new_markdown_cell("""# TEKNOFEST Healthcare AI — Phase 01: Exploratory Data Analysis
## Missense Variant Pathogenicity Classification

**Objective**: Understand the dataset deeply before any modeling.

**Datasets**:
- MASTER: General training set
- CFTR: Cystic Fibrosis gene panel
- PAH: Phenylalanine Hydroxylase gene panel
- KANSER: Cancer gene panel
"""))

# Setup
cells.append(new_code_cell("""import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
%matplotlib inline

print("Libraries loaded successfully.")
"""))

# Load data
cells.append(new_markdown_cell("## A. Dataset Loading"))
cells.append(new_code_cell("""from pathlib import Path
from collections import OrderedDict

DATA_DIR = Path("EĞİTİM (TRAIN) SETLERİ 2")

datasets = OrderedDict()
for fname in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(fname)
    key = fname.stem.replace("YARISMA_TRAIN_", "")
    datasets[key] = df
    print(f"{key}: {df.shape[0]} rows x {df.shape[1]} cols — {fname.stat().st_size / 1024:.0f} KB")

master = datasets["MASTER"]
"""))

# Column groups
cells.append(new_markdown_cell("## B. Schema Inspection"))
cells.append(new_code_cell("""al_cols = [c for c in master.columns if c.startswith("AL_")]
cat_cols = [c for c in master.columns if c.startswith("CAT_")]
ek_cols = [c for c in master.columns if c.startswith("EK_")]
aa_cols = [c for c in master.columns if c.startswith("AA_")]

print(f"AL features: {len(al_cols)}")
print(f"CAT features: {len(cat_cols)}")
print(f"EK features: {len(ek_cols)}")
print(f"AA features: {len(aa_cols)}")
print(f"Other: Variant_ID, Label")
print(f"\\nTotal columns: {master.shape[1]}")

print("\\n--- Data Types ---")
print(master.dtypes.value_counts())

print("\\n--- Constant columns per dataset ---")
for key, df in datasets.items():
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    print(f"  {key}: {len(const)} constant columns")

print("\\n--- Duplicate rows per dataset ---")
for key, df in datasets.items():
    print(f"  {key}: {df.duplicated().sum()} duplicate rows")
"""))

# Dtypes table
cells.append(new_code_cell("""print("\\n--- Schema overview (first 5 + last 5 columns) ---")
schema = pd.DataFrame({
    "Column": master.columns,
    "Dtype": master.dtypes.astype(str),
    "Non-Null": master.notna().sum(),
    "Missing%": (master.isnull().mean() * 100).round(2),
    "Unique": master.nunique()
})
display(pd.concat([schema.head(10), schema.tail(10)]))
"""))

# Target Analysis
cells.append(new_markdown_cell("## C. Target Label Analysis"))
cells.append(new_code_cell("""class_dist = []
for key, df in datasets.items():
    vc = df["Label"].value_counts()
    total = len(df)
    n_path = int(vc.get(1, 0))
    n_ben = int(vc.get(0, 0))
    pos_ratio = n_path / total
    imb_ratio = max(n_path, n_ben) / min(n_path, n_ben) if min(n_path, n_ben) > 0 else float('inf')
    class_dist.append({
        "Dataset": key, "Total": total,
        "Pathogenic (1)": n_path, "Benign (0)": n_ben,
        "Positive Ratio": round(pos_ratio, 4),
        "Imbalance Ratio": round(imb_ratio, 2)
    })

class_dist_df = pd.DataFrame(class_dist)
display(class_dist_df)
"""))

cells.append(new_code_cell("""fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for idx, (key, df) in enumerate(datasets.items()):
    vc = df["Label"].value_counts().sort_index()
    axes[idx].bar(["Benign (0)", "Pathogenic (1)"], [vc.get(0, 0), vc.get(1, 0)],
                  color=["#2196F3", "#F44336"])
    axes[idx].set_title(f"{key} (n={len(df)})")
    axes[idx].set_ylabel("Count")
    for j, v in enumerate([vc.get(0, 0), vc.get(1, 0)]):
        axes[idx].text(j, v + 2, str(v), ha='center', fontsize=10)
plt.suptitle("Class Distribution Across Datasets", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

# Missing Values
cells.append(new_markdown_cell("## D. Missing Value Analysis"))
cells.append(new_code_cell("""for key, df in datasets.items():
    miss_pct = df.isnull().mean() * 100
    bands = {
        "0%": (miss_pct == 0).sum(),
        "0-5%": ((miss_pct > 0) & (miss_pct <= 5)).sum(),
        "5-20%": ((miss_pct > 5) & (miss_pct <= 20)).sum(),
        "20-50%": ((miss_pct > 20) & (miss_pct <= 50)).sum(),
        "50-80%": ((miss_pct > 50) & (miss_pct <= 80)).sum(),
        "80-95%": ((miss_pct > 80) & (miss_pct <= 95)).sum(),
        ">95%": (miss_pct > 95).sum()
    }
    print(f"\\n--- {key} ---")
    for band, cnt in bands.items():
        print(f"  {band}: {cnt} columns")

    row_miss = df.isnull().mean(axis=1) * 100
    print(f"  Row missingness: mean={row_miss.mean():.1f}%, median={row_miss.median():.1f}%, max={row_miss.max():.1f}%")
"""))

cells.append(new_code_cell("""# Top 50 most-missing features in MASTER
miss_pct_master = master[al_cols + ek_cols].isnull().mean().sort_values(ascending=False)
top50 = miss_pct_master.head(50)

fig, ax = plt.subplots(figsize=(14, 8))
ax.barh(range(len(top50)), top50.values * 100, color='#FF9800')
ax.set_yticks(range(len(top50)))
ax.set_yticklabels(top50.index, fontsize=7)
ax.set_xlabel("Missing %")
ax.set_title("Top 50 Features by Missing Percentage (MASTER)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""))

# Differential missingness
cells.append(new_code_cell("""# Differential missingness by target
path_mask = master["Label"] == 1
ben_mask = master["Label"] == 0

diff_miss = []
for col in al_cols + ek_cols:
    miss_path = master.loc[path_mask, col].isnull().mean()
    miss_ben = master.loc[ben_mask, col].isnull().mean()
    diff_miss.append({
        "Column": col,
        "Miss% Pathogenic": round(miss_path * 100, 2),
        "Miss% Benign": round(miss_ben * 100, 2),
        "Abs Diff": round(abs(miss_path - miss_ben) * 100, 2)
    })

diff_miss_df = pd.DataFrame(diff_miss).sort_values("Abs Diff", ascending=False)
print("Top 15 features with differential missingness:")
display(diff_miss_df.head(15))
"""))

# Panel missingness comparison
cells.append(new_code_cell("""fig, ax = plt.subplots(figsize=(12, 5))
for key, df in datasets.items():
    miss_profile = df[al_cols].isnull().mean().sort_values()
    ax.plot(range(len(miss_profile)), miss_profile.values * 100, label=key, alpha=0.8)
ax.set_xlabel("Feature Index (sorted by missingness)")
ax.set_ylabel("Missing %")
ax.set_title("Missingness Profile Across Datasets")
ax.legend()
plt.tight_layout()
plt.show()
"""))

# Distribution Analysis
cells.append(new_markdown_cell("## E. Distribution and Outlier Analysis"))
cells.append(new_code_cell("""numeric_feats = [c for c in al_cols + ek_cols
                 if c in master.columns and master[c].dtype in [np.float64, np.int64, float, int]]

desc_stats = []
for col in numeric_feats:
    s = master[col].dropna()
    if len(s) == 0:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()

    feat_type = "numeric"
    if s.min() >= 0 and s.max() <= 1.001:
        feat_type = "probability/score [0,1]"
    elif s.nunique() == 2:
        feat_type = "binary"
    elif s.min() >= 0 and s.max() > 1000:
        feat_type = "count/position-like"

    desc_stats.append({
        "Column": col,
        "Count": len(s),
        "Missing%": round(master[col].isnull().mean()*100, 2),
        "Mean": round(s.mean(), 6),
        "Std": round(s.std(), 6),
        "Min": round(s.min(), 6),
        "Max": round(s.max(), 6),
        "Skewness": round(float(s.skew()), 4),
        "Inferred Type": feat_type
    })

desc_df = pd.DataFrame(desc_stats)
print("Feature type breakdown:")
display(desc_df["Inferred Type"].value_counts().to_frame("Count"))
print(f"\\nHighly skewed (|skew|>3): {(desc_df['Skewness'].abs() > 3).sum()}")
"""))

cells.append(new_code_cell("""# Feature type pie chart
type_counts = desc_df["Inferred Type"].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
       colors=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
ax.set_title("Feature Type Distribution (MASTER)")
plt.tight_layout()
plt.show()
"""))

# Feature-Target Association
cells.append(new_markdown_cell("## F. Feature-Target Association"))
cells.append(new_code_cell("""path_idx = master["Label"] == 1
ben_idx = master["Label"] == 0

assoc_results = []
for col in numeric_feats:
    s_path = master.loc[path_idx, col].dropna()
    s_ben = master.loc[ben_idx, col].dropna()
    if len(s_path) < 5 or len(s_ben) < 5:
        continue
    med_diff = s_path.median() - s_ben.median()
    try:
        u_stat, p_val = mannwhitneyu(s_path, s_ben, alternative='two-sided')
        n1, n2 = len(s_path), len(s_ben)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
    except:
        p_val, effect_size = np.nan, np.nan

    assoc_results.append({
        "Column": col,
        "Med Pathogenic": round(float(s_path.median()), 6),
        "Med Benign": round(float(s_ben.median()), 6),
        "Med Diff": round(float(med_diff), 6),
        "p-value": float(p_val),
        "Effect Size (r)": round(float(effect_size), 4),
        "Abs Effect": round(abs(float(effect_size)), 4)
    })

assoc_df = pd.DataFrame(assoc_results).sort_values("Abs Effect", ascending=False)
print(f"Total features tested: {len(assoc_df)}")
print(f"Significant (p<0.001): {(assoc_df['p-value'] < 0.001).sum()}")
print(f"Large effect (|r|>0.5): {(assoc_df['Abs Effect'] > 0.5).sum()}")
print("\\nTop 20 features:")
display(assoc_df.head(20))
"""))

cells.append(new_code_cell("""# Top 20 barplot
top20 = assoc_df.head(20)
fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#F44336' if d > 0 else '#2196F3' for d in top20["Effect Size (r)"].values]
ax.barh(range(len(top20)), top20["Abs Effect"].values, color=colors)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["Column"].values, fontsize=9)
ax.set_xlabel("Absolute Effect Size (rank-biserial r)")
ax.set_title("Top 20 Features by Class Separation")
ax.invert_yaxis()
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#F44336', label='Higher in Pathogenic'),
                   Patch(facecolor='#2196F3', label='Higher in Benign')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.show()
"""))

# Correlation
cells.append(new_markdown_cell("## G. Correlation and Redundancy"))
cells.append(new_code_cell("""# Correlation heatmap of top 30 features
top30_feats = assoc_df.head(30)["Column"].tolist()
top30_avail = [f for f in top30_feats if f in master.columns]

corr_sub = master[top30_avail].corr(method='spearman')
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_sub, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=True, yticklabels=True, ax=ax, square=True, linewidths=0.5)
ax.set_title("Spearman Correlation: Top 30 Predictive Features")
plt.xticks(fontsize=7, rotation=90)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.show()
"""))

cells.append(new_code_cell("""# Count highly correlated pairs
feat_for_corr = [c for c in numeric_feats if master[c].notna().sum() > 100]
corr_full = master[feat_for_corr].corr(method='spearman')

pairs_85 = 0
pairs_95 = 0
for i in range(len(feat_for_corr)):
    for j in range(i+1, len(feat_for_corr)):
        r = abs(corr_full.iloc[i, j])
        if r > 0.95:
            pairs_95 += 1
        if r > 0.85:
            pairs_85 += 1

print(f"Pairs with |r| > 0.85: {pairs_85}")
print(f"Pairs with |r| > 0.95: {pairs_95}")
"""))

# Panel Comparison
cells.append(new_markdown_cell("## H. Panel Comparison"))
cells.append(new_code_cell("""panel_comp = []
for key, df in datasets.items():
    n_path = (df["Label"] == 1).sum()
    n_ben = (df["Label"] == 0).sum()
    miss_al = df[al_cols].isnull().mean().mean() * 100
    n_const = sum(df[c].nunique(dropna=True) <= 1 for c in df.columns)
    panel_comp.append({
        "Dataset": key,
        "Rows": len(df),
        "Pathogenic": n_path,
        "Benign": n_ben,
        "Pos Ratio": round(n_path/len(df), 3),
        "Mean Missing% (AL)": round(miss_al, 1),
        "Constant Cols": n_const
    })

display(pd.DataFrame(panel_comp))

# Variant_ID overlaps
print("\\nVariant_ID Overlap Matrix:")
overlap = {}
for k1, d1 in datasets.items():
    overlap[k1] = {}
    for k2, d2 in datasets.items():
        overlap[k1][k2] = len(set(d1["Variant_ID"]) & set(d2["Variant_ID"]))
display(pd.DataFrame(overlap))
"""))

# EK features
cells.append(new_markdown_cell("## EK Features Deep Dive (Leakage Check)"))
cells.append(new_code_cell("""ek_numeric = [c for c in ek_cols if c in master.columns and master[c].dtype in [np.float64, float]]
n_ek = len(ek_numeric)
if n_ek > 0:
    fig, axes = plt.subplots(1, min(n_ek, 6), figsize=(4 * min(n_ek, 6), 4))
    if n_ek == 1:
        axes = [axes]
    for idx, col in enumerate(ek_numeric[:6]):
        ax = axes[idx]
        master[master["Label"]==0][col].dropna().plot(kind='kde', ax=ax, label='Benign', color='#2196F3')
        master[master["Label"]==1][col].dropna().plot(kind='kde', ax=ax, label='Pathogenic', color='#F44336')
        ax.set_title(col)
        ax.legend()
    plt.suptitle("EK Feature Distributions by Class", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

for col in ek_numeric:
    s = master[col].dropna()
    print(f"{col}: range [{s.min():.3f}, {s.max():.3f}], mean={s.mean():.3f}, missing={master[col].isnull().mean()*100:.1f}%")
"""))

# Leakage
cells.append(new_markdown_cell("## I. Leakage Risk Register"))
cells.append(new_code_cell("""leakage_risks = pd.read_csv("reports/phase_01_data_understanding/leakage_risk_register.csv")
display(leakage_risks[["Risk", "Severity", "Mitigation"]])
"""))

# Recommendations
cells.append(new_markdown_cell("""## J. Phase 2 Recommendations

### Baseline Model
- **LightGBM or XGBoost** with default hyperparameters
- Native missing value handling

### Validation
- **Stratified 5-Fold CV** on MASTER
- Variant_ID-aware splitting

### Metrics
- **Primary**: ROC-AUC
- **Secondary**: PR-AUC, Sensitivity, F1
- Sensitivity-focused thresholding

### Strategy
1. Train global model on MASTER
2. Evaluate per-panel performance
3. Build models WITH and WITHOUT EK features
4. If panel performance is poor: fine-tune panel-specific models

### Risks to Monitor
1. EK features may cause circular prediction
2. Small panel sizes = high variance
3. Distribution shift between MASTER and panels
4. Anonymized features prevent validation
5. Differential missingness may confound results
"""))

# CAT analysis
cells.append(new_markdown_cell("## Categorical Feature Analysis"))
cells.append(new_code_cell("""for col in cat_cols + aa_cols:
    print(f"\\n--- {col} ---")
    print(f"  Unique values: {master[col].nunique(dropna=True)}")
    print(f"  Missing: {master[col].isnull().mean()*100:.1f}%")
    print(f"  Sample values: {list(master[col].dropna().unique()[:10])}")
    print(f"  Value counts (top 5):")
    print(master[col].value_counts().head(5).to_string())
"""))

nb.cells = cells

nb_path = REPORT_DIR / "Phase_01_EDA_Notebook.ipynb"
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"  Notebook saved: {nb_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"\nOutput directory: {REPORT_DIR}")
print("\nGenerated files:")
for f in sorted(REPORT_DIR.glob("*")):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    elif f.is_dir():
        for ff in sorted(f.glob("*")):
            print(f"  {f.name}/{ff.name} ({ff.stat().st_size / 1024:.1f} KB)")
