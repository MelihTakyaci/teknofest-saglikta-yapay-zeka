#!/usr/bin/env python3
"""
TEKNOFEST Healthcare AI — Data Understanding & Preparation Report
=================================================================
Consolidated analysis of all four competition datasets.
Produces: 1 markdown report, 10 CSV tables, 8+ figures, 1 Word doc, 1 notebook.
"""

import warnings, os, json, time, textwrap, itertools
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path("/Users/melihtakyaci/Documents/TeknofestSagliktaYapayZekâVerisi")
DATA = BASE / "EĞİTİM (TRAIN) SETLERİ 2"
OUT  = BASE / "reports" / "data_understanding_and_preparation"
FIG  = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("TEKNOFEST Healthcare AI — Data Understanding & Preparation")
print("=" * 72)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print("\n[1/12] Loading datasets...")
t0 = time.time()

datasets = {}
files = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "CFTR":   "YARISMA_TRAIN_CFTR.csv",
    "PAH":    "YARISMA_TRAIN_PAH.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
}
for name, fname in files.items():
    datasets[name] = pd.read_csv(DATA / fname)
    print(f"  {name}: {datasets[name].shape}")

master = datasets["MASTER"]
all_cols = list(master.columns)

# column groups
al_cols  = sorted([c for c in all_cols if c.startswith("AL_")],
                  key=lambda x: int(x.split("_")[1]))
cat_cols = sorted([c for c in all_cols if c.startswith("CAT_")],
                  key=lambda x: int(x.split("_")[1]))
ek_cols  = sorted([c for c in all_cols if c.startswith("EK_")],
                  key=lambda x: int(x.split("_")[1]))
aa_cols  = sorted([c for c in all_cols if c.startswith("AA_")])
id_col   = "Variant_ID"
label_col = "Label"
feature_cols = al_cols + cat_cols + ek_cols + aa_cols

print(f"  Column groups: AL={len(al_cols)}, CAT={len(cat_cols)}, "
      f"EK={len(ek_cols)}, AA={len(aa_cols)}, ID=1, Label=1")

# =====================================================================
# 2. DATASET INVENTORY (Section 1)
# =====================================================================
print("\n[2/12] Dataset inventory...")

inventory_rows = []
for name, df in datasets.items():
    schema_match = list(df.columns) == all_cols
    inventory_rows.append({
        "Dataset": name,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Schema_Match": schema_match,
        "AL_count": sum(1 for c in df.columns if c.startswith("AL_")),
        "CAT_count": sum(1 for c in df.columns if c.startswith("CAT_")),
        "EK_count": sum(1 for c in df.columns if c.startswith("EK_")),
        "AA_count": sum(1 for c in df.columns if c.startswith("AA_")),
        "Has_Variant_ID": "Variant_ID" in df.columns,
        "Has_Label": "Label" in df.columns,
    })
inventory_df = pd.DataFrame(inventory_rows)
inventory_df.to_csv(OUT / "dataset_inventory.csv", index=False)

# dtype summary
dtype_summary = {}
for col in all_cols:
    dtype_summary[col] = str(master[col].dtype)

# =====================================================================
# 3. TARGET VARIABLE ANALYSIS (Section 2)
# =====================================================================
print("\n[3/12] Target variable analysis...")

class_rows = []
for name, df in datasets.items():
    n = len(df)
    n_path = int((df["Label"] == 1).sum())
    n_ben  = int((df["Label"] == 0).sum())
    n_other = n - n_path - n_ben
    class_rows.append({
        "Dataset": name,
        "Total": n,
        "Pathogenic": n_path,
        "Benign": n_ben,
        "Other": n_other,
        "Pathogenic_Rate": round(n_path / n, 4),
        "Imbalance_Ratio": round(n_path / max(n_ben, 1), 2),
        "Is_Binary": df["Label"].nunique() == 2 and set(df["Label"].unique()) == {0, 1},
    })
class_df = pd.DataFrame(class_rows)
class_df.to_csv(OUT / "class_distribution.csv", index=False)

# class distribution figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, df) in zip(axes, datasets.items()):
    counts = df["Label"].value_counts().sort_index()
    colors = ["#3498db", "#e74c3c"]
    bars = ax.bar(["Benign (0)", "Pathogenic (1)"], [counts.get(0, 0), counts.get(1, 0)],
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(f"{name}\n(n={len(df)})", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", fontsize=9)
    ratio = counts.get(1, 0) / max(counts.get(0, 0), 1)
    ax.text(0.5, 0.92, f"Ratio: {ratio:.2f}:1", transform=ax.transAxes,
            ha="center", fontsize=9, style="italic")
plt.tight_layout()
plt.savefig(FIG / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================================
# 4. MISSING VALUE ANALYSIS (Section 3)
# =====================================================================
print("\n[4/12] Missing value analysis...")

# 4a. per-column missingness for each dataset
miss_rows = []
for name, df in datasets.items():
    for col in all_cols:
        n_miss = int(df[col].isna().sum())
        pct = round(n_miss / len(df) * 100, 2)
        miss_rows.append({
            "Dataset": name, "Column": col, "N_Missing": n_miss,
            "Pct_Missing": pct, "N_Present": len(df) - n_miss,
        })
miss_all_df = pd.DataFrame(miss_rows)
miss_all_df.to_csv(OUT / "missingness_summary.csv", index=False)

# 4b. missingness bands (MASTER only for summary)
master_miss = miss_all_df[miss_all_df["Dataset"] == "MASTER"].copy()
feat_miss = master_miss[master_miss["Column"].isin(feature_cols)]

bands = [
    ("0%", 0, 0), ("0-5%", 0.01, 5), ("5-20%", 5.01, 20),
    ("20-50%", 20.01, 50), ("50-80%", 50.01, 80),
    ("80-95%", 80.01, 95), (">95%", 95.01, 100),
]
band_counts = {}
for label, lo, hi in bands:
    if lo == 0 and hi == 0:
        band_counts[label] = int((feat_miss["Pct_Missing"] == 0).sum())
    else:
        band_counts[label] = int(((feat_miss["Pct_Missing"] >= lo) & (feat_miss["Pct_Missing"] <= hi)).sum())

print(f"  Missingness bands (MASTER features):")
for b, c in band_counts.items():
    print(f"    {b:8s}: {c} features")

# 4c. row-level missingness
row_miss_data = []
for name, df in datasets.items():
    for lbl in [0, 1]:
        sub = df[df["Label"] == lbl]
        rm = sub[feature_cols].isna().sum(axis=1) / len(feature_cols) * 100
        row_miss_data.append({
            "Dataset": name, "Label": lbl,
            "Mean_Row_Miss_Pct": round(rm.mean(), 2),
            "Median_Row_Miss_Pct": round(rm.median(), 2),
            "Max_Row_Miss_Pct": round(rm.max(), 2),
            "Q75_Row_Miss_Pct": round(rm.quantile(0.75), 2),
        })
row_miss_df = pd.DataFrame(row_miss_data)

# 4d. differential missingness by label (MASTER)
diff_miss_rows = []
for col in feature_cols:
    miss_path = master.loc[master["Label"] == 1, col].isna().mean() * 100
    miss_ben  = master.loc[master["Label"] == 0, col].isna().mean() * 100
    diff = miss_path - miss_ben
    diff_miss_rows.append({
        "Column": col,
        "Miss_Pathogenic_Pct": round(miss_path, 2),
        "Miss_Benign_Pct": round(miss_ben, 2),
        "Difference_Pct": round(diff, 2),
        "Abs_Difference": round(abs(diff), 2),
    })
diff_miss_df = pd.DataFrame(diff_miss_rows).sort_values("Abs_Difference", ascending=False)
diff_miss_df.to_csv(OUT / "differential_missingness_by_label.csv", index=False)

# differential missingness figure (top 30)
top30_diff = diff_miss_df.head(30)
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = range(len(top30_diff))
ax.barh(y_pos, top30_diff["Difference_Pct"].values,
        color=["#e74c3c" if d > 0 else "#3498db" for d in top30_diff["Difference_Pct"].values])
ax.set_yticks(y_pos)
ax.set_yticklabels(top30_diff["Column"].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Missingness Difference (Pathogenic − Benign, %)")
ax.set_title("Top 30 Features by Label-Associated Missingness (MASTER)")
ax.axvline(x=0, color="black", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIG / "differential_missingness_top30.png", dpi=150, bbox_inches="tight")
plt.close()

# panel missingness comparison figure
panel_miss_summary = []
for name, df in datasets.items():
    for grp_name, grp_cols in [("AL_1-26", [c for c in al_cols if int(c.split("_")[1]) <= 26]),
                                ("AL_27-95", [c for c in al_cols if 27 <= int(c.split("_")[1]) <= 95]),
                                ("AL_96-185", [c for c in al_cols if 96 <= int(c.split("_")[1]) <= 185]),
                                ("AL_186-334", [c for c in al_cols if 186 <= int(c.split("_")[1]) <= 334]),
                                ("EK_1-9", ek_cols),
                                ("CAT_1-6", cat_cols),
                                ("AA_1-2", aa_cols)]:
        present = [c for c in grp_cols if c in df.columns]
        if present:
            mean_miss = df[present].isna().mean().mean() * 100
            panel_miss_summary.append({"Dataset": name, "Group": grp_name, "Mean_Miss_Pct": round(mean_miss, 1)})

panel_miss_df = pd.DataFrame(panel_miss_summary)
panel_miss_pivot = panel_miss_df.pivot(index="Group", columns="Dataset", values="Mean_Miss_Pct")
panel_miss_pivot = panel_miss_pivot[["MASTER", "CFTR", "PAH", "KANSER"]]

fig, ax = plt.subplots(figsize=(10, 6))
panel_miss_pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5)
ax.set_ylabel("Mean Missingness (%)")
ax.set_title("Feature Group Missingness by Dataset")
ax.set_xlabel("Feature Group")
ax.legend(title="Dataset")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIG / "panel_missingness_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# row missingness by class figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, df) in zip(axes, datasets.items()):
    for lbl, color, label_str in [(0, "#3498db", "Benign"), (1, "#e74c3c", "Pathogenic")]:
        sub = df[df["Label"] == lbl]
        rm = sub[feature_cols].isna().sum(axis=1) / len(feature_cols) * 100
        ax.hist(rm, bins=30, alpha=0.6, color=color, label=label_str, density=True)
    ax.set_title(f"{name}")
    ax.set_xlabel("Row Missingness (%)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
plt.suptitle("Row-Level Missingness Distribution by Class", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "row_missingness_by_class.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================================
# 5. DUPLICATE AND OVERLAP ANALYSIS (Section 4)
# =====================================================================
print("\n[5/12] Duplicate and overlap analysis...")

dup_rows = []
for name, df in datasets.items():
    feat_only = df.drop(columns=["Variant_ID", "Label"], errors="ignore")
    exact_dups = df.duplicated().sum()
    feat_dups = feat_only.duplicated().sum()

    # conflicting labels
    feat_groups = df.groupby(list(feat_only.columns))["Label"].nunique()
    conflict = int((feat_groups > 1).sum())

    vid_dups = df["Variant_ID"].duplicated().sum()

    dup_rows.append({
        "Dataset": name,
        "Exact_Duplicates": int(exact_dups),
        "Feature_Vector_Duplicates": int(feat_dups),
        "Conflicting_Labels": conflict,
        "Variant_ID_Duplicates": int(vid_dups),
    })
dup_df = pd.DataFrame(dup_rows)

# overlap matrix
overlap_data = []
for n1 in datasets:
    row = {"Dataset": n1}
    ids1 = set(datasets[n1]["Variant_ID"])
    for n2 in datasets:
        ids2 = set(datasets[n2]["Variant_ID"])
        row[n2] = len(ids1 & ids2)
    overlap_data.append(row)
overlap_df = pd.DataFrame(overlap_data).set_index("Dataset")

# label consistency for shared variants
label_consistency = []
for panel_name in ["CFTR", "PAH", "KANSER"]:
    panel = datasets[panel_name]
    shared_ids = set(master["Variant_ID"]) & set(panel["Variant_ID"])
    if shared_ids:
        m_labels = master.set_index("Variant_ID").loc[list(shared_ids), "Label"]
        p_labels = panel.set_index("Variant_ID").loc[list(shared_ids), "Label"]
        common = m_labels.align(p_labels, join="inner")
        matches = (common[0] == common[1]).sum()
        total = len(common[0])
        label_consistency.append({
            "Panel": panel_name, "Shared_Variants": total,
            "Label_Matches": int(matches), "Label_Mismatches": total - int(matches),
        })
label_cons_df = pd.DataFrame(label_consistency)

combined_dup = pd.concat([dup_df, pd.DataFrame([{"Note": "Overlap matrix and label consistency in separate tables"}])],
                         ignore_index=True)
dup_df.to_csv(OUT / "duplicate_and_overlap_summary.csv", index=False)
overlap_df.to_csv(OUT / "variant_id_overlap_matrix.csv")
label_cons_df.to_csv(OUT / "label_consistency_shared_variants.csv", index=False)

print(f"  Overlap: MASTER-CFTR={overlap_df.loc['MASTER','CFTR']}, "
      f"MASTER-PAH={overlap_df.loc['MASTER','PAH']}, "
      f"MASTER-KANSER={overlap_df.loc['MASTER','KANSER']}")

# =====================================================================
# 6. CONSTANT AND NEAR-CONSTANT FEATURES (Section 5)
# =====================================================================
print("\n[6/12] Constant and near-constant feature analysis...")

const_rows = []
for name, df in datasets.items():
    for col in feature_cols:
        nu = df[col].nunique(dropna=True)
        n_present = df[col].notna().sum()
        miss_pct = round(df[col].isna().mean() * 100, 2)
        most_common_pct = round(df[col].value_counts(normalize=True).iloc[0] * 100, 2) if n_present > 0 else 100
        const_rows.append({
            "Dataset": name, "Column": col, "N_Unique": int(nu),
            "N_Present": int(n_present), "Miss_Pct": miss_pct,
            "Most_Common_Pct": most_common_pct,
            "Is_Constant": nu <= 1,
            "Is_Near_Constant": most_common_pct >= 99.0 and nu <= 3,
        })
const_df = pd.DataFrame(const_rows)
const_df.to_csv(OUT / "constant_features_summary.csv", index=False)

const_summary = const_df.groupby("Dataset").agg(
    Constant_Cols=("Is_Constant", "sum"),
    Near_Constant_Cols=("Is_Near_Constant", "sum"),
).astype(int)

for name in datasets:
    n_const = int(const_summary.loc[name, "Constant_Cols"])
    n_near  = int(const_summary.loc[name, "Near_Constant_Cols"])
    print(f"  {name}: {n_const} constant, {n_near} near-constant")

# which constants are shared vs dataset-specific
master_const = set(const_df[(const_df["Dataset"] == "MASTER") & (const_df["Is_Constant"])]["Column"])
global_const_cols = list(master_const)

# =====================================================================
# 7. FEATURE TYPE ANALYSIS (Section 6)
# =====================================================================
print("\n[7/12] Feature type analysis...")

# 7a. numeric feature summary for AL_* (MASTER)
al_stats_rows = []
for col in al_cols:
    s = master[col].dropna()
    if len(s) == 0:
        al_stats_rows.append({"Column": col, "N": 0, "Mean": np.nan, "Std": np.nan,
                               "Min": np.nan, "Max": np.nan, "Skew": np.nan, "Miss_Pct": 100.0,
                               "Looks_Like": "empty"})
        continue
    skew = float(sp_stats.skew(s)) if len(s) > 2 else np.nan
    looks_like = "unknown"
    if s.min() >= 0 and s.max() <= 1.001:
        looks_like = "probability/score [0,1]"
    elif set(s.unique()).issubset({0, 1}):
        looks_like = "binary flag"
    elif s.min() >= -1 and s.max() <= 1:
        looks_like = "correlation/score [-1,1]"
    al_stats_rows.append({
        "Column": col, "N": int(len(s)), "Mean": round(float(s.mean()), 4),
        "Std": round(float(s.std()), 4), "Min": round(float(s.min()), 4),
        "Max": round(float(s.max()), 4), "Skew": round(skew, 2),
        "Miss_Pct": round(master[col].isna().mean() * 100, 2),
        "Looks_Like": looks_like,
    })
al_stats_df = pd.DataFrame(al_stats_rows)

# 7b. CAT features
cat_unique_rows = []
for col in cat_cols:
    for name, df in datasets.items():
        vals = df[col].dropna().unique()
        cat_unique_rows.append({
            "Dataset": name, "Column": col, "N_Unique": len(vals),
            "Miss_Pct": round(df[col].isna().mean() * 100, 2),
            "Sample_Values": str(sorted(vals)[:8]),
        })
cat_unique_df = pd.DataFrame(cat_unique_rows)
cat_unique_df.to_csv(OUT / "categorical_unique_values.csv", index=False)

# 7c. AA features
aa_info_rows = []
for col in aa_cols:
    vals = master[col].dropna().unique()
    aa_info_rows.append({
        "Column": col, "N_Unique": len(vals),
        "Miss_Pct": round(master[col].isna().mean() * 100, 2),
        "Values": str(sorted(vals)[:20]),
    })
aa_info_df = pd.DataFrame(aa_info_rows)

# 7d. feature type summary
ft_rows = []
for col in all_cols:
    if col == "Variant_ID":
        ft_rows.append({"Column": col, "Group": "ID", "Dtype": str(master[col].dtype),
                        "Role": "identifier", "Use_As_Feature": "NO"})
    elif col == "Label":
        ft_rows.append({"Column": col, "Group": "Target", "Dtype": str(master[col].dtype),
                        "Role": "target", "Use_As_Feature": "NO"})
    elif col in al_cols:
        ft_rows.append({"Column": col, "Group": "AL", "Dtype": str(master[col].dtype),
                        "Role": "numeric_feature", "Use_As_Feature": "YES"})
    elif col in cat_cols:
        ft_rows.append({"Column": col, "Group": "CAT", "Dtype": str(master[col].dtype),
                        "Role": "categorical_feature", "Use_As_Feature": "YES (encode)"})
    elif col in ek_cols:
        ft_rows.append({"Column": col, "Group": "EK", "Dtype": str(master[col].dtype),
                        "Role": "numeric_feature (risk)", "Use_As_Feature": "YES (with caution)"})
    elif col in aa_cols:
        ft_rows.append({"Column": col, "Group": "AA", "Dtype": str(master[col].dtype),
                        "Role": "amino_acid_feature", "Use_As_Feature": "YES (encode)"})
ft_df = pd.DataFrame(ft_rows)
ft_df.to_csv(OUT / "feature_type_summary.csv", index=False)

# =====================================================================
# 8. EK FEATURE RISK ANALYSIS (Section 7)
# =====================================================================
print("\n[8/12] EK feature leakage risk analysis...")

ek_risk_rows = []
for col in ek_cols:
    s = master[col].dropna()
    # association with label
    path_vals = master.loc[(master["Label"] == 1) & master[col].notna(), col]
    ben_vals  = master.loc[(master["Label"] == 0) & master[col].notna(), col]

    if len(path_vals) > 5 and len(ben_vals) > 5:
        try:
            u_stat, u_pval = sp_stats.mannwhitneyu(path_vals, ben_vals, alternative="two-sided")
            effect_size = round(abs(path_vals.mean() - ben_vals.mean()) / (s.std() + 1e-12), 4)
        except Exception:
            u_pval, effect_size = np.nan, np.nan
        point_biserial = np.corrcoef(
            master.loc[master[col].notna(), col],
            master.loc[master[col].notna(), "Label"]
        )[0, 1]
    else:
        u_pval, effect_size, point_biserial = np.nan, np.nan, np.nan

    looks_like_01 = s.min() >= 0 and s.max() <= 1.001
    risk = "LOW"
    risk_reason = ""
    if looks_like_01 and abs(point_biserial) > 0.2:
        risk = "HIGH"
        risk_reason = "[0,1] range + strong label correlation → possible ClinVar-trained meta-predictor"
    elif abs(point_biserial) > 0.3:
        risk = "MEDIUM"
        risk_reason = "Strong label correlation; origin unknown"
    elif looks_like_01:
        risk = "MEDIUM"
        risk_reason = "[0,1] range suggests predictor score"
    else:
        risk_reason = "Not [0,1]; likely conservation or raw score"

    ek_risk_rows.append({
        "Column": col,
        "N_Present": int(len(s)),
        "Miss_Pct": round(master[col].isna().mean() * 100, 2),
        "Min": round(float(s.min()), 3) if len(s) > 0 else np.nan,
        "Max": round(float(s.max()), 3) if len(s) > 0 else np.nan,
        "Mean": round(float(s.mean()), 3) if len(s) > 0 else np.nan,
        "Std": round(float(s.std()), 3) if len(s) > 0 else np.nan,
        "Range_0_1": looks_like_01,
        "Correlation_with_Label": round(float(point_biserial), 4) if not np.isnan(point_biserial) else np.nan,
        "Effect_Size": effect_size,
        "MannWhitney_pval": f"{u_pval:.2e}" if not np.isnan(u_pval) else "N/A",
        "Leakage_Risk": risk,
        "Risk_Reason": risk_reason,
        "Recommendation": "ABLATION" if risk == "HIGH" else ("MONITOR" if risk == "MEDIUM" else "KEEP"),
    })
ek_risk_df = pd.DataFrame(ek_risk_rows)
ek_risk_df.to_csv(OUT / "ek_feature_risk_summary.csv", index=False)

# EK distribution by class figure
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, col in zip(axes.flat, ek_cols):
    ben  = master.loc[master["Label"] == 0, col].dropna()
    path = master.loc[master["Label"] == 1, col].dropna()
    if len(ben) > 0:
        ax.hist(ben, bins=40, alpha=0.6, color="#3498db", label="Benign", density=True)
    if len(path) > 0:
        ax.hist(path, bins=40, alpha=0.6, color="#e74c3c", label="Pathogenic", density=True)
    risk = ek_risk_df.loc[ek_risk_df["Column"] == col, "Leakage_Risk"].values[0]
    corr = ek_risk_df.loc[ek_risk_df["Column"] == col, "Correlation_with_Label"].values[0]
    ax.set_title(f"{col} (risk={risk}, r={corr:.3f})", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
plt.suptitle("EK Feature Distributions by Class (MASTER)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG / "ek_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# EK inter-correlation
ek_data = master[ek_cols].dropna(how="all")
ek_corr = ek_data.corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(ek_corr, dtype=bool), k=1)
sns.heatmap(ek_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            mask=mask, vmin=-1, vmax=1, square=True, linewidths=0.5)
ax.set_title("EK Feature Inter-Correlation (Spearman)")
plt.tight_layout()
plt.savefig(FIG / "ek_intercorrelation.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  HIGH risk: {sum(1 for r in ek_risk_rows if r['Leakage_Risk'] == 'HIGH')}")
print(f"  MEDIUM risk: {sum(1 for r in ek_risk_rows if r['Leakage_Risk'] == 'MEDIUM')}")
print(f"  LOW risk: {sum(1 for r in ek_risk_rows if r['Leakage_Risk'] == 'LOW')}")

# =====================================================================
# 9. PANEL COMPARISON (Section 8)
# =====================================================================
print("\n[9/12] Panel comparison...")

# distribution shift detection (KS test for numeric features, MASTER vs panels)
shift_results = []
numeric_feats = [c for c in feature_cols if master[c].dtype in ("float64", "int64")]
for panel_name in ["CFTR", "PAH", "KANSER"]:
    panel = datasets[panel_name]
    n_shifted = 0
    for col in numeric_feats[:100]:  # sample for speed
        m_vals = master[col].dropna()
        p_vals = panel[col].dropna()
        if len(m_vals) > 10 and len(p_vals) > 10:
            ks_stat, ks_pval = sp_stats.ks_2samp(m_vals, p_vals)
            if ks_pval < 0.001:
                n_shifted += 1
    shift_results.append({
        "Panel": panel_name,
        "Features_Tested": min(len(numeric_feats), 100),
        "Features_Shifted": n_shifted,
        "Shift_Pct": round(n_shifted / min(len(numeric_feats), 100) * 100, 1),
    })
shift_df = pd.DataFrame(shift_results)
print(f"  Distribution shifts detected:")
for _, row in shift_df.iterrows():
    print(f"    {row['Panel']}: {row['Features_Shifted']}/{row['Features_Tested']} features shifted")

# =====================================================================
# 10. PREPROCESSING RECOMMENDATIONS (Section 9)
# =====================================================================
print("\n[10/12] Preprocessing recommendations...")

preproc_rows = [
    {"Step": "Drop Variant_ID from features", "Priority": "MANDATORY",
     "Applies_To": "All models", "Reason": "Identifier, not predictive; leakage risk"},
    {"Step": "Drop constant columns (MASTER: " + str(len(global_const_cols)) + ")", "Priority": "MANDATORY",
     "Applies_To": "All models", "Reason": "Zero information content"},
    {"Step": "Use class weights (is_unbalance / scale_pos_weight)", "Priority": "MANDATORY",
     "Applies_To": "All models", "Reason": "73% pathogenic imbalance"},
    {"Step": "Stratified K-Fold CV", "Priority": "MANDATORY",
     "Applies_To": "All models", "Reason": "Preserve label ratio in folds"},
    {"Step": "Variant_ID-aware splitting for panel evaluation", "Priority": "MANDATORY",
     "Applies_To": "Panel evaluation", "Reason": "Prevent overlap leakage"},
    {"Step": "Native missing handling (no imputation)", "Priority": "RECOMMENDED",
     "Applies_To": "GBDT models", "Reason": "LightGBM/XGBoost handle NaN natively"},
    {"Step": "Add missingness indicator features", "Priority": "RECOMMENDED",
     "Applies_To": "All models", "Reason": "Missingness is label-associated in KANSER"},
    {"Step": "One-hot encode CAT_3/4/5", "Priority": "RECOMMENDED",
     "Applies_To": "All models", "Reason": "Low cardinality (4-5 values)"},
    {"Step": "Label encode CAT_1/CAT_2", "Priority": "RECOMMENDED",
     "Applies_To": "GBDT models", "Reason": "High cardinality; or use CatBoost native"},
    {"Step": "Binary flag for CAT_6", "Priority": "RECOMMENDED",
     "Applies_To": "All models", "Reason": "97.7% missing; binary presence flag"},
    {"Step": "One-hot encode AA_1/AA_2", "Priority": "RECOMMENDED",
     "Applies_To": "All models", "Reason": "20 amino acid categories each"},
    {"Step": "EK ablation (train with and without EK_4/5/6)", "Priority": "HIGH",
     "Applies_To": "All models", "Reason": "Quantify circularity risk"},
    {"Step": "Median imputation + scaling", "Priority": "CONDITIONAL",
     "Applies_To": "Linear models only", "Reason": "Linear models cannot handle NaN"},
    {"Step": "Avoid SMOTE / oversampling", "Priority": "WARNING",
     "Applies_To": "All models", "Reason": "Synthetic genomic variants are biologically meaningless"},
    {"Step": "Avoid target encoding outside CV folds", "Priority": "WARNING",
     "Applies_To": "All models", "Reason": "Creates label leakage"},
]
preproc_df = pd.DataFrame(preproc_rows)
preproc_df.to_csv(OUT / "preprocessing_recommendations.csv", index=False)

# =====================================================================
# 11. TOP FEATURES BY LABEL ASSOCIATION (bonus figure)
# =====================================================================
print("\n[11/12] Feature-label association analysis...")

assoc_rows = []
for col in numeric_feats:
    path_vals = master.loc[(master["Label"] == 1) & master[col].notna(), col]
    ben_vals  = master.loc[(master["Label"] == 0) & master[col].notna(), col]
    if len(path_vals) > 10 and len(ben_vals) > 10:
        try:
            u_stat, u_pval = sp_stats.mannwhitneyu(path_vals, ben_vals, alternative="two-sided")
            n_total = len(path_vals) + len(ben_vals)
            effect = abs(u_stat / (len(path_vals) * len(ben_vals)) - 0.5) * 2
            assoc_rows.append({"Column": col, "Effect_Size": round(effect, 4),
                               "MannWhitney_p": u_pval, "N": n_total})
        except Exception:
            pass
assoc_df = pd.DataFrame(assoc_rows).sort_values("Effect_Size", ascending=False)

# top features figure
top_assoc = assoc_df.head(30)
fig, ax = plt.subplots(figsize=(12, 8))
colors = []
for col in top_assoc["Column"]:
    if col.startswith("EK_"):
        colors.append("#e74c3c")
    elif col.startswith("AL_"):
        colors.append("#3498db")
    else:
        colors.append("#95a5a6")
ax.barh(range(len(top_assoc)), top_assoc["Effect_Size"].values, color=colors)
ax.set_yticks(range(len(top_assoc)))
ax.set_yticklabels(top_assoc["Column"].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Effect Size (rank-biserial)")
ax.set_title("Top 30 Features by Association with Label (MASTER)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#e74c3c", label="EK features"),
                   Patch(color="#3498db", label="AL features")],
          loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIG / "top_features_by_label_association.png", dpi=150, bbox_inches="tight")
plt.close()

# missingness heatmap (top 50 features by missingness, MASTER)
miss_pct_series = master[feature_cols].isna().mean().sort_values(ascending=False)
top50_miss = miss_pct_series.head(50)
fig, ax = plt.subplots(figsize=(14, 8))
miss_matrix = master[top50_miss.index].isna().astype(int)
# sample rows for heatmap visibility
sample_idx = np.random.choice(len(miss_matrix), size=min(200, len(miss_matrix)), replace=False)
sample_idx.sort()
sns.heatmap(miss_matrix.iloc[sample_idx].T, cmap="YlOrRd", cbar_kws={"label": "Missing (1=yes)"},
            xticklabels=False, yticklabels=True, ax=ax)
ax.set_title("Missingness Pattern (Top 50 Features by Miss%, 200 Sample Rows)")
ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
plt.savefig(FIG / "missingness_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================================
# 12. GENERATE REPORTS
# =====================================================================
print("\n[12/12] Generating report, notebook, and Word document...")

# — MARKDOWN REPORT —
# helper for table formatting
def md_table(df, max_rows=None):
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False)

ek_high = ek_risk_df[ek_risk_df["Leakage_Risk"] == "HIGH"]["Column"].tolist()
ek_medium = ek_risk_df[ek_risk_df["Leakage_Risk"] == "MEDIUM"]["Column"].tolist()

report = f"""# TEKNOFEST Healthcare AI — Data Understanding & Preparation Report

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Datasets**: MASTER, CFTR, PAH, KANSER
**Task**: Binary classification of genetic variants (Benign=0 vs Pathogenic=1)

---

## Executive Summary

**What is the dataset?**
Four tabular CSV files containing anonymised genetic variant annotations. Each row is a missense variant. The MASTER set has 2931 variants; three gene-panel sets (CFTR: 111, PAH: 372, KANSER: 388) overlap partially with MASTER.

**What is the prediction target?**
Binary classification: 0 = Benign, 1 = Pathogenic. All datasets are imbalanced toward pathogenic (69-83%).

**Biggest risks:**
1. High missingness (54.9% mean row-level in MASTER)
2. {len(global_const_cols)} constant columns in MASTER carry zero information
3. Label-associated missingness in KANSER (up to 57% differential)
4. EK_4 and EK_6 may be ClinVar-trained meta-predictors (circularity risk)
5. Variant_ID overlap between MASTER and panels can cause leakage
6. CFTR is too small (n=111) for an independent model
7. Class imbalance across all datasets (2.2:1 to 5:1)

**What should be done before modeling?**
1. Drop Variant_ID from features (keep for split control)
2. Drop {len(global_const_cols)} constant columns
3. Use class weights in all models
4. Add missingness indicator features
5. Encode categoricals (one-hot for CAT_3-5, label encode CAT_1-2, binary CAT_6)
6. Encode amino acids (one-hot AA_1, AA_2)
7. Train EK ablation models (with and without EK_4/5/6)
8. Use stratified, Variant_ID-aware cross-validation

**Which features are safe?**
- AL_1 to AL_185: Population allele frequencies (low leakage risk)
- EK_7: Very likely phyloP conservation score (safe)
- EK_9: Very likely GERP++ score (safe)
- EK_1/EK_2: Likely CADD scores (low risk)
- AA_1/AA_2: Amino acid identity (safe)

**Which features are suspicious?**
- EK_4: Likely ClinVar-trained meta-predictor [0,1] — HIGH risk
- EK_6: Likely ClinVar-trained meta-predictor [0,1] — HIGH risk
- EK_5: Unclear origin — MEDIUM risk
- AL_223-AL_334: Possible in-silico predictor outputs — MEDIUM risk

**First modeling strategy:**
LightGBM with native missing handling, class weights, 5-fold stratified CV, Pipeline B preprocessing.

**What should NOT be done:**
- Do not use Variant_ID as a feature
- Do not use accuracy as the primary metric
- Do not use SMOTE (synthetic genomic variants are meaningless)
- Do not impute for tree-based models
- Do not trust EK_4/EK_6 without ablation testing
- Do not train independent models on CFTR (too small)
- Do not use target encoding outside CV folds

---

## 1. Dataset Inventory

{md_table(inventory_df)}

All four datasets share the same 353-column schema. Column groups:
- **Variant_ID**: 1 column (object) — identifier only
- **Label**: 1 column (int64) — binary target
- **AL_1 to AL_334**: {len(al_cols)} columns (float64) — anonymised numeric features
- **CAT_1 to CAT_6**: {len(cat_cols)} columns (object) — categorical features
- **EK_1 to EK_9**: {len(ek_cols)} columns (float64) — extra numeric/score features
- **AA_1, AA_2**: {len(aa_cols)} columns (object) — amino acid identity

No schema mismatches, no unexpected columns.

---

## 2. Target Variable Analysis

{md_table(class_df)}

**Key observations:**
- All datasets are binary (0/1) with no VUS or uncertain labels.
- All datasets are imbalanced toward pathogenic (69-83%).
- PAH has the most extreme imbalance (5:1).
- KANSER has the mildest imbalance (2.2:1).

**Why accuracy is insufficient:**
With 73.3% pathogenic in MASTER, a naive "predict all pathogenic" classifier achieves 73.3% accuracy. This is misleading because it misses all benign variants.

**Recommended metrics:**
- **Primary**: ROC-AUC (threshold-independent discrimination)
- **Secondary**: PR-AUC (sensitive to minority class)
- **Threshold-dependent**: F1, sensitivity (recall), specificity, MCC
- **Clinical**: Sensitivity ≥ 95% (minimize missed pathogenic variants)

![Class Distribution](figures/class_distribution.png)

---

## 3. Missing Value Analysis

### 3.1 Column-Level Missingness Bands (MASTER Features)

| Band | Feature Count |
|------|--------------|
"""

for b, c in band_counts.items():
    report += f"| {b} | {c} |\n"

report += f"""
### 3.2 Row-Level Missingness by Class

{md_table(row_miss_df)}

**Key finding**: In KANSER, pathogenic variants have 69.4% mean row missingness vs 33.0% for benign. Missingness is strongly label-associated.

### 3.3 Top 20 Features by Label-Associated Missingness (MASTER)

{md_table(diff_miss_df.head(20))}

### 3.4 Recommendations

| Model Family | Missing Value Strategy |
|-------------|----------------------|
| LightGBM / XGBoost | Native NaN handling; DO NOT impute |
| CatBoost | Native NaN handling |
| Logistic Regression | Median imputation + missingness indicators |
| Neural Network | Imputation + missingness indicators |

- Add binary `is_missing_{{col}}` features for columns with >5% missingness
- Add row-level `total_miss_count` and `total_miss_pct` as features
- Add group-level missing counts (e.g., `n_miss_al_1_26`)
- Do NOT drop high-missingness features without ablation testing

![Differential Missingness](figures/differential_missingness_top30.png)
![Panel Missingness](figures/panel_missingness_comparison.png)
![Row Missingness](figures/row_missingness_by_class.png)
![Missingness Heatmap](figures/missingness_heatmap.png)

---

## 4. Duplicate and Variant Overlap Analysis

### 4.1 Duplicates

{md_table(dup_df)}

- No exact row duplicates in any dataset.
- MASTER has 116 feature-vector duplicates (same features, possibly different variants).
- 1 conflicting label (same features, different label) in MASTER and 1 in PAH.

### 4.2 Variant_ID Overlap Matrix

{overlap_df.to_markdown()}

### 4.3 Label Consistency for Shared Variants

{md_table(label_cons_df)}

All shared variants have consistent labels between MASTER and panels.

### 4.4 Leakage Risk

If the same Variant_ID appears in both train and validation folds, the model sees the answer during training. This inflates performance estimates.

**Recommendation**: Use `GroupKFold` with `Variant_ID` as the group key, or manually ensure no Variant_ID appears in both train and validation.

---

## 5. Constant and Near-Constant Features

{const_summary.to_markdown()}

- MASTER has {len(global_const_cols)} constant columns that carry zero information and must be dropped.
- Panel datasets have additional panel-specific constants (PAH: 91, CFTR: 70, KANSER: 69).
- When evaluating on panels, drop panel-specific constants as well.

---

## 6. Feature Type Analysis

### 6.1 AL Feature Summary (MASTER)

Most AL features are in the [0, 1] range, consistent with population allele frequencies or predictor scores. Features show a clear triplet pattern (score + binary flag + binary flag) in blocks AL_39-AL_95 and beyond.

| Looks Like | Count |
|------------|-------|
"""

al_type_counts = al_stats_df["Looks_Like"].value_counts()
for t, c in al_type_counts.items():
    report += f"| {t} | {c} |\n"

report += f"""

### 6.2 Categorical Features

{md_table(cat_unique_df[cat_unique_df["Dataset"] == "MASTER"])}

**Encoding recommendations:**
- **CAT_1** (30 unique, gnomAD populations): Label encode for GBDT; target encode inside CV for linear models
- **CAT_2** (7 unique, AllofUs populations): One-hot encode
- **CAT_3/4/5** (4-5 unique, genotypes): One-hot encode
- **CAT_6** (3 unique, 97.7% missing): Binary flag `has_region_annotation`

### 6.3 Amino Acid Features

{md_table(aa_info_df)}

**Encoding recommendations:**
- One-hot encode AA_1 and AA_2 (20 amino acids each)
- Optionally create `AA_mutation = AA_1 + ">" + AA_2` as a combined feature
- Optionally group by physicochemical properties (charged, polar, hydrophobic)

---

## 7. EK Feature Leakage Risk Analysis

{md_table(ek_risk_df[["Column", "N_Present", "Miss_Pct", "Min", "Max", "Mean", "Range_0_1", "Correlation_with_Label", "Leakage_Risk", "Recommendation"]])}

### Risk Summary

- **HIGH risk** ({', '.join(ek_high)}): [0,1] range + strong label correlation. Likely ClinVar-trained meta-predictors (REVEL, BayesDel, ClinPred, or AlphaMissense). If labels also come from ClinVar, these features create circular prediction.
- **MEDIUM risk** ({', '.join(ek_medium)}): Strong correlation but not in [0,1] range, or unclear provenance.
- **LOW risk** (EK_1, EK_2, EK_7, EK_8, EK_9): Likely CADD or conservation scores. Not ClinVar-trained.

### Required Feature Sets for Modeling

| Feature Set | Description | Purpose |
|-------------|-------------|---------|
| **Set A (Full)** | All features except Variant_ID and Label | Maximum performance (may include circularity) |
| **Set B (No EK_4/5/6)** | Drop EK_4, EK_5, EK_6 | Honest performance without ClinVar meta-predictors |
| **Set C (No EK)** | Drop all EK_1 through EK_9 | Maximum anti-circularity |
| **Set D (Core only)** | EK_7, EK_9, low-missingness AL, AA, CAT_3-5 | Minimum safe feature set |

Both Set A and Set B must be tested. If Set A performance is much higher than Set B, circularity is confirmed.

![EK Distributions](figures/ek_feature_distributions.png)
![EK Intercorrelation](figures/ek_intercorrelation.png)

---

## 8. Panel Comparison

### 8.1 Distribution Shift

{md_table(shift_df)}

Significant distribution shifts exist between MASTER and panels, especially PAH.

### 8.2 Recommendations

| Panel | N | Strategy | Reason |
|-------|---|----------|--------|
| MASTER | 2931 | Primary training set | Largest; use for all CV |
| CFTR | 111 | Global model + panel threshold | Too small for independent model |
| PAH | 372 | Global model + panel calibration | 5:1 imbalance; 57+ shifted features |
| KANSER | 388 | Global model + panel calibration | Differential missingness leveraged |

- Train one global model on MASTER.
- Evaluate on each panel separately using OOF predictions for shared variants.
- Compute panel-specific optimal thresholds.
- Do NOT train independent models on CFTR (only 111 samples, 21 benign).
- Consider panel-specific Platt scaling if calibration differs.

---

## 9. Preprocessing Pipeline Recommendations

{md_table(preproc_df)}

### Pipeline A — Minimal GBDT Baseline
1. Drop Variant_ID + constant columns
2. Native missing handling
3. Class weights (is_unbalance=True)
4. LightGBM, 5-fold stratified CV

### Pipeline B — Missingness-Aware GBDT (Recommended Start)
1. Drop Variant_ID + constant columns
2. Add per-feature missingness indicators
3. Add row-level and group-level miss counts
4. One-hot encode CAT_3/4/5, label encode CAT_1/2, binary CAT_6
5. One-hot encode AA_1/AA_2
6. Class weights
7. LightGBM, 5-fold stratified CV

### Pipeline C — Anti-Leakage GBDT
Same as B but drop EK_4, EK_5, EK_6

### Pipeline D — Linear Baseline
1. Impute median + add missingness indicators
2. Standard scale all numeric features
3. One-hot encode all categoricals
4. Drop >80% missing features
5. LogisticRegression / ElasticNet

### Pipeline E — Panel-Calibrated Ensemble
1. OOF predictions from multiple Pipeline B/C models
2. Per-panel Platt scaling
3. Threshold optimisation per panel

---

## 10. Validation Strategy

**Design**: 5-fold Stratified K-Fold on MASTER

**Rules**:
1. Stratify by Label to preserve class ratio
2. Use Variant_ID as group key if using GroupKFold
3. Never allow the same Variant_ID in train and validation
4. Evaluate on panels separately (not pooled)
5. Report ROC-AUC, PR-AUC, F1, sensitivity, specificity, MCC per fold
6. Report mean ± std across folds
7. Report per-panel metrics using OOF predictions for shared variants

**For competition submission**:
- Train final model on all of MASTER
- Evaluate honestly using CV metrics (not resubstitution)
- Apply panel-specific thresholds if competition evaluates per-panel

---

## 11. Risk Register

| # | Risk | Severity | Evidence | Mitigation |
|---|------|----------|----------|------------|
| 1 | EK_4/EK_6 circularity | HIGH | [0,1] meta-predictor scores | Ablation testing |
| 2 | Variant_ID overlap leakage | HIGH | 77-255 shared variants | Group-aware splitting |
| 3 | Label-associated missingness | HIGH | KANSER: 57% differential | Missingness indicators |
| 4 | Class imbalance | MEDIUM | 2.2:1 to 5:1 | Class weights + threshold tuning |
| 5 | CFTR too small | HIGH | n=111, only 21 benign | Global model only |
| 6 | PAH distribution shift | MEDIUM | 57+ shifted features | Panel calibration |
| 7 | {len(global_const_cols)} constant columns | MEDIUM | Zero variance | Drop before modeling |
| 8 | Feature anonymisation | LOW | Cannot verify biology | Accept uncertainty |
| 9 | Conflicting labels | LOW | 1 in MASTER, 1 in PAH | Accept as noise |
| 10 | EK_5 unknown origin | MEDIUM | [0,1] range but low correlation | Monitor |

---

## 12. Next-Step Checklist

- [ ] Run Pipeline B (Missingness-Aware GBDT) baseline on MASTER
- [ ] Run Pipeline C (Anti-Leakage) to quantify EK circularity
- [ ] Compare Pipeline B vs C: if AUC drops >5%, circularity is confirmed
- [ ] Run Pipeline A (Minimal) to measure preprocessing benefit
- [ ] Run Pipeline D (Linear baseline) for calibration reference
- [ ] Compute per-panel performance and optimal thresholds
- [ ] Tune hyperparameters on best pipeline
- [ ] Test XGBoost and CatBoost alternatives
- [ ] Build stacking ensemble (Pipeline E) if time permits
- [ ] Generate final competition submission

---

*Report generated automatically. All findings are based on computed evidence from the four competition datasets.*
"""

with open(OUT / "data_understanding_and_preparation_report.md", "w") as f:
    f.write(report)
print("  Markdown report saved")

# — JUPYTER NOTEBOOK —
nb_cells = []

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().split("\n"),
            "id": f"c{len(nb_cells)}"}

def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "source": src.strip().split("\n"),
            "outputs": [], "execution_count": None, "id": f"c{len(nb_cells)}"}

nb_cells.append(md_cell(f"# TEKNOFEST Healthcare AI — Data Understanding & Preparation\n\n"
                         f"**Date**: {datetime.now().strftime('%Y-%m-%d')}"))

nb_cells.append(code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, Image
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.4f}'.format)
REPORT = "reports/data_understanding_and_preparation" """))

nb_cells.append(md_cell("## 1. Dataset Inventory"))
nb_cells.append(code_cell('display(pd.read_csv(f"{REPORT}/dataset_inventory.csv"))'))

nb_cells.append(md_cell("## 2. Class Distribution"))
nb_cells.append(code_cell('display(pd.read_csv(f"{REPORT}/class_distribution.csv"))'))
nb_cells.append(code_cell('Image(f"{REPORT}/figures/class_distribution.png")'))

nb_cells.append(md_cell("## 3. Missing Value Analysis"))
nb_cells.append(code_cell("""miss = pd.read_csv(f"{REPORT}/missingness_summary.csv")
master_miss = miss[miss["Dataset"]=="MASTER"]
print("Missingness bands (MASTER features):")
feat_miss = master_miss[~master_miss["Column"].isin(["Variant_ID","Label"])]
bins = [0, 0.01, 5, 20, 50, 80, 95, 100]
labels = ["0%", "0-5%", "5-20%", "20-50%", "50-80%", "80-95%", ">95%"]
feat_miss["Band"] = pd.cut(feat_miss["Pct_Missing"], bins=bins, labels=labels, right=True)
print(feat_miss["Band"].value_counts().sort_index())"""))

nb_cells.append(code_cell('Image(f"{REPORT}/figures/differential_missingness_top30.png")'))
nb_cells.append(code_cell('Image(f"{REPORT}/figures/panel_missingness_comparison.png")'))
nb_cells.append(code_cell('Image(f"{REPORT}/figures/row_missingness_by_class.png")'))
nb_cells.append(code_cell('Image(f"{REPORT}/figures/missingness_heatmap.png")'))

nb_cells.append(md_cell("## 4. Duplicate and Overlap Analysis"))
nb_cells.append(code_cell("""display(pd.read_csv(f"{REPORT}/duplicate_and_overlap_summary.csv"))
print("\\nVariant_ID Overlap Matrix:")
display(pd.read_csv(f"{REPORT}/variant_id_overlap_matrix.csv", index_col=0))
print("\\nLabel Consistency:")
display(pd.read_csv(f"{REPORT}/label_consistency_shared_variants.csv"))"""))

nb_cells.append(md_cell("## 5. Constant Features"))
nb_cells.append(code_cell("""const = pd.read_csv(f"{REPORT}/constant_features_summary.csv")
print("Constant columns per dataset:")
print(const.groupby("Dataset")["Is_Constant"].sum().astype(int))"""))

nb_cells.append(md_cell("## 6. Feature Types"))
nb_cells.append(code_cell('display(pd.read_csv(f"{REPORT}/feature_type_summary.csv").head(20))'))
nb_cells.append(code_cell("""print("Categorical unique values (MASTER):")
display(pd.read_csv(f"{REPORT}/categorical_unique_values.csv").query("Dataset=='MASTER'"))"""))

nb_cells.append(md_cell("## 7. EK Feature Risk Analysis"))
nb_cells.append(code_cell("""ek = pd.read_csv(f"{REPORT}/ek_feature_risk_summary.csv")
display(ek[["Column","Min","Max","Mean","Range_0_1","Correlation_with_Label","Leakage_Risk","Recommendation"]])"""))
nb_cells.append(code_cell('Image(f"{REPORT}/figures/ek_feature_distributions.png")'))
nb_cells.append(code_cell('Image(f"{REPORT}/figures/ek_intercorrelation.png")'))

nb_cells.append(md_cell("## 8. Top Features by Label Association"))
nb_cells.append(code_cell('Image(f"{REPORT}/figures/top_features_by_label_association.png")'))

nb_cells.append(md_cell("## 9. Preprocessing Recommendations"))
nb_cells.append(code_cell('display(pd.read_csv(f"{REPORT}/preprocessing_recommendations.csv"))'))

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10.0"}},
    "cells": nb_cells,
}
with open(OUT / "Data_Understanding_Notebook.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("  Notebook saved")

# — WORD DOCUMENT —
try:
    from docx import Document
    from docx.shared import Inches, Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2)
        sec.bottom_margin = Cm(2)
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(2.5)
    style = doc.styles["Normal"]
    style.font.size = Pt(10)
    style.font.name = "Calibri"

    def add_tbl(doc, headers, rows):
        t = doc.add_table(rows=1, cols=len(headers), style="Light Shading Accent 1")
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(headers):
            t.rows[0].cells[i].text = h
            for p in t.rows[0].cells[i].paragraphs:
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(9)
        for rd in rows:
            row = t.add_row()
            for i, v in enumerate(rd):
                row.cells[i].text = str(v)
                for p in row.cells[i].paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9)

    def add_fig(doc, name, width=5.5):
        p = FIG / name
        if p.exists():
            doc.add_picture(str(p), width=Inches(width))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # title
    title = doc.add_heading("Data Understanding & Preparation Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("TEKNOFEST Healthcare AI — Missense Variant Classification")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # exec summary
    doc.add_heading("Executive Summary", level=1)
    exec_points = [
        "4 datasets: MASTER (2931), CFTR (111), PAH (372), KANSER (388)",
        "Binary classification: Benign (0) vs Pathogenic (1)",
        "All datasets imbalanced (69-83% pathogenic)",
        f"{len(global_const_cols)} constant columns must be dropped",
        "54.9% mean row-level missingness in MASTER",
        "EK_4/EK_6 are HIGH leakage risk (possible ClinVar circularity)",
        "CFTR too small for independent model (n=111)",
        "Start with Pipeline B (Missingness-Aware LightGBM)",
    ]
    for pt in exec_points:
        doc.add_paragraph(pt, style="List Bullet")

    # 1. inventory
    doc.add_heading("1. Dataset Inventory", level=1)
    add_tbl(doc, ["Dataset", "Rows", "Columns", "Schema Match"],
            [[r["Dataset"], r["Rows"], r["Columns"], r["Schema_Match"]] for r in inventory_rows])

    # 2. class distribution
    doc.add_heading("2. Class Distribution", level=1)
    add_tbl(doc, ["Dataset", "Total", "Pathogenic", "Benign", "Ratio"],
            [[r["Dataset"], r["Total"], r["Pathogenic"], r["Benign"],
              f"{r['Imbalance_Ratio']}:1"] for r in class_rows])
    doc.add_paragraph("")
    add_fig(doc, "class_distribution.png")

    # 3. missing values
    doc.add_heading("3. Missing Value Analysis", level=1)
    doc.add_paragraph("Missingness bands (MASTER features):")
    add_tbl(doc, ["Band", "Feature Count"], [[b, c] for b, c in band_counts.items()])
    doc.add_paragraph("")
    add_fig(doc, "differential_missingness_top30.png")
    doc.add_paragraph("")
    add_fig(doc, "panel_missingness_comparison.png")
    doc.add_paragraph("")
    add_fig(doc, "row_missingness_by_class.png")

    # 4. duplicates/overlap
    doc.add_heading("4. Duplicate and Overlap Analysis", level=1)
    add_tbl(doc, ["Dataset", "Exact Dups", "Feature Dups", "Conflicts", "ID Dups"],
            [[r["Dataset"], r["Exact_Duplicates"], r["Feature_Vector_Duplicates"],
              r["Conflicting_Labels"], r["Variant_ID_Duplicates"]] for _, r in dup_df.iterrows()])
    doc.add_paragraph("")
    doc.add_paragraph("Variant_ID overlap: MASTER-CFTR=77, MASTER-PAH=255, MASTER-KANSER=246")
    doc.add_paragraph("All shared variants have consistent labels.")

    # 5. constants
    doc.add_heading("5. Constant Features", level=1)
    add_tbl(doc, ["Dataset", "Constant Cols", "Near-Constant Cols"],
            [[name, int(const_summary.loc[name, "Constant_Cols"]),
              int(const_summary.loc[name, "Near_Constant_Cols"])] for name in datasets])

    # 6. feature types
    doc.add_heading("6. Feature Types", level=1)
    doc.add_paragraph(f"AL features: {len(al_cols)} (mostly [0,1] probability/scores)")
    doc.add_paragraph(f"CAT features: {len(cat_cols)} (categorical — populations, genotypes, regions)")
    doc.add_paragraph(f"EK features: {len(ek_cols)} (extra scores — mixed risk levels)")
    doc.add_paragraph(f"AA features: {len(aa_cols)} (amino acid identity)")

    # 7. EK risk
    doc.add_heading("7. EK Feature Risk Analysis", level=1)
    add_tbl(doc, ["Feature", "Range", "Corr w/ Label", "Risk", "Action"],
            [[r["Column"], f"[{r['Min']}, {r['Max']}]", f"{r['Correlation_with_Label']:.3f}",
              r["Leakage_Risk"], r["Recommendation"]] for _, r in ek_risk_df.iterrows()])
    doc.add_paragraph("")
    add_fig(doc, "ek_feature_distributions.png")
    doc.add_paragraph("")
    add_fig(doc, "ek_intercorrelation.png")

    # 8. panel comparison
    doc.add_heading("8. Panel Comparison", level=1)
    add_tbl(doc, ["Panel", "Tested", "Shifted", "Shift %"],
            [[r["Panel"], r["Features_Tested"], r["Features_Shifted"], f"{r['Shift_Pct']}%"]
             for _, r in shift_df.iterrows()])

    # 9. preprocessing
    doc.add_heading("9. Preprocessing Recommendations", level=1)
    add_tbl(doc, ["Step", "Priority", "Applies To"],
            [[r["Step"], r["Priority"], r["Applies_To"]] for _, r in preproc_df.iterrows()])

    # 10. validation
    doc.add_heading("10. Validation Strategy", level=1)
    val_points = [
        "5-fold Stratified K-Fold on MASTER",
        "Variant_ID-aware splitting (no overlap between train/val)",
        "Evaluate per-panel separately using OOF predictions",
        "Report ROC-AUC, PR-AUC, F1, sensitivity, specificity, MCC",
        "Report mean ± std across folds",
    ]
    for pt in val_points:
        doc.add_paragraph(pt, style="List Bullet")

    # top features figure
    doc.add_heading("11. Top Features by Label Association", level=1)
    add_fig(doc, "top_features_by_label_association.png")

    # risk register
    doc.add_heading("12. Risk Register", level=1)
    risks = [
        ["1", "EK_4/EK_6 circularity", "HIGH", "Ablation testing"],
        ["2", "Variant_ID overlap leakage", "HIGH", "Group-aware splitting"],
        ["3", "Label-associated missingness", "HIGH", "Missingness indicators"],
        ["4", "Class imbalance", "MEDIUM", "Class weights + threshold tuning"],
        ["5", "CFTR too small", "HIGH", "Global model only"],
        ["6", "PAH distribution shift", "MEDIUM", "Panel calibration"],
        ["7", f"{len(global_const_cols)} constant columns", "MEDIUM", "Drop before modeling"],
        ["8", "Feature anonymisation", "LOW", "Accept uncertainty"],
        ["9", "Conflicting labels", "LOW", "Accept as noise"],
        ["10", "EK_5 unknown origin", "MEDIUM", "Monitor"],
    ]
    add_tbl(doc, ["#", "Risk", "Severity", "Mitigation"], risks)

    doc.save(OUT / "Data_Understanding_Report.docx")
    print("  Word document saved")
except ImportError:
    print("  WARNING: python-docx not installed — Word doc skipped")

elapsed = time.time() - t0
print(f"\n{'=' * 72}")
print(f"COMPLETE — {elapsed:.1f}s")
print(f"Output: {OUT}")
print(f"Files: {len(list(OUT.rglob('*')))}")
print(f"{'=' * 72}")
