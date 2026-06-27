#!/usr/bin/env python3
"""
Phase 3 — Pipeline B: Missingness-Aware GBDT (LightGBM)
=========================================================
TEKNOFEST Healthcare AI Competition
Missense variant pathogenicity classification

Pipeline B steps:
  1. Drop Variant_ID
  2. Drop global constant columns (nunique <= 1 in MASTER)
  3. Add missingness indicators (binary per feature)
  4. Add row-level missing count + missing percentage
  5. One-hot encode CAT_3, CAT_4, CAT_5
  6. Label encode CAT_1, CAT_2
  7. Binary flag for CAT_6
  8. One-hot encode AA_1, AA_2
  9. Train LightGBM with is_unbalance=True
 10. 5-fold Stratified CV with OOF predictions
 11. Evaluate: ROC-AUC, PR-AUC, F1, Sensitivity, Specificity, MCC
 12. Feature importance (gain + split)
 13. Per-panel evaluation
 14. Threshold optimisation
 15. Generate reports, notebook, Word doc
"""

import warnings, os, json, time, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, balanced_accuracy_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
    precision_score, recall_score, accuracy_score,
    log_loss, brier_score_loss,
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path("/Users/melihtakyaci/Documents/TeknofestSagliktaYapayZekâVerisi")
DATA = BASE / "EĞİTİM (TRAIN) SETLERİ 2"
OUT  = BASE / "reports" / "phase_03_pipeline_b"
FIG  = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 3 — PIPELINE B: Missingness-Aware GBDT")
print("=" * 70)

# ── 1. load data ───────────────────────────────────────────────────────
t0 = time.time()
print("\n[1/15] Loading datasets...")

master = pd.read_csv(DATA / "YARISMA_TRAIN_MASTER.csv")
cftr   = pd.read_csv(DATA / "YARISMA_TRAIN_CFTR.csv")
pah    = pd.read_csv(DATA / "YARISMA_TRAIN_PAH.csv")
kanser = pd.read_csv(DATA / "YARISMA_TRAIN_KANSER.csv")

panels = {"CFTR": cftr, "PAH": pah, "KANSER": kanser}
print(f"  MASTER: {master.shape}")
for name, df in panels.items():
    print(f"  {name}: {df.shape}")

# ── 2. identify constant columns ──────────────────────────────────────
print("\n[2/15] Identifying constant columns in MASTER...")

feature_cols = [c for c in master.columns if c not in ("Variant_ID", "Label")]
constant_cols = []
for c in feature_cols:
    if master[c].nunique(dropna=True) <= 1:
        constant_cols.append(c)

print(f"  Constant columns to drop: {len(constant_cols)}")

# ── 3. preprocessing function ─────────────────────────────────────────
print("\n[3/15] Building preprocessing pipeline...")

CAT_ONEHOT = ["CAT_3", "CAT_4", "CAT_5"]
CAT_LABEL  = ["CAT_1", "CAT_2"]
CAT_BINARY = ["CAT_6"]
AA_COLS    = ["AA_1", "AA_2"]

cat1_mapping = None
cat2_mapping = None

def preprocess(df, fit=False):
    """Apply Pipeline B preprocessing. fit=True learns encodings from data."""
    global cat1_mapping, cat2_mapping

    out = df.copy()

    # drop Variant_ID (save it first)
    variant_ids = out["Variant_ID"].values if "Variant_ID" in out.columns else None
    out = out.drop(columns=["Variant_ID"], errors="ignore")

    # drop constant columns
    out = out.drop(columns=[c for c in constant_cols if c in out.columns], errors="ignore")

    # separate label
    label = out.pop("Label") if "Label" in out.columns else None

    # identify numeric feature columns (after dropping constants, Variant_ID, Label)
    num_cols = [c for c in out.columns
                if c not in CAT_ONEHOT + CAT_LABEL + CAT_BINARY + AA_COLS
                and out[c].dtype in ("float64", "int64", "float32", "int32")]

    # ── missingness indicators ──
    miss_features = {}
    for c in num_cols:
        miss_col = f"miss_{c}"
        miss_features[miss_col] = out[c].isna().astype(np.int8)

    miss_df = pd.DataFrame(miss_features)

    # row-level missing stats
    miss_df["total_miss_count"] = out[num_cols].isna().sum(axis=1).astype(np.int16)
    miss_df["total_miss_pct"]   = (miss_df["total_miss_count"] / len(num_cols)).astype(np.float32)

    # group-level missing counts
    al_1_26  = [c for c in num_cols if c.startswith("AL_") and int(c.split("_")[1]) <= 26]
    al_27_38 = [c for c in num_cols if c.startswith("AL_") and 27 <= int(c.split("_")[1]) <= 38]
    al_39_95 = [c for c in num_cols if c.startswith("AL_") and 39 <= int(c.split("_")[1]) <= 95]
    al_96_185 = [c for c in num_cols if c.startswith("AL_") and 96 <= int(c.split("_")[1]) <= 185]
    al_186_222 = [c for c in num_cols if c.startswith("AL_") and 186 <= int(c.split("_")[1]) <= 222]
    al_223_334 = [c for c in num_cols if c.startswith("AL_") and 223 <= int(c.split("_")[1]) <= 334]
    ek_cols = [c for c in num_cols if c.startswith("EK_")]

    for grp_name, grp_cols in [
        ("miss_grp_al_1_26", al_1_26), ("miss_grp_al_27_38", al_27_38),
        ("miss_grp_al_39_95", al_39_95), ("miss_grp_al_96_185", al_96_185),
        ("miss_grp_al_186_222", al_186_222), ("miss_grp_al_223_334", al_223_334),
        ("miss_grp_ek", ek_cols),
    ]:
        if grp_cols:
            miss_df[grp_name] = out[grp_cols].isna().sum(axis=1).astype(np.int16)

    # ── CAT_3/4/5 one-hot ──
    cat_oh_df = pd.DataFrame(index=out.index)
    for c in CAT_ONEHOT:
        if c in out.columns:
            filled = out[c].fillna("MISSING")
            dummies = pd.get_dummies(filled, prefix=c, dtype=np.int8)
            cat_oh_df = pd.concat([cat_oh_df, dummies], axis=1)

    # ── CAT_1/CAT_2 label encode ──
    if fit:
        if "CAT_1" in out.columns:
            vals = out["CAT_1"].dropna().unique()
            cat1_mapping = {v: i + 1 for i, v in enumerate(sorted(vals))}
        if "CAT_2" in out.columns:
            vals = out["CAT_2"].dropna().unique()
            cat2_mapping = {v: i + 1 for i, v in enumerate(sorted(vals))}

    cat_le_df = pd.DataFrame(index=out.index)
    if "CAT_1" in out.columns and cat1_mapping is not None:
        cat_le_df["CAT_1_enc"] = out["CAT_1"].map(cat1_mapping).fillna(0).astype(np.int16)
        cat_le_df["CAT_1_miss"] = out["CAT_1"].isna().astype(np.int8)
    if "CAT_2" in out.columns and cat2_mapping is not None:
        cat_le_df["CAT_2_enc"] = out["CAT_2"].map(cat2_mapping).fillna(0).astype(np.int16)
        cat_le_df["CAT_2_miss"] = out["CAT_2"].isna().astype(np.int8)

    # ── CAT_6 binary flag ──
    cat6_df = pd.DataFrame(index=out.index)
    if "CAT_6" in out.columns:
        cat6_df["has_region_flag"] = out["CAT_6"].notna().astype(np.int8)

    # ── AA_1/AA_2 one-hot ──
    aa_df = pd.DataFrame(index=out.index)
    for c in AA_COLS:
        if c in out.columns:
            filled = out[c].fillna("MISSING")
            dummies = pd.get_dummies(filled, prefix=c, dtype=np.int8)
            aa_df = pd.concat([aa_df, dummies], axis=1)

    # drop original categorical/AA columns from numeric frame
    out = out.drop(columns=CAT_ONEHOT + CAT_LABEL + CAT_BINARY + AA_COLS, errors="ignore")

    # ── assemble final feature matrix ──
    final = pd.concat([out, miss_df, cat_oh_df, cat_le_df, cat6_df, aa_df], axis=1)

    return final, label, variant_ids


# ── 4. preprocess MASTER ──────────────────────────────────────────────
print("\n[4/15] Preprocessing MASTER dataset...")

X, y, variant_ids = preprocess(master, fit=True)
print(f"  Feature matrix: {X.shape}")
print(f"  Label: {y.shape}, positive rate: {y.mean():.4f}")
print(f"  Missingness indicator features: {sum(1 for c in X.columns if c.startswith('miss_'))}")
print(f"  Total features: {X.shape[1]}")

# ── 5. LightGBM hyperparameters ──────────────────────────────────────
print("\n[5/15] Setting LightGBM parameters...")

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "n_estimators": 2000,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

print(f"  Key params: lr={lgb_params['learning_rate']}, "
      f"leaves={lgb_params['num_leaves']}, "
      f"is_unbalance=True, "
      f"n_estimators={lgb_params['n_estimators']}")

# ── 6. 5-fold stratified CV ──────────────────────────────────────────
print("\n[6/15] Running 5-fold Stratified Cross-Validation...")

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_proba = np.zeros(len(X))
oof_preds = np.zeros(len(X))
fold_models = []
fold_metrics = []
feature_importance_gain = np.zeros(X.shape[1])
feature_importance_split = np.zeros(X.shape[1])
fold_best_iters = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}:")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print(f"    Train: {len(train_idx)} (pos={y_train.sum():.0f}, "
          f"neg={len(y_train)-y_train.sum():.0f})")
    print(f"    Val:   {len(val_idx)} (pos={y_val.sum():.0f}, "
          f"neg={len(y_val)-y_val.sum():.0f})")

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = model.best_iteration_
    fold_best_iters.append(best_iter)
    print(f"    Best iteration: {best_iter}")

    val_proba = model.predict_proba(X_val)[:, 1]
    oof_proba[val_idx] = val_proba

    val_auc = roc_auc_score(y_val, val_proba)
    val_prauc = average_precision_score(y_val, val_proba)
    val_logloss = log_loss(y_val, val_proba)
    val_brier = brier_score_loss(y_val, val_proba)

    val_pred = (val_proba >= 0.5).astype(int)
    val_f1 = f1_score(y_val, val_pred)
    val_sens = recall_score(y_val, val_pred)
    val_spec = recall_score(y_val, val_pred, pos_label=0)
    val_mcc = matthews_corrcoef(y_val, val_pred)
    val_bacc = balanced_accuracy_score(y_val, val_pred)

    fold_metrics.append({
        "fold": fold_idx + 1,
        "best_iteration": best_iter,
        "roc_auc": val_auc,
        "pr_auc": val_prauc,
        "log_loss": val_logloss,
        "brier_score": val_brier,
        "f1": val_f1,
        "sensitivity": val_sens,
        "specificity": val_spec,
        "mcc": val_mcc,
        "balanced_accuracy": val_bacc,
    })

    print(f"    ROC-AUC={val_auc:.4f}  PR-AUC={val_prauc:.4f}  "
          f"F1={val_f1:.4f}  Sens={val_sens:.4f}  Spec={val_spec:.4f}")

    feature_importance_gain += model.booster_.feature_importance(importance_type="gain")
    feature_importance_split += model.booster_.feature_importance(importance_type="split")
    fold_models.append(model)

feature_importance_gain /= N_FOLDS
feature_importance_split /= N_FOLDS

# ── 7. overall OOF metrics ───────────────────────────────────────────
print("\n[7/15] Computing overall OOF metrics...")

oof_pred_05 = (oof_proba >= 0.5).astype(int)

overall_metrics = {
    "roc_auc": roc_auc_score(y, oof_proba),
    "pr_auc": average_precision_score(y, oof_proba),
    "log_loss": log_loss(y, oof_proba),
    "brier_score": brier_score_loss(y, oof_proba),
    "f1_at_05": f1_score(y, oof_pred_05),
    "sensitivity_at_05": recall_score(y, oof_pred_05),
    "specificity_at_05": recall_score(y, oof_pred_05, pos_label=0),
    "precision_at_05": precision_score(y, oof_pred_05),
    "mcc_at_05": matthews_corrcoef(y, oof_pred_05),
    "balanced_accuracy_at_05": balanced_accuracy_score(y, oof_pred_05),
    "accuracy_at_05": accuracy_score(y, oof_pred_05),
}

print(f"  OOF ROC-AUC:  {overall_metrics['roc_auc']:.4f}")
print(f"  OOF PR-AUC:   {overall_metrics['pr_auc']:.4f}")
print(f"  OOF Log Loss:  {overall_metrics['log_loss']:.4f}")
print(f"  OOF F1@0.5:    {overall_metrics['f1_at_05']:.4f}")
print(f"  OOF Sens@0.5:  {overall_metrics['sensitivity_at_05']:.4f}")
print(f"  OOF Spec@0.5:  {overall_metrics['specificity_at_05']:.4f}")
print(f"  OOF MCC@0.5:   {overall_metrics['mcc_at_05']:.4f}")

# ── 8. threshold optimisation ─────────────────────────────────────────
print("\n[8/15] Optimising classification threshold...")

precisions, recalls, thresholds_pr = precision_recall_curve(y, oof_proba)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds_pr[best_f1_idx]
best_f1 = f1_scores[best_f1_idx]

# sensitivity-constrained: find threshold where sensitivity >= 0.95
fpr, tpr, thresholds_roc = roc_curve(y, oof_proba)
sens95_mask = tpr >= 0.95
if sens95_mask.any():
    sens95_idx = np.where(sens95_mask)[0]
    spec_at_sens95 = 1 - fpr[sens95_idx]
    best_spec_idx = sens95_idx[np.argmax(spec_at_sens95)]
    sens95_threshold = thresholds_roc[best_spec_idx]
    sens95_sensitivity = tpr[best_spec_idx]
    sens95_specificity = 1 - fpr[best_spec_idx]
else:
    sens95_threshold = 0.5
    sens95_sensitivity = recall_score(y, oof_pred_05)
    sens95_specificity = recall_score(y, oof_pred_05, pos_label=0)

# Youden's J
j_scores = tpr - fpr
best_j_idx = np.argmax(j_scores)
youden_threshold = thresholds_roc[best_j_idx]
youden_sens = tpr[best_j_idx]
youden_spec = 1 - fpr[best_j_idx]

# MCC optimisation
mcc_scores = []
test_thresholds = np.arange(0.1, 0.95, 0.01)
for t in test_thresholds:
    pred = (oof_proba >= t).astype(int)
    mcc_scores.append(matthews_corrcoef(y, pred))
mcc_scores = np.array(mcc_scores)
best_mcc_idx = np.argmax(mcc_scores)
best_mcc_threshold = test_thresholds[best_mcc_idx]
best_mcc = mcc_scores[best_mcc_idx]

threshold_results = {
    "default_05": {"threshold": 0.5,
        "sensitivity": overall_metrics["sensitivity_at_05"],
        "specificity": overall_metrics["specificity_at_05"],
        "f1": overall_metrics["f1_at_05"],
        "mcc": overall_metrics["mcc_at_05"]},
    "best_f1": {"threshold": float(best_f1_threshold),
        "f1": float(best_f1),
        "sensitivity": float(recalls[best_f1_idx]),
        "specificity": float("nan")},
    "sensitivity_95": {"threshold": float(sens95_threshold),
        "sensitivity": float(sens95_sensitivity),
        "specificity": float(sens95_specificity)},
    "youden_j": {"threshold": float(youden_threshold),
        "sensitivity": float(youden_sens),
        "specificity": float(youden_spec)},
    "best_mcc": {"threshold": float(best_mcc_threshold),
        "mcc": float(best_mcc)},
}

# compute full metrics at best thresholds
for key in ["best_f1", "sensitivity_95", "youden_j", "best_mcc"]:
    t = threshold_results[key]["threshold"]
    pred = (oof_proba >= t).astype(int)
    threshold_results[key]["sensitivity"] = float(recall_score(y, pred))
    threshold_results[key]["specificity"] = float(recall_score(y, pred, pos_label=0))
    threshold_results[key]["f1"] = float(f1_score(y, pred))
    threshold_results[key]["mcc"] = float(matthews_corrcoef(y, pred))
    threshold_results[key]["precision"] = float(precision_score(y, pred))
    threshold_results[key]["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))

print(f"  Best F1 threshold:     {best_f1_threshold:.3f} → F1={best_f1:.4f}")
print(f"  Sensitivity≥95% thr:   {sens95_threshold:.3f} → Sens={sens95_sensitivity:.4f}, Spec={sens95_specificity:.4f}")
print(f"  Youden's J threshold:  {youden_threshold:.3f} → Sens={youden_sens:.4f}, Spec={youden_spec:.4f}")
print(f"  Best MCC threshold:    {best_mcc_threshold:.3f} → MCC={best_mcc:.4f}")

# ── 9. feature importance ─────────────────────────────────────────────
print("\n[9/15] Computing feature importance...")

fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance_gain": feature_importance_gain,
    "importance_split": feature_importance_split,
})
fi_df = fi_df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
fi_df["rank_gain"] = fi_df.index + 1
fi_df = fi_df.sort_values("importance_split", ascending=False)
fi_df["rank_split"] = range(1, len(fi_df) + 1)
fi_df = fi_df.sort_values("rank_gain")

# categorise features
def categorise_feature(name):
    if name.startswith("miss_grp_"):
        return "missingness_group"
    if name.startswith("miss_"):
        return "missingness_indicator"
    if name in ("total_miss_count", "total_miss_pct"):
        return "missingness_aggregate"
    if name.startswith("CAT_"):
        return "categorical"
    if name.startswith("AA_"):
        return "amino_acid"
    if name.startswith("EK_"):
        return "extra_knowledge"
    if name.startswith("AL_"):
        return "allele_annotation"
    if name == "has_region_flag":
        return "categorical"
    return "other"

fi_df["category"] = fi_df["feature"].apply(categorise_feature)

top30_gain = fi_df.head(30)
print(f"  Top 5 by gain:")
for _, row in fi_df.head(5).iterrows():
    print(f"    {row['feature']:30s}  gain={row['importance_gain']:10.1f}  ({row['category']})")

n_zero = (fi_df["importance_gain"] == 0).sum()
print(f"  Features with zero gain importance: {n_zero}/{len(fi_df)}")

# ── 10. per-panel evaluation ──────────────────────────────────────────
print("\n[10/15] Per-panel evaluation...")

panel_results = {}
master_ids = set(master["Variant_ID"].values)

for panel_name, panel_df in panels.items():
    print(f"\n  {panel_name} (n={len(panel_df)}):")

    panel_ids = panel_df["Variant_ID"].values
    shared_mask = np.array([vid in master_ids for vid in panel_ids])
    shared_ids = set(panel_ids[shared_mask])

    # for shared variants: use OOF predictions from MASTER
    master_id_to_idx = {vid: i for i, vid in enumerate(variant_ids)}
    shared_oof_proba = []
    shared_labels = []
    shared_variant_list = []

    for _, row in panel_df[shared_mask].iterrows():
        vid = row["Variant_ID"]
        if vid in master_id_to_idx:
            idx = master_id_to_idx[vid]
            shared_oof_proba.append(oof_proba[idx])
            shared_labels.append(row["Label"])
            shared_variant_list.append(vid)

    # for non-shared variants: predict using average of fold models
    nonshared_df = panel_df[~shared_mask]
    nonshared_proba = []
    nonshared_labels = []

    if len(nonshared_df) > 0:
        X_ns, y_ns, _ = preprocess(nonshared_df, fit=False)
        # align columns
        missing_cols = set(X.columns) - set(X_ns.columns)
        for mc in missing_cols:
            X_ns[mc] = 0
        X_ns = X_ns[X.columns]

        for model in fold_models:
            p = model.predict_proba(X_ns)[:, 1]
            nonshared_proba.append(p)
        nonshared_proba = np.mean(nonshared_proba, axis=0)
        nonshared_labels = y_ns.values if y_ns is not None else nonshared_df["Label"].values

    # combine
    all_proba = np.array(shared_oof_proba + list(nonshared_proba))
    all_labels = np.array(shared_labels + list(nonshared_labels))

    if len(np.unique(all_labels)) < 2:
        print(f"    SKIP — only one class present")
        continue

    p_auc = roc_auc_score(all_labels, all_proba)
    p_prauc = average_precision_score(all_labels, all_proba)
    p_pred = (all_proba >= 0.5).astype(int)
    p_f1 = f1_score(all_labels, p_pred)
    p_sens = recall_score(all_labels, p_pred)
    p_spec = recall_score(all_labels, p_pred, pos_label=0)
    p_mcc = matthews_corrcoef(all_labels, p_pred)

    # optimal threshold for this panel
    pr_p, re_p, th_p = precision_recall_curve(all_labels, all_proba)
    f1_p = 2 * pr_p * re_p / (pr_p + re_p + 1e-12)
    best_p_idx = np.argmax(f1_p)
    panel_best_threshold = th_p[best_p_idx] if best_p_idx < len(th_p) else 0.5

    p_pred_opt = (all_proba >= panel_best_threshold).astype(int)
    p_f1_opt = f1_score(all_labels, p_pred_opt)
    p_sens_opt = recall_score(all_labels, p_pred_opt)
    p_spec_opt = recall_score(all_labels, p_pred_opt, pos_label=0)

    panel_results[panel_name] = {
        "n": len(all_labels),
        "n_shared": len(shared_oof_proba),
        "n_nonshared": len(nonshared_proba) if len(nonshared_df) > 0 else 0,
        "pathogenic_rate": all_labels.mean(),
        "roc_auc": p_auc,
        "pr_auc": p_prauc,
        "f1_at_05": p_f1,
        "sensitivity_at_05": p_sens,
        "specificity_at_05": p_spec,
        "mcc_at_05": p_mcc,
        "optimal_threshold": float(panel_best_threshold),
        "f1_at_optimal": p_f1_opt,
        "sensitivity_at_optimal": p_sens_opt,
        "specificity_at_optimal": p_spec_opt,
    }

    print(f"    Shared/Non-shared: {len(shared_oof_proba)}/{len(nonshared_proba) if len(nonshared_df) > 0 else 0}")
    print(f"    ROC-AUC={p_auc:.4f}  PR-AUC={p_prauc:.4f}  F1@0.5={p_f1:.4f}")
    print(f"    Sens@0.5={p_sens:.4f}  Spec@0.5={p_spec:.4f}  MCC={p_mcc:.4f}")
    print(f"    Optimal threshold={panel_best_threshold:.3f} → F1={p_f1_opt:.4f}  Sens={p_sens_opt:.4f}")

# ── 11. calibration analysis ──────────────────────────────────────────
print("\n[11/15] Calibration analysis...")

prob_true, prob_pred = calibration_curve(y, oof_proba, n_bins=10, strategy="uniform")
calibration_data = pd.DataFrame({
    "bin_predicted": prob_pred,
    "bin_actual": prob_true,
    "gap": np.abs(prob_true - prob_pred),
})
ece = np.mean(np.abs(prob_true - prob_pred))
print(f"  Expected Calibration Error (ECE): {ece:.4f}")

# ── 12. figures ───────────────────────────────────────────────────────
print("\n[12/15] Generating figures...")

# fig 1: ROC curve
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"OOF ROC-AUC = {overall_metrics['roc_auc']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].scatter([1 - youden_spec], [youden_sens], c="red", s=100, zorder=5,
                label=f"Youden (t={youden_threshold:.3f})")
axes[0].scatter([1 - sens95_specificity], [sens95_sensitivity], c="green", s=100, zorder=5,
                label=f"Sens≥95% (t={sens95_threshold:.3f})")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (OOF)")
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# fig 2: PR curve
axes[1].plot(recalls, precisions, "r-", linewidth=2,
             label=f"OOF PR-AUC = {overall_metrics['pr_auc']:.4f}")
axes[1].axhline(y=y.mean(), color="k", linestyle="--", alpha=0.5, label=f"Baseline = {y.mean():.3f}")
axes[1].scatter([recalls[best_f1_idx]], [precisions[best_f1_idx]], c="blue", s=100, zorder=5,
                label=f"Best F1 (t={best_f1_threshold:.3f})")
axes[1].set_xlabel("Recall (Sensitivity)")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve (OOF)")
axes[1].legend(loc="lower left")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# fig 3: confusion matrix at multiple thresholds
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (thr_name, thr_val) in zip(axes, [
    ("Default (0.5)", 0.5),
    (f"Youden ({youden_threshold:.3f})", youden_threshold),
    (f"Sens≥95% ({sens95_threshold:.3f})", sens95_threshold),
]):
    pred = (oof_proba >= thr_val).astype(int)
    cm = confusion_matrix(y, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Benign", "Pathogenic"],
                yticklabels=["Benign", "Pathogenic"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Threshold: {thr_name}\nF1={f1_score(y, pred):.3f}")
plt.tight_layout()
plt.savefig(FIG / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# fig 4: feature importance top 30
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
top30 = fi_df.head(30)

colors = {"extra_knowledge": "#e74c3c", "allele_annotation": "#3498db",
          "missingness_indicator": "#2ecc71", "missingness_aggregate": "#27ae60",
          "missingness_group": "#16a085", "categorical": "#9b59b6",
          "amino_acid": "#f39c12", "other": "#95a5a6"}

bar_colors = [colors.get(c, "#95a5a6") for c in top30["category"]]

axes[0].barh(range(30), top30["importance_gain"].values, color=bar_colors)
axes[0].set_yticks(range(30))
axes[0].set_yticklabels(top30["feature"].values, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean Gain")
axes[0].set_title("Top 30 Features by Gain Importance")

top30_split = fi_df.sort_values("importance_split", ascending=False).head(30)
bar_colors_s = [colors.get(c, "#95a5a6") for c in top30_split["category"]]
axes[1].barh(range(30), top30_split["importance_split"].values, color=bar_colors_s)
axes[1].set_yticks(range(30))
axes[1].set_yticklabels(top30_split["feature"].values, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_xlabel("Mean Split Count")
axes[1].set_title("Top 30 Features by Split Importance")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items() if k in fi_df["category"].values]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(FIG / "feature_importance_top30.png", dpi=150, bbox_inches="tight")
plt.close()

# fig 5: probability distribution by class
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(oof_proba[y == 0], bins=50, alpha=0.6, color="blue", label=f"Benign (n={int((y==0).sum())})", density=True)
ax.hist(oof_proba[y == 1], bins=50, alpha=0.6, color="red", label=f"Pathogenic (n={int((y==1).sum())})", density=True)
ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold=0.5")
ax.axvline(x=youden_threshold, color="green", linestyle="--", label=f"Youden={youden_threshold:.3f}")
ax.set_xlabel("Predicted Probability (Pathogenic)")
ax.set_ylabel("Density")
ax.set_title("OOF Prediction Distribution by True Class")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "prediction_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# fig 6: calibration plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(prob_pred, prob_true, "ro-", linewidth=2, label=f"LightGBM (ECE={ece:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve (OOF)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# fig 7: fold stability
fig, ax = plt.subplots(figsize=(10, 6))
fold_df = pd.DataFrame(fold_metrics)
metrics_to_plot = ["roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "mcc"]
x = np.arange(N_FOLDS)
width = 0.12
for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, fold_df[metric], width, label=metric, alpha=0.8)
ax.set_xlabel("Fold")
ax.set_ylabel("Score")
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
ax.set_title("Per-Fold Metric Stability")
ax.legend(loc="lower right", fontsize=8)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIG / "fold_stability.png", dpi=150, bbox_inches="tight")
plt.close()

# fig 8: threshold sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sweep_thresholds = np.arange(0.05, 0.96, 0.01)
sweep_sens, sweep_spec, sweep_f1, sweep_mcc_vals = [], [], [], []
for t in sweep_thresholds:
    p = (oof_proba >= t).astype(int)
    sweep_sens.append(recall_score(y, p))
    sweep_spec.append(recall_score(y, p, pos_label=0))
    sweep_f1.append(f1_score(y, p))
    sweep_mcc_vals.append(matthews_corrcoef(y, p))

axes[0].plot(sweep_thresholds, sweep_sens, "r-", label="Sensitivity", linewidth=2)
axes[0].plot(sweep_thresholds, sweep_spec, "b-", label="Specificity", linewidth=2)
axes[0].plot(sweep_thresholds, sweep_f1, "g-", label="F1", linewidth=2)
axes[0].axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
axes[0].axvline(x=youden_threshold, color="purple", linestyle="--", alpha=0.5, label=f"Youden={youden_threshold:.3f}")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Score")
axes[0].set_title("Sensitivity / Specificity / F1 vs Threshold")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(sweep_thresholds, sweep_mcc_vals, "m-", linewidth=2)
axes[1].axvline(x=best_mcc_threshold, color="red", linestyle="--",
                label=f"Best MCC={best_mcc:.4f} @ t={best_mcc_threshold:.3f}")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("MCC")
axes[1].set_title("Matthews Correlation Coefficient vs Threshold")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "threshold_sweep.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved 8 figures to {FIG}")

# ── 13. CSV outputs ───────────────────────────────────────────────────
print("\n[13/15] Saving CSV outputs...")

# fold metrics
fold_df.to_csv(OUT / "fold_metrics.csv", index=False)

# overall metrics
pd.DataFrame([overall_metrics]).to_csv(OUT / "overall_oof_metrics.csv", index=False)

# threshold results
thr_rows = []
for name, vals in threshold_results.items():
    row = {"strategy": name}
    row.update(vals)
    thr_rows.append(row)
pd.DataFrame(thr_rows).to_csv(OUT / "threshold_optimisation.csv", index=False)

# feature importance
fi_df.to_csv(OUT / "feature_importance.csv", index=False)

# top 50 feature importance
fi_df.head(50).to_csv(OUT / "feature_importance_top50.csv", index=False)

# panel results
panel_rows = []
for name, vals in panel_results.items():
    row = {"panel": name}
    row.update(vals)
    panel_rows.append(row)
pd.DataFrame(panel_rows).to_csv(OUT / "panel_evaluation.csv", index=False)

# calibration data
calibration_data.to_csv(OUT / "calibration_data.csv", index=False)

# OOF predictions
oof_df = pd.DataFrame({
    "Variant_ID": variant_ids,
    "Label": y.values,
    "oof_proba": oof_proba,
    "oof_pred_05": oof_pred_05,
    "oof_pred_youden": (oof_proba >= youden_threshold).astype(int),
    "oof_pred_sens95": (oof_proba >= sens95_threshold).astype(int),
})
oof_df.to_csv(OUT / "oof_predictions.csv", index=False)

# feature category importance summary
cat_imp = fi_df.groupby("category").agg(
    n_features=("feature", "count"),
    mean_gain=("importance_gain", "mean"),
    total_gain=("importance_gain", "sum"),
    max_gain=("importance_gain", "max"),
    mean_split=("importance_split", "mean"),
    top_feature=("feature", "first"),
).sort_values("total_gain", ascending=False)
cat_imp.to_csv(OUT / "feature_category_importance.csv")

# missingness feature contribution
miss_fi = fi_df[fi_df["category"].str.startswith("missingness")]
miss_fi.to_csv(OUT / "missingness_feature_importance.csv", index=False)

print(f"  Saved CSVs to {OUT}")

# ── 14. markdown reports ──────────────────────────────────────────────
print("\n[14/15] Generating markdown reports...")

# — main results report —
fold_df_r = pd.DataFrame(fold_metrics)
fold_summary = fold_df_r[["roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "mcc"]].describe().loc[["mean", "std"]]

md_results = f"""# Phase 3 — Pipeline B: Missingness-Aware GBDT Results

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | LightGBM (GBDT) |
| Learning Rate | {lgb_params['learning_rate']} |
| Num Leaves | {lgb_params['num_leaves']} |
| Max Estimators | {lgb_params['n_estimators']} |
| Early Stopping | 100 rounds |
| Class Balance | is_unbalance=True |
| CV Strategy | 5-fold Stratified |
| Total Features | {X.shape[1]} |
| Training Samples | {len(X)} |

## Preprocessing (Pipeline B)

1. Dropped Variant_ID
2. Dropped {len(constant_cols)} constant columns
3. Added {sum(1 for c in X.columns if c.startswith('miss_') and not c.startswith('miss_grp'))} per-feature missingness indicators
4. Added {sum(1 for c in X.columns if c.startswith('miss_grp_'))} group-level missingness counts
5. Added total_miss_count + total_miss_pct
6. One-hot encoded CAT_3, CAT_4, CAT_5
7. Label encoded CAT_1, CAT_2 (with missingness flags)
8. Binary flag for CAT_6
9. One-hot encoded AA_1, AA_2

## Overall OOF Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | **{overall_metrics['roc_auc']:.4f}** |
| PR-AUC | **{overall_metrics['pr_auc']:.4f}** |
| Log Loss | {overall_metrics['log_loss']:.4f} |
| Brier Score | {overall_metrics['brier_score']:.4f} |
| F1 @ 0.5 | {overall_metrics['f1_at_05']:.4f} |
| Sensitivity @ 0.5 | {overall_metrics['sensitivity_at_05']:.4f} |
| Specificity @ 0.5 | {overall_metrics['specificity_at_05']:.4f} |
| MCC @ 0.5 | {overall_metrics['mcc_at_05']:.4f} |
| Balanced Accuracy @ 0.5 | {overall_metrics['balanced_accuracy_at_05']:.4f} |
| ECE (Calibration) | {ece:.4f} |

## Per-Fold Stability

| Fold | ROC-AUC | PR-AUC | F1 | Sensitivity | Specificity | MCC | Best Iter |
|------|---------|--------|----|-------------|-------------|-----|-----------|
"""
for _, row in fold_df_r.iterrows():
    md_results += (f"| {int(row['fold'])} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | "
                   f"{row['f1']:.4f} | {row['sensitivity']:.4f} | {row['specificity']:.4f} | "
                   f"{row['mcc']:.4f} | {int(row['best_iteration'])} |\n")

md_results += f"""
**Mean ± Std**:
- ROC-AUC: {fold_summary.loc['mean','roc_auc']:.4f} ± {fold_summary.loc['std','roc_auc']:.4f}
- PR-AUC:  {fold_summary.loc['mean','pr_auc']:.4f} ± {fold_summary.loc['std','pr_auc']:.4f}
- F1:      {fold_summary.loc['mean','f1']:.4f} ± {fold_summary.loc['std','f1']:.4f}
- MCC:     {fold_summary.loc['mean','mcc']:.4f} ± {fold_summary.loc['std','mcc']:.4f}

## Threshold Optimisation

| Strategy | Threshold | Sensitivity | Specificity | F1 | MCC |
|----------|-----------|-------------|-------------|----|-----|
| Default (0.5) | 0.500 | {threshold_results['default_05']['sensitivity']:.4f} | {threshold_results['default_05']['specificity']:.4f} | {threshold_results['default_05']['f1']:.4f} | {threshold_results['default_05']['mcc']:.4f} |
| Best F1 | {threshold_results['best_f1']['threshold']:.3f} | {threshold_results['best_f1']['sensitivity']:.4f} | {threshold_results['best_f1']['specificity']:.4f} | {threshold_results['best_f1']['f1']:.4f} | {threshold_results['best_f1']['mcc']:.4f} |
| Sensitivity ≥ 95% | {threshold_results['sensitivity_95']['threshold']:.3f} | {threshold_results['sensitivity_95']['sensitivity']:.4f} | {threshold_results['sensitivity_95']['specificity']:.4f} | {threshold_results['sensitivity_95']['f1']:.4f} | {threshold_results['sensitivity_95']['mcc']:.4f} |
| Youden's J | {threshold_results['youden_j']['threshold']:.3f} | {threshold_results['youden_j']['sensitivity']:.4f} | {threshold_results['youden_j']['specificity']:.4f} | {threshold_results['youden_j']['f1']:.4f} | {threshold_results['youden_j']['mcc']:.4f} |
| Best MCC | {threshold_results['best_mcc']['threshold']:.3f} | {threshold_results['best_mcc']['sensitivity']:.4f} | {threshold_results['best_mcc']['specificity']:.4f} | {threshold_results['best_mcc']['f1']:.4f} | {threshold_results['best_mcc']['mcc']:.4f} |

## Top 30 Features by Gain Importance

| Rank | Feature | Category | Gain | Split Count |
|------|---------|----------|------|-------------|
"""
for _, row in fi_df.head(30).iterrows():
    md_results += (f"| {int(row['rank_gain'])} | {row['feature']} | {row['category']} | "
                   f"{row['importance_gain']:.1f} | {row['importance_split']:.0f} |\n")

md_results += f"""
## Feature Category Importance

| Category | N Features | Mean Gain | Total Gain | Top Feature |
|----------|-----------|-----------|------------|-------------|
"""
for cat, row in cat_imp.iterrows():
    md_results += (f"| {cat} | {int(row['n_features'])} | {row['mean_gain']:.1f} | "
                   f"{row['total_gain']:.0f} | {row['top_feature']} |\n")

md_results += f"""
## Missingness Feature Contribution

Total missingness-derived features: {len(miss_fi)}
Non-zero importance: {(miss_fi['importance_gain'] > 0).sum()}
Total gain from missingness features: {miss_fi['importance_gain'].sum():.1f}
Percentage of total gain: {miss_fi['importance_gain'].sum() / fi_df['importance_gain'].sum() * 100:.1f}%

## Per-Panel Performance

| Panel | N | Shared | ROC-AUC | PR-AUC | F1@0.5 | Sens@0.5 | Spec@0.5 | Optimal Thr | F1@Opt |
|-------|---|--------|---------|--------|--------|----------|----------|-------------|--------|
"""
for name, vals in panel_results.items():
    md_results += (f"| {name} | {vals['n']} | {vals['n_shared']} | "
                   f"{vals['roc_auc']:.4f} | {vals['pr_auc']:.4f} | "
                   f"{vals['f1_at_05']:.4f} | {vals['sensitivity_at_05']:.4f} | "
                   f"{vals['specificity_at_05']:.4f} | {vals['optimal_threshold']:.3f} | "
                   f"{vals['f1_at_optimal']:.4f} |\n")

md_results += """
## Figures

- ![ROC & PR Curves](figures/roc_pr_curves.png)
- ![Confusion Matrices](figures/confusion_matrices.png)
- ![Feature Importance](figures/feature_importance_top30.png)
- ![Prediction Distribution](figures/prediction_distribution.png)
- ![Calibration Curve](figures/calibration_curve.png)
- ![Fold Stability](figures/fold_stability.png)
- ![Threshold Sweep](figures/threshold_sweep.png)
"""

with open(OUT / "pipeline_b_results.md", "w") as f:
    f.write(md_results)

# — executive summary —
md_exec = f"""# Phase 3 — Pipeline B: Executive Summary

## Key Result

**OOF ROC-AUC = {overall_metrics['roc_auc']:.4f}** using LightGBM with missingness-aware preprocessing.

## Performance at a Glance

| Metric | Value |
|--------|-------|
| ROC-AUC | {overall_metrics['roc_auc']:.4f} |
| PR-AUC | {overall_metrics['pr_auc']:.4f} |
| F1 (Youden threshold) | {threshold_results['youden_j']['f1']:.4f} |
| Sensitivity @ ≥95% target | {threshold_results['sensitivity_95']['sensitivity']:.4f} (threshold={threshold_results['sensitivity_95']['threshold']:.3f}) |
| Specificity @ ≥95% sensitivity | {threshold_results['sensitivity_95']['specificity']:.4f} |
| MCC (best) | {threshold_results['best_mcc']['mcc']:.4f} |
| Fold Stability (AUC std) | {fold_summary.loc['std','roc_auc']:.4f} |
| ECE (calibration error) | {ece:.4f} |

## What Worked

1. **Missingness indicators**: Contributed {miss_fi['importance_gain'].sum() / fi_df['importance_gain'].sum() * 100:.1f}% of total feature importance gain
2. **Native missing handling**: LightGBM handled 54.9% mean row missingness without imputation
3. **Class weighting**: is_unbalance=True handled the 73.3% pathogenic imbalance
4. **Early stopping**: Models converged at ~{int(np.mean(fold_best_iters))} iterations (of 2000 max)

## Top 5 Most Important Features

| Rank | Feature | Category | Gain |
|------|---------|----------|------|
"""
for _, row in fi_df.head(5).iterrows():
    md_exec += f"| {int(row['rank_gain'])} | {row['feature']} | {row['category']} | {row['importance_gain']:.1f} |\n"

md_exec += f"""
## Panel Performance

| Panel | ROC-AUC | F1@0.5 | Sens@0.5 | Notes |
|-------|---------|--------|----------|-------|
"""
for name, vals in panel_results.items():
    note = ""
    if name == "CFTR":
        note = "Small sample (n=111); high variance expected"
    elif name == "PAH":
        note = "High imbalance (5:1); threshold tuning critical"
    elif name == "KANSER":
        note = "Differential missingness leveraged"
    md_exec += (f"| {name} | {vals['roc_auc']:.4f} | {vals['f1_at_05']:.4f} | "
                f"{vals['sensitivity_at_05']:.4f} | {note} |\n")

md_exec += f"""
## Recommended Threshold

For clinical use (minimize missed pathogenic variants):
- **Threshold = {threshold_results['sensitivity_95']['threshold']:.3f}** → Sensitivity = {threshold_results['sensitivity_95']['sensitivity']:.4f}, Specificity = {threshold_results['sensitivity_95']['specificity']:.4f}

For balanced classification:
- **Threshold = {threshold_results['youden_j']['threshold']:.3f}** → Sensitivity = {threshold_results['youden_j']['sensitivity']:.4f}, Specificity = {threshold_results['youden_j']['specificity']:.4f}

## Next Steps

1. **Pipeline C (Anti-Leakage)**: Drop EK_4/5/6 and re-train to quantify circularity
2. **Pipeline A comparison**: Minimal baseline without missingness features
3. **Hyperparameter tuning**: Bayesian optimisation on Pipeline B
4. **XGBoost/CatBoost comparison**: Alternative GBDT implementations
5. **Stacking ensemble (Pipeline E)**: Combine multiple GBDT models
"""

with open(OUT / "pipeline_b_executive_summary.md", "w") as f:
    f.write(md_exec)

print("  Saved markdown reports")

# ── 15. notebook + Word doc ───────────────────────────────────────────
print("\n[15/15] Generating Jupyter notebook and Word document...")

# — notebook —
nb_cells = []

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().split("\n"),
            "id": f"cell_{len(nb_cells)}"}

def code_cell(source):
    return {"cell_type": "code", "metadata": {}, "source": source.strip().split("\n"),
            "outputs": [], "execution_count": None, "id": f"cell_{len(nb_cells)}"}

nb_cells.append(md_cell(f"# TEKNOFEST Healthcare AI — Phase 3: Pipeline B Results\n\n"
                         f"**Model**: LightGBM GBDT | **CV**: 5-fold Stratified | **Date**: {datetime.now().strftime('%Y-%m-%d')}"))

nb_cells.append(code_cell("""import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, Image
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.4f}'.format)"""))

nb_cells.append(md_cell("## Overall OOF Metrics"))
nb_cells.append(code_cell("""metrics = pd.read_csv("reports/phase_03_pipeline_b/overall_oof_metrics.csv")
display(metrics.T.rename(columns={0: "Value"}))"""))

nb_cells.append(md_cell("## Per-Fold Stability"))
nb_cells.append(code_cell("""folds = pd.read_csv("reports/phase_03_pipeline_b/fold_metrics.csv")
display(folds)
print(f"\\nMean ROC-AUC: {folds['roc_auc'].mean():.4f} ± {folds['roc_auc'].std():.4f}")
print(f"Mean PR-AUC:  {folds['pr_auc'].mean():.4f} ± {folds['pr_auc'].std():.4f}")"""))

nb_cells.append(md_cell("## ROC & PR Curves"))
nb_cells.append(code_cell('Image("reports/phase_03_pipeline_b/figures/roc_pr_curves.png")'))

nb_cells.append(md_cell("## Confusion Matrices at Different Thresholds"))
nb_cells.append(code_cell('Image("reports/phase_03_pipeline_b/figures/confusion_matrices.png")'))

nb_cells.append(md_cell("## Threshold Optimisation"))
nb_cells.append(code_cell("""thresholds = pd.read_csv("reports/phase_03_pipeline_b/threshold_optimisation.csv")
display(thresholds)"""))

nb_cells.append(code_cell('Image("reports/phase_03_pipeline_b/figures/threshold_sweep.png")'))

nb_cells.append(md_cell("## Top 30 Feature Importance"))
nb_cells.append(code_cell("""fi = pd.read_csv("reports/phase_03_pipeline_b/feature_importance_top50.csv")
display(fi.head(30)[['rank_gain', 'feature', 'category', 'importance_gain', 'importance_split']])"""))

nb_cells.append(code_cell('Image("reports/phase_03_pipeline_b/figures/feature_importance_top30.png")'))

nb_cells.append(md_cell("## Feature Category Importance"))
nb_cells.append(code_cell("""cat_imp = pd.read_csv("reports/phase_03_pipeline_b/feature_category_importance.csv")
display(cat_imp)"""))

nb_cells.append(md_cell("## Missingness Feature Contribution"))
nb_cells.append(code_cell("""miss_fi = pd.read_csv("reports/phase_03_pipeline_b/missingness_feature_importance.csv")
print(f"Total missingness features: {len(miss_fi)}")
print(f"Non-zero importance: {(miss_fi['importance_gain'] > 0).sum()}")
print(f"Total gain: {miss_fi['importance_gain'].sum():.1f}")
display(miss_fi[miss_fi['importance_gain'] > 0].head(20))"""))

nb_cells.append(md_cell("## Prediction Distribution"))
nb_cells.append(code_cell('Image("reports/phase_03_pipeline_b/figures/prediction_distribution.png")'))

nb_cells.append(md_cell("## Calibration"))
nb_cells.append(code_cell("""cal = pd.read_csv("reports/phase_03_pipeline_b/calibration_data.csv")
display(cal)"""))
nb_cells.append(code_cell('Image("reports/phase_03_pipeline_b/figures/calibration_curve.png")'))

nb_cells.append(md_cell("## Per-Panel Performance"))
nb_cells.append(code_cell("""panels = pd.read_csv("reports/phase_03_pipeline_b/panel_evaluation.csv")
display(panels)"""))

nb_cells.append(md_cell("## Fold Stability"))
nb_cells.append(code_cell('Image("reports/phase_03_pipeline_b/figures/fold_stability.png")'))

nb_cells.append(md_cell("## OOF Predictions Sample"))
nb_cells.append(code_cell("""oof = pd.read_csv("reports/phase_03_pipeline_b/oof_predictions.csv")
print(f"Total predictions: {len(oof)}")
display(oof.head(10))"""))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": nb_cells,
}
with open(OUT / "Phase_03_Pipeline_B_Notebook.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

# — Word document —
try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()
    style = doc.styles["Normal"]
    style.font.size = Pt(10)
    style.font.name = "Calibri"

    doc.add_heading("TEKNOFEST Healthcare AI — Phase 3: Pipeline B Results", 0)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph(f"Model: LightGBM GBDT | CV: 5-fold Stratified | Features: {X.shape[1]}")

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(f"OOF ROC-AUC = {overall_metrics['roc_auc']:.4f}")
    doc.add_paragraph(f"OOF PR-AUC = {overall_metrics['pr_auc']:.4f}")
    doc.add_paragraph(f"Best F1 = {threshold_results['best_f1']['f1']:.4f} "
                      f"(threshold = {threshold_results['best_f1']['threshold']:.3f})")
    doc.add_paragraph(f"Sensitivity ≥ 95% achieved at threshold = "
                      f"{threshold_results['sensitivity_95']['threshold']:.3f} "
                      f"(specificity = {threshold_results['sensitivity_95']['specificity']:.4f})")
    doc.add_paragraph(f"Fold stability: ROC-AUC std = {fold_summary.loc['std','roc_auc']:.4f}")
    doc.add_paragraph(f"ECE (calibration error) = {ece:.4f}")

    doc.add_heading("Overall OOF Metrics", level=1)
    tbl = doc.add_table(rows=1, cols=2, style="Light Shading Accent 1")
    tbl.rows[0].cells[0].text = "Metric"
    tbl.rows[0].cells[1].text = "Value"
    for k, v in overall_metrics.items():
        row = tbl.add_row()
        row.cells[0].text = k
        row.cells[1].text = f"{v:.4f}"

    doc.add_heading("Per-Fold Results", level=1)
    cols = ["fold", "roc_auc", "pr_auc", "f1", "sensitivity", "specificity", "mcc"]
    tbl = doc.add_table(rows=1, cols=len(cols), style="Light Shading Accent 1")
    for i, c in enumerate(cols):
        tbl.rows[0].cells[i].text = c
    for _, row_data in fold_df_r.iterrows():
        row = tbl.add_row()
        for i, c in enumerate(cols):
            val = row_data[c]
            row.cells[i].text = f"{int(val)}" if c == "fold" else f"{val:.4f}"

    doc.add_heading("Threshold Optimisation", level=1)
    tbl = doc.add_table(rows=1, cols=6, style="Light Shading Accent 1")
    for i, h in enumerate(["Strategy", "Threshold", "Sensitivity", "Specificity", "F1", "MCC"]):
        tbl.rows[0].cells[i].text = h
    for name, vals in threshold_results.items():
        row = tbl.add_row()
        row.cells[0].text = name
        row.cells[1].text = f"{vals['threshold']:.3f}"
        row.cells[2].text = f"{vals.get('sensitivity', 0):.4f}"
        row.cells[3].text = f"{vals.get('specificity', 0):.4f}"
        row.cells[4].text = f"{vals.get('f1', 0):.4f}"
        row.cells[5].text = f"{vals.get('mcc', 0):.4f}"

    doc.add_heading("Top 30 Features by Importance", level=1)
    tbl = doc.add_table(rows=1, cols=4, style="Light Shading Accent 1")
    for i, h in enumerate(["Rank", "Feature", "Category", "Gain"]):
        tbl.rows[0].cells[i].text = h
    for _, row_data in fi_df.head(30).iterrows():
        row = tbl.add_row()
        row.cells[0].text = str(int(row_data["rank_gain"]))
        row.cells[1].text = row_data["feature"]
        row.cells[2].text = row_data["category"]
        row.cells[3].text = f"{row_data['importance_gain']:.1f}"

    doc.add_heading("Per-Panel Performance", level=1)
    tbl = doc.add_table(rows=1, cols=7, style="Light Shading Accent 1")
    for i, h in enumerate(["Panel", "N", "ROC-AUC", "PR-AUC", "F1@0.5", "Sens@0.5", "Spec@0.5"]):
        tbl.rows[0].cells[i].text = h
    for name, vals in panel_results.items():
        row = tbl.add_row()
        row.cells[0].text = name
        row.cells[1].text = str(vals["n"])
        row.cells[2].text = f"{vals['roc_auc']:.4f}"
        row.cells[3].text = f"{vals['pr_auc']:.4f}"
        row.cells[4].text = f"{vals['f1_at_05']:.4f}"
        row.cells[5].text = f"{vals['sensitivity_at_05']:.4f}"
        row.cells[6].text = f"{vals['specificity_at_05']:.4f}"

    doc.add_heading("Figures", level=1)
    for fig_name in [
        "roc_pr_curves.png", "confusion_matrices.png", "feature_importance_top30.png",
        "prediction_distribution.png", "calibration_curve.png", "fold_stability.png",
        "threshold_sweep.png",
    ]:
        fig_path = FIG / fig_name
        if fig_path.exists():
            doc.add_paragraph(fig_name.replace(".png", "").replace("_", " ").title())
            doc.add_picture(str(fig_path), width=Inches(6.0))

    doc.save(OUT / "Phase_03_Pipeline_B_Report.docx")
    print("  Word document generated")
except ImportError:
    print("  WARNING: python-docx not installed — Word doc skipped")

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"PHASE 3 PIPELINE B COMPLETE — {elapsed:.1f}s")
print(f"{'=' * 70}")
print(f"\nOutput directory: {OUT}")
print(f"Total files generated: {len(list(OUT.rglob('*')))}")
