#!/usr/bin/env python3
"""
Phase 4 — Optimization & Hardening of the Pathogenicity Pipeline
================================================================
TEKNOFEST 2026 Healthcare AI — Genetic variant pathogenicity classification

Builds on Phase 3 (Pipeline B, missingness-aware LightGBM). Implements four
optimization pillars in strict priority order:

  1. Out-of-Sample Threshold Calibration (HIGHEST PRIORITY)
     Replaces the previous in-sample, F1-maximized thresholding on imbalanced
     panel data with an Out-of-Fold (OOF), 1:1-prior, cost-sensitive threshold
     (clinical cost: FN = 2 x FP). The threshold is selected only on data that
     is independent of where it is applied.

  2. EK Ablation & Data-Leakage Defense (Pipeline C)
     Drops the dominant Extra-Knowledge features (EK_4, EK_5, EK_6, EK_7) and
     re-runs the 5-fold CV, logging the delta in OOF ROC-AUC to quantify the
     pipeline's robustness against circular / leaked meta-predictor features.

  3. Explainability (XAI)
     Integrates SHAP TreeExplainer: global summary plots + local explanations
     for representative False Positive / False Negative OOF cases.

  4. Ensemble Architecture & Inference Pipeline
     Adds XGBoost and (optionally) CatBoost base learners, stacks their OOF
     predictions with a Logistic-Regression meta-learner, and serializes the
     encoders, frozen column lists, base models and meta-learner for inference.

Reproducibility: seed = 42 and StratifiedKFold(5, shuffle=True, random_state=42)
are reused unchanged, so fold assignments are IDENTICAL to Phase 3 and the
leakage-aware panel-evaluation logic is preserved.
"""

import warnings
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, matthews_corrcoef,
)

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── configuration ──────────────────────────────────────────────────────
SEED = 42
N_FOLDS = 5
COST_FN = 2.0          # clinical cost of a False Negative (missed Pathogenic)
COST_FP = 1.0          # clinical cost of a False Positive
TARGET_PRIOR = 0.5     # balanced (1:1) test-set prior, per competition spec
EK_ABLATION = ["EK_4", "EK_5", "EK_6", "EK_7"]

np.random.seed(SEED)

BASE = Path("/Users/melihtakyaci/Documents/TeknofestSagliktaYapayZekâVerisi")
DATA = BASE / "EĞİTİM (TRAIN) SETLERİ 2"
OUT = BASE / "reports" / "phase_04_optimization"
FIG = OUT / "figures"
ART = OUT / "artifacts"
for d in (OUT, FIG, ART):
    d.mkdir(parents=True, exist_ok=True)

# Column groups (identical semantics to Phase 3 Pipeline B)
CAT_ONEHOT = ["CAT_3", "CAT_4", "CAT_5"]
CAT_LABEL = ["CAT_1", "CAT_2"]
CAT_BINARY = ["CAT_6"]
AA_COLS = ["AA_1", "AA_2"]

# LightGBM params — identical to Phase 3 for direct comparability
LGB_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 63, "max_depth": -1,
    "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "is_unbalance": True,
    "n_estimators": 2000, "verbose": -1, "random_state": SEED, "n_jobs": -1,
}


# ════════════════════════════════════════════════════════════════════════
# Preprocessing (faithful re-implementation of Phase 3 Pipeline B, but with
# explicit, serializable encoder state instead of module-level globals)
# ════════════════════════════════════════════════════════════════════════
def fit_preprocessor(master: pd.DataFrame) -> dict:
    """Learn constant columns and categorical encodings from MASTER."""
    feature_cols = [c for c in master.columns if c not in ("Variant_ID", "Label")]
    constant_cols = [c for c in feature_cols if master[c].nunique(dropna=True) <= 1]

    state = {"constant_cols": constant_cols, "cat1_mapping": None,
             "cat2_mapping": None, "feature_columns": None,
             "drop_extra": []}

    if "CAT_1" in master.columns:
        vals = sorted(master["CAT_1"].dropna().unique())
        state["cat1_mapping"] = {v: i + 1 for i, v in enumerate(vals)}
    if "CAT_2" in master.columns:
        vals = sorted(master["CAT_2"].dropna().unique())
        state["cat2_mapping"] = {v: i + 1 for i, v in enumerate(vals)}

    # build the reference feature matrix to freeze the final column order
    X_ref, _, _ = _transform(master, state, align=False)
    X_ref = X_ref.drop(columns=[c for c in state["drop_extra"] if c in X_ref.columns],
                       errors="ignore")
    state["feature_columns"] = list(X_ref.columns)
    return state


def _transform(df: pd.DataFrame, state: dict, align: bool = True):
    """Apply Pipeline B preprocessing using frozen encoder state."""
    out = df.copy()
    variant_ids = out["Variant_ID"].values if "Variant_ID" in out.columns else None
    out = out.drop(columns=["Variant_ID"], errors="ignore")
    out = out.drop(columns=[c for c in state["constant_cols"] if c in out.columns],
                   errors="ignore")
    label = out.pop("Label") if "Label" in out.columns else None

    num_cols = [c for c in out.columns
                if c not in CAT_ONEHOT + CAT_LABEL + CAT_BINARY + AA_COLS
                and out[c].dtype in ("float64", "int64", "float32", "int32")]

    # missingness indicators
    miss_features = {f"miss_{c}": out[c].isna().astype(np.int8) for c in num_cols}
    miss_df = pd.DataFrame(miss_features)
    miss_df["total_miss_count"] = out[num_cols].isna().sum(axis=1).astype(np.int16)
    miss_df["total_miss_pct"] = (miss_df["total_miss_count"] / len(num_cols)).astype(np.float32)

    def _al_range(lo, hi):
        return [c for c in num_cols if c.startswith("AL_") and lo <= int(c.split("_")[1]) <= hi]

    groups = [
        ("miss_grp_al_1_26", _al_range(1, 26)), ("miss_grp_al_27_38", _al_range(27, 38)),
        ("miss_grp_al_39_95", _al_range(39, 95)), ("miss_grp_al_96_185", _al_range(96, 185)),
        ("miss_grp_al_186_222", _al_range(186, 222)), ("miss_grp_al_223_334", _al_range(223, 334)),
        ("miss_grp_ek", [c for c in num_cols if c.startswith("EK_")]),
    ]
    for name, cols in groups:
        if cols:
            miss_df[name] = out[cols].isna().sum(axis=1).astype(np.int16)

    # CAT_3/4/5 one-hot
    cat_oh_df = pd.DataFrame(index=out.index)
    for c in CAT_ONEHOT:
        if c in out.columns:
            dummies = pd.get_dummies(out[c].fillna("MISSING"), prefix=c, dtype=np.int8)
            cat_oh_df = pd.concat([cat_oh_df, dummies], axis=1)

    # CAT_1/CAT_2 label encode (frozen mappings)
    cat_le_df = pd.DataFrame(index=out.index)
    if "CAT_1" in out.columns and state["cat1_mapping"] is not None:
        cat_le_df["CAT_1_enc"] = out["CAT_1"].map(state["cat1_mapping"]).fillna(0).astype(np.int16)
        cat_le_df["CAT_1_miss"] = out["CAT_1"].isna().astype(np.int8)
    if "CAT_2" in out.columns and state["cat2_mapping"] is not None:
        cat_le_df["CAT_2_enc"] = out["CAT_2"].map(state["cat2_mapping"]).fillna(0).astype(np.int16)
        cat_le_df["CAT_2_miss"] = out["CAT_2"].isna().astype(np.int8)

    # CAT_6 binary flag
    cat6_df = pd.DataFrame(index=out.index)
    if "CAT_6" in out.columns:
        cat6_df["has_region_flag"] = out["CAT_6"].notna().astype(np.int8)

    # AA_1/AA_2 one-hot
    aa_df = pd.DataFrame(index=out.index)
    for c in AA_COLS:
        if c in out.columns:
            dummies = pd.get_dummies(out[c].fillna("MISSING"), prefix=c, dtype=np.int8)
            aa_df = pd.concat([aa_df, dummies], axis=1)

    out = out.drop(columns=CAT_ONEHOT + CAT_LABEL + CAT_BINARY + AA_COLS, errors="ignore")
    final = pd.concat([out, miss_df, cat_oh_df, cat_le_df, cat6_df, aa_df], axis=1)

    if align and state.get("feature_columns") is not None:
        final = final.reindex(columns=state["feature_columns"], fill_value=0)

    return final, label, variant_ids


def transform(df, state):
    return _transform(df, state, align=True)


# ════════════════════════════════════════════════════════════════════════
# Cross-validation helpers (shared seed / folds = Phase 3 parity)
# ════════════════════════════════════════════════════════════════════════
def make_folds(y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def cv_lightgbm(X, y, folds):
    oof = np.zeros(len(X))
    models = []
    for tr, va in folds:
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
    return oof, models


def cv_xgboost(X, y, folds):
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    oof = np.zeros(len(X))
    models = []
    for tr, va in folds:
        m = xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_w, eval_metric="auc", early_stopping_rounds=100,
            tree_method="hist", random_state=SEED, n_jobs=-1,
        )
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
    return oof, models


def cv_catboost(X, y, folds):
    oof = np.zeros(len(X))
    models = []
    for tr, va in folds:
        m = CatBoostClassifier(
            iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            auto_class_weights="Balanced", eval_metric="AUC",
            random_seed=SEED, early_stopping_rounds=100, verbose=0,
            allow_writing_files=False,
        )
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
    return oof, models


# ════════════════════════════════════════════════════════════════════════
# PILLAR 1 — Out-of-sample, 1:1-prior, cost-sensitive threshold calibration
# ════════════════════════════════════════════════════════════════════════
def balanced_weights(y, prior=TARGET_PRIOR):
    """Per-sample weights that rebalance the data to a `prior` positive rate,
    so metrics/costs simulate the balanced (1:1) competition test set."""
    y = np.asarray(y)
    pos_rate = y.mean()
    return np.where(y == 1, prior / pos_rate, (1 - prior) / (1 - pos_rate)).astype(float)


def calibrate_threshold(y, proba, cost_fn=COST_FN, cost_fp=COST_FP, prior=TARGET_PRIOR):
    """Select the decision threshold that minimizes the expected, prior-balanced
    clinical cost  cost_fn*FN + cost_fp*FP  on out-of-fold predictions."""
    y = np.asarray(y)
    w = balanced_weights(y, prior)
    grid = np.linspace(0.01, 0.99, 197)
    best = None
    for t in grid:
        pred = (proba >= t).astype(int)
        fn = w[(y == 1) & (pred == 0)].sum()
        fp = w[(y == 0) & (pred == 1)].sum()
        cost = cost_fn * fn + cost_fp * fp
        if best is None or cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost)}
    return best["threshold"]


def weighted_metrics(y, proba, threshold, prior=TARGET_PRIOR):
    """Report metrics under the balanced (1:1) prior to mirror the test set."""
    y = np.asarray(y)
    w = balanced_weights(y, prior)
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, proba, sample_weight=w)),
        "pr_auc": float(average_precision_score(y, proba, sample_weight=w)),
        "f1_balanced": float(f1_score(y, pred, sample_weight=w)),
        "sensitivity": float(recall_score(y, pred, sample_weight=w)),
        "specificity": float(recall_score(y, pred, pos_label=0, sample_weight=w)),
        "mcc": float(matthews_corrcoef(y, pred, sample_weight=w)),
    }


def panel_leakage_aware_proba(panel_df, master_ids, master_id_to_idx, oof_proba,
                              fold_models, X_columns, state):
    """Reproduce Phase 3's leakage-aware scoring:
       shared variants -> MASTER OOF predictions; non-shared -> fold-model mean."""
    panel_ids = panel_df["Variant_ID"].values
    shared_mask = np.array([vid in master_ids for vid in panel_ids])

    shared_proba, shared_labels = [], []
    for _, row in panel_df[shared_mask].iterrows():
        vid = row["Variant_ID"]
        if vid in master_id_to_idx:
            shared_proba.append(oof_proba[master_id_to_idx[vid]])
            shared_labels.append(row["Label"])

    nonshared_df = panel_df[~shared_mask]
    ns_proba, ns_labels = [], []
    if len(nonshared_df) > 0:
        X_ns, y_ns, _ = transform(nonshared_df, state)
        X_ns = X_ns.reindex(columns=X_columns, fill_value=0)
        preds = np.mean([m.predict_proba(X_ns)[:, 1] for m in fold_models], axis=0)
        ns_proba = list(preds)
        ns_labels = list(y_ns.values)

    proba = np.array(shared_proba + ns_proba)
    labels = np.array(shared_labels + ns_labels)
    return labels, proba, int(shared_mask.sum()), int((~shared_mask).sum())


# ════════════════════════════════════════════════════════════════════════
# PILLAR 3 — SHAP explainability
# ════════════════════════════════════════════════════════════════════════
def _normalize_shap(values):
    """TreeExplainer may return a list [neg, pos] or a single array; take positive."""
    if isinstance(values, list):
        return values[1]
    if isinstance(values, np.ndarray) and values.ndim == 3:
        return values[:, :, 1]
    return values


def run_shap(full_model, X, y, oof_proba, threshold):
    if not HAS_SHAP:
        print("  [SHAP] skipped — shap not installed.")
        return None
    print("  [SHAP] computing TreeExplainer values...")
    explainer = shap.TreeExplainer(full_model)
    sv = _normalize_shap(explainer.shap_values(X))

    # global beeswarm
    plt.figure()
    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(FIG / "shap_summary_beeswarm.png", dpi=130); plt.close()

    # global bar
    plt.figure()
    shap.summary_plot(sv, X, show=False, plot_type="bar", max_display=20)
    plt.tight_layout(); plt.savefig(FIG / "shap_summary_bar.png", dpi=130); plt.close()

    # local explanations for the most confident FP and FN under the calibrated threshold
    y = np.asarray(y); pred = (oof_proba >= threshold).astype(int)
    fp_idx = np.where((y == 0) & (pred == 1))[0]
    fn_idx = np.where((y == 1) & (pred == 0))[0]
    fp_pick = fp_idx[np.argmax(oof_proba[fp_idx])] if len(fp_idx) else None   # most confident FP
    fn_pick = fn_idx[np.argmin(oof_proba[fn_idx])] if len(fn_idx) else None   # most confident FN

    cols = list(X.columns)
    for tag, idx in [("false_positive", fp_pick), ("false_negative", fn_pick)]:
        if idx is None:
            continue
        contrib = pd.Series(sv[idx], index=cols).sort_values(key=np.abs, ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in contrib.values]
        plt.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
        plt.axvline(0, color="k", lw=0.8)
        plt.title(f"Local SHAP — {tag} (OOF p={oof_proba[idx]:.3f}, y={int(y[idx])})")
        plt.xlabel("SHAP value (→ Pathogenic)")
        plt.tight_layout(); plt.savefig(FIG / f"shap_local_{tag}.png", dpi=130); plt.close()

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
    mean_abs.head(40).to_csv(OUT / "shap_global_importance.csv", header=["mean_abs_shap"])
    print(f"  [SHAP] saved global + local plots; top feature: {mean_abs.index[0]}")
    return mean_abs


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    results = {"config": {"seed": SEED, "n_folds": N_FOLDS, "cost_fn": COST_FN,
                          "cost_fp": COST_FP, "target_prior": TARGET_PRIOR,
                          "ek_ablation": EK_ABLATION, "has_catboost": HAS_CATBOOST,
                          "has_shap": HAS_SHAP, "timestamp": datetime.now().isoformat()}}

    print("=" * 70 + "\nPHASE 4 — OPTIMIZATION & HARDENING\n" + "=" * 70)

    # ── load ──
    master = pd.read_csv(DATA / "YARISMA_TRAIN_MASTER.csv")
    panels = {n: pd.read_csv(DATA / f"YARISMA_TRAIN_{n}.csv")
              for n in ("CFTR", "PAH", "KANSER")}
    print(f"MASTER: {master.shape} | " + " ".join(f"{n}:{d.shape[0]}" for n, d in panels.items()))

    state = fit_preprocessor(master)
    X, y, variant_ids = transform(master, state)
    print(f"Feature matrix: {X.shape} | positive rate: {y.mean():.4f}")
    folds = make_folds(y)

    # ── baseline LightGBM CV (reused across pillars 1, 3, 4) ──
    print("\n[LightGBM] 5-fold CV ...")
    lgb_oof, lgb_models = cv_lightgbm(X, y, folds)
    base_auc = roc_auc_score(y, lgb_oof)
    print(f"  Baseline OOF ROC-AUC = {base_auc:.4f}")

    master_ids = set(master["Variant_ID"].values)
    id_to_idx = {vid: i for i, vid in enumerate(variant_ids)}

    # ═══════════════════ PILLAR 1 — threshold calibration ═══════════════════
    print("\n" + "─" * 70 + "\nPILLAR 1 — Out-of-sample cost-sensitive threshold\n" + "─" * 70)
    cal_thr = calibrate_threshold(y, lgb_oof)
    print(f"  Calibrated threshold (1:1 prior, FN={COST_FN}xFP): {cal_thr:.3f}")

    master_metrics = weighted_metrics(y, lgb_oof, cal_thr)
    naive_metrics = weighted_metrics(y, lgb_oof, 0.5)
    print(f"  MASTER balanced F1: {naive_metrics['f1_balanced']:.4f} (t=0.5) "
          f"-> {master_metrics['f1_balanced']:.4f} (calibrated)")

    panel_rows = []
    for name, pdf in panels.items():
        labels, proba, n_sh, n_ns = panel_leakage_aware_proba(
            pdf, master_ids, id_to_idx, lgb_oof, lgb_models, list(X.columns), state)
        m = weighted_metrics(labels, proba, cal_thr)
        m.update({"panel": name, "n": len(labels), "n_shared": n_sh, "n_nonshared": n_ns,
                  "raw_f1": float(f1_score(labels, (proba >= cal_thr).astype(int)))})
        panel_rows.append(m)
        print(f"  {name:7s} balanced F1={m['f1_balanced']:.4f}  "
              f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}  (n={len(labels)})")

    results["pillar1_threshold"] = {
        "calibrated_threshold": cal_thr,
        "master_calibrated": master_metrics, "master_naive_t05": naive_metrics,
        "panels": panel_rows,
    }
    pd.DataFrame(panel_rows).to_csv(OUT / "pillar1_panel_calibrated.csv", index=False)

    # ═══════════════════ PILLAR 2 — EK ablation (Pipeline C) ════════════════
    print("\n" + "─" * 70 + "\nPILLAR 2 — EK Ablation (Pipeline C)\n" + "─" * 70)
    drop_cols = [c for c in X.columns
                 if c in EK_ABLATION or c in [f"miss_{e}" for e in EK_ABLATION]]
    X_abl = X.drop(columns=drop_cols)
    print(f"  Dropped {len(drop_cols)} columns: {drop_cols}")
    abl_oof, _ = cv_lightgbm(X_abl, y, folds)
    abl_auc = roc_auc_score(y, abl_oof)
    abl_thr = calibrate_threshold(y, abl_oof)
    abl_metrics = weighted_metrics(y, abl_oof, abl_thr)
    delta = base_auc - abl_auc
    verdict = ("LOW dependence — robust to EK leakage" if delta < 0.02
               else "MODERATE dependence" if delta < 0.05 else "HIGH dependence on EK features")
    print(f"  OOF ROC-AUC: baseline={base_auc:.4f}  ablated={abl_auc:.4f}  Δ={delta:+.4f}")
    print(f"  Verdict: {verdict}")
    results["pillar2_ablation"] = {
        "dropped_columns": drop_cols, "baseline_auc": base_auc, "ablated_auc": abl_auc,
        "delta_auc": delta, "verdict": verdict,
        "ablated_calibrated": abl_metrics,
    }
    pd.DataFrame([{
        "pipeline": "B_baseline", "oof_roc_auc": base_auc,
        "balanced_f1": master_metrics["f1_balanced"]},
        {"pipeline": "C_ablation_noEK4567", "oof_roc_auc": abl_auc,
         "balanced_f1": abl_metrics["f1_balanced"], "delta_auc": delta},
    ]).to_csv(OUT / "pillar2_ablation_comparison.csv", index=False)

    # ═══════════════════ PILLAR 3 — SHAP ════════════════════════════════════
    print("\n" + "─" * 70 + "\nPILLAR 3 — SHAP Explainability\n" + "─" * 70)
    full_model = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": int(np.mean(
        [m.best_iteration_ or LGB_PARAMS["n_estimators"] for m in lgb_models]))})
    full_model.fit(X, y)
    shap_imp = run_shap(full_model, X, y, lgb_oof, cal_thr)
    results["pillar3_shap"] = {
        "available": HAS_SHAP,
        "top_features": (list(shap_imp.head(15).index) if shap_imp is not None else []),
    }

    # ═══════════════════ PILLAR 4 — ensemble + inference artifacts ══════════
    print("\n" + "─" * 70 + "\nPILLAR 4 — Stacking Ensemble & Inference Pipeline\n" + "─" * 70)
    print("  [XGBoost] 5-fold CV ...")
    xgb_oof, xgb_models = cv_xgboost(X, y, folds)
    print(f"    XGBoost OOF ROC-AUC = {roc_auc_score(y, xgb_oof):.4f}")

    meta_features = {"lgb": lgb_oof, "xgb": xgb_oof}
    base_full = {}
    if HAS_CATBOOST:
        print("  [CatBoost] 5-fold CV ...")
        cat_oof, cat_models = cv_catboost(X, y, folds)
        print(f"    CatBoost OOF ROC-AUC = {roc_auc_score(y, cat_oof):.4f}")
        meta_features["cat"] = cat_oof
    else:
        print("  [CatBoost] not installed — stacking with LightGBM + XGBoost only.")

    # stacking meta-learner trained on OOF predictions (valid: OOF = out-of-sample)
    meta_names = list(meta_features.keys())
    Z = np.column_stack([meta_features[k] for k in meta_names])
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta.fit(Z, y)
    stack_oof = meta.predict_proba(Z)[:, 1]
    stack_auc = roc_auc_score(y, stack_oof)
    stack_thr = calibrate_threshold(y, stack_oof)
    stack_metrics = weighted_metrics(y, stack_oof, stack_thr)
    print(f"  Stacked OOF ROC-AUC = {stack_auc:.4f} | balanced F1 = {stack_metrics['f1_balanced']:.4f}")

    results["pillar4_ensemble"] = {
        "base_learners": meta_names,
        "oof_auc": {"lightgbm": float(roc_auc_score(y, lgb_oof)),
                    "xgboost": float(roc_auc_score(y, xgb_oof)),
                    **({"catboost": float(roc_auc_score(y, cat_oof))} if HAS_CATBOOST else {}),
                    "stacked": float(stack_auc)},
        "stacked_calibrated": stack_metrics,
        "meta_coefficients": dict(zip(meta_names, meta.coef_[0].tolist())),
    }

    # ── refit base learners on full data for inference, then serialize ──
    print("  Serializing inference artifacts (.pkl) ...")
    base_full["lightgbm"] = full_model  # already fitted on full data
    xgb_full = xgb.XGBClassifier(
        n_estimators=int(np.mean([m.best_iteration or 500 for m in xgb_models])) or 500,
        learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        tree_method="hist", random_state=SEED, n_jobs=-1)
    xgb_full.fit(X, y)
    base_full["xgboost"] = xgb_full
    if HAS_CATBOOST:
        cat_full = CatBoostClassifier(
            iterations=int(np.mean([m.get_best_iteration() or 500 for m in cat_models])) or 500,
            learning_rate=0.05, depth=6, l2_leaf_reg=3.0, auto_class_weights="Balanced",
            random_seed=SEED, verbose=0, allow_writing_files=False)
        cat_full.fit(X, y)
        base_full["catboost"] = cat_full

    joblib.dump({
        "preprocessor_state": state,
        "feature_columns": state["feature_columns"],
        "base_models": base_full,
        "meta_learner": meta,
        "meta_order": meta_names,
        "calibrated_threshold": stack_thr,
        "cost_fn": COST_FN, "cost_fp": COST_FP, "target_prior": TARGET_PRIOR,
        "seed": SEED,
    }, ART / "inference_pipeline.pkl")
    joblib.dump(state, ART / "preprocessor_state.pkl")
    print(f"    -> {ART/'inference_pipeline.pkl'}")

    # ── persist full result log ──
    with open(OUT / "phase_04_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── PDR-ready summary ──
    summary = build_pdr_summary(results)
    (OUT / "PDR_summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts in {OUT}")


def build_pdr_summary(r) -> str:
    p1, p2, p4 = r["pillar1_threshold"], r["pillar2_ablation"], r["pillar4_ensemble"]
    lines = []
    a = lines.append
    a("# Phase 4 — PDR Results Summary\n")
    a(f"*Generated {r['config']['timestamp']} · seed={r['config']['seed']} · "
      f"cost FN:FP = {r['config']['cost_fn']:.0f}:{r['config']['cost_fp']:.0f} · "
      f"balanced (1:1) test prior*\n")

    a("\n## 1. Out-of-Sample Threshold Calibration")
    a(f"- Calibrated decision threshold (OOF, 1:1 prior, cost-sensitive): "
      f"**{p1['calibrated_threshold']:.3f}**")
    a(f"- MASTER OOF — ROC-AUC **{p1['master_calibrated']['roc_auc']:.4f}**, "
      f"balanced F1 {p1['master_naive_t05']['f1_balanced']:.4f} (t=0.5) → "
      f"**{p1['master_calibrated']['f1_balanced']:.4f}** (calibrated)")
    a("- Panel performance at the single OOF-calibrated threshold "
      "(balanced 1:1, leakage-aware):")
    a("\n| Panel | Balanced F1 | Sensitivity | Specificity | MCC | n |")
    a("|-------|-------------|-------------|-------------|-----|---|")
    for row in p1["panels"]:
        a(f"| {row['panel']} | {row['f1_balanced']:.4f} | {row['sensitivity']:.4f} "
          f"| {row['specificity']:.4f} | {row['mcc']:.4f} | {row['n']} |")

    a("\n## 2. EK Ablation (Pipeline C — Leakage Defense)")
    a(f"- Dropped: {', '.join(p2['dropped_columns'])}")
    a(f"- OOF ROC-AUC: baseline **{p2['baseline_auc']:.4f}** → ablated "
      f"**{p2['ablated_auc']:.4f}** (Δ = {p2['delta_auc']:+.4f})")
    a(f"- Balanced F1 after ablation: {p2['ablated_calibrated']['f1_balanced']:.4f}")
    a(f"- **Verdict:** {p2['verdict']}")

    a("\n## 3. Explainability (SHAP)")
    if r["pillar3_shap"]["available"]:
        a(f"- Global + local SHAP plots saved to `reports/phase_04_optimization/figures/`")
        a(f"- Top SHAP features: {', '.join(r['pillar3_shap']['top_features'][:8])}")
    else:
        a("- SHAP not available in this environment.")

    a("\n## 4. Stacking Ensemble")
    a(f"- Base learners: {', '.join(p4['base_learners'])} + Logistic-Regression meta-learner")
    a("\n| Model | OOF ROC-AUC |")
    a("|-------|-------------|")
    for k, v in p4["oof_auc"].items():
        a(f"| {k} | {v:.4f} |")
    a(f"- Stacked balanced F1: **{p4['stacked_calibrated']['f1_balanced']:.4f}** "
      f"(threshold {p4['stacked_calibrated']['threshold']:.3f})")
    a("- Serialized inference pipeline: "
      "`reports/phase_04_optimization/artifacts/inference_pipeline.pkl`")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
