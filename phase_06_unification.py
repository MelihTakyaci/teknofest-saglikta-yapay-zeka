#!/usr/bin/env python3
"""
Phase 6 — Threshold Unification, Error Profiling & PDR Metric Consolidation
===========================================================================
TEKNOFEST 2026 Healthcare AI

Enforces a single, internally consistent operating point for the PDR: the
STACKED ENSEMBLE evaluated at its calibrated threshold 0.305 (1:1 test prior,
2:1 clinical cost). No model is retrained for deployment.

Leakage discipline:
  - The frozen base models in inference_pipeline.pkl were refit on the FULL
    MASTER set. Therefore panel variants that also occur in MASTER (shared)
    MUST NOT be scored with the frozen models (that would leak training data).
  - Shared panel variants are scored with leakage-free out-of-fold (OOF)
    stacked probabilities, reproduced with the identical seed=42 / 5-fold
    splits and the FROZEN meta-learner from the pkl.
  - Non-shared panel variants and the held-out cohort are scored with the
    frozen pkl ensemble (they were never in MASTER training).
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from phase_04_optimization import (
    transform, make_folds, cv_lightgbm, cv_xgboost, cv_catboost,
)
from phase_05_external_validation import ensemble_predict, metrics_block, load_pipeline

BASE = Path("/Users/melihtakyaci/Documents/TeknofestSagliktaYapayZekâVerisi")
DATA = BASE / "EĞİTİM (TRAIN) SETLERİ 2"
OUT = BASE / "reports" / "phase_06_unification"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.305          # single unified operating point (stacked ensemble)
PANELS = ["CFTR", "KANSER", "PAH"]


def leakage_free_stacked_oof(X, y, pkl):
    """Reproduce per-variant stacked OOF probabilities (seed=42, identical
    folds) and apply the FROZEN meta-learner. No deployment model is retrained."""
    folds = make_folds(y)
    lgb_oof, _ = cv_lightgbm(X, y, folds)
    xgb_oof, _ = cv_xgboost(X, y, folds)
    cat_oof, _ = cv_catboost(X, y, folds)
    base = {"lgb": lgb_oof, "xgb": xgb_oof, "cat": cat_oof}
    Z = np.column_stack([base[k] for k in pkl["meta_order"]])
    return pkl["meta_learner"].predict_proba(Z)[:, 1], lgb_oof, xgb_oof, cat_oof


def panel_metrics(panel_df, master_ids, id_to_idx, stacked_oof, pkl):
    """Leakage-aware stacked-ensemble scoring for a full panel at THRESHOLD."""
    pids = panel_df["Variant_ID"].values
    shared_mask = np.array([v in master_ids for v in pids])

    proba, labels = [], []
    for _, row in panel_df[shared_mask].iterrows():
        if row["Variant_ID"] in id_to_idx:
            proba.append(stacked_oof[id_to_idx[row["Variant_ID"]]])
            labels.append(row["Label"])
    ns = panel_df[~shared_mask]
    if len(ns) > 0:
        p = ensemble_predict(ns, pkl)
        proba += list(p)
        labels += list(ns["Label"].values)

    proba, labels = np.array(proba), np.array(labels)
    m = metrics_block(labels, proba, THRESHOLD)
    m["n_shared"] = int(shared_mask.sum())
    m["n_nonshared"] = int((~shared_mask).sum())
    return m


def error_profile(y, proba, X, master, panels_raw):
    """Systematic FP / FN profiling at THRESHOLD on the MASTER OOF set."""
    y = np.asarray(y)
    pred = (proba >= THRESHOLD).astype(int)
    fp = (y == 0) & (pred == 1)
    fn = (y == 1) & (pred == 0)
    correct = y == pred

    # 1) missingness bands (row-level missing fraction)
    miss = X["total_miss_pct"].values
    bands = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    band_rows = []
    for lo, hi in bands:
        sel = (miss >= lo) & (miss < hi)
        n = int(sel.sum())
        err = int((fp | fn)[sel].sum())
        band_rows.append({
            "band": f"{int(lo*100)}-{int(hi*100 if hi <= 1 else 100)}%",
            "n": n, "errors": err,
            "error_rate": round(err / n, 4) if n else 0.0,
            "fp": int(fp[sel].sum()), "fn": int(fn[sel].sum()),
        })

    # 2) EK_7 signal in errors vs correct (conflicting-signal diagnosis)
    ek7 = X["EK_7"].values
    def _mean(mask):
        v = ek7[mask]; v = v[~np.isnan(v)]
        return round(float(v.mean()), 3) if len(v) else None
    ek7_profile = {"correct": _mean(correct), "false_positive": _mean(fp),
                   "false_negative": _mean(fn)}

    # 3) overlap of MASTER errors with the hard PAH panel
    master_ids = master["Variant_ID"].values
    pah_ids = set(panels_raw["PAH"]["Variant_ID"].values)
    in_pah = np.array([v in pah_ids for v in master_ids])
    pah_err_rate = round(float((fp | fn)[in_pah].mean()), 4) if in_pah.sum() else 0.0
    nonpah_err_rate = round(float((fp | fn)[~in_pah].mean()), 4)

    return {
        "n_fp": int(fp.sum()), "n_fn": int(fn.sum()),
        "fn_fp_ratio": round(int(fn.sum()) / max(int(fp.sum()), 1), 2),
        "missingness_bands": band_rows,
        "ek7_mean": ek7_profile,
        "pah_overlap_error_rate": pah_err_rate,
        "nonpah_error_rate": nonpah_err_rate,
    }


def main():
    print("=" * 70 + "\nPHASE 6 — THRESHOLD UNIFICATION @ 0.305 (STACKED ENSEMBLE)\n" + "=" * 70)
    pkl = load_pipeline()
    state = pkl["preprocessor_state"]

    master = pd.read_csv(DATA / "YARISMA_TRAIN_MASTER.csv")
    panels_raw = {n: pd.read_csv(DATA / f"YARISMA_TRAIN_{n}.csv") for n in PANELS}
    master_ids = set(master["Variant_ID"].values)

    X, y, vids = transform(master, state)
    X = X.reindex(columns=pkl["feature_columns"], fill_value=0)
    id_to_idx = {v: i for i, v in enumerate(vids)}

    print("Reproducing leakage-free stacked OOF (seed=42, frozen meta-learner)...")
    stacked_oof, lgb_oof, xgb_oof, cat_oof = leakage_free_stacked_oof(X, y, pkl)

    from sklearn.metrics import roc_auc_score
    res = {"timestamp": datetime.now().isoformat(), "threshold": THRESHOLD}
    res["oof_auc"] = {
        "lightgbm": round(float(roc_auc_score(y, lgb_oof)), 4),
        "xgboost": round(float(roc_auc_score(y, xgb_oof)), 4),
        "catboost": round(float(roc_auc_score(y, cat_oof)), 4),
        "stacked": round(float(roc_auc_score(y, stacked_oof)), 4),
    }

    # MASTER stacked metrics at 0.305
    res["master"] = metrics_block(y, stacked_oof, THRESHOLD)
    print(f"MASTER stacked OOF @0.305: ROC-AUC={res['master']['roc_auc']:.4f} "
          f"Bal-F1={res['master']['f1_balanced']:.4f} "
          f"Sens={res['master']['sensitivity']:.4f} Spec={res['master']['specificity']:.4f}")

    # Panels (leakage-aware, stacked, @0.305)
    res["panels"] = {}
    for name in PANELS:
        m = panel_metrics(panels_raw[name], master_ids, id_to_idx, stacked_oof, pkl)
        res["panels"][name] = m
        print(f"  {name:7s} n={m['n']:3d} ROC-AUC={m['roc_auc']:.4f} "
              f"Bal-F1={m['f1_balanced']:.4f} Sens={m['sensitivity']:.4f} "
              f"Spec={m['specificity']:.4f} MCC={m['mcc']:.4f}")

    # Held-out cohort (293 non-shared variants) @0.305, frozen ensemble
    held_frames = [panels_raw[n][~panels_raw[n]["Variant_ID"].isin(master_ids)] for n in PANELS]
    cohort = pd.concat(held_frames, ignore_index=True)
    assert cohort["Variant_ID"].isin(master_ids).sum() == 0
    co_proba = ensemble_predict(cohort, pkl)
    res["held_out_union"] = metrics_block(cohort["Label"].values, co_proba, THRESHOLD)
    res["held_out_panels"] = {}
    for name in PANELS:
        held = panels_raw[name][~panels_raw[name]["Variant_ID"].isin(master_ids)]
        res["held_out_panels"][name] = metrics_block(
            held["Label"].values, ensemble_predict(held, pkl), THRESHOLD)
    hu = res["held_out_union"]
    print(f"HELD-OUT union n={hu['n']} ROC-AUC={hu['roc_auc']:.4f} "
          f"Bal-F1={hu['f1_balanced']:.4f} Sens={hu['sensitivity']:.4f} Spec={hu['specificity']:.4f}")

    # Error analysis @0.305 on MASTER OOF
    res["error_analysis"] = error_profile(y, stacked_oof, X, master, panels_raw)
    ea = res["error_analysis"]
    print(f"Errors @0.305: FP={ea['n_fp']} FN={ea['n_fn']} (FN:FP={ea['fn_fp_ratio']}); "
          f"EK_7 mean correct={ea['ek7_mean']['correct']} FN={ea['ek7_mean']['false_negative']} "
          f"FP={ea['ek7_mean']['false_positive']}")

    (OUT / "phase_06_unified_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'phase_06_unified_results.json'}")


if __name__ == "__main__":
    main()
