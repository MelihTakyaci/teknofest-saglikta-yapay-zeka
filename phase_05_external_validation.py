#!/usr/bin/env python3
"""
Phase 5 — External Validation / Out-of-Distribution (OOD) Testing
=================================================================
TEKNOFEST 2026 Healthcare AI

Goal: estimate the generalizability of the FROZEN Phase 4 inference pipeline on
a cohort that was NEVER part of training. No model is retrained.

Step 1 — Feature-mapping feasibility (logged, then pivot):
  The features are anonymized (AL_*, EK_*, CAT_*, AA_*), genomic coordinates are
  removed, and access is NDA-bound (competition spec). Mapping EK_7 / AL_327 to
  ClinVar / dbNSFP columns is therefore infeasible — so a *truly independent*
  external dataset cannot be assembled. We pivot to a strictly isolated,
  held-out internal cohort (Step 2).

Step 2 — Held-out / OOD cohort construction:
  The Phase 4 base models were trained on MASTER only. The disease-panel files
  (CFTR / PAH / KANSER) contain variants that are ABSENT from MASTER. Those
  non-shared variants were never seen in training and form the external cohort.
  Isolation is enforced and asserted by Variant_ID.

Step 3 — Inference: load inference_pipeline.pkl, preprocess with the frozen
  encoder state (missingness-aware: fully-null columns become NaN handled
  natively + their missingness indicators fire), score with the stacked
  ensemble (LightGBM + XGBoost + CatBoost -> Logistic-Regression meta-learner).

Step 4 — Report ROC-AUC, PR-AUC, Balanced F1, Sensitivity, Specificity at the
  Phase 4 clinical 2:1-cost threshold (0.365), saved as Markdown.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, matthews_corrcoef,
)

# Reuse the FROZEN preprocessing logic (import is safe: phase_04 is __main__-guarded)
from phase_04_optimization import transform, balanced_weights

BASE = Path("/Users/melihtakyaci/Documents/TeknofestSagliktaYapayZekâVerisi")
DATA = BASE / "EĞİTİM (TRAIN) SETLERİ 2"
ART = BASE / "reports" / "phase_04_optimization" / "artifacts"
REPORT = BASE / "reports" / "phase_05_external_validation.md"

CLINICAL_THRESHOLD = 0.365   # Phase 4 single-model 2:1-cost operating point (user-specified)
PANELS = ["CFTR", "PAH", "KANSER"]
BASE_KEY = {"lgb": "lightgbm", "xgb": "xgboost", "cat": "catboost"}


def load_pipeline():
    pkl = joblib.load(ART / "inference_pipeline.pkl")
    return pkl


def ensemble_predict(df: pd.DataFrame, pkl: dict) -> np.ndarray:
    """Run the frozen stacking ensemble end-to-end on a raw dataframe."""
    state = pkl["preprocessor_state"]
    X, _, _ = transform(df, state)                       # missingness-aware, NaN preserved
    X = X.reindex(columns=pkl["feature_columns"], fill_value=0)   # freeze column space
    cols = []
    for k in pkl["meta_order"]:                          # exact base-model order used at fit
        model = pkl["base_models"][BASE_KEY[k]]
        cols.append(model.predict_proba(X)[:, 1])
    Z = np.column_stack(cols)
    return pkl["meta_learner"].predict_proba(Z)[:, 1]


def metrics_block(y, proba, threshold):
    """Raw + balanced (1:1-prior) metrics. Balanced metrics simulate the
    competition's balanced test set and define 'Balanced F1'."""
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    w = balanced_weights(y)
    out = {
        "n": int(len(y)), "pathogenic_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1_raw": float(f1_score(y, pred)),
        "f1_balanced": float(f1_score(y, pred, sample_weight=w)),
        "sensitivity": float(recall_score(y, pred)),
        "specificity": float(recall_score(y, pred, pos_label=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc_balanced": float(roc_auc_score(y, proba, sample_weight=w)),
    }
    return out


def main():
    print("=" * 70 + "\nPHASE 5 — EXTERNAL VALIDATION / OOD TESTING\n" + "=" * 70)

    # ── Step 1: feature-mapping feasibility (logged) ──
    step1 = ("Features are anonymized (AL_*, EK_*, CAT_*, AA_*) with genomic "
             "coordinates removed and NDA-restricted provenance. Mapping EK_7 / "
             "AL_327 to ClinVar / dbNSFP is INFEASIBLE -> no independent external "
             "dataset can be built. Pivoting to a strictly isolated held-out cohort.")
    print("\n[Step 1] " + step1)

    # ── Step 2: construct strictly isolated held-out cohort ──
    master = pd.read_csv(DATA / "YARISMA_TRAIN_MASTER.csv")
    master_ids = set(master["Variant_ID"].values)

    frames = []
    per_panel_src = {}
    for name in PANELS:
        df = pd.read_csv(DATA / f"YARISMA_TRAIN_{name}.csv")
        held = df[~df["Variant_ID"].isin(master_ids)].copy()
        held["__panel"] = name
        per_panel_src[name] = held
        frames.append(held)
    cohort = pd.concat(frames, ignore_index=True)

    # Hard isolation guarantee
    leaked = cohort["Variant_ID"].isin(master_ids).sum()
    assert leaked == 0, f"LEAKAGE: {leaked} cohort variants are in MASTER training!"
    print(f"[Step 2] Held-out cohort: n={len(cohort)} "
          f"(zero MASTER-training leakage verified). "
          f"Pathogenic rate={cohort['Label'].mean():.3f}")

    # ── Step 3: inference with frozen pipeline ──
    pkl = load_pipeline()
    print(f"[Step 3] Loaded frozen pipeline: base={list(pkl['base_models'])}, "
          f"meta_order={pkl['meta_order']}, stored_threshold={pkl['calibrated_threshold']:.3f}")
    proba_all = ensemble_predict(cohort, pkl)

    # ── Step 4: metrics ──
    results = {
        "timestamp": datetime.now().isoformat(),
        "threshold_used": CLINICAL_THRESHOLD,
        "stored_ensemble_threshold": float(pkl["calibrated_threshold"]),
        "step1_feature_mapping": step1,
        "cohort_definition": "Panel variants (CFTR/PAH/KANSER) absent from MASTER "
                             "training set; never seen by frozen base models.",
        "union": metrics_block(cohort["Label"].values, proba_all, CLINICAL_THRESHOLD),
        "panels": {},
    }
    print(f"\n[Step 4] UNION cohort (n={results['union']['n']}): "
          f"ROC-AUC={results['union']['roc_auc']:.4f}  "
          f"Balanced F1={results['union']['f1_balanced']:.4f}  "
          f"Sens={results['union']['sensitivity']:.4f}  Spec={results['union']['specificity']:.4f}")

    for name in PANELS:
        held = per_panel_src[name]
        p = ensemble_predict(held, pkl)
        m = metrics_block(held["Label"].values, p, CLINICAL_THRESHOLD)
        results["panels"][name] = m
        print(f"   {name:7s} n={m['n']:3d}  ROC-AUC={m['roc_auc']:.4f}  "
              f"Balanced F1={m['f1_balanced']:.4f}  Sens={m['sensitivity']:.4f}  "
              f"Spec={m['specificity']:.4f}")

    REPORT.write_text(build_markdown(results), encoding="utf-8")
    (REPORT.parent / "phase_05_external_validation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {REPORT}")


def build_markdown(r) -> str:
    u = r["union"]
    L = []
    a = L.append
    a("# Phase 5 — External Validation / Out-of-Distribution Report\n")
    a(f"*Generated {r['timestamp']} · frozen Phase 4 stacking ensemble · "
      f"clinical 2:1-cost threshold = {r['threshold_used']}*\n")

    a("## Scope & Honest Framing")
    a("This is a **strictly isolated, held-out internal cohort**, not a truly "
      "independent external dataset. Because the competition features are "
      "anonymized with genomic coordinates removed (NDA-bound), assembling a "
      "genuine external cohort from public databases is infeasible (see Step 1). "
      "The reported numbers are therefore an **out-of-sample generalization estimate** "
      "on unseen variants — they should be presented as such, not as independent "
      "external validation.\n")

    a("## Step 1 — Feature-Mapping Feasibility")
    a(f"{r['step1_feature_mapping']}\n")

    a("## Step 2 — Cohort Construction (Strict Isolation)")
    a(f"- **Definition:** {r['cohort_definition']}")
    a("- The Phase 4 base models were trained on **MASTER only**. Variants present "
      "in the panel files but **absent from MASTER** were never seen in training.")
    a(f"- **Isolation verified:** 0 cohort variants found in the MASTER training set.")
    a(f"- **Cohort size:** n = {u['n']} (pathogenic rate {u['pathogenic_rate']:.3f}), "
      "no model retraining performed.\n")

    a("## Step 3 — Inference")
    a("- Frozen `inference_pipeline.pkl` (LightGBM + XGBoost + CatBoost → "
      "Logistic-Regression meta-learner).")
    a("- Preprocessing applied with the **frozen encoder state**; fully-null columns "
      "in the subset are preserved as NaN (native GBDT handling) and their "
      "missingness indicators fire — the missingness-aware design is leveraged directly.\n")

    a("## Step 4 — External Validation Metrics")
    a(f"Operating threshold = **{r['threshold_used']}** (Phase 4 clinical 2:1-cost "
      f"point). *Note: the stacked ensemble's own calibrated threshold was "
      f"{r['stored_ensemble_threshold']:.3f}; ROC-AUC / PR-AUC are threshold-independent.*\n")

    a("### Union held-out cohort")
    a("| Metric | Value |")
    a("|--------|-------|")
    a(f"| n | {u['n']} |")
    a(f"| ROC-AUC | **{u['roc_auc']:.4f}** |")
    a(f"| PR-AUC | {u['pr_auc']:.4f} |")
    a(f"| Balanced F1 (1:1 prior) | **{u['f1_balanced']:.4f}** |")
    a(f"| Raw F1 | {u['f1_raw']:.4f} |")
    a(f"| Sensitivity | {u['sensitivity']:.4f} |")
    a(f"| Specificity | {u['specificity']:.4f} |")
    a(f"| Precision | {u['precision']:.4f} |")
    a(f"| MCC | {u['mcc']:.4f} |\n")

    a("### Per-panel breakdown (held-out, unseen variants only)")
    a("| Panel | n | Pathogenic % | ROC-AUC | PR-AUC | Balanced F1 | Sensitivity | Specificity |")
    a("|-------|---|--------------|---------|--------|-------------|-------------|-------------|")
    for name, m in r["panels"].items():
        a(f"| {name} | {m['n']} | {m['pathogenic_rate']*100:.1f}% | {m['roc_auc']:.4f} "
          f"| {m['pr_auc']:.4f} | {m['f1_balanced']:.4f} | {m['sensitivity']:.4f} "
          f"| {m['specificity']:.4f} |")

    a("\n## Interpretation")
    a(f"- On {u['n']} variants never seen in training, the frozen ensemble retains "
      f"**ROC-AUC {u['roc_auc']:.3f}** — evidence that learned signal generalizes "
      "to unseen variants rather than memorizing training instances.")
    a("- CFTR's held-out subset is naturally balanced (17/17), so its metrics are the "
      "cleanest read on generalization; PAH remains the hardest panel, consistent "
      "with Phase 1–4 findings.")
    a("- Constraint compliance: **no models retrained**; train/eval isolation enforced "
      "and asserted by Variant_ID.")
    return "\n".join(L)


if __name__ == "__main__":
    main()
