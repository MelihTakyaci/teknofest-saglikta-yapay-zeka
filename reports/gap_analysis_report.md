# Gap Analysis Report — Roadmap vs. Implementation

**Project**: TEKNOFEST 2026 Sağlıkta Yapay Zeka — Genetic Variant Pathogenicity Classification
**Roadmap analyzed**: `reports/teknofest_data_analysis_report.md` (v2026-06-02) + `reports/teknofest_data_analysis_summary.md`
**Source audited**: `phase_01_eda.py`, `phase_02_research.py`, `phase_02_5_preprocessing_audit.py`, `phase_03_pipeline_b.py`, `data_understanding_and_preparation.py`, `analysis/data_analysis.py`
**Audit date**: 2026-06-25

---

## Executive Summary

The project is in a **strong analytical / single-baseline state** but has **not yet entered the modeling phase the roadmap defines as the path to a competitive submission**. The EDA, data-quality, preprocessing-audit, and model-research phases (Phases 1, 2, 2.5) are complete and produce extensive reports, but they are **report generators** — the advanced techniques they describe (ensembling, Optuna, SHAP, EK ablation) exist only as pseudocode inside Markdown strings, not as executable code.

Exactly **one** real training pipeline is implemented and executed: `phase_03_pipeline_b.py`, a missingness-aware LightGBM with 5-fold stratified CV and per-panel evaluation. It is well-engineered and faithfully implements the roadmap's preprocessing and "global model + panel calibration" architecture, achieving OOF **ROC-AUC 0.846 / F1 0.873**. Verified against on-disk results in `reports/phase_03_pipeline_b/`.

**Headline gaps:**
1. **No ensemble** — the roadmap's primary modeling recommendation (LightGBM + XGBoost + CatBoost stacking) is unimplemented. Only LightGBM exists.
2. **No hyperparameter optimization** — Optuna is recommended but params are hardcoded.
3. **No EK leakage ablation (Pipeline C)** — flagged HIGH-priority in the roadmap; EK_7 is by far the single strongest feature, so circularity risk is unquantified.
4. **Threshold calibration is misaligned** — thresholds are optimized in-sample on imbalanced panel data, directly contradicting the roadmap's mandate to calibrate for the balanced (1:1) test distribution.
5. **No SHAP explainability and no formal error-analysis script** — both required roadmap deliverables.

Estimated completion against the roadmap's modeling scope: **~40%** (baseline + infrastructure done; the differentiation work that drives ranking is outstanding, with the final report `Raports/2026_PDR_RaporSon (1).docx` due 2026-06-29).

---

## Successfully Implemented

Components in code that align with the roadmap, verified by execution artifacts.

### 1. Task framing and metric (Roadmap §1, §3)
- Binary Pathogenic(1)/Benign(0) classification with **F1 as primary metric**, plus ROC-AUC, PR-AUC, MCC, sensitivity/specificity — `phase_03_pipeline_b.py:38-45`, results in `overall_oof_metrics.csv`.

### 2. Missingness-aware feature engineering (Roadmap §6.7, §10 Step 3) — fully aligned
- **No imputation** for the tree model; LightGBM native NaN handling used (`is_unbalance=True`, `phase_03_pipeline_b.py:212-229`).
- Per-feature binary missingness indicators (`miss_{col}`), row-level `total_miss_count` / missing %, and group-level missing counts (`miss_grp_*`) — `phase_03_pipeline_b.py:124-130`; confirmed in `feature_category_importance.csv` (`missingness_aggregate`, `missingness_group`, `missingness_indicator` categories present).
- The roadmap's MNAR insight is validated empirically: `total_miss_count` is a high-gain feature (mean gain 144.8).

### 3. Constant-column removal & encoding (Roadmap §5.2, §10 Steps 2,4)
- Drops `Variant_ID` and MASTER constant columns (`phase_03_pipeline_b.py:83-89, 108-113`).
- CAT_3/4/5 one-hot, CAT_1/2 label-encoded, CAT_6 binary flag, AA_1/AA_2 one-hot — `phase_03_pipeline_b.py:94-97` and downstream, matching Roadmap §8.5 exactly.

### 4. Validation protocol & class imbalance (Roadmap §7.5, §11.3)
- 5-fold **StratifiedKFold**, fixed seed 42, OOF predictions, early stopping — `phase_03_pipeline_b.py:240-308`.
- `is_unbalance=True` for class weighting; no SMOTE, no undersampling (compliant with roadmap mandates).

### 5. "Global model + panel calibration" architecture (Roadmap §9.4)
- A single global LightGBM trained on MASTER, then evaluated on each panel — the roadmap's explicitly recommended approach over 4 independent models — `phase_03_pipeline_b.py:462-555`.

### 6. Leakage-aware panel evaluation (Roadmap §11.5)
- For panel variants overlapping MASTER, **OOF predictions are reused** (no train/eval leakage); only non-shared variants are scored with the fold-model average — `phase_03_pipeline_b.py:471-510`. This correctly implements the roadmap's Variant_ID-aware split concern.

### 7. Reproducibility & artifact persistence (Roadmap §11.6)
- Seed 42; OOF predictions, fold metrics, feature importances, thresholds, confusion data, and figures all saved to `reports/phase_03_pipeline_b/`.

---

## Partial or Misaligned Implementations

### P1. Threshold calibration — MISALIGNED with the balanced-test mandate (Roadmap §7.3, §14)
The roadmap repeatedly stresses: test sets are **balanced 1:1**, training is imbalanced (up to 5:1), so thresholds must be **re-calibrated for the balanced distribution**. The code instead selects each panel threshold by maximizing F1 **on the imbalanced in-sample panel data** (`phase_03_pipeline_b.py:524-531`). Evidence of the failure mode in `panel_evaluation.csv`:
- **PAH** (83% pathogenic): "optimal" threshold drops to **0.205**, pushing sensitivity to 0.98 but specificity to **0.34** — i.e. predict-almost-everything-pathogenic. F1 looks inflated (0.928) only because positives dominate; on a 1:1 test set this threshold would collapse F1.
- **CFTR**: "optimal" threshold (0.71) actually yields *lower* F1 (0.933) than the default 0.5 (0.937) — in-sample selection on n=111 is noise-fitting.

This is both a **selection-on-the-evaluation-set** issue (optimistic) and a **distribution-mismatch** issue. The roadmap's prescribed fix — calibrate on a held-out/leave-one-panel-out set assuming 1:1 balance — is not implemented.

### P2. Panel strategy — calibration head only, no fine-tuning fallback (Roadmap §9.4 alternative)
The global-model + threshold approach is implemented, but the roadmap's documented escalation path (panel-specific fine-tuning / MASTER+panel combined training / stacking global preds as a feature) for weak panels is absent. PAH remains the weakest panel (ROC-AUC **0.772**, exactly as predicted) with no remediation code.

### P3. Explainability — partial (Roadmap §14 step 8)
Gain/split feature importance is computed and saved (`feature_importance.csv`, `feature_category_importance.csv`), confirming EK dominance (EK_7 mean gain 5335, ~4× any other feature). But **SHAP**, which the roadmap names for the Proje Detay Raporu, is referenced only in report text — no `shap` import or computation exists in any executed script.

### P4. Phases 1/2/2.5 are documentation, not modeling code
`phase_01_eda.py`, `phase_02_research.py`, and `phase_02_5_preprocessing_audit.py` are thorough but are **report/notebook generators**. Their references to XGBoost, CatBoost, Optuna, stacking, `CalibratedClassifierCV`, and SMOTE are **pseudocode inside triple-quoted Markdown strings** (e.g. `phase_02_research.py:1426-1449`, `phase_02_5_preprocessing_audit.py:950-962`), not executed logic. They satisfy the roadmap's *analysis* deliverables but contribute zero modeling capability.

---

## Missing Objectives

Roadmap goals with **no executable implementation** anywhere in the codebase.

| # | Missing objective | Roadmap ref | Evidence of absence |
|---|-------------------|-------------|---------------------|
| M1 | **Ensemble (LightGBM + XGBoost + CatBoost) with stacking meta-learner** — the roadmap's *primary* recommended architecture | §11.2, §14.3 | No `import xgboost / catboost`; only `lightgbm` imported (`phase_03_pipeline_b.py:47`). Stacking exists only as pseudocode strings. |
| M2 | **EK ablation — "Pipeline C" without EK_4/EK_5/EK_6** to quantify circularity/leakage | §8.3, §11.5, §14.1 (HIGH priority) | No script trains without EK features. EK_7 is the top feature, so the risk is real and unmeasured. |
| M3 | **Hyperparameter optimization (Optuna / Bayesian)** | §11.2, §14.5 | No `optuna`; `lgb_params` are hardcoded (`phase_03_pipeline_b.py:212-229`). |
| M4 | **Balanced-distribution threshold recalibration** | §7.3, §14.2 | No code assumes/simulates a 1:1 test distribution (see P1). |
| M5 | **Formal error analysis of misclassified variants** | §14.7 | OOF predictions are saved (enabling it) but no analysis script consumes them. |
| M6 | **SHAP explainability** | §14.8 | No `shap` import in executed code (see P3). |
| M7 | **Inference / test-set prediction pipeline** | §3.5, §9 | All code trains/evaluates on the 4 train CSVs; no module loads a held-out test set and emits a submission. Required before finals. |

---

## Prioritized Action Plan

Ordered by competition impact (F1 on balanced test) per unit effort.

### Priority 1 — Fix threshold calibration (correctness, low effort, high impact)
Addresses M4 / P1. Re-derive each panel's decision threshold under a **1:1 class prior**, selected on data *not* used to evaluate it (leave-one-panel-out, or a stratified calibration hold-out). Concretely: down-weight positives to a 50/50 prior when computing the F1-maximizing threshold, or evaluate F1 on a re-balanced bootstrap of each panel. This is the single change most likely to prevent a large finals F1 drop, since current PAH/CFTR thresholds are tuned to the wrong (imbalanced, in-sample) distribution.

### Priority 2 — EK ablation "Pipeline C" (de-risk leakage, low effort)
Addresses M2. Clone the Pipeline B flow, drop `EK_4`, `EK_5`, `EK_6` (and test dropping all `EK_*`), and compare OOF ROC-AUC/F1. If the drop is small (<0.02 AUC), prefer the safer feature set; if large, document the dependence and justify retention. This directly answers the roadmap's open question on circularity and is required for jury defensibility.

### Priority 3 — Build the ensemble (primary architecture, medium effort)
Addresses M1. Add XGBoost (`scale_pos_weight`) and CatBoost (`class_weight='balanced'`, native categoricals) alongside the existing LightGBM, sharing the same StratifiedKFold splits and OOF scheme. Stack the three OOF probability vectors with a logistic-regression meta-learner. Reuse the existing preprocessing and panel-evaluation harness — the infrastructure already supports per-fold OOF.

### Priority 4 — Hyperparameter optimization (medium effort)
Addresses M3. Wrap the LightGBM (and later ensemble base learners) in an Optuna study over `num_leaves`, `min_child_samples`, `learning_rate`, `feature_fraction`, `bagging_fraction`, optimizing OOF F1 with the same 5-fold splits. Persist best params for reproducibility.

### Priority 5 — Error analysis + SHAP (explainability deliverables, medium effort)
Addresses M5 / M6 / P3. Build one analysis script consuming `oof_predictions.csv`: profile false positives/negatives by panel, missingness band, and EK ranges. Add SHAP (TreeExplainer on the final model) for global and per-panel attribution. Both feed directly into the Proje Detay Raporu (due **2026-06-29**).

### Priority 6 — Inference / submission pipeline (required for finals, medium effort)
Addresses M7. Factor `preprocess()` and the trained fold-model ensemble into a reusable module that ingests an unseen test CSV (same 353-col schema), applies frozen encoders/constant-column lists, and emits per-panel predictions at the calibrated thresholds. Currently the encoders are learned via module-level globals (`cat1_mapping`, `cat2_mapping`), which must be serialized to guarantee train/test consistency.

### Sequencing note
P1 and P2 are quick, high-leverage, and unblock honest metric reporting before the 2026-06-29 report deadline. P3–P6 are the substantive modeling build-out and should follow the roadmap's 4-week timeline (§14), with P6 mandatory before the August–September finals regardless of ensemble progress.
