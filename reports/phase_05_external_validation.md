# Phase 5 — External Validation / Out-of-Distribution Report

*Generated 2026-06-25T11:20:22.333406 · frozen Phase 4 stacking ensemble · clinical 2:1-cost threshold = 0.365*

## Scope & Honest Framing
This is a **strictly isolated, held-out internal cohort**, not a truly independent external dataset. Because the competition features are anonymized with genomic coordinates removed (NDA-bound), assembling a genuine external cohort from public databases is infeasible (see Step 1). The reported numbers are therefore an **out-of-sample generalization estimate** on unseen variants — they should be presented as such, not as independent external validation.

## Step 1 — Feature-Mapping Feasibility
Features are anonymized (AL_*, EK_*, CAT_*, AA_*) with genomic coordinates removed and NDA-restricted provenance. Mapping EK_7 / AL_327 to ClinVar / dbNSFP is INFEASIBLE -> no independent external dataset can be built. Pivoting to a strictly isolated held-out cohort.

## Step 2 — Cohort Construction (Strict Isolation)
- **Definition:** Panel variants (CFTR/PAH/KANSER) absent from MASTER training set; never seen by frozen base models.
- The Phase 4 base models were trained on **MASTER only**. Variants present in the panel files but **absent from MASTER** were never seen in training.
- **Isolation verified:** 0 cohort variants found in the MASTER training set.
- **Cohort size:** n = 293 (pathogenic rate 0.444), no model retraining performed.

## Step 3 — Inference
- Frozen `inference_pipeline.pkl` (LightGBM + XGBoost + CatBoost → Logistic-Regression meta-learner).
- Preprocessing applied with the **frozen encoder state**; fully-null columns in the subset are preserved as NaN (native GBDT handling) and their missingness indicators fire — the missingness-aware design is leveraged directly.

## Step 4 — External Validation Metrics
Operating threshold = **0.365** (Phase 4 clinical 2:1-cost point). *Note: the stacked ensemble's own calibrated threshold was 0.305; ROC-AUC / PR-AUC are threshold-independent.*

### Union held-out cohort
| Metric | Value |
|--------|-------|
| n | 293 |
| ROC-AUC | **0.8934** |
| PR-AUC | 0.8557 |
| Balanced F1 (1:1 prior) | **0.8292** |
| Raw F1 | 0.8000 |
| Sensitivity | 0.9385 |
| Specificity | 0.6748 |
| Precision | 0.6971 |
| MCC | 0.6213 |

### Per-panel breakdown (held-out, unseen variants only)
| Panel | n | Pathogenic % | ROC-AUC | PR-AUC | Balanced F1 | Sensitivity | Specificity |
|-------|---|--------------|---------|--------|-------------|-------------|-------------|
| CFTR | 34 | 50.0% | 0.9619 | 0.9640 | 0.8235 | 0.8235 | 0.8235 |
| PAH | 117 | 58.1% | 0.8187 | 0.8272 | 0.7950 | 0.9559 | 0.5510 |
| KANSER | 142 | 31.7% | 0.9416 | 0.8848 | 0.8516 | 0.9556 | 0.7113 |

## Interpretation
- On 293 variants never seen in training, the frozen ensemble retains **ROC-AUC 0.893** — evidence that learned signal generalizes to unseen variants rather than memorizing training instances.
- CFTR's held-out subset is naturally balanced (17/17), so its metrics are the cleanest read on generalization; PAH remains the hardest panel, consistent with Phase 1–4 findings.
- Constraint compliance: **no models retrained**; train/eval isolation enforced and asserted by Variant_ID.