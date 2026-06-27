# Phase 4 — PDR Results Summary

*Generated 2026-06-25T10:38:47.759470 · seed=42 · cost FN:FP = 2:1 · balanced (1:1) test prior*


## 1. Out-of-Sample Threshold Calibration
- Calibrated decision threshold (OOF, 1:1 prior, cost-sensitive): **0.365**
- MASTER OOF — ROC-AUC **0.8461**, balanced F1 0.7915 (t=0.5) → **0.7808** (calibrated)
- Panel performance at the single OOF-calibrated threshold (balanced 1:1, leakage-aware):

| Panel | Balanced F1 | Sensitivity | Specificity | MCC | n |
|-------|-------------|-------------|-------------|-----|---|
| CFTR | 0.8412 | 0.9333 | 0.7143 | 0.6637 | 111 |
| PAH | 0.7389 | 0.9355 | 0.4032 | 0.4001 | 372 |
| KANSER | 0.8018 | 0.9925 | 0.5167 | 0.5790 | 388 |

## 2. EK Ablation (Pipeline C — Leakage Defense)
- Dropped: EK_4, EK_5, EK_6, EK_7, miss_EK_4, miss_EK_5, miss_EK_6, miss_EK_7
- OOF ROC-AUC: baseline **0.8461** → ablated **0.8349** (Δ = +0.0112)
- Balanced F1 after ablation: 0.7740
- **Verdict:** LOW dependence — robust to EK leakage

## 3. Explainability (SHAP)
- Global + local SHAP plots saved to `reports/phase_04_optimization/figures/`
- Top SHAP features: EK_7, AL_327, AA_1_R, EK_2, AL_1, AL_16, EK_6, EK_9

## 4. Stacking Ensemble
- Base learners: lgb, xgb, cat + Logistic-Regression meta-learner

| Model | OOF ROC-AUC |
|-------|-------------|
| lightgbm | 0.8461 |
| xgboost | 0.8491 |
| catboost | 0.8538 |
| stacked | 0.8581 |
- Stacked balanced F1: **0.7974** (threshold 0.305)
- Serialized inference pipeline: `reports/phase_04_optimization/artifacts/inference_pipeline.pkl`