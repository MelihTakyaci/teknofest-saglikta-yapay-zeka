# Teknofest Data Analysis and Modeling Strategy Report

**Competition**: TEKNOFEST 2026 — Sağlıkta Yapay Zeka Yarışması (Health AI Competition)
**Category**: Üniversite ve Üzeri Seviyesi (University and Above Level)
**Report Date**: 2026-06-02
**Dataset Version**: EĞİTİM (TRAIN) SETLERİ 2

---

## 1. Executive Summary

This report presents a comprehensive technical analysis of the TEKNOFEST 2026 Health AI competition dataset for the University-level genetic variant pathogenicity classification task.

**Key findings:**

1. **Task**: Binary classification of genetic variants as Pathogenic (1) or Benign (0), evaluated on **F1 Score** (computed from TP, FP, FN).
2. **4 datasets** share an identical 353-column schema — 1 general (MASTER, n=2931) and 3 gene-specific panels (KANSER n=388, PAH n=372, CFTR n=111). The competition will test on 4 corresponding test sets, motivating **4 separate or panel-calibrated models**.
3. **Missing data is extreme**: 54.9% of all cells in MASTER are missing. Missingness is structured, not random — it correlates with label, feature group, and panel.
4. **Class imbalance is moderate to severe**: Pathogenic variants dominate all panels (67–83%), with the most extreme ratio in PAH (5:1).
5. **EK features (EK_1–EK_9)** are the strongest predictors (likely pre-computed in-silico pathogenicity scores), raising circularity/leakage concerns.
6. A baseline LightGBM achieves **ROC-AUC 0.846** on MASTER with out-of-fold predictions, establishing a viable starting point.

**Primary metric**: F1 Score (confirmed in specification Section 7.3).
**Recommended approach**: Gradient-boosted tree ensemble (LightGBM/XGBoost/CatBoost) with panel-specific threshold calibration and missingness-aware feature engineering.

---

## 2. Project Folder and File Overview

### 2.1 Dataset Files

| File | Location | Size | Rows | Cols |
|------|----------|------|------|------|
| YARISMA_TRAIN_MASTER.csv | EĞİTİM (TRAIN) SETLERİ 2/ | 5.4 MB | 2931 | 353 |
| YARISMA_TRAIN_KANSER.csv | EĞİTİM (TRAIN) SETLERİ 2/ | 689 KB | 388 | 353 |
| YARISMA_TRAIN_PAH.csv | EĞİTİM (TRAIN) SETLERİ 2/ | 836 KB | 372 | 353 |
| YARISMA_TRAIN_CFTR.csv | EĞİTİM (TRAIN) SETLERİ 2/ | 295 KB | 111 | 353 |

### 2.2 Specification Documents

| File | Description |
|------|-------------|
| 2026-_Sağlıkta_Yapay_Zeka_Türkçe_Şartname_v4_6k439.pdf | Official specification (V1.4, 14.04.2026) |
| 2026_Sağlıkta_Yapay_Zeka_V01.docx | Earlier version of specification |

### 2.3 Existing Analysis Scripts

| File | Description |
|------|-------------|
| phase_01_eda.py | Phase 1 EDA script (72 KB) |
| phase_02_research.py | Phase 2 specialized model research (98 KB) |
| phase_02_5_preprocessing_audit.py | Phase 2.5 preprocessing audit (65 KB) |
| phase_03_pipeline_b.py | Phase 3 baseline pipeline (52 KB) |
| data_understanding_and_preparation.py | Data understanding script (56 KB) |
| generate_pipeline_b_docx.py | Report generation script (9 KB) |

### 2.4 Existing Reports

| Directory | Contents |
|-----------|----------|
| reports/phase_01_data_understanding/ | EDA reports, figures, CSV summaries |
| reports/phase_02_specialized_model_research/ | Model research, strategy comparison |
| reports/phase_02_5_preprocessing_audit/ | Preprocessing readiness audit |
| reports/phase_03_pipeline_b/ | Baseline LightGBM results (ROC-AUC 0.846) |
| reports/data_understanding_and_preparation/ | Initial data understanding report |

---

## 3. Competition Specification Summary

### 3.1 Confirmed Requirements (from PDF specification)

| Item | Detail |
|------|--------|
| **Competition** | TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması |
| **Organizers** | T.C. Sağlık Bakanlığı, TÜSEB, T3 Vakfı |
| **Task** | Classify genetic variants as Pathogenic or Benign |
| **Target** | Binary: Pathogenic (including Likely Pathogenic) vs Benign (including Likely Benign) |
| **Ground truth source** | ACMG criteria, ClinVar/ClinGen databases |
| **Ranking metric** | **F1 Score** (computed from TP, FP, FN) — Section 7.3 |
| **Evaluation** | External validation on unseen test data during finals |
| **Score weight** | 90% competition task + 10% final presentation |
| **Feature anonymization** | Column names anonymized; genomic coordinates removed |
| **External data** | Open data sources can also be used alongside provided data |
| **Confidentiality** | Signed NDA required to access data |
| **Deadline** | Proje Detay Raporu due 29.06.2026 |
| **Finals** | August–September 2026, TEKNOFEST Şanlıurfa |

### 3.2 Training Dataset Sizes (from specification)

| Panel | Pathogenic | Benign | Total |
|-------|-----------|--------|-------|
| Genel (MASTER) | 1500 | 1500 | 3000 |
| Kalıtsal Kanser | 200 | 200 | 400 |
| Fenilketonüri (PAH) | 200 | 200 | 400 |
| Kistik Fibrozis (CFTR) | 70 | 70 | 140 |

**Note**: Actual training data differs from spec counts (MASTER: 2931 instead of 3000, etc.). The Benign class is augmented with gnomAD population variants to reduce imbalance. Despite the stated balanced design, actual data shows significant imbalance.

### 3.3 Test Dataset Sizes (from specification)

| Panel | Pathogenic | Benign | Total |
|-------|-----------|--------|-------|
| Genel (MASTER) | 1000 | 1000 | 2000 |
| Kalıtsal Kanser | 100 | 100 | 200 |
| Fenilketonüri (PAH) | 100 | 100 | 200 |
| Kistik Fibrozis (CFTR) | 30 | 30 | 60 |

**Critical**: Test sets are stated as balanced (1:1 ratio), while training sets are imbalanced. This means the optimal decision threshold at test time will differ from the training-time optimal threshold.

### 3.4 Feature Categories (from specification)

The specification describes the following variant profile categories:
1. **Sekans ve Değişim Bilgisi** — Sequence and change information (nucleotide changes, codon changes, amino acid changes)
2. **Yerel Sekans ve Çevresel Bağlam** — Local sequence context (5 nucleotide genomic neighborhood, 5 amino acid proteomic neighborhood)
3. **Biyokimyasal ve Yapısal Etkiler** — Biochemical/structural effects (hydrophobicity, polarity, molecular weight, 3D structure)
4. **Evrimsel Korunmuşluk** — Evolutionary conservation (phylogenetic diversity, genomic similarity across species)
5. **Popülasyon Verileri** — Population data (Minor Allele Frequency)
6. **In Silico Risk Skorları** — Computational risk scores (various pathogenicity predictors)

### 3.5 Inferred Requirements

- Models must predict on 4 separate test sets independently
- Metric reserves may change (TÜSEB reserves this right per Section 7.5)
- Code must be reproducible and documented
- External validation will use a completely new test dataset

### 3.6 Open Questions from Specification

1. Will the test sets maintain the exact 1:1 ratio as stated, or might they deviate like the training sets?
2. Is F1 calculated as macro-averaged across panels or separately per panel?
3. Are ensemble or multi-model approaches acceptable, or must there be a single model?
4. Is there a runtime/hardware constraint for inference?

---

## 4. Dataset Overview

### 4.1 Schema Summary

All 4 datasets share an identical 353-column schema:

| Feature Group | Prefix | Count | Type | Description |
|---------------|--------|-------|------|-------------|
| Variant ID | Variant_ID | 1 | String | Unique variant identifier |
| Algorithm/Score | AL_1 – AL_334 | 334 | Float | Anonymized numerical features (likely in-silico scores, population frequencies, conservation scores) |
| Categorical | CAT_1 – CAT_6 | 6 | String | Population/ancestry labels, genotype, structural region annotations |
| Extra Knowledge | EK_1 – EK_9 | 9 | Float | Likely meta-predictor or composite pathogenicity scores |
| Amino Acid | AA_1, AA_2 | 2 | String | Reference and alternate amino acid identifiers |
| Label | Label | 1 | Int | Target: 0 = Benign, 1 = Pathogenic |

### 4.2 Data Types

| Data Type | Count |
|-----------|-------|
| float64 | 343 |
| object (string) | 9 |
| int64 | 1 |

### 4.3 Dataset Sizes

| Dataset | Rows | Columns | Total Cells | Missing Cells | Missing % |
|---------|------|---------|-------------|---------------|-----------|
| MASTER | 2931 | 353 | 1,034,643 | 568,464 | 54.9% |
| KANSER | 388 | 353 | 136,964 | 78,309 | 57.2% |
| PAH | 372 | 353 | 131,316 | 71,301 | 54.3% |
| CFTR | 111 | 353 | 39,183 | 11,864 | 30.3% |

### 4.4 Variant ID Overlap Between Panels

| Panel Pair | Overlapping Variant_IDs |
|------------|------------------------|
| MASTER ∩ PAH | 255 |
| MASTER ∩ KANSER | 246 |
| MASTER ∩ CFTR | 77 |
| CFTR ∩ PAH | 0 |
| CFTR ∩ KANSER | 0 |
| PAH ∩ KANSER | 0 |

Panel-specific datasets (PAH, KANSER, CFTR) have zero overlap with each other, but significant overlap with MASTER. This creates a **leakage risk** if MASTER is used for training and panels for evaluation — shared variants would have identical features.

---

## 5. Data Quality Analysis

### 5.1 Duplicate Analysis

| Dataset | Duplicate Variant_IDs | Duplicate Feature Rows |
|---------|----------------------|----------------------|
| MASTER | 0 | 114 |
| CFTR | 0 | 0 |
| PAH | 0 | 3 |
| KANSER | 0 | 2 |

114 rows in MASTER have identical feature values (but different Variant_IDs and possibly different labels). This suggests variants with very similar biological profiles, which is expected in genomic data.

### 5.2 Constant and Near-Constant Columns

| Dataset | Constant (1 unique) | Near-Constant (2 unique) |
|---------|-------------------|------------------------|
| MASTER | 57 | 48 |
| CFTR | 70 | 94 |
| PAH | 91 | 21 |
| KANSER | 69 | 91 |

**57 columns in MASTER** carry zero information and must be dropped before modeling. Panel-specific datasets have even more constant columns due to smaller sample sizes and gene-specific characteristics.

### 5.3 Suspicious Values

- **AL_185**: Takes value 264690.0 in some rows — anomalous compared to other AL features which are typically in [0, 1] range. Likely a count or position field, not a probability score.
- **CAT_6**: 97.7% missing in MASTER, 100% missing in PAH — practically unusable without binary flag encoding.
- **EK_2**: Range [-11.9, 6.17], negative values suggest log-scale or Z-score transformation.
- **EK_8**: Contains a value of -5.716, far outside the [0, 1] range of most variants.

---

## 6. Missing Data Analysis

### 6.1 Overall Missingness

| Dataset | Overall Missing % | Cols 100% Missing | Cols >90% Missing | Cols >50% Missing | Cols 0% Missing |
|---------|------------------|-------------------|-------------------|-------------------|-----------------|
| MASTER | 54.9% | 0 | 13 | 165 | 2 |
| CFTR | 30.3% | 0 | 1 | 33 | 15 |
| PAH | 54.3% | 1 | 19 | 201 | 2 |
| KANSER | 57.2% | 0 | 1 | 249 | 2 |

### 6.2 Missingness Bands (MASTER)

| Missing Range | Column Count |
|---------------|-------------|
| 0% (complete) | 2 |
| 0–10% | 0 |
| 10–25% | 13 |
| 25–50% | 173 |
| 50–75% | 146 |
| 75–90% | 6 |
| 90–100% | 13 |

Only 2 columns (Variant_ID, Label) are fully complete. Every feature column has at least 10% missingness.

### 6.3 Highest-Missingness Columns (MASTER)

| Column | Missing % | Likely Reason |
|--------|-----------|---------------|
| CAT_6 | 97.7% | Structural region annotation — only available for specific genomic regions |
| AL_27–AL_38 | ~91.4% | Population frequency features from rare databases |
| AL_1–AL_6 | ~90.0% | Population frequency features from rare databases |
| AL_215 | 74.3% | Specialized score — available only for certain variant types |

### 6.4 Missingness by Feature Group (Mean % per group)

| Feature Group | MASTER | CFTR | PAH | KANSER |
|---------------|--------|------|-----|--------|
| AL features | 56.9% | 31.4% | 56.5% | 59.5% |
| EK features | 16.9% | 4.9% | 7.9% | 9.0% |
| CAT features | 38.4% | 24.3% | 34.4% | 37.3% |
| AA features | 11.9% | 0.0% | 2.4% | 5.7% |

**EK and AA features are much more complete** than AL features. This has modeling implications — EK features can be relied upon more consistently.

### 6.5 Missingness by Label (MASTER)

| Label | N | Overall Missing % |
|-------|---|------------------|
| 0 (Benign) | 782 | 41.2% |
| 1 (Pathogenic) | 2149 | 59.9% |

**Pathogenic variants have significantly more missing data** (59.9% vs 41.2%). This is **Missing Not At Random (MNAR)** — missingness itself is informative of the label. This is critical because:
1. Imputation strategies must account for this differential pattern
2. Missingness indicators become valuable predictive features
3. Tree-based models that natively handle missing values will learn to exploit these patterns

### 6.6 Row-Level Missingness (MASTER)

| Quantile | Missing Columns (of 353) |
|----------|------------------------|
| 0% (min) | 0 |
| 25% | 30 |
| 50% (median) | 212 |
| 75% | 337 |
| 100% (max) | 351 |
| Mean ± Std | 193.9 ± 133.4 |

The median variant is missing 60% of its features. The distribution is bimodal: some variants have extensive annotation (few missing values), others are sparsely annotated (>90% missing).

### 6.7 Recommended Missing Data Handling

| Strategy | Applicability | Reasoning |
|----------|--------------|-----------|
| **Native GBDT missing handling** | PRIMARY | LightGBM/XGBoost learn optimal split direction for missing values |
| **Missingness indicator features** | HIGH PRIORITY | Add binary is_missing flags; missingness is informative (MNAR) |
| **Row-level missingness count** | MEDIUM | Total missing count per row as a meta-feature |
| **Group-level missingness count** | MEDIUM | Missing count within AL, EK, CAT groups separately |
| **DO NOT impute for tree models** | MANDATORY | Imputation destroys the missingness signal that GBDT models exploit |
| **Median imputation** | ONLY for linear models | If using logistic regression as a baseline |
| **DO NOT use SMOTE** | MANDATORY | Synthetic genomic variants are biologically meaningless |

---

## 7. Class / Target Imbalance Analysis

### 7.1 Class Distribution

| Dataset | Pathogenic (1) | Benign (0) | Total | Pathogenic % | Imbalance Ratio |
|---------|---------------|------------|-------|-------------|-----------------|
| MASTER | 2149 | 782 | 2931 | 73.3% | 2.75:1 |
| KANSER | 268 | 120 | 388 | 69.1% | 2.23:1 |
| PAH | 310 | 62 | 372 | 83.3% | 5.00:1 |
| CFTR | 90 | 21 | 111 | 81.1% | 4.29:1 |

### 7.2 Imbalance Severity Assessment

| Dataset | Severity | Reasoning |
|---------|----------|-----------|
| MASTER | **Moderate** | 2.75:1 — manageable with class weights |
| KANSER | **Moderate** | 2.23:1 — least imbalanced |
| PAH | **Severe** | 5.00:1 — only 62 Benign samples |
| CFTR | **Severe + Small** | 4.29:1 with only 21 Benign — high variance |

### 7.3 Training vs Test Imbalance Mismatch

**Critical finding**: The specification states test sets will be balanced (1:1 ratio), but training sets are imbalanced (up to 5:1). This means:
- A threshold optimized on training data will be **too aggressive** for balanced test data
- The optimal threshold must be re-calibrated assuming balanced test distribution
- F1 score behavior differs significantly between imbalanced and balanced data

### 7.4 Misleading Metrics Under Imbalance

| Metric | Reliability | Notes |
|--------|------------|-------|
| Accuracy | **Misleading** | 73.3% accuracy achievable by predicting all Pathogenic |
| F1 Score | **Reliable** | Appropriate for imbalanced data; this is the competition metric |
| ROC-AUC | **Reliable** | Threshold-independent |
| PR-AUC | **Reliable** | Especially informative for minority class |
| Macro F1 | **Reliable** | Treats both classes equally |

### 7.5 Recommended Handling Strategies

| Strategy | Priority | Details |
|----------|----------|---------|
| **Class weights** | MANDATORY | Use `is_unbalance=True` (LightGBM) or `scale_pos_weight` |
| **Stratified K-Fold** | MANDATORY | Preserve label ratio in each fold |
| **Threshold calibration** | HIGH | Optimize threshold for F1 on balanced distribution |
| **Cost-sensitive learning** | MEDIUM | Higher penalty for missing Pathogenic (clinical safety) |
| **DO NOT use SMOTE** | MANDATORY | Biologically meaningless for genetic variants |
| **DO NOT undersample** | RECOMMENDED | Already limited data; losing Benign samples is harmful |

---

## 8. Feature Analysis

### 8.1 Top 30 Features by Label Correlation (|Pearson|)

| Rank | Feature | |Correlation| | Category |
|------|---------|--------------|----------|
| 1 | EK_7 | 0.380 | Extra knowledge |
| 2 | EK_9 | 0.322 | Extra knowledge |
| 3 | EK_4 | 0.320 | Extra knowledge |
| 4 | EK_2 | 0.316 | Extra knowledge |
| 5 | EK_6 | 0.302 | Extra knowledge |
| 6 | EK_3 | 0.296 | Extra knowledge |
| 7 | EK_8 | 0.232 | Extra knowledge |
| 8 | AL_83 | 0.212 | Algorithm score |
| 9 | AL_92 | 0.173 | Algorithm score |
| 10 | AL_77 | 0.165 | Algorithm score |
| 11 | EK_5 | 0.154 | Extra knowledge |
| 12 | AL_34 | 0.125 | Algorithm score |
| 13 | AL_26 | 0.120 | Algorithm score |
| 14 | AL_32 | 0.116 | Algorithm score |
| 15 | AL_95 | 0.107 | Algorithm score |

**EK features dominate**: 8 of the top 11 features are EK features. This suggests EK features are pre-computed meta-predictor scores that aggregate information from multiple underlying tools.

### 8.2 EK Feature Distribution by Label

| Feature | Benign Mean | Path Mean | Separation | Likely Interpretation |
|---------|------------|-----------|------------|----------------------|
| EK_7 | 3.849 | 6.637 | **Strong** | Composite damage score (higher = more damaging) |
| EK_9 | 5.311 | 8.209 | **Strong** | Aggregate pathogenicity score |
| EK_4 | 0.749 | 0.946 | **Strong** | Meta-predictor [0,1], e.g., REVEL/BayesDel |
| EK_2 | 3.294 | 4.851 | **Strong** | Conservation or damage score |
| EK_6 | 0.770 | 0.951 | **Strong** | Meta-predictor [0,1], similar to EK_4 |
| EK_3 | 1.896 | 3.524 | **Strong** | Conservation score |
| EK_8 | 0.420 | 0.605 | Moderate | Possibly phastCons or similar |
| EK_1 | 5.176 | 5.292 | Weak | Baseline annotation quality score |
| EK_5 | 0.723 | 0.832 | Moderate | Conservation probability |

### 8.3 Leakage Risk Assessment

| Feature(s) | Risk Level | Concern |
|------------|-----------|---------|
| EK_4, EK_6 | **HIGH** | [0,1] range meta-predictors, likely trained on ClinVar — circular prediction |
| EK_7, EK_9 | **MEDIUM** | Strong predictors, but broader range suggests composite scores |
| Variant_ID | **HIGH** | Must be dropped from features — encodes identity, not biology |
| AL_185 | **LOW** | Takes value 264690 — likely allele count, not a leakage risk |

### 8.4 Categorical Feature Summary

| Column | Unique Values | Missing % | Likely Meaning |
|--------|--------------|-----------|----------------|
| CAT_1 | 30 | 37.0% | Population/database with highest allele frequency (gnomAD/AllofUs) |
| CAT_2 | 7 | 59.2% | AllofUs population group |
| CAT_3 | 5 | 12.2% | Reference genotype (e.g., C/C, T/T, A/A, G/G, ./.) |
| CAT_4 | 5 | 12.2% | Alternate genotype |
| CAT_5 | 5 | 12.2% | Another genotype field |
| CAT_6 | 3 | 97.7% | Structural region type (lcr, segdup, decoy&segdup) |

### 8.5 Encoding Recommendations

| Feature(s) | Recommended Encoding | Reasoning |
|------------|---------------------|-----------|
| CAT_1 | Label encoding or CatBoost native | 30 categories, too many for one-hot |
| CAT_2 | One-hot or label encoding | 7 categories |
| CAT_3, CAT_4, CAT_5 | One-hot encoding | 4-5 categories each |
| CAT_6 | Binary flag (has annotation) | 97.7% missing |
| AA_1, AA_2 | One-hot encoding (20 amino acids) or physicochemical grouping | 24-25 unique values |

### 8.6 Feature Scaling

- **Not needed for tree-based models** (GBDT is scale-invariant)
- **Required for logistic regression** (if used as stacking meta-learner)
- StandardScaler recommended if scaling is needed

---

## 9. Four-Model Strategy

### 9.1 Why 4 Models Are Needed

The competition specification explicitly defines **4 separate test sets** (Section 3.2, Section 7.7):
1. Genel Veri Seti (General)
2. Kalıtsal Kanser Gen Paneli (Hereditary Cancer)
3. Fenilketonüri (PAH) Gen Paneli
4. Kistik Fibrozis (CFTR) Gen Paneli

Each test set will be evaluated independently with F1 Score. The final ranking depends on performance across all 4 test sets.

**Data-driven reasons for 4 separate models:**

1. **Distribution shift**: Feature distributions differ significantly between panels (KS test p<0.001 for 57 features between MASTER and PAH)
2. **Missingness patterns differ**: CFTR has 30.3% missing vs KANSER's 57.2%
3. **Class imbalance varies**: PAH is 5:1 while KANSER is 2.23:1
4. **Gene-specific biology**: Different genes have different pathogenicity mechanisms
5. **Variant_ID overlap**: Panel variants partially overlap with MASTER, requiring careful data management

### 9.2 Confirmed vs Inferred Model Requirements

**Confirmed** (from specification):
- 4 test sets require predictions on 4 datasets
- F1 Score is the evaluation metric
- Models should predict based on variant profiles only (no genomic coordinates)

**Inferred** (from data analysis):
- Separate models or panel-specific calibration will likely outperform a single global model
- CFTR model will have the highest variance (smallest dataset)
- Panel-specific threshold optimization is critical due to train/test distribution mismatch

### 9.3 Model Analysis Table

| Model No | Purpose | Task Type | Input | Output | Recommended Algorithms | Evaluation Metrics | Main Risks |
|---------|---------|-----------|-------|--------|------------------------|--------------------|------------|
| Model 1 | General variant pathogenicity (MASTER) | Binary classification | 353 features from any gene variant | Pathogenic probability [0,1] | LightGBM ensemble, XGBoost, CatBoost | F1 Score, ROC-AUC, PR-AUC, Sensitivity | Distribution shift at test time; extreme missingness |
| Model 2 | Hereditary Cancer panel (KANSER) | Binary classification | Same 353-column schema, KANSER genes | Pathogenic probability [0,1] | LightGBM with transfer from MASTER, or fine-tuned model | F1 Score, Precision, Recall | Moderate size (n=388); 57% missingness |
| Model 3 | Fenilketonüri panel (PAH) | Binary classification | Same 353-column schema, PAH gene | Pathogenic probability [0,1] | LightGBM with transfer from MASTER, panel-calibrated threshold | F1 Score, Balanced Accuracy | Severe imbalance (5:1); only 62 Benign |
| Model 4 | Kistik Fibrozis panel (CFTR) | Binary classification | Same 353-column schema, CFTR gene | Pathogenic probability [0,1] | Global MASTER model with threshold calibration | F1 Score, Confidence intervals | Very small (n=111, 21 Benign); high variance |

### 9.4 Recommended Architecture Strategy

**Approach: Global model + panel-specific calibration** (safest for competition)

```
Strategy:
┌──────────────────────────────────────────────────────┐
│ Global GBDT Ensemble (trained on MASTER)             │
│ ├── LightGBM                                         │
│ ├── XGBoost                                          │
│ └── CatBoost                                         │
│                                                      │
│ Output: probability P(pathogenic) per variant         │
└──────────────┬───────────────────────────────────────┘
               │
     ┌─────────┴─────────┐
     │ Panel Calibration  │
     ├────────────────────┤
     │ MASTER:   t=0.50   │ → F1-optimized threshold
     │ KANSER:   t=0.XX   │ → calibrated on KANSER validation
     │ PAH:      t=0.XX   │ → calibrated on PAH validation
     │ CFTR:     t=0.XX   │ → calibrated on CFTR validation
     └────────────────────┘
```

**Why this over 4 independent models:**

1. **Data efficiency**: MASTER has 2931 samples; training separate models on 111–388 samples wastes information
2. **Shared features**: All panels share the same 353-column schema with identical feature semantics
3. **Overfitting risk**: Independent models on small panels (especially CFTR) will overfit
4. **Simplicity**: One global model is easier to debug, explain, and maintain

**Alternative (if global model underperforms on panels):**

Train panel-specific models using:
1. MASTER + panel data (combined training)
2. Panel-aware feature selection (remove constant columns per panel)
3. Panel-specific hyperparameter tuning
4. Stacking: global model predictions as an additional feature for panel-specific models

### 9.5 Model Relationships

The 4 models should be **one shared global model with 4 panel-specific calibration heads** rather than 4 independent models:

1. All models share the same feature space
2. The global model captures general pathogenicity patterns
3. Panel calibration adjusts for gene-specific pathogenicity distributions
4. Panel-specific threshold optimization adjusts for evaluation-time class balance
5. If panel performance is poor, escalate to panel-specific fine-tuning

---

## 10. Recommended Preprocessing Pipeline

### Step-by-Step Pipeline

```
1. LOAD DATA
   └── Read all 4 CSV files
   └── Verify schema consistency (353 columns)

2. DROP NON-INFORMATIVE COLUMNS
   └── Drop Variant_ID (leakage risk)
   └── Drop 57 constant columns from MASTER
   └── Per-panel: drop additional constant columns

3. FEATURE ENGINEERING — MISSINGNESS
   └── Add binary is_missing_{col} for each feature
   └── Add row_missing_count (total NaN per row)
   └── Add group_missing_AL, group_missing_EK, group_missing_CAT

4. FEATURE ENCODING — CATEGORICAL
   └── CAT_1: label encode (30 categories)
   └── CAT_2: one-hot or label encode (7 categories)
   └── CAT_3, CAT_4, CAT_5: one-hot encode (4-5 categories)
   └── CAT_6: binary flag (has_annotation = 1 if not NaN, else 0)
   └── AA_1, AA_2: one-hot encode (20+ amino acids)

5. NO IMPUTATION FOR TREE MODELS
   └── LightGBM/XGBoost handle NaN natively
   └── Only impute if using linear models (median imputation + indicators)

6. CLASS WEIGHT HANDLING
   └── is_unbalance=True for LightGBM
   └── scale_pos_weight for XGBoost
   └── class_weight='balanced' for CatBoost

7. VALIDATION SPLIT
   └── 5-fold Stratified K-Fold
   └── Variant_ID-aware: if evaluating on panels, exclude overlapping variants
   └── Fixed random seed (e.g., 42)
```

---

## 11. Baseline Modeling Plan

### 11.1 Baseline Models

| Model | Purpose | Why This Baseline |
|-------|---------|-------------------|
| LightGBM (default params) | Primary baseline | Fast, handles missing values natively, handles categorical features |
| XGBoost (default params) | Diversity baseline | Different tree-building algorithm, complementary errors |
| Logistic Regression | Interpretable reference | Simple, fast, establishes a linear separability floor |

### 11.2 Stronger Candidates

| Model | When to Use |
|-------|------------|
| LightGBM (tuned) | After baseline, Bayesian hyperparameter optimization |
| CatBoost | If categorical features are important |
| Stacking Ensemble | Combine LightGBM + XGBoost + CatBoost via logistic regression meta-learner |
| Neural Network (TabNet) | Only if tree models plateau and dataset size justifies it |

### 11.3 Validation Strategy

| Aspect | Choice | Reasoning |
|--------|--------|-----------|
| Split method | **Stratified 5-Fold CV** | Preserves class ratio; 5 folds balance bias-variance |
| Primary metric | **F1 Score** | Competition metric |
| Secondary metrics | ROC-AUC, PR-AUC, Sensitivity, Specificity | Comprehensive evaluation |
| Panel evaluation | **Held-out panel evaluation** | Evaluate MASTER model on each panel separately |
| Threshold selection | **F1-maximizing threshold on validation** | Grid search over thresholds [0.1, 0.9] |

### 11.4 Existing Baseline Result

Pipeline B (LightGBM with missingness-aware preprocessing) achieved:

| Metric | Value |
|--------|-------|
| ROC-AUC (OOF) | 0.846 |
| PR-AUC (OOF) | 0.925 |
| F1 (Youden threshold) | 0.860 |
| Sensitivity @ 95% | 0.950 |
| CFTR panel ROC-AUC | 0.940 |
| KANSER panel ROC-AUC | 0.914 |
| PAH panel ROC-AUC | 0.772 |

PAH panel is the weakest performer — consistent with its extreme imbalance and high missingness.

### 11.5 Data Leakage Prevention

| Risk | Mitigation |
|------|-----------|
| Variant_ID as feature | Drop before training |
| EK_4/EK_6 circularity | Train ablation model (Pipeline C) without these features |
| Train/test overlap | Variant_ID-aware splits when evaluating on panels |
| Target encoding leakage | Only fit encoders within CV fold training splits |
| Feature selection leakage | Only select features within CV fold training splits |

### 11.6 Reproducibility Requirements

- Fixed random seed: 42
- Train/validation/test separation preserved
- Versioned preprocessing pipeline (scikit-learn Pipeline or manual steps documented)
- All metrics saved to CSV
- OOF predictions saved for error analysis
- Confusion matrices saved for each fold and panel

---

## 12. Risk Analysis

### 12.1 Dataset Risks

| Risk | Severity | Impact | Mitigation |
|------|----------|--------|-----------|
| 54.9% missing data | HIGH | Reduces effective information per variant | Native GBDT missing handling + missingness features |
| MNAR missingness (label-correlated) | HIGH | Imputation could introduce bias | Do NOT impute for tree models; use missingness as signal |
| CFTR sample size (n=111, 21 Benign) | HIGH | Unreliable model estimates | Use global model + threshold calibration |
| 57 constant columns | MEDIUM | Noise if not removed | Drop before training |
| 114 duplicate feature rows in MASTER | LOW | Minor overfitting risk | Monitor, but likely benign |

### 12.2 Label Quality Risks

| Risk | Severity | Evidence |
|------|----------|----------|
| ClinVar label noise | LOW | ACMG criteria applied; Expert Panel / Practice Guideline status required |
| Conflicting labels in shared variants | LOW | 1 conflicting pair detected |
| Pathogenic/Likely Pathogenic merge | LOW | Merging is standard ACMG practice |

### 12.3 Leakage Risks

| Risk | Severity | Impact |
|------|----------|--------|
| EK_4/EK_6 circular prediction | HIGH | If these are ClinVar-trained meta-predictors, they contain the label |
| Variant_ID overlap MASTER↔Panels | MEDIUM | Shared variants in train and panel evaluation inflate metrics |
| Anonymized feature provenance | MEDIUM | Cannot verify whether any feature is derived from ClinVar labels |

### 12.4 Competition Compliance Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Metric change by TÜSEB | MEDIUM | Build models that optimize multiple metrics; keep threshold flexible |
| Train/Test distribution mismatch | HIGH | Test is balanced (1:1); train is imbalanced (up to 5:1). Calibrate thresholds. |
| Runtime constraints unknown | LOW | Tree models are fast for inference; <1 second per variant |
| Code reproducibility requirement | MEDIUM | Document all preprocessing; use fixed seeds; save pipeline artifacts |

### 12.5 Model Performance Risks

| Risk | Severity | Impact |
|------|----------|--------|
| PAH panel underperformance | HIGH | Only 62 Benign training samples; 5:1 imbalance |
| Overfitting on small panels | HIGH | CFTR n=111; cross-validation variance will be high |
| EK ablation performance drop | MEDIUM | Removing EK_4/5/6 may drop ROC-AUC by 0.02-0.05 |
| Distribution shift to test data | HIGH | Models may not generalize to unseen test variants |

---

## 13. Open Questions

### Must-Answer Before Final Model Development

1. **F1 calculation method**: Is F1 computed per-panel separately and then averaged, or computed globally across all 4 test sets? This affects threshold optimization strategy.
2. **EK feature provenance**: Are EK_4 and EK_6 trained on ClinVar data? If yes, their inclusion is circular. The competition should clarify this during Q&A sessions.
3. **Test set balance confirmation**: Will all 4 test sets be exactly 1:1 balanced as stated in the specification?
4. **Single model or multiple submissions**: Can teams submit separate models per panel, or must it be a single unified model?
5. **Runtime constraints**: Are there hardware or time limits for inference during the competition finals?
6. **Feature column mapping**: Is there any additional documentation mapping anonymized feature names (AL_*, EK_*) to their biological meaning?

### Nice-to-Know

7. What is the source of Benign variants augmented from gnomAD? Are they confirmed Benign or merely common?
8. Are there any additional Q&A session clarifications since V1.4 of the specification?
9. Will the external validation use the exact same feature extraction pipeline?

---

## 14. Final Recommendations

### Immediate Next Steps (Priority Order)

1. **Resolve EK circularity**: Train Pipeline C (without EK_4/EK_5/EK_6) and compare performance. If the drop is small (<0.02 AUC), consider excluding them for safety.

2. **Optimize for F1 on balanced data**: Since test sets are balanced (1:1), re-optimize thresholds assuming balanced class distribution. Current thresholds are tuned for imbalanced training data.

3. **Build ensemble**: Train XGBoost and CatBoost alongside LightGBM. Stack their predictions with a logistic regression meta-learner.

4. **Panel-specific calibration**: For each panel, find the F1-maximizing threshold using the overlapping variants as a calibration set (or leave-one-panel-out validation).

5. **Hyperparameter tuning**: Run Bayesian optimization (Optuna) on the global LightGBM model. Focus on: `num_leaves`, `min_child_samples`, `learning_rate`, `feature_fraction`, `bagging_fraction`.

6. **Feature ablation study**: Systematically remove feature groups (AL_1-6, AL_27-38, high-missingness blocks) to identify which features add signal vs noise.

7. **Error analysis**: Examine misclassified variants from OOF predictions. Identify patterns — are errors concentrated in specific panels, missingness profiles, or feature ranges?

8. **Prepare documentation**: Begin writing the Proje Detay Raporu. Include: problem definition, literature review, methodology, model architecture, validation results, explainability (SHAP), and clinical relevance discussion.

### Development Timeline Suggestion

| Week | Task |
|------|------|
| Week 1 | EK ablation (Pipeline C), ensemble training (XGBoost, CatBoost) |
| Week 2 | Stacking ensemble, threshold calibration per panel |
| Week 3 | Hyperparameter optimization, error analysis, feature engineering experiments |
| Week 4 | Final model selection, robustness testing, documentation |

### Critical Reminders

- **F1 Score is the metric** — optimize for F1, not ROC-AUC
- **Test data is balanced** — calibrate thresholds for 1:1 ratio
- **Do NOT impute** for tree models — native handling is superior
- **Do NOT use SMOTE** — biologically meaningless for genetic variants
- **Save everything** — OOF predictions, metrics, feature importances, thresholds
- **Code must be reproducible** — the jury can request code re-execution

---

*Report generated on 2026-06-02. Based on TEKNOFEST 2026 Sağlıkta Yapay Zeka competition specification V1.4 and training dataset version "EĞİTİM (TRAIN) SETLERİ 2".*
