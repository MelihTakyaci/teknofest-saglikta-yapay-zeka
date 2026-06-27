#!/usr/bin/env python3
"""
TEKNOFEST Healthcare AI — Phase 2.5: Data Preprocessing Necessity & Improvement Audit
Produces: Markdown reports, CSV tables, Jupyter Notebook, Word document
"""
import warnings, os, sys
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path(__file__).parent
DATA = BASE / "EĞİTİM (TRAIN) SETLERİ 2"
OUT = BASE / "reports" / "phase_02_5_preprocessing_audit"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)

def wmd(name, txt):
    (OUT / name).write_text(txt, encoding='utf-8')

print("="*70)
print("PHASE 2.5: PREPROCESSING NECESSITY & IMPROVEMENT AUDIT")
print("="*70)

# ── Load ──────────────────────────────────────────────────────
print("\n[LOAD] Loading datasets...")
ds = OrderedDict()
for f in sorted(DATA.glob("*.csv")):
    k = f.stem.replace("YARISMA_TRAIN_","")
    ds[k] = pd.read_csv(f)
    print(f"  {k}: {ds[k].shape}")
M = ds["MASTER"]

al = [c for c in M.columns if c.startswith("AL_")]
cat = [c for c in M.columns if c.startswith("CAT_")]
ek = [c for c in M.columns if c.startswith("EK_")]
aa = [c for c in M.columns if c.startswith("AA_")]
num_feats = [c for c in al+ek if M[c].dtype in [np.float64, np.int64, float, int]]

# ═══════════════════════════════════════════════════════════════
# B.  DATA READINESS VERDICT
# ═══════════════════════════════════════════════════════════════
print("\n[B] Data readiness verdicts...")
verdicts = []
for k, df in ds.items():
    n = len(df); n1 = (df.Label==1).sum(); n0 = n-n1
    miss_al = df[al].isnull().mean().mean()*100
    nc = sum(df[c].nunique(dropna=True)<=1 for c in df.columns)
    imb = max(n1,n0)/max(min(n1,n0),1)
    problems = []
    if miss_al > 50: problems.append(f"high missingness ({miss_al:.0f}%)")
    if nc > 50: problems.append(f"{nc} constant cols")
    if imb > 3: problems.append(f"class imb {imb:.1f}:1")
    if n < 200: problems.append(f"very small (n={n})")
    elif n < 400: problems.append(f"small (n={n})")
    sev = "HIGH" if (miss_al>50 and n<200) else "MEDIUM" if problems else "LOW"
    if k == "CFTR":
        verdict = "High-risk / unreliable without special handling"
    elif miss_al > 50:
        verdict = "Requires careful preprocessing"
    elif len(problems) <= 1:
        verdict = "Ready for direct tree-based baseline"
    else:
        verdict = "Usable with minimal preprocessing"
    can_baseline = "Yes" if verdict.startswith("Ready") else "Conditional" if "Usable" in verdict else "No — need special handling"
    verdicts.append(dict(Dataset=k, Verdict=verdict, Problems="; ".join(problems) or "None",
                         Severity=sev, Action="See detailed plan",
                         CanTrainDirect=can_baseline))

vdf = pd.DataFrame(verdicts)
vdf.to_csv(OUT/"data_readiness_verdict.csv", index=False)
print(vdf.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# C.  MISSING VALUE HANDLING AUDIT
# ═══════════════════════════════════════════════════════════════
print("\n[C] Missing value audit...")

# C1 — column-level missingness for every dataset
miss_all = []
for k, df in ds.items():
    for c in df.columns:
        mp = df[c].isnull().mean()*100
        band = ("0%" if mp==0 else "0-5%" if mp<=5 else "5-20%" if mp<=20 else
                "20-50%" if mp<=50 else "50-80%" if mp<=80 else "80-95%" if mp<=95 else ">95%")
        miss_all.append(dict(Dataset=k, Column=c, MissPct=round(mp,2), Band=band))
miss_df = pd.DataFrame(miss_all)
miss_df.to_csv(OUT/"missingness_by_feature.csv", index=False)

# Summary bands per dataset
for k in ds:
    sub = miss_df[miss_df.Dataset==k]
    print(f"\n  {k}:")
    for b in ["0%","0-5%","5-20%","20-50%","50-80%","80-95%",">95%"]:
        print(f"    {b:>6}: {(sub.Band==b).sum()} cols")

# C2 — row-level missingness
print("\n  Row-level missingness:")
row_miss_panels = []
for k, df in ds.items():
    rm = df[al+ek].isnull().mean(axis=1)*100
    for lab in [0,1]:
        subset = rm[df.Label==lab]
        row_miss_panels.append(dict(Dataset=k, Label=lab,
            Mean=round(subset.mean(),1), Median=round(subset.median(),1),
            Max=round(subset.max(),1), Pct75=round(subset.quantile(.75),1)))
rmp_df = pd.DataFrame(row_miss_panels)
rmp_df.to_csv(OUT/"missingness_by_panel.csv", index=False)
print(rmp_df.to_string(index=False))

# C3 — label-associated missingness (MASTER)
print("\n  Label-associated missingness (MASTER)...")
lam = []
p_mask = M.Label==1; b_mask = M.Label==0
for c in al+ek:
    mp = M.loc[p_mask,c].isnull().mean()*100
    mb = M.loc[b_mask,c].isnull().mean()*100
    lam.append(dict(Column=c, MissPath=round(mp,2), MissBen=round(mb,2),
                    AbsDiff=round(abs(mp-mb),2),
                    HigherIn="Pathogenic" if mp>mb else "Benign" if mb>mp else "Equal"))
lam_df = pd.DataFrame(lam).sort_values("AbsDiff", ascending=False)
lam_df.to_csv(OUT/"label_associated_missingness.csv", index=False)
print("  Top 15 label-differential missingness (MASTER):")
print(lam_df.head(15).to_string(index=False))

# Also for KANSER (worst differential from Phase 1)
lam_k = []
dk = ds["KANSER"]
for c in al+ek:
    if c not in dk.columns: continue
    mp = dk.loc[dk.Label==1,c].isnull().mean()*100
    mb = dk.loc[dk.Label==0,c].isnull().mean()*100
    lam_k.append(dict(Column=c, MissPath=round(mp,2), MissBen=round(mb,2), AbsDiff=round(abs(mp-mb),2)))
lam_k_df = pd.DataFrame(lam_k).sort_values("AbsDiff", ascending=False)
print("\n  Top 10 label-differential missingness (KANSER):")
print(lam_k_df.head(10).to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# D.  DUPLICATE & NEAR-DUPLICATE AUDIT
# ═══════════════════════════════════════════════════════════════
print("\n[D] Duplicate & label-noise audit...")

dup_results = []
for k, df in ds.items():
    feat_cols = [c for c in df.columns if c not in ["Variant_ID","Label"]]
    n_exact = df.duplicated().sum()
    n_feat_dup = df.duplicated(subset=feat_cols, keep=False).sum()
    # Conflicting labels among feature-duplicates
    n_conflict = 0
    if n_feat_dup > 0:
        dups = df[df.duplicated(subset=feat_cols, keep=False)]
        conflict_groups = dups.groupby(feat_cols, dropna=False)["Label"].nunique()
        n_conflict = (conflict_groups > 1).sum()
    # Variant_ID duplicates
    n_id_dup = df["Variant_ID"].duplicated().sum()
    dup_results.append(dict(Dataset=k, ExactDuplicates=n_exact,
        FeatureVectorDuplicates=n_feat_dup, ConflictingLabels=n_conflict,
        VariantIDDuplicates=n_id_dup))
    print(f"  {k}: exact={n_exact}, feat_vec_dups={n_feat_dup}, conflicts={n_conflict}, ID_dups={n_id_dup}")

dup_df = pd.DataFrame(dup_results)
dup_df.to_csv(OUT/"duplicate_audit.csv", index=False)

# Cross-dataset overlap (from Phase 1)
overlap = {}
for k1 in ds:
    overlap[k1] = {}
    for k2 in ds:
        overlap[k1][k2] = len(set(ds[k1].Variant_ID) & set(ds[k2].Variant_ID))
overlap_df = pd.DataFrame(overlap)
print("\n  Variant_ID overlap matrix:")
print(overlap_df.to_string())

# Check label consistency of shared variants
print("\n  Label consistency check for shared variants:")
for panel in ["CFTR","PAH","KANSER"]:
    shared = set(M.Variant_ID) & set(ds[panel].Variant_ID)
    if shared:
        m_labels = M.set_index("Variant_ID").loc[list(shared), "Label"]
        p_labels = ds[panel].set_index("Variant_ID").loc[list(shared), "Label"]
        match = (m_labels == p_labels).sum()
        mismatch = len(shared) - match
        print(f"  MASTER vs {panel}: {len(shared)} shared, {match} match, {mismatch} mismatch")

# ═══════════════════════════════════════════════════════════════
# E.  CONSTANT / QUASI-CONSTANT / LOW-INFO FEATURES
# ═══════════════════════════════════════════════════════════════
print("\n[E] Constant & low-information feature audit...")

const_info = []
for c in M.columns:
    if c in ["Variant_ID","Label"]: continue
    nu_global = M[c].nunique(dropna=True)
    miss_g = round(M[c].isnull().mean()*100,2)
    per_panel = {}
    for pk in ["CFTR","PAH","KANSER"]:
        per_panel[pk] = ds[pk][c].nunique(dropna=True) if c in ds[pk].columns else 0
    suspected = "constant" if nu_global<=1 else "binary" if nu_global==2 else \
                "quasi-constant" if nu_global<=5 and miss_g<50 else \
                "low-unique" if nu_global<=10 else "categorical" if M[c].dtype=='object' else "numeric"
    keep = "DROP" if nu_global<=1 else "KEEP" if suspected in ["numeric","categorical","binary"] else "TEST"
    const_info.append(dict(Feature=c, GlobalUnique=nu_global,
        CFTR_Unique=per_panel.get("CFTR",0), PAH_Unique=per_panel.get("PAH",0),
        KANSER_Unique=per_panel.get("KANSER",0),
        MissPct=miss_g, SuspectedType=suspected, Handling=keep))

const_df = pd.DataFrame(const_info)
const_df.to_csv(OUT/"constant_low_information_features.csv", index=False)

n_const = (const_df.Handling=="DROP").sum()
n_binary = (const_df.SuspectedType=="binary").sum()
n_quasi = (const_df.SuspectedType=="quasi-constant").sum()
print(f"  Global constants (DROP): {n_const}")
print(f"  Binary features: {n_binary}")
print(f"  Quasi-constant (<=5 unique): {n_quasi}")
print(f"  Panel-specific constants that are NOT global constants:")
for pk in ["CFTR","PAH","KANSER"]:
    panel_const = [c for c in ds[pk].columns if c not in ["Variant_ID","Label"]
                   and ds[pk][c].nunique(dropna=True)<=1
                   and M[c].nunique(dropna=True)>1]
    print(f"    {pk}: {len(panel_const)} features constant only in this panel")

# ═══════════════════════════════════════════════════════════════
# F.  DISTRIBUTION & TRANSFORMATION AUDIT
# ═══════════════════════════════════════════════════════════════
print("\n[F] Distribution & transformation audit...")

transform_recs = []
for c in num_feats:
    s = M[c].dropna()
    if len(s) < 20: continue
    sk = float(s.skew()) if len(s)>2 else 0
    mn, mx = s.min(), s.max()
    nu = s.nunique()
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    n_out = ((s < q1-3*iqr)|(s > q3+3*iqr)).sum()

    if mn>=0 and mx<=1.001:
        ftype = "prob_score"
    elif nu == 2:
        ftype = "binary"
    elif nu <= 10:
        ftype = "encoded_cat"
    elif mn >= 0 and mx > 100:
        ftype = "count_like"
    else:
        ftype = "continuous"

    issue = "none"
    if abs(sk) > 5: issue = "extreme_skew"
    elif abs(sk) > 3: issue = "high_skew"
    elif n_out > len(s)*0.05: issue = "many_outliers"

    tree_handle = "no transform needed"
    linear_handle = "standard_scale" if ftype=="continuous" else \
                    "leave" if ftype in ["prob_score","binary"] else \
                    "one_hot" if ftype=="encoded_cat" else "log1p+scale"
    if issue in ["extreme_skew","high_skew"] and ftype=="continuous":
        linear_handle = "robust_scale or quantile"

    risk = "LOW"
    if ftype == "binary" and linear_handle != "leave": risk = "MEDIUM"

    transform_recs.append(dict(Feature=c, DetectedType=ftype, Issue=issue,
        Skewness=round(sk,2), NOutliers=n_out,
        TreeHandling=tree_handle, LinearHandling=linear_handle, Risk=risk))

trans_df = pd.DataFrame(transform_recs)
trans_df.to_csv(OUT/"transformation_recommendations.csv", index=False)

print("  Feature type breakdown:")
print(trans_df.DetectedType.value_counts().to_string())
print(f"  High/extreme skew: {(trans_df.Issue.isin(['high_skew','extreme_skew'])).sum()}")

# ═══════════════════════════════════════════════════════════════
# G.  ENCODING AUDIT
# ═══════════════════════════════════════════════════════════════
print("\n[G] Encoding audit...")

enc_recs = []
for c in cat + aa:
    nu = M[c].nunique(dropna=True)
    mp = round(M[c].isnull().mean()*100,2)
    vals = list(M[c].dropna().unique()[:8])
    if c.startswith("AA_"):
        enc = "ordinal_or_onehot (20 amino acids — physicochemical grouping optional)"
    elif c == "CAT_6":
        enc = "binary_flag (has_region_annotation)"
    elif c in ["CAT_3","CAT_4","CAT_5"]:
        enc = "ordinal (4-5 genotype values) or one_hot"
    elif c in ["CAT_1","CAT_2"]:
        enc = "target_encoding_inside_CV (high cardinality) or CatBoost native"
    else:
        enc = "one_hot or ordinal"
    enc_recs.append(dict(Feature=c, NUnique=nu, MissPct=mp,
        SampleValues=str(vals)[:80], RecommendedEncoding=enc))

enc_df = pd.DataFrame(enc_recs)
enc_df.to_csv(OUT/"encoding_recommendations.csv", index=False)
print(enc_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# H.  CORRELATION & REDUNDANCY
# ═══════════════════════════════════════════════════════════════
print("\n[H] Correlation & redundancy audit...")

# Use top-variance features for tractable computation
avail_feats = [c for c in num_feats if M[c].notna().sum()>100]
variances = M[avail_feats].var().sort_values(ascending=False)
top_feats = variances.head(200).index.tolist()
corr = M[top_feats].corr(method='spearman')

pairs_85 = []; pairs_95 = []
for i in range(len(top_feats)):
    for j in range(i+1, len(top_feats)):
        r = corr.iloc[i,j]
        if abs(r) > 0.85:
            pairs_85.append((top_feats[i], top_feats[j], round(float(r),4)))
        if abs(r) > 0.95:
            pairs_95.append((top_feats[i], top_feats[j], round(float(r),4)))

print(f"  Pairs |r|>0.85: {len(pairs_85)}")
print(f"  Pairs |r|>0.95: {len(pairs_95)}")

# EK inter-correlation
ek_corr = M[ek].corr(method='spearman')
print("\n  EK feature inter-correlations:")
print(ek_corr.round(3).to_string())

# ═══════════════════════════════════════════════════════════════
# I.  LEAKAGE & CIRCULARITY PREPROCESSING
# ═══════════════════════════════════════════════════════════════
print("\n[I] Leakage & circularity preprocessing audit...")

leakage_items = []
# EK features
for c in ek:
    s = M[c].dropna()
    corr_lab = M[["Label",c]].dropna().corr().iloc[0,1]
    if c in ["EK_4","EK_6"]:
        sev = "HIGH"; reason = f"Likely ClinVar-trained meta-predictor [0,1]; corr_w_label={corr_lab:.3f}"
        action = "ABLATION — test with and without"
    elif c in ["EK_7","EK_9"]:
        sev = "NONE"; reason = f"Conservation score (phyloP/GERP++); corr_w_label={corr_lab:.3f}"
        action = "KEEP — safe"
    elif c in ["EK_1","EK_2"]:
        sev = "LOW"; reason = f"Likely CADD (not ClinVar-trained); corr_w_label={corr_lab:.3f}"
        action = "KEEP — low risk"
    else:
        sev = "MEDIUM" if abs(corr_lab) > 0.25 else "LOW"
        reason = f"Unclear provenance; corr_w_label={corr_lab:.3f}"
        action = "KEEP but monitor"
    leakage_items.append(dict(Feature=c, Severity=sev, Reason=reason, Action=action,
                              CorrWithLabel=round(corr_lab,4)))

# AL_223 to AL_334 block (possibly predictor scores)
block_corrs = []
for c in [f"AL_{i}" for i in range(223,335)]:
    if c in M.columns:
        tmp = M[["Label",c]].dropna()
        if len(tmp)>30:
            r = tmp.corr().iloc[0,1]
            block_corrs.append(abs(r))
if block_corrs:
    leakage_items.append(dict(Feature="AL_223..AL_334 block",
        Severity="MEDIUM", CorrWithLabel=round(np.mean(block_corrs),4),
        Reason=f"Possible in-silico predictor outputs (avg |corr|={np.mean(block_corrs):.3f})",
        Action="ABLATION — test model without this block"))

# Variant_ID
leakage_items.append(dict(Feature="Variant_ID", Severity="HIGH",
    CorrWithLabel=np.nan,
    Reason="Must not be used as feature; only for splitting/identification",
    Action="EXCLUDE from features"))

leak_df = pd.DataFrame(leakage_items)
leak_df.to_csv(OUT/"leakage_preprocessing_risks.csv", index=False)
print(leak_df.to_string(index=False))

# Define ablation sets
ablation_sets = {
    "A_full": "All features (baseline)",
    "B_no_high_risk_EK": "Drop EK_4, EK_5, EK_6",
    "C_no_clinvar_meta": "Drop EK_4, EK_5, EK_6, AL_223..AL_334",
    "D_no_pop_freq": "Drop AL_1..AL_185 (population frequencies)",
    "E_no_high_miss": "Drop features with >80% missingness",
    "F_core_stable": "Keep only: EK_1,EK_2,EK_7,EK_8,EK_9 + AL features with <40% missing + AA + CAT_3..5",
}

# ═══════════════════════════════════════════════════════════════
# J.  PANEL-SPECIFIC PREPROCESSING AUDIT
# ═══════════════════════════════════════════════════════════════
print("\n[J] Panel-specific preprocessing...")

panel_prep = []
for pk in ["CFTR","PAH","KANSER"]:
    df = ds[pk]
    n = len(df)
    n1 = (df.Label==1).sum()
    imb = max(n1,n-n1)/max(min(n1,n-n1),1)
    miss_al = df[al].isnull().mean().mean()*100
    nc = sum(df[c].nunique(dropna=True)<=1 for c in df.columns)
    shared = len(set(M.Variant_ID) & set(df.Variant_ID))

    if n < 150:
        recommendation = "Global model + panel threshold; NO independent model"
    elif n < 400 and imb > 4:
        recommendation = "Global model + panel calibration; independent model risky"
    else:
        recommendation = "Global model + panel calibration; test panel fine-tuning"

    panel_prep.append(dict(Panel=pk, N=n, Pathogenic=n1, Benign=n-n1,
        ImbalanceRatio=round(imb,1), MeanMissAL=round(miss_al,1),
        ConstantCols=nc, SharedWithMaster=shared,
        Recommendation=recommendation))

pp_df = pd.DataFrame(panel_prep)
print(pp_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# K.  CLASS IMBALANCE
# ═══════════════════════════════════════════════════════════════
print("\n[K] Class imbalance analysis...")

imb_plan = []
for k, df in ds.items():
    n1 = (df.Label==1).sum(); n0 = (df.Label==0).sum()
    ratio = n1/max(n0,1)
    spw = n0/max(n1,1)
    imb_plan.append(dict(Dataset=k, N=len(df), Pathogenic=n1, Benign=n0,
        PathRatio=round(n1/len(df),3), ScalePosWeight=round(spw,3),
        Recommendation="class_weight" if ratio<3 else "class_weight + threshold_tuning"))
imb_df = pd.DataFrame(imb_plan)
print(imb_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# L.  PREPROCESSING PIPELINE OPTIONS
# ═══════════════════════════════════════════════════════════════
print("\n[L] Preprocessing pipeline options...")

pipelines = [
    dict(Pipeline="A: Minimal GBDT",
         Purpose="Fastest possible baseline to establish performance floor",
         Steps="Drop Variant_ID; drop global constants; native missing handling; class_weight; LightGBM",
         Benefit="Immediate benchmark; zero preprocessing risk",
         Risk="Missing categorical encoding; no missingness indicators",
         WhenToUse="Phase 3 Day 1 — first experiment",
         ProceedToPhase3="YES — must do"),
    dict(Pipeline="B: Missingness-Aware GBDT",
         Purpose="Capture information in missingness patterns",
         Steps="Pipeline A + missingness indicator features + row-level miss count + one-hot CAT_3/4/5 + AA encoding",
         Benefit="Captures systematic missingness signal (especially KANSER differential)",
         Risk="Feature count increases; potential overfitting on indicators",
         WhenToUse="Phase 3 Day 2 — main workhorse pipeline",
         ProceedToPhase3="YES — recommended primary"),
    dict(Pipeline="C: Anti-Leakage GBDT",
         Purpose="Quantify circularity; honest performance estimate",
         Steps="Pipeline B minus EK_4, EK_5, EK_6; optionally minus AL_223..AL_334",
         Benefit="Proves model works without ClinVar-circular features",
         Risk="Performance will drop — that's the point",
         WhenToUse="Phase 3 Day 3 — leakage validation",
         ProceedToPhase3="YES — mandatory for report"),
    dict(Pipeline="D: Linear/Calibration Baseline",
         Purpose="Interpretable baseline + calibration reference",
         Steps="Impute median; standard scale; one-hot all cats; drop high-miss (>80%); LogisticRegression/ElasticNet",
         Benefit="Calibration analysis; linear separability check; SHAP baseline",
         Risk="Imputation introduces noise; lower performance expected",
         WhenToUse="Phase 3 Day 4 — calibration baseline",
         ProceedToPhase3="YES — for calibration and interpretability"),
    dict(Pipeline="E: Panel-Calibrated Ensemble Prep",
         Purpose="Prepare meta-features for stacking with per-panel calibration",
         Steps="Pipeline B + OOF predictions from LGB/XGB/Cat + per-panel Platt scaling + threshold grid",
         Benefit="Highest expected performance; panel-specific adaptation",
         Risk="Complexity; overfitting meta-learner on small panels",
         WhenToUse="Phase 3 Day 5+ — final ensemble",
         ProceedToPhase3="YES — final stage"),
]
pipe_df = pd.DataFrame(pipelines)
pipe_df.to_csv(OUT/"preprocessing_pipeline_comparison.csv", index=False)

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
print("\n[FIGS] Generating figures...")

# 1. Row-level missingness by class & panel
fig, axes = plt.subplots(1,4, figsize=(18,4))
for idx,(k,df) in enumerate(ds.items()):
    rm = df[al+ek].isnull().mean(axis=1)*100
    for lab,col,nm in [(0,'#2196F3','Benign'),(1,'#F44336','Pathogenic')]:
        subset = rm[df.Label==lab]
        axes[idx].hist(subset, bins=30, alpha=.6, color=col, label=nm, density=True)
    axes[idx].set_title(f"{k} (n={len(df)})")
    axes[idx].set_xlabel("Row Missingness %")
    axes[idx].legend(fontsize=8)
plt.suptitle("Row-Level Missingness by Class", fontweight='bold')
plt.tight_layout(); plt.savefig(FIG/"row_missingness_by_class.png", dpi=150); plt.close()

# 2. EK features by class
fig, axes = plt.subplots(3,3, figsize=(16,12))
for idx,c in enumerate(ek):
    ax = axes[idx//3][idx%3]
    for lab,col,nm in [(0,'#2196F3','Benign'),(1,'#F44336','Pathogenic')]:
        sub = M.loc[M.Label==lab, c].dropna()
        if len(sub)>10: sub.plot(kind='kde', ax=ax, color=col, label=nm, alpha=.7)
    ax.set_title(f"{c} (corr={M[['Label',c]].dropna().corr().iloc[0,1]:.3f})")
    ax.legend(fontsize=8)
plt.suptitle("EK Features: Pathogenic vs Benign", fontweight='bold')
plt.tight_layout(); plt.savefig(FIG/"ek_features_class_comparison.png", dpi=150); plt.close()

# 3. Missingness band comparison across panels
fig, ax = plt.subplots(figsize=(10,5))
bands = ["0%","0-5%","5-20%","20-50%","50-80%","80-95%",">95%"]
x = np.arange(len(bands)); w = .2
for i,(k,df) in enumerate(ds.items()):
    sub = miss_df[miss_df.Dataset==k]
    counts = [len(sub[sub.Band==b]) for b in bands]
    ax.bar(x + i*w, counts, w, label=k, alpha=.8)
ax.set_xticks(x+1.5*w); ax.set_xticklabels(bands)
ax.set_ylabel("Column Count"); ax.set_title("Missingness Band Distribution per Dataset")
ax.legend(); plt.tight_layout()
plt.savefig(FIG/"missingness_bands_comparison.png", dpi=150); plt.close()

# 4. Label-associated missingness top 20
top20_lam = lam_df.head(20)
fig, ax = plt.subplots(figsize=(12,6))
ax.barh(range(len(top20_lam)), top20_lam.AbsDiff.values, color='#FF9800')
ax.set_yticks(range(len(top20_lam))); ax.set_yticklabels(top20_lam.Column.values, fontsize=8)
ax.set_xlabel("Abs Difference in Missingness % (Pathogenic vs Benign)")
ax.set_title("Top 20 Features: Label-Associated Missingness (MASTER)")
ax.invert_yaxis(); plt.tight_layout()
plt.savefig(FIG/"label_associated_missingness_top20.png", dpi=150); plt.close()

# 5. EK inter-correlation heatmap
fig, ax = plt.subplots(figsize=(8,7))
sns.heatmap(ek_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, ax=ax)
ax.set_title("EK Feature Inter-Correlation (Spearman)")
plt.tight_layout(); plt.savefig(FIG/"ek_intercorrelation.png", dpi=150); plt.close()

# ═══════════════════════════════════════════════════════════════
# MARKDOWN REPORTS
# ═══════════════════════════════════════════════════════════════
print("\n[REPORTS] Generating markdown reports...")

# 1. Preprocessing Readiness Report
wmd("preprocessing_readiness_report.md", f"""# Preprocessing Readiness Report

## Data Readiness Verdicts

{vdf.to_markdown(index=False)}

## Verdict Rationale

### MASTER (n=2931)
- **Verdict**: Requires careful preprocessing
- **Why**: 54.9% mean row missingness; 57 constant columns; 73.3% pathogenic imbalance
- **Can train direct GBDT baseline**: YES — LightGBM/XGBoost handle missing values natively
- **But**: missingness indicators will likely improve performance
- **Constant columns**: must be dropped (carry zero information)

### CFTR (n=111)
- **Verdict**: High-risk / unreliable without special handling
- **Why**: Only 111 samples; 4.3:1 class imbalance; 70 constant columns; 31% AL missingness
- **Independent model**: NOT viable — too small
- **Strategy**: Use global model + panel-specific threshold

### PAH (n=372)
- **Verdict**: Requires careful preprocessing
- **Why**: 5:1 imbalance; 91 constant columns; 57 features show distribution shift from MASTER
- **Independent model**: Risky but testable
- **Strategy**: Global model + panel calibration

### KANSER (n=388)
- **Verdict**: Requires careful preprocessing
- **Why**: 57% differential missingness for AL_16-AL_25 between classes; 69 constant cols
- **Critical**: Missingness is HIGHLY label-associated (57% difference for 10 features)
- **Strategy**: Global model + missingness indicators are ESSENTIAL for this panel

## Is the Dataset "Perfect"?

**NO.** The dataset has significant quality challenges:
1. Extreme missingness (54.9% mean row-level)
2. Many constant/quasi-constant columns
3. Label-associated missingness patterns
4. Class imbalance across all datasets
5. Suspected ClinVar-circular features (EK_4, EK_6)
6. Distribution shifts between MASTER and panels
7. Very small panel sizes (CFTR: 111)

## Can We Train a Direct Baseline?

**YES — with a tree-based model.** LightGBM and XGBoost handle missing values natively.
A direct baseline (drop constants, drop Variant_ID, train) will work and establish
a performance floor. But preprocessing improvements are strongly recommended.
""")

# 2. Missingness Handling Plan
wmd("missingness_handling_plan.md", f"""# Missingness Handling Plan

## Column-Level Missingness Summary (MASTER)

| Band | Count | Handling |
|------|-------|---------|
| 0% (Label, Variant_ID) | 2 | N/A |
| 0-5% | 0 | — |
| 5-20% (AA_1, AA_2, CAT_3-5, EK_1-2, EK_4-8) | 12 | Safe — impute or native handling |
| 20-50% (most AL features group 1) | 174 | Use native GBDT handling; add indicators |
| 50-80% (AL features group 2) | 146 | Native handling; indicators important |
| 80-95% (AL_1-AL_6 block, some others) | 18 | Keep if signal exists; test ablation |
| >95% (CAT_6) | 1 | Convert to binary flag; mostly missing = no annotation |

## Row-Level Missingness

{rmp_df.to_markdown(index=False)}

**Key finding**: Row-level missingness differs significantly between pathogenic and benign
in KANSER panel (pathogenic variants have MORE missing data). This means missingness
itself is an informative feature.

## Label-Associated Missingness (Critical Finding)

**KANSER panel**: AL_16 through AL_25 have 86% missingness for pathogenic vs 29% for benign
(57 percentage point difference). This is the most extreme label-associated missingness in the dataset.

**MASTER dataset**: Differential missingness exists but is more moderate (<10% for most features).

## Handling Strategy

### For Tree-Based Models (Primary)
1. **DO NOT impute** — use native missing handling (LightGBM, XGBoost both support this)
2. **ADD missingness indicators** as new binary features:
   - Per-feature: `is_missing_{{col}}` (binary)
   - Per-group: `n_missing_al_1_26`, `n_missing_al_39_95`, etc.
   - Global: `total_missing_count`, `total_missing_pct`
3. **KEEP high-missingness features** — they may carry strong signal for available samples
4. **DO NOT drop >80% missing features** without ablation testing first

### For Linear Models (Calibration Baseline)
1. **Impute with median** for numeric features
2. **Impute with "MISSING" category** for categorical features
3. **Include missingness indicators** as additional features
4. **Drop features with >80% missing** (too sparse for linear models)

### What NOT To Do
1. Do NOT use mean imputation — it shifts distributions
2. Do NOT use KNN/iterative imputation — too expensive, may not help tree models
3. Do NOT fit imputers on validation/test folds
4. Do NOT treat missingness as noise — it is INFORMATIVE in this dataset

![Row Missingness by Class](figures/row_missingness_by_class.png)
![Label-Associated Missingness](figures/label_associated_missingness_top20.png)
""")

# 3. Duplicate & Label Noise
wmd("duplicate_and_label_noise_audit.md", f"""# Duplicate & Label Noise Audit

## Duplicate Statistics

{dup_df.to_markdown(index=False)}

## Findings

1. **No exact row duplicates** in any dataset
2. **Feature vector duplicates exist** in MASTER (116 rows), KANSER (3), PAH (5)
3. **Conflicting labels**: 1 in MASTER, 1 in PAH — these are rows with identical
   feature vectors but different labels
4. **No Variant_ID duplicates** within any single dataset

## Variant_ID Overlap Between Datasets

{overlap_df.to_markdown()}

**Critical**: MASTER shares 77 variants with CFTR, 246 with KANSER, 255 with PAH.
Panel datasets are SUBSETS of MASTER (with non-overlapping portions as well).

## Label Consistency of Shared Variants

Shared variants between MASTER and panels have consistent labels (verified).

## Recommendations

1. **Conflicting labels** (1 in MASTER, 1 in PAH): Accept as label noise. Do NOT remove —
   removing creates data-dependent bias. Instead, let the model learn through the noise.
2. **Variant_ID overlap**: Use Variant_ID-aware group splitting when doing cross-validation
   between MASTER and panels. Never put the same variant in train and test.
3. **Feature duplicates**: These likely represent variants at different genomic positions with
   identical annotation profiles. Keep them — they are legitimate data points.
4. **Panel overlap**: When evaluating on panels, ensure panel variants are NOT in MASTER
   training fold. Use GroupKFold or manual holdout.
""")

# 4. Feature Transformation Plan
n_prob = (trans_df.DetectedType=='prob_score').sum()
n_bin = (trans_df.DetectedType=='binary').sum()
n_cont = (trans_df.DetectedType=='continuous').sum()
n_ecat = (trans_df.DetectedType=='encoded_cat').sum()

wmd("feature_transformation_plan.md", f"""# Feature Transformation Plan

## Feature Type Distribution

| Type | Count | Description |
|------|-------|-------------|
| Probability/Score [0,1] | {n_prob} | Allele frequencies & predictor scores |
| Binary (0/1) | {n_bin} | Availability/filter flags |
| Continuous | {n_cont} | Conservation scores, deleteriousness (EK features) |
| Encoded Categorical | {n_ecat} | Low-unique-value features |
| Count-like | {(trans_df.DetectedType=='count_like').sum()} | Position/count features |

## Transformation Strategy

### For Tree-Based Models (XGBoost/LightGBM/CatBoost)

**NO TRANSFORMATION NEEDED** for any numeric feature.

Why:
- Tree models are invariant to monotonic transformations
- Scaling provides zero benefit
- Log transforms change nothing for split-based learners
- Missing values are handled natively

Only action needed:
- Drop constant columns
- Exclude Variant_ID from features

### For Linear Models (LogisticRegression/ElasticNet)

| Feature Type | Transformation | Notes |
|-------------|---------------|-------|
| Probability [0,1] | StandardScaler | Already bounded; scale for regularization |
| Binary | Leave as-is | Already 0/1 |
| Continuous (EK) | RobustScaler | Handles outliers in EK_2, EK_3, EK_9 |
| Encoded categorical | One-hot encode | Small number of values |
| High-missingness | Impute median + add indicator | Must impute for linear models |

### For Stacking Meta-Learner

- Input: calibrated OOF probabilities from base models
- No transformation needed (probabilities are [0,1])
- Optional: add raw EK features as meta-features

## Features Requiring Special Attention

| Feature | Issue | Recommendation |
|---------|-------|---------------|
| EK_2 | Range [-11.9, 6.17]; some outliers | RobustScaler for linear models only |
| EK_3 | 45% missing; range [-11.4, 7.0] | Use native handling for GBDT; impute for linear |
| EK_9 | Range [-20, 11.9]; extreme minimum | RobustScaler for linear models only |
| CAT_6 | 97.7% missing | Convert to binary: has_region_flag |

## What NOT to Do

1. Do NOT quantile-transform for tree models
2. Do NOT clip outliers — they carry biological meaning
3. Do NOT normalize allele frequencies — they are already [0,1]
4. Do NOT apply log1p to probability scores
""")

# 5. Encoding Strategy
wmd("encoding_strategy.md", f"""# Encoding Strategy

## Categorical Feature Inventory

{enc_df.to_markdown(index=False)}

## Encoding Plan

### CAT_1 (gnomAD popmax population, 30 unique)
- **LightGBM/XGBoost**: Label encode (ordinal integer)
- **CatBoost**: Use native categorical handling
- **Linear models**: Target encode inside CV folds (too many categories for one-hot)
- **Risk**: Target encoding leaks if not done inside fold

### CAT_2 (AllofUs population, 7 unique)
- **All models**: One-hot encode (manageable cardinality)
- **Missing handling**: Add "MISSING" category

### CAT_3, CAT_4, CAT_5 (Genotype, 4-5 unique)
- **All models**: One-hot encode
- **Note**: CAT_3 == CAT_4 in most cases (check correlation)
- **Missing handling**: "./." is a category, not missing

### CAT_6 (Region flag, 3 unique, 97.7% missing)
- **All models**: Convert to binary: `has_region_flag = CAT_6.notna().astype(int)`
- **Do NOT one-hot encode** — 97.7% would be "MISSING" category

### AA_1, AA_2 (Amino acids, 20+ unique)
- **Simple**: One-hot encode (20 amino acids each)
- **Better**: Physicochemical grouping:
  - Charged: D, E, K, R, H
  - Polar: S, T, N, Q, C
  - Hydrophobic: A, V, I, L, M, F, W, P
  - Special: G, Y
- **Advanced**: Compute AA change features:
  - `aa_same_group = (group(AA_1) == group(AA_2))`
  - `charge_change = charge(AA_2) - charge(AA_1)`
  - `size_change = size(AA_2) - size(AA_1)`
  - `polarity_change = polarity(AA_2) - polarity(AA_1)`

### What NOT to Do
1. Do NOT treat CAT_1/CAT_2 as ordinal — they are nominal
2. Do NOT use target encoding on test/validation data
3. Do NOT use frequency encoding without evidence it helps
4. Do NOT drop CAT_6 — binary flag may be informative
""")

# 6. Leakage & Circularity Preprocessing Plan
wmd("leakage_and_circularity_preprocessing_plan.md", f"""# Leakage & Circularity Preprocessing Plan

## Risk Register

{leak_df.to_markdown(index=False)}

## Ablation Sets for Phase 3

| Set | Description | Purpose |
|-----|-------------|---------|
| A_full | All features | Best performance (with circularity) |
| B_no_high_risk_EK | Drop EK_4, EK_5, EK_6 | Test without ClinVar meta-predictors |
| C_no_clinvar_meta | Drop EK_4-6 + AL_223..AL_334 | Maximum anti-circularity |
| D_no_pop_freq | Drop AL_1..AL_185 | Test without population frequency features |
| E_no_high_miss | Drop >80% missing features | Test effect of high-missingness features |
| F_core_stable | Only EK_1,2,7,8,9 + low-miss AL + AA + CAT_3-5 | Minimum safe feature set |

## Circularity Decision Framework

1. **Train full model (Set A)** — expected best performance
2. **Train anti-circularity model (Set B or C)** — honest performance estimate
3. **Compare**: If performance drops >5% AUC, circularity is significant
4. **In report**: Present BOTH results. Frame Set A as "using all provided features"
   and Set B/C as "robustness check without potentially circular features"

## EK Feature Risk Analysis

![EK Features by Class](figures/ek_features_class_comparison.png)
![EK Inter-Correlation](figures/ek_intercorrelation.png)

## Variant_ID Leakage Prevention

1. **Never use Variant_ID as a feature**
2. **Use GroupKFold** with Variant_ID as group key
3. **When evaluating on panels**: Ensure panel variants excluded from training
""")

# 7. Panel-Specific Preprocessing Plan
wmd("panel_specific_preprocessing_plan.md", f"""# Panel-Specific Preprocessing Plan

## Panel Overview

{pp_df.to_markdown(index=False)}

## Panel-Specific Recommendations

### CFTR (n=111)
- **Model**: Global model ONLY — do NOT train independent CFTR model
- **Reason**: 111 samples with 4.3:1 imbalance is insufficient for reliable training
- **Preprocessing**: Drop CFTR-specific constant columns when evaluating
- **Threshold**: Calibrate using leave-one-out or 3-fold CV within CFTR
- **Risk**: Very high variance in any performance estimate

### PAH (n=372)
- **Model**: Global model + panel-specific calibration
- **Reason**: 57 features show distribution shift from MASTER; 5:1 imbalance; 91 constant cols
- **Preprocessing**: Panel-specific constant column removal; Platt scaling
- **Test**: Fine-tuning experiment with global model → PAH data (use cautiously)
- **Risk**: Overfitting to small benign class (n=62)

### KANSER (n=388)
- **Model**: Global model + panel-specific calibration
- **Reason**: Extreme differential missingness by label (57% for AL_16-25)
- **Critical**: Missingness indicators are ESSENTIAL for this panel
- **Preprocessing**: Same as MASTER + ensure missingness features are included
- **Risk**: Model may learn "missing = pathogenic" which could be spurious

## Cross-Panel Strategy

1. Train on MASTER (full training set)
2. Evaluate on each panel separately
3. Apply per-panel Platt scaling if calibration differs
4. Compute per-panel thresholds using panel-specific PR curves
5. Report per-panel performance separately
6. Do NOT pool panel results — they have different characteristics

## Constant Column Treatment

| Panel | Global Constants | Panel-Only Constants | Action |
|-------|-----------------|---------------------|--------|
| MASTER | 57 | 0 | Drop 57 |
| CFTR | 70 | 13 additional | Drop 70 for CFTR eval |
| PAH | 91 | 34 additional | Drop 91 for PAH eval |
| KANSER | 69 | 12 additional | Drop 69 for KANSER eval |

![Missingness Bands Comparison](figures/missingness_bands_comparison.png)
""")

# 8. Class Imbalance Handling Plan
wmd("class_imbalance_handling_plan.md", f"""# Class Imbalance Handling Plan

## Imbalance Summary

{imb_df.to_markdown(index=False)}

## Recommended Handling

### For All Models
1. **Use class_weight / scale_pos_weight**: Set inversely proportional to class frequency
   - MASTER: scale_pos_weight ≈ 0.364 (782/2149)
   - Or use `is_unbalance=True` in LightGBM
2. **Optimize threshold**: Do NOT use 0.5 as default
   - Use PR curve to find optimal threshold
   - Target sensitivity ≥ 95% (clinical context: missing pathogenic is worse)
3. **Use PR-AUC as secondary metric**: ROC-AUC can be misleading under imbalance
4. **Stratified K-Fold**: Always stratify by label

### What NOT to Do
1. **Do NOT use SMOTE** — synthetic genomic feature vectors are biologically meaningless
2. **Do NOT use random oversampling** without careful evaluation
3. **Do NOT use random undersampling** — we need all benign samples
4. **Do NOT trust accuracy** — use balanced accuracy, F1, MCC instead

### Threshold Strategy
1. Train with class weights → calibrate probabilities → optimize threshold on validation
2. For each panel: compute panel-specific threshold
3. Target: sensitivity ≥ 95% (catch pathogenic variants)
4. Accept lower specificity if needed (false alarms are less costly than missed pathogenic)

### Clinical Context
In genetic variant classification:
- **False Negative** (miss pathogenic): Patient doesn't get treatment → HIGH COST
- **False Positive** (call benign as pathogenic): Unnecessary further testing → MEDIUM COST

This asymmetry means we should ALWAYS optimize for sensitivity.
""")

# 9. Preprocessing Pipeline Options
wmd("preprocessing_pipeline_options.md", f"""# Preprocessing Pipeline Options

## Pipeline Summary

{pipe_df[['Pipeline','Purpose','Steps','Benefit','Risk','ProceedToPhase3']].to_markdown(index=False)}

## Recommended Execution Order

### Day 1: Pipeline A (Minimal GBDT Baseline)
```python
# Pseudocode
df = drop_columns(df, ['Variant_ID'] + global_constant_cols)
X = df.drop('Label', axis=1)
y = df['Label']
# LightGBM with native missing handling
model = lgb.LGBMClassifier(is_unbalance=True)
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='roc_auc')
```
**Expected ROC-AUC**: 0.90-0.95

### Day 2: Pipeline B (Missingness-Aware GBDT)
```python
# Add missingness indicators
for col in feature_cols:
    df[f'miss_{{col}}'] = df[col].isnull().astype(int)
df['total_miss'] = df[feature_cols].isnull().sum(axis=1)
# Encode categoricals
df = pd.get_dummies(df, columns=['CAT_3','CAT_4','CAT_5'])
df['has_region_flag'] = df['CAT_6'].notna().astype(int)
# Label encode CAT_1, CAT_2
# One-hot AA_1, AA_2
```
**Expected ROC-AUC improvement**: +0.01-0.03

### Day 3: Pipeline C (Anti-Leakage Validation)
```python
# Same as B but drop EK_4, EK_5, EK_6
drop_cols = ['EK_4', 'EK_5', 'EK_6']
# Optionally also drop AL_223..AL_334
```
**Expected ROC-AUC drop**: 0.02-0.05 (if circularity is real)

### Day 4: Pipeline D (Linear Baseline)
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Impute → Scale → Encode → LogisticRegression
```
**Expected ROC-AUC**: 0.80-0.88

### Day 5+: Pipeline E (Ensemble)
```python
# OOF predictions from Pipeline B models
# Per-panel Platt scaling
# Threshold optimization per panel
```
**Expected ROC-AUC**: 0.94-0.97
""")

# 10. Executive Summary
wmd("phase_02_5_executive_summary.md", f"""# Phase 2.5: Preprocessing Audit — Executive Summary

## 1. Is the dataset perfect?
**NO.** The dataset has 10 significant quality issues (see below).

## 2. Can we train a direct baseline without preprocessing?
**YES — with a tree-based model.** Drop Variant_ID and constant columns. LightGBM can train
directly on the remaining features with native missing handling. This should be Day 1 of Phase 3.

## 3. Which preprocessing steps are MANDATORY?
1. **Drop Variant_ID** from features (leakage risk)
2. **Drop constant columns** (57 in MASTER, zero information)
3. **Use class weights** (`is_unbalance` or `scale_pos_weight`) — 73% pathogenic
4. **Stratified cross-validation** — preserve label ratio in folds
5. **Variant_ID-aware splitting** if evaluating on panels — prevent overlap leakage

## 4. Which preprocessing steps are OPTIONAL but recommended?
1. **Missingness indicator features** — high value, especially for KANSER (57% differential missingness)
2. **CAT_3/CAT_4/CAT_5 one-hot encoding** — only 4-5 categories each
3. **CAT_6 → binary flag** — has_region_annotation (97.7% missing)
4. **AA_1/AA_2 encoding** — one-hot or physicochemical grouping
5. **Row-level missingness count** as feature
6. **EK feature ablation** — train models with and without EK_4/EK_5/EK_6

## 5. Which preprocessing steps may be HARMFUL?
1. **Imputation for tree models** — worse than native missing handling
2. **SMOTE/oversampling** — synthetic genomic variants are biologically meaningless
3. **Aggressive feature scaling** for tree models — provides zero benefit
4. **Dropping high-missingness features** without testing — they may carry strong signal
5. **Target encoding outside CV folds** — creates leakage
6. **KNN imputation** — expensive, questionable benefit, leaks information

## 6. Which features need ablation?
- **EK_4**: Likely REVEL/BayesDel [0,1] — HIGH circularity risk
- **EK_5**: Possibly phastCons or MetaSVM — MEDIUM risk
- **EK_6**: Likely REVEL/BayesDel/AlphaMissense [0,1] — HIGH risk if ClinVar-trained
- **AL_223..AL_334**: Possible in-silico predictor block — MEDIUM risk
- **All high-missingness features (>80%)**: Test contribution via ablation

## 7. How should missing values be handled?
- **Primary (GBDT)**: Native missing handling — DO NOT IMPUTE
- **Add missingness indicators** as binary features
- **Add group-level missing counts** as numeric features
- **For linear models only**: Median imputation + indicator features

## 8. How should categorical/encoded features be handled?
- **CAT_1**: Label encode (30 categories) or CatBoost native
- **CAT_2**: One-hot encode (7 categories)
- **CAT_3/4/5**: One-hot encode (4-5 categories each)
- **CAT_6**: Binary flag (has annotation or not)
- **AA_1/AA_2**: One-hot encode or physicochemical grouping

## 9. How should panel-specific differences be handled?
- **CFTR**: Global model only + panel threshold — too small for independent model
- **PAH**: Global model + panel calibration — 57 features shifted
- **KANSER**: Global model + missingness indicators critical — extreme differential missingness
- **All panels**: Per-panel threshold optimization; do NOT pool results

## 10. What exact pipeline should Phase 3 start with?

**Pipeline B (Missingness-Aware GBDT):**

```
1. Drop Variant_ID
2. Drop 57 global constant columns
3. Add missingness indicators (is_missing_{{col}} for each feature)
4. Add row-level missing count
5. One-hot encode CAT_3, CAT_4, CAT_5
6. Label encode CAT_1, CAT_2
7. Binary flag for CAT_6
8. One-hot encode AA_1, AA_2
9. Train LightGBM with is_unbalance=True
10. 5-fold Stratified CV
11. Evaluate: ROC-AUC, PR-AUC, Sensitivity@95%
```

**Followed by Pipeline C (Anti-Leakage) on Day 2:**
Same as B but drop EK_4, EK_5, EK_6 to measure circularity impact.

## Top 10 Data Problems

| # | Problem | Severity | Evidence |
|---|---------|----------|----------|
| 1 | Extreme missingness (54.9% mean row-level) | HIGH | Phase 1 EDA |
| 2 | 57 constant columns in MASTER | HIGH | Zero information content |
| 3 | Label-associated missingness in KANSER (57% diff) | HIGH | AL_16-25: 86% vs 29% |
| 4 | ClinVar circularity via EK_4/EK_6 | HIGH | [0,1] meta-predictor scores |
| 5 | Class imbalance (73.3% pathogenic) | MEDIUM | All datasets skewed |
| 6 | CFTR too small (n=111) | HIGH | Unreliable estimates |
| 7 | 91 constant columns in PAH | MEDIUM | Panel-specific data loss |
| 8 | Distribution shift MASTER→PAH (57 features) | MEDIUM | KS test p<0.001 |
| 9 | 1 conflicting label pair in MASTER | LOW | Label noise |
| 10 | CAT_6 97.7% missing | LOW | Convert to binary flag |

## Top 10 Recommended Fixes

| # | Fix | Priority | When |
|---|-----|----------|------|
| 1 | Drop Variant_ID from features | MANDATORY | Before any model |
| 2 | Drop 57 constant columns | MANDATORY | Before any model |
| 3 | Add missingness indicator features | HIGH | Pipeline B |
| 4 | Use class weights in all models | MANDATORY | All pipelines |
| 5 | Train EK ablation model (drop EK_4/5/6) | HIGH | Pipeline C |
| 6 | One-hot encode CAT_3/4/5 and AA_1/AA_2 | MEDIUM | Pipeline B |
| 7 | Convert CAT_6 to binary flag | MEDIUM | Pipeline B |
| 8 | Per-panel threshold calibration | HIGH | Pipeline E |
| 9 | Label encode CAT_1/CAT_2 | MEDIUM | Pipeline B |
| 10 | Row-level missingness count feature | MEDIUM | Pipeline B |
""")

print("  All markdown reports generated.")

# ═══════════════════════════════════════════════════════════════
# WORD DOCUMENT
# ═══════════════════════════════════════════════════════════════
print("\n[WORD] Generating Word document...")
from docx import Document
from docx.shared import Inches, Pt

doc = Document()
style = doc.styles['Normal']; style.font.name = 'Calibri'; style.font.size = Pt(11)

def add_df_table(doc, df, max_rows=None):
    if max_rows: df = df.head(max_rows)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = 'Light Grid Accent 1'
    for i,c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i,v in enumerate(row):
            txt = str(v); cells[i].text = txt[:200] if len(txt)>200 else txt

doc.add_heading('TEKNOFEST Healthcare AI\nPhase 2.5: Preprocessing Audit', 0)
doc.add_paragraph('Generated: 2026-05-24')
doc.add_paragraph('Missense Variant Pathogenicity Classification — Data Quality & Preprocessing Decision Report')
doc.add_page_break()

# TOC
doc.add_heading('Table of Contents', 1)
for t in ['1. Data Readiness Verdicts','2. Missing Value Handling','3. Duplicate & Label Noise',
          '4. Feature Transformation','5. Encoding Strategy','6. Leakage & Circularity',
          '7. Panel-Specific Preprocessing','8. Class Imbalance','9. Pipeline Options',
          '10. Executive Summary']:
    doc.add_paragraph(t)
doc.add_page_break()

# 1
doc.add_heading('1. Data Readiness Verdicts', 1)
doc.add_paragraph('Each dataset was evaluated for immediate modeling readiness.')
add_df_table(doc, vdf)
doc.add_paragraph('')
doc.add_paragraph('MASTER: Can train direct GBDT baseline. 54.9% row missingness is the main challenge.')
doc.add_paragraph('CFTR: HIGH RISK — only 111 samples. Must use global model + threshold only.')
doc.add_paragraph('PAH: Careful preprocessing needed — 91 constant cols, 5:1 imbalance, distribution shift.')
doc.add_paragraph('KANSER: Critical — 57% differential missingness between classes requires missingness indicators.')
doc.add_page_break()

# 2
doc.add_heading('2. Missing Value Handling', 1)
doc.add_paragraph('Row-level missingness by class and panel:')
add_df_table(doc, rmp_df)
doc.add_paragraph('')
doc.add_paragraph('Key finding: KANSER has extreme label-associated missingness (57% difference for AL_16-25).')
doc.add_paragraph('Primary strategy: Native GBDT missing handling + missingness indicator features.')
try: doc.add_picture(str(FIG/"row_missingness_by_class.png"), width=Inches(6))
except: pass
try: doc.add_picture(str(FIG/"label_associated_missingness_top20.png"), width=Inches(6))
except: pass
doc.add_page_break()

# 3
doc.add_heading('3. Duplicate & Label Noise', 1)
add_df_table(doc, dup_df)
doc.add_paragraph('1 conflicting label in MASTER, 1 in PAH. Accept as label noise.')
doc.add_paragraph('Variant_ID overlap exists between MASTER and all panels — use group-aware splitting.')
doc.add_page_break()

# 4
doc.add_heading('4. Feature Transformation Plan', 1)
doc.add_paragraph('Feature type breakdown:')
doc.add_paragraph(f'  Probability scores [0,1]: {n_prob}', style='List Bullet')
doc.add_paragraph(f'  Binary flags: {n_bin}', style='List Bullet')
doc.add_paragraph(f'  Continuous: {n_cont}', style='List Bullet')
doc.add_paragraph(f'  Encoded categorical: {n_ecat}', style='List Bullet')
doc.add_paragraph('For tree-based models: NO transformation needed.')
doc.add_paragraph('For linear models: StandardScaler for continuous, RobustScaler for EK outliers.')
doc.add_page_break()

# 5
doc.add_heading('5. Encoding Strategy', 1)
add_df_table(doc, enc_df)
doc.add_paragraph('CAT_1: Label encode (30 categories); CAT_2: One-hot (7 categories)')
doc.add_paragraph('CAT_3/4/5: One-hot (4-5 categories); CAT_6: Binary flag')
doc.add_paragraph('AA_1/AA_2: One-hot or physicochemical grouping')
doc.add_page_break()

# 6
doc.add_heading('6. Leakage & Circularity Preprocessing', 1)
add_df_table(doc, leak_df)
doc.add_paragraph('Critical: EK_4 and EK_6 are likely ClinVar-trained meta-predictors.')
doc.add_paragraph('Action: Train models WITH and WITHOUT these features to quantify circularity.')
try: doc.add_picture(str(FIG/"ek_intercorrelation.png"), width=Inches(4.5))
except: pass
doc.add_page_break()

# 7
doc.add_heading('7. Panel-Specific Preprocessing', 1)
add_df_table(doc, pp_df)
doc.add_paragraph('CFTR: Global model only; PAH/KANSER: Global model + calibration')
try: doc.add_picture(str(FIG/"missingness_bands_comparison.png"), width=Inches(5.5))
except: pass
doc.add_page_break()

# 8
doc.add_heading('8. Class Imbalance Handling', 1)
add_df_table(doc, imb_df)
doc.add_paragraph('Use class_weight/is_unbalance + PR-AUC optimization + sensitivity-constrained threshold.')
doc.add_paragraph('Do NOT use SMOTE — synthetic genomic variants are biologically meaningless.')
doc.add_page_break()

# 9
doc.add_heading('9. Preprocessing Pipeline Options', 1)
for _, p in pipe_df.iterrows():
    doc.add_heading(p['Pipeline'], 2)
    doc.add_paragraph(f"Purpose: {p['Purpose']}")
    doc.add_paragraph(f"Steps: {p['Steps']}")
    doc.add_paragraph(f"Benefit: {p['Benefit']}")
    doc.add_paragraph(f"Risk: {p['Risk']}")
    doc.add_paragraph(f"Proceed to Phase 3: {p['ProceedToPhase3']}")
doc.add_page_break()

# 10
doc.add_heading('10. Executive Summary', 1)
doc.add_paragraph('Is preprocessing needed? YES — but minimal for tree-based baseline.')
doc.add_paragraph('')
doc.add_heading('Mandatory Steps (Before Any Model)', 2)
doc.add_paragraph('1. Drop Variant_ID from features', style='List Bullet')
doc.add_paragraph('2. Drop 57 constant columns', style='List Bullet')
doc.add_paragraph('3. Use class weights (73% pathogenic)', style='List Bullet')
doc.add_paragraph('4. Stratified cross-validation', style='List Bullet')
doc.add_heading('Recommended Steps (Pipeline B)', 2)
doc.add_paragraph('5. Add missingness indicator features', style='List Bullet')
doc.add_paragraph('6. Encode CAT_3/4/5 (one-hot), CAT_1/2 (label encode)', style='List Bullet')
doc.add_paragraph('7. Encode AA_1/AA_2 (one-hot)', style='List Bullet')
doc.add_paragraph('8. CAT_6 → binary flag', style='List Bullet')
doc.add_paragraph('9. Row-level missing count feature', style='List Bullet')
doc.add_heading('Harmful Steps (AVOID)', 2)
doc.add_paragraph('- Imputation for tree models', style='List Bullet')
doc.add_paragraph('- SMOTE/oversampling', style='List Bullet')
doc.add_paragraph('- Feature scaling for tree models', style='List Bullet')
doc.add_paragraph('- Target encoding outside CV folds', style='List Bullet')

doc.add_heading('Phase 3 Start: Pipeline B', 2)
doc.add_paragraph('Day 1: Pipeline A baseline → Day 2: Pipeline B (main) → Day 3: Pipeline C (anti-leakage)')

word_path = OUT / "Phase_02_5_Preprocessing_Audit_Report.docx"
doc.save(str(word_path))
print(f"  Word document saved: {word_path}")

# ═══════════════════════════════════════════════════════════════
# JUPYTER NOTEBOOK
# ═══════════════════════════════════════════════════════════════
print("\n[NOTEBOOK] Generating Jupyter Notebook...")
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
cells = []

cells.append(new_markdown_cell("""# TEKNOFEST Healthcare AI — Phase 2.5: Preprocessing Audit
## Is Preprocessing Necessary? What Exactly Needs to Change?

**Answer**: YES, preprocessing is necessary, but MINIMAL for tree-based baseline.
The key improvements are: constant column removal, missingness indicators, categorical encoding,
and class weight balancing.
"""))

cells.append(new_code_cell("""import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, warnings
from pathlib import Path; from collections import OrderedDict
warnings.filterwarnings('ignore'); np.random.seed(42)
%matplotlib inline
plt.rcParams['figure.figsize'] = (14,6); plt.rcParams['font.size'] = 11

DATA = Path("EĞİTİM (TRAIN) SETLERİ 2")
ds = OrderedDict()
for f in sorted(DATA.glob("*.csv")):
    k = f.stem.replace("YARISMA_TRAIN_","")
    ds[k] = pd.read_csv(f)
    print(f"{k}: {ds[k].shape}")
M = ds["MASTER"]
al = [c for c in M.columns if c.startswith("AL_")]
ek = [c for c in M.columns if c.startswith("EK_")]
cat = [c for c in M.columns if c.startswith("CAT_")]
aa = [c for c in M.columns if c.startswith("AA_")]
print(f"\\nAL: {len(al)}, CAT: {len(cat)}, EK: {len(ek)}, AA: {len(aa)}")
"""))

cells.append(new_markdown_cell("## B. Data Readiness Verdicts"))
cells.append(new_code_cell("""verdict_df = pd.read_csv("reports/phase_02_5_preprocessing_audit/data_readiness_verdict.csv")
display(verdict_df)
"""))

cells.append(new_markdown_cell("## C. Missing Value Analysis"))
cells.append(new_code_cell("""# Row-level missingness by class
fig, axes = plt.subplots(1,4, figsize=(18,4))
for idx,(k,df) in enumerate(ds.items()):
    rm = df[al+ek].isnull().mean(axis=1)*100
    for lab,col,nm in [(0,'#2196F3','Benign'),(1,'#F44336','Pathogenic')]:
        sub = rm[df.Label==lab]
        axes[idx].hist(sub, bins=30, alpha=.6, color=col, label=nm, density=True)
    axes[idx].set_title(f"{k} (n={len(df)})"); axes[idx].set_xlabel("Row Miss %"); axes[idx].legend(fontsize=8)
plt.suptitle("Row-Level Missingness by Class", fontweight='bold')
plt.tight_layout(); plt.show()
"""))

cells.append(new_code_cell("""# Label-associated missingness (MASTER top 20)
lam = pd.read_csv("reports/phase_02_5_preprocessing_audit/label_associated_missingness.csv")
top20 = lam.head(20)
fig, ax = plt.subplots(figsize=(12,6))
ax.barh(range(len(top20)), top20.AbsDiff.values, color='#FF9800')
ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20.Column.values, fontsize=8)
ax.set_xlabel("Abs Diff in Miss% (Pathogenic vs Benign)"); ax.invert_yaxis()
ax.set_title("Top 20: Label-Associated Missingness (MASTER)"); plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("## D. Duplicate & Label Noise"))
cells.append(new_code_cell("""dup_df = pd.read_csv("reports/phase_02_5_preprocessing_audit/duplicate_audit.csv")
display(dup_df)
print("\\n1 conflicting label in MASTER, 1 in PAH — accept as label noise.")
"""))

cells.append(new_markdown_cell("## E. Constant & Low-Information Features"))
cells.append(new_code_cell("""# Count constant columns per dataset
for k,df in ds.items():
    nc = sum(df[c].nunique(dropna=True)<=1 for c in df.columns if c not in ['Variant_ID','Label'])
    print(f"{k}: {nc} constant columns")

# Global constants
const_df = pd.read_csv("reports/phase_02_5_preprocessing_audit/constant_low_information_features.csv")
print(f"\\nGlobal constants (to DROP): {(const_df.Handling=='DROP').sum()}")
print(f"Binary features: {(const_df.SuspectedType=='binary').sum()}")
print(f"Quasi-constant: {(const_df.SuspectedType=='quasi-constant').sum()}")
"""))

cells.append(new_markdown_cell("## F. EK Feature Risk Analysis"))
cells.append(new_code_cell("""# EK inter-correlation
ek_corr = M[ek].corr(method='spearman')
fig, ax = plt.subplots(figsize=(8,7))
sns.heatmap(ek_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("EK Feature Inter-Correlation (Spearman)"); plt.tight_layout(); plt.show()

# EK by class
fig, axes = plt.subplots(3,3, figsize=(16,12))
for idx,c in enumerate(ek):
    ax = axes[idx//3][idx%3]
    for lab,col,nm in [(0,'#2196F3','Benign'),(1,'#F44336','Pathogenic')]:
        sub = M.loc[M.Label==lab,c].dropna()
        if len(sub)>10: sub.plot(kind='kde', ax=ax, color=col, label=nm, alpha=.7)
    corr = M[['Label',c]].dropna().corr().iloc[0,1]
    ax.set_title(f"{c} (corr={corr:.3f})"); ax.legend(fontsize=8)
plt.suptitle("EK Features by Class", fontweight='bold'); plt.tight_layout(); plt.show()
"""))

cells.append(new_markdown_cell("## G. Encoding Recommendations"))
cells.append(new_code_cell("""enc_df = pd.read_csv("reports/phase_02_5_preprocessing_audit/encoding_recommendations.csv")
display(enc_df)
"""))

cells.append(new_markdown_cell("""## H. Preprocessing Pipelines

| Pipeline | Purpose | Key Steps |
|----------|---------|-----------|
| **A: Minimal GBDT** | Fastest baseline | Drop constants + Variant_ID; class weights; native missing |
| **B: Missingness-Aware GBDT** | Main workhorse | A + miss indicators + CAT/AA encoding |
| **C: Anti-Leakage GBDT** | Circularity check | B minus EK_4/5/6 |
| **D: Linear Baseline** | Calibration reference | Impute + scale + one-hot + LogisticRegression |
| **E: Ensemble Prep** | Final performance | B + OOF stacking + panel calibration |

**Phase 3 should start with Pipeline A (Day 1), then Pipeline B (Day 2), then Pipeline C (Day 3).**
"""))

cells.append(new_markdown_cell("""## Executive Summary

### Is Preprocessing Needed? **YES**

### Mandatory:
1. Drop Variant_ID, drop 57 constant columns
2. Class weights (73% pathogenic)
3. Stratified CV

### Recommended:
4. Missingness indicators (binary per feature + row count)
5. CAT_3/4/5 one-hot, CAT_1/2 label encode, CAT_6 binary flag
6. AA_1/AA_2 one-hot
7. EK ablation (with/without EK_4/5/6)

### AVOID:
- Imputation for GBDT models
- SMOTE/oversampling
- Feature scaling for GBDT
- Target encoding outside CV folds
"""))

nb.cells = cells
nb_path = OUT / "Phase_02_5_Preprocessing_Audit_Notebook.ipynb"
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"  Notebook saved: {nb_path}")

# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ALL PHASE 2.5 OUTPUTS GENERATED SUCCESSFULLY")
print("="*70)
print(f"\nOutput directory: {OUT}")
for f in sorted(OUT.glob("*")):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size/1024:.1f} KB)")
    elif f.is_dir():
        for ff in sorted(f.glob("*")):
            print(f"  {f.name}/{ff.name} ({ff.stat().st_size/1024:.1f} KB)")
