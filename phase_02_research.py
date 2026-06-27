#!/usr/bin/env python3
"""
TEKNOFEST Healthcare AI - Phase 02: Specialized Model and Literature-Based Strategy Research
Generates: Markdown reports, CSV tables, Jupyter Notebook, Word document
"""

import os
import sys
import warnings
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
REPORT_DIR = BASE_DIR / "reports" / "phase_02_specialized_model_research"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PHASE 02: SPECIALIZED MODEL & LITERATURE-BASED STRATEGY RESEARCH")
print("=" * 70)

# ============================================================
# A. PROJECT CONTEXT FROM PHASE 1
# ============================================================
print("\n[A] Extracting project context from Phase 1...")

project_context = {
    "datasets": {
        "MASTER": {"rows": 2931, "cols": 353, "pathogenic": 2149, "benign": 782},
        "CFTR": {"rows": 111, "cols": 353, "pathogenic": 90, "benign": 21},
        "PAH": {"rows": 372, "cols": 353, "pathogenic": 310, "benign": 62},
        "KANSER": {"rows": 388, "cols": 353, "pathogenic": 268, "benign": 120},
    },
    "feature_groups": {
        "AL_features": 334,
        "CAT_features": 6,
        "EK_features": 9,
        "AA_features": 2,
        "Variant_ID": 1,
        "Label": 1,
    },
    "al_pattern": "Triplet: (allele_frequency_score [0,1], binary_flag, secondary_flag) per population",
    "cat_meanings": {
        "CAT_1": "gnomAD population with highest AF (gnomADe_EAS, gnomADg_NFE, etc.)",
        "CAT_2": "AllofUs population (AllofUs_EUR, AllofUs_AFR, etc.)",
        "CAT_3": "Genotype reference allele (C/C, T/T, G/G)",
        "CAT_4": "Genotype alternate allele",
        "CAT_5": "Genotype (possibly in another database)",
        "CAT_6": "Genomic region flag (lcr, segdup, decoy) — 97.7% missing",
    },
    "ek_ranges": {
        "EK_1": "[0.856, 6.170] — likely conservation/deleteriousness score",
        "EK_2": "[-11.9, 6.17] — likely raw conservation/deleteriousness score",
        "EK_3": "[-11.37, 7.01] — similar to EK_2, ~45% missing",
        "EK_4": "[0, 1] — probability meta-predictor score",
        "EK_5": "[0, 1] — probability meta-predictor score",
        "EK_6": "[0, 1] — probability meta-predictor score",
        "EK_7": "[-3.32, 10.0] — likely conservation score (phyloP/GERP-like)",
        "EK_8": "[-5.72, 0.756] — possibly log-scaled score",
        "EK_9": "[-20, 11.93] — likely GERP++ or similar",
    },
    "constraints": [
        "Feature names are anonymized (AL_*, CAT_*, EK_*)",
        "No genomic coordinates available",
        "No gene names available (except panel identity)",
        "Variant_ID format VAR_XXXXXX — may not map to external DBs",
        "AA_1, AA_2 provide reference/alternate amino acids",
        "Binary labels only (0=Benign, 1=Pathogenic), no VUS",
        "External database lookup prohibited by competition rules (assumed)",
        "73.3% pathogenic in MASTER — imbalanced",
        "54.9% mean row-level missingness in MASTER",
    ],
}

print("  Project context extracted successfully.")

# ============================================================
# B. SPECIALIZED MODEL CATALOGUE
# ============================================================
print("\n[B] Building specialized model catalogue...")

models = [
    # Classical / Rule-based / Score-based
    {
        "Model": "SIFT", "Year": 2003, "Category": "Classical score-based",
        "Reference": "Ng & Henikoff, Genome Res 2003",
        "Input Requirements": "Protein sequence + substitution position",
        "Output": "Score 0-1 (lower = more damaging), binary tolerated/damaging",
        "Prediction Target": "Functional impact (tolerance)",
        "Missense Specific": "Yes",
        "Disease Specific": "No (general)",
        "Pretrained Scores Available": "Yes (dbNSFP, VEP)",
        "Source Code Available": "Yes (sift.bii.a-star.edu.sg)",
        "Works Without Coordinates": "No — needs protein alignment",
        "Works Without Protein Seq": "No",
        "Works From Tabular Only": "No (needs sequence context)",
        "Likely Overlap With Dataset": "Possibly embedded as an AL feature (score [0,1], inverted)",
        "Leakage Risk": "Medium — trained independently of ClinVar labels",
        "Competition Usability": "Low (cannot query without coordinates)",
        "Recommendation": "Likely already embedded; use as conceptual reference"
    },
    {
        "Model": "PolyPhen-2", "Year": 2010, "Category": "Classical score-based",
        "Reference": "Adzhubei et al., Nat Methods 2010",
        "Input Requirements": "Protein sequence + substitution + structure features",
        "Output": "Score 0-1 (higher = more damaging), HumDiv/HumVar models",
        "Prediction Target": "Functional impact / pathogenicity",
        "Missense Specific": "Yes",
        "Disease Specific": "No (HumVar is disease-oriented)",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Yes (genetics.bwh.harvard.edu/pph2)",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "No",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possibly embedded as AL feature ([0,1] score)",
        "Leakage Risk": "Medium — HumVar trained on disease mutations",
        "Competition Usability": "Low",
        "Recommendation": "Likely already embedded; cite in report"
    },
    {
        "Model": "MutationTaster", "Year": 2010, "Category": "Classical score-based",
        "Reference": "Schwarz et al., Nat Methods 2010",
        "Input Requirements": "Genomic coordinates + alleles",
        "Output": "Score + categorical prediction (disease_causing, polymorphism)",
        "Prediction Target": "Pathogenicity",
        "Missense Specific": "No (handles all variant types)",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Web-only (mutationtaster.org)",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "No",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possibly embedded",
        "Leakage Risk": "High — uses ClinVar/HGMD directly",
        "Competition Usability": "Low",
        "Recommendation": "Cite only; HIGH circularity risk if labels from ClinVar"
    },
    {
        "Model": "MutationAssessor", "Year": 2011, "Category": "Classical score-based",
        "Reference": "Reva et al., Genome Res 2011",
        "Input Requirements": "Protein variant (UniProt ID + AA change)",
        "Output": "Functional impact score (-5 to +5), categorical",
        "Prediction Target": "Functional impact",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Limited",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "No",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possibly embedded",
        "Leakage Risk": "Low — conservation-based, not ClinVar-trained",
        "Competition Usability": "Low",
        "Recommendation": "Possibly embedded; cite in report"
    },
    {
        "Model": "PROVEAN", "Year": 2012, "Category": "Classical score-based",
        "Reference": "Choi et al., PLoS One 2012",
        "Input Requirements": "Protein sequence + substitution",
        "Output": "Score (lower = more deleterious), threshold -2.5",
        "Prediction Target": "Functional impact (deleteriousness)",
        "Missense Specific": "No (handles indels too)",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Yes (provean.jcvi.org)",
        "Works Without Coordinates": "No — needs protein sequence",
        "Works Without Protein Seq": "No",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possibly embedded",
        "Leakage Risk": "Low — sequence-based, not label-trained",
        "Competition Usability": "Low",
        "Recommendation": "Possibly embedded; cite in report"
    },
    {
        "Model": "FATHMM", "Year": 2013, "Category": "Classical score-based",
        "Reference": "Shihab et al., Hum Mutat 2013",
        "Input Requirements": "Protein ID + substitution OR genomic coordinates",
        "Output": "Score (-20 to +20), lower = more damaging",
        "Prediction Target": "Functional impact / pathogenicity",
        "Missense Specific": "Yes (FATHMM-MKL handles non-coding)",
        "Disease Specific": "No (but disease-specific weighting exists)",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Web-based (fathmm.biocompute.org.uk)",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "No",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possibly embedded",
        "Leakage Risk": "Medium",
        "Competition Usability": "Low",
        "Recommendation": "Possibly embedded; cite in report"
    },
    # Ensemble / Meta-predictors
    {
        "Model": "REVEL", "Year": 2016, "Category": "Ensemble meta-predictor",
        "Reference": "Ioannidis et al., AJHG 2016",
        "Input Requirements": "Precomputed — needs genomic coordinates for lookup",
        "Output": "Score 0-1 (higher = more pathogenic)",
        "Prediction Target": "Pathogenicity",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (widely available, dbNSFP)",
        "Source Code Available": "Precomputed scores only",
        "Works Without Coordinates": "No — lookup-based",
        "Works Without Protein Seq": "Lookup only",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "HIGH — EK_4 or EK_6 likely REVEL (range [0,1], mean ~0.9)",
        "Leakage Risk": "HIGH — meta-predictor trained on disease-associated variants, potential circularity if ClinVar-derived labels",
        "Competition Usability": "Low (cannot query); but likely already IN the dataset",
        "Recommendation": "Almost certainly already an EK feature; use as-is but report circularity risk"
    },
    {
        "Model": "BayesDel", "Year": 2019, "Category": "Ensemble meta-predictor",
        "Reference": "Feng, Bioinformatics 2019",
        "Input Requirements": "Precomputed scores via dbNSFP",
        "Output": "Score (continuous), addAF and noAF variants",
        "Prediction Target": "Pathogenicity / deleteriousness",
        "Missense Specific": "Yes (primarily)",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Limited",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Lookup only",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible — another EK feature candidate",
        "Leakage Risk": "HIGH — uses ClinVar-derived training data",
        "Competition Usability": "Low",
        "Recommendation": "Possibly embedded as EK feature; cite in report"
    },
    {
        "Model": "MetaSVM / MetaLR", "Year": 2015, "Category": "Ensemble meta-predictor",
        "Reference": "Dong et al., Hum Mol Genet 2015",
        "Input Requirements": "Precomputed via dbNSFP",
        "Output": "Score + binary prediction",
        "Prediction Target": "Pathogenicity",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Limited",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Lookup",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible EK feature",
        "Leakage Risk": "HIGH — ClinVar-trained",
        "Competition Usability": "Low",
        "Recommendation": "Possibly embedded; cite only"
    },
    {
        "Model": "ClinPred", "Year": 2018, "Category": "Ensemble meta-predictor",
        "Reference": "Alirezaie et al., AJHG 2018",
        "Input Requirements": "Precomputed via dbNSFP",
        "Output": "Score 0-1",
        "Prediction Target": "Clinical pathogenicity",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Limited",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Lookup",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible EK feature",
        "Leakage Risk": "VERY HIGH — explicitly trained on ClinVar labels",
        "Competition Usability": "Low",
        "Recommendation": "Highest circularity risk; if embedded, must report this"
    },
    {
        "Model": "M-CAP", "Year": 2016, "Category": "Ensemble meta-predictor",
        "Reference": "Jagadeesh et al., Nat Genet 2016",
        "Input Requirements": "Precomputed",
        "Output": "Score 0-1",
        "Prediction Target": "Pathogenicity",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Limited",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Lookup",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible",
        "Leakage Risk": "HIGH",
        "Competition Usability": "Low",
        "Recommendation": "Cite only"
    },
    # Deleteriousness / Annotation
    {
        "Model": "CADD", "Year": 2014/2019, "Category": "Deleteriousness annotation",
        "Reference": "Kircher et al., Nat Genet 2014; Rentzsch et al., 2019",
        "Input Requirements": "Genomic coordinates + alleles",
        "Output": "Raw score + Phred-scaled score (1-99)",
        "Prediction Target": "Deleteriousness (general, not pathogenicity)",
        "Missense Specific": "No (all variant types)",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (cadd.gs.washington.edu)",
        "Source Code Available": "Yes",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Needs coordinates",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "HIGH — EK_1 (range [0.86, 6.17]) or EK_2 (range [-11.9, 6.17]) may be CADD phred or raw score",
        "Leakage Risk": "Low — trained on simulated vs fixed variants, NOT ClinVar",
        "Competition Usability": "Low (cannot query)",
        "Recommendation": "Very likely already an EK feature; low circularity — safe to use"
    },
    {
        "Model": "DANN", "Year": 2015, "Category": "Deleteriousness annotation",
        "Reference": "Quang et al., Bioinformatics 2015",
        "Input Requirements": "Genomic coordinates",
        "Output": "Score 0-1",
        "Prediction Target": "Deleteriousness",
        "Missense Specific": "No",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Limited",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Needs coordinates",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible",
        "Leakage Risk": "Low — similar training to CADD",
        "Competition Usability": "Low",
        "Recommendation": "Possibly embedded; cite only"
    },
    {
        "Model": "Eigen", "Year": 2016, "Category": "Deleteriousness annotation",
        "Reference": "Ionita-Laza et al., Nat Genet 2016",
        "Input Requirements": "Genomic coordinates",
        "Output": "Eigen score + Eigen-PC score",
        "Prediction Target": "Deleteriousness (unsupervised)",
        "Missense Specific": "No",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Yes",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Needs coordinates",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible",
        "Leakage Risk": "Low — unsupervised approach",
        "Competition Usability": "Low",
        "Recommendation": "Cite only"
    },
    # Missense-specific ML
    {
        "Model": "VARITY", "Year": 2022, "Category": "Missense-specific ML",
        "Reference": "Wu et al., Science 2021",
        "Input Requirements": "Variant identifier (protein + AA change)",
        "Output": "Score 0-1 (pathogenicity probability)",
        "Prediction Target": "Pathogenicity",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (varity.msl.ubc.ca)",
        "Source Code Available": "Yes (GitHub)",
        "Works Without Coordinates": "No — needs protein variant ID",
        "Works Without Protein Seq": "Needs variant identification",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possibly an EK feature",
        "Leakage Risk": "Medium — uses multiple data sources including some clinical data",
        "Competition Usability": "Low",
        "Recommendation": "Cite in report as benchmark; cannot use directly"
    },
    {
        "Model": "MVP / gMVP", "Year": 2021, "Category": "Missense-specific ML",
        "Reference": "Qi et al., bioRxiv 2021",
        "Input Requirements": "Precomputed via dbNSFP",
        "Output": "Score 0-1",
        "Prediction Target": "Pathogenicity",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP)",
        "Source Code Available": "Limited",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Lookup",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible EK feature",
        "Leakage Risk": "Medium",
        "Competition Usability": "Low",
        "Recommendation": "Possibly embedded; cite only"
    },
    {
        "Model": "MutPred2", "Year": 2017, "Category": "Missense-specific ML",
        "Reference": "Pejaver et al., Nat Commun 2020",
        "Input Requirements": "Protein sequence + substitution",
        "Output": "Score 0-1 + molecular mechanism predictions",
        "Prediction Target": "Pathogenicity + functional impact mechanisms",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (dbNSFP for score only)",
        "Source Code Available": "Yes (mutpred.org)",
        "Works Without Coordinates": "Needs protein sequence",
        "Works Without Protein Seq": "No",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible",
        "Leakage Risk": "Medium",
        "Competition Usability": "Low",
        "Recommendation": "Cite in report; potentially valuable for explainability discussion"
    },
    # Protein LM / Deep Learning
    {
        "Model": "AlphaMissense", "Year": 2023, "Category": "Protein language model",
        "Reference": "Cheng et al., Science 2023",
        "Input Requirements": "Protein sequence + variant position (or precomputed lookup)",
        "Output": "Score 0-1 (pathogenicity probability)",
        "Prediction Target": "Pathogenicity",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (downloadable for all possible missense variants)",
        "Source Code Available": "Yes (GitHub, Zenodo)",
        "Works Without Coordinates": "Needs protein + position for lookup",
        "Works Without Protein Seq": "Precomputed lookup available",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible EK feature (score [0,1]); HIGH probability if dataset is recent (2024+)",
        "Leakage Risk": "LOW — trained on primate sequence data + protein structure, NOT ClinVar labels",
        "Competition Usability": "Low (cannot look up without coordinates)",
        "Recommendation": "STATE OF THE ART for missense; likely embedded as EK feature; very safe from circularity"
    },
    {
        "Model": "EVE", "Year": 2021, "Category": "Protein language model / evolutionary",
        "Reference": "Frazer et al., Nature 2021",
        "Input Requirements": "Gene-specific model + variant position",
        "Output": "EVE score (evolutionary index), binary classification",
        "Prediction Target": "Pathogenicity via evolutionary constraint",
        "Missense Specific": "Yes",
        "Disease Specific": "No (evolutionary, gene-specific)",
        "Pretrained Scores Available": "Yes (evemodel.org)",
        "Source Code Available": "Yes (GitHub)",
        "Works Without Coordinates": "Needs gene + protein position",
        "Works Without Protein Seq": "Lookup available for ~3,000 genes",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible",
        "Leakage Risk": "LOW — evolutionary model, no clinical label training",
        "Competition Usability": "Low",
        "Recommendation": "Cite in report; conceptually important for understanding unsupervised approaches"
    },
    {
        "Model": "ESM1b / ESM-1v / ESM2", "Year": 2021-2023, "Category": "Protein language model",
        "Reference": "Rives et al., PNAS 2021; Meier et al., NeurIPS 2021; Lin et al., Science 2023",
        "Input Requirements": "Protein sequence",
        "Output": "Log-likelihood ratio (variant effect score)",
        "Prediction Target": "Functional impact / fitness effect",
        "Missense Specific": "Yes (via log-likelihood scoring)",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Model weights available; scores computable",
        "Source Code Available": "Yes (GitHub: facebookresearch/esm)",
        "Works Without Coordinates": "Needs protein sequence",
        "Works Without Protein Seq": "No",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Unlikely — too new/specialized for most dbNSFP versions",
        "Leakage Risk": "LOW — unsupervised protein LM, no clinical labels",
        "Competition Usability": "Low (cannot apply without protein sequence)",
        "Recommendation": "Cite in report for discussion of PLM approaches; cannot use directly"
    },
    {
        "Model": "PrimateAI / PrimateAI-3D", "Year": 2018/2023, "Category": "Deep learning",
        "Reference": "Sundaram et al., Nat Genet 2018; PrimateAI-3D, Nat Genet 2023",
        "Input Requirements": "Genomic coordinates OR precomputed",
        "Output": "Score 0-1",
        "Prediction Target": "Pathogenicity (learned from primate variation)",
        "Missense Specific": "Yes",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Limited (PrimateAI); available for PrimateAI-3D via dbNSFP4.5+",
        "Source Code Available": "Limited",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Lookup",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Possible if dbNSFP version is recent",
        "Leakage Risk": "LOW — primate-trained, no ClinVar labels",
        "Competition Usability": "Low",
        "Recommendation": "Cite in report; possibly embedded"
    },
    # Conservation scores (not predictors, but relevant features)
    {
        "Model": "phyloP", "Year": 2010, "Category": "Conservation score",
        "Reference": "Pollard et al., Genome Res 2010",
        "Input Requirements": "Genomic position",
        "Output": "Conservation score (negative = accelerated, positive = conserved)",
        "Prediction Target": "Evolutionary conservation",
        "Missense Specific": "No (position-based)",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (UCSC, dbNSFP)",
        "Source Code Available": "Yes (phast package)",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Yes (genomic)",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "HIGH — EK_7 (range [-3.32, 10.0]) matches phyloP range exactly",
        "Leakage Risk": "NONE — purely evolutionary, no labels",
        "Competition Usability": "Already embedded (very likely)",
        "Recommendation": "Almost certainly EK_7; safe to use"
    },
    {
        "Model": "GERP++", "Year": 2010, "Category": "Conservation score",
        "Reference": "Davydov et al., PLoS Comput Biol 2010",
        "Input Requirements": "Genomic position",
        "Output": "RS score (rejected substitutions)",
        "Prediction Target": "Evolutionary constraint",
        "Missense Specific": "No",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (UCSC, dbNSFP)",
        "Source Code Available": "Yes",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Yes",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "HIGH — EK_9 (range [-20, 11.93]) matches GERP++ RS score range",
        "Leakage Risk": "NONE — purely evolutionary",
        "Competition Usability": "Already embedded (very likely)",
        "Recommendation": "Almost certainly EK_9; safe to use"
    },
    {
        "Model": "phastCons", "Year": 2005, "Category": "Conservation score",
        "Reference": "Siepel et al., Genome Res 2005",
        "Input Requirements": "Genomic position",
        "Output": "Score 0-1 (conservation probability)",
        "Prediction Target": "Evolutionary conservation",
        "Missense Specific": "No",
        "Disease Specific": "No",
        "Pretrained Scores Available": "Yes (UCSC, dbNSFP)",
        "Source Code Available": "Yes",
        "Works Without Coordinates": "No",
        "Works Without Protein Seq": "Yes",
        "Works From Tabular Only": "No",
        "Likely Overlap With Dataset": "Moderate — possibly an EK feature with [0,1] range",
        "Leakage Risk": "NONE",
        "Competition Usability": "Likely embedded",
        "Recommendation": "Possibly EK_5 or EK_8; safe to use"
    },
]

model_df = pd.DataFrame(models)
model_df.to_csv(REPORT_DIR / "specialized_model_catalogue.csv", index=False)
print(f"  Catalogued {len(model_df)} specialized models/tools")

# ============================================================
# C. APPLICABILITY AUDIT
# ============================================================
print("\n[C] Applicability audit...")

applicability = []
for m in models:
    name = m["Model"]
    direct = "NO" if "No" in m.get("Works From Tabular Only", "No") else "YES"
    indirect = "YES" if "embedded" in m.get("Likely Overlap With Dataset", "").lower() or \
               "likely" in m.get("Likely Overlap With Dataset", "").lower() else \
               "MAYBE" if "possible" in m.get("Likely Overlap With Dataset", "").lower() else "NO"
    missing_info = []
    if "No" in m.get("Works Without Coordinates", ""):
        missing_info.append("genomic coordinates")
    if "No" in m.get("Works Without Protein Seq", ""):
        missing_info.append("protein sequence")
    risk = m.get("Leakage Risk", "Unknown")

    if "already" in m.get("Recommendation", "").lower() or \
       "embedded" in m.get("Recommendation", "").lower():
        decision = "USE AS EMBEDDED FEATURE"
    elif "cite" in m.get("Recommendation", "").lower():
        decision = "CITE IN REPORT ONLY"
    elif "avoid" in m.get("Recommendation", "").lower():
        decision = "AVOID"
    else:
        decision = "CITE IN REPORT ONLY"

    applicability.append({
        "Model": name,
        "Direct Usability": direct,
        "Indirect (Embedded)": indirect,
        "Missing Information": "; ".join(missing_info) if missing_info else "None",
        "Competition Risk": risk,
        "Final Decision": decision
    })

applicability_df = pd.DataFrame(applicability)
applicability_df.to_csv(REPORT_DIR / "applicability_matrix.csv", index=False)
print(f"  Applicability audit complete for {len(applicability_df)} models")

# ============================================================
# D. FEATURE OVERLAP AUDIT
# ============================================================
print("\n[D] Feature overlap audit...")

feature_overlap = [
    {"Feature Group": "AL_1 to AL_26",
     "Count": 26,
     "Inferred Identity": "Population-specific allele frequencies (gnomAD exome sub-populations)",
     "Evidence": "Values [0,1], right-skewed, grouped by missingness bands matching gnomAD population coverage; CAT_1 contains gnomAD population labels",
     "Confidence": "HIGH",
     "Known Tool": "gnomAD (exome)",
     "Leakage Risk": "LOW — allele frequencies are legitimate features"},

    {"Feature Group": "AL_27 to AL_38 (low count, ~91% missing)",
     "Count": 12,
     "Inferred Identity": "Rare population allele frequencies or sub-population annotations from smaller databases",
     "Evidence": "Very high missingness, small number available (~252), similar [0,1] range with binary flags",
     "Confidence": "MEDIUM",
     "Known Tool": "Possibly gnomAD genome or very small sub-populations",
     "Leakage Risk": "LOW"},

    {"Feature Group": "AL_39 to AL_95 (triplet pattern: score + binary + binary)",
     "Count": 57,
     "Inferred Identity": "Population allele frequencies from multiple databases with availability and filtering flags",
     "Evidence": "Clear triplet pattern: continuous AF [0,0.9], binary data-present flag, binary filter/pass flag. n~1514 or n~1197.",
     "Confidence": "HIGH",
     "Known Tool": "gnomAD genome populations, possibly AllofUs sub-populations, TopMed, others",
     "Leakage Risk": "LOW — population frequencies are legitimate"},

    {"Feature Group": "AL_96 to AL_98",
     "Count": 3,
     "Inferred Identity": "Additional population frequency annotation or quality metrics",
     "Evidence": "Similar pattern to AL_39-AL_95 but slightly different structure",
     "Confidence": "MEDIUM",
     "Known Tool": "Unknown database/annotation",
     "Leakage Risk": "LOW"},

    {"Feature Group": "AL_99 to AL_185 (continued triplet pattern)",
     "Count": 87,
     "Inferred Identity": "Additional population-specific allele frequencies from AllofUs and possibly other databases",
     "Evidence": "Triplet pattern continues; CAT_2 encodes AllofUs population labels; n~1486-1197",
     "Confidence": "HIGH",
     "Known Tool": "AllofUs, gnomAD v4, possibly BRAVO/TopMed",
     "Leakage Risk": "LOW"},

    {"Feature Group": "AL_186 to AL_222",
     "Count": 37,
     "Inferred Identity": "Further population frequencies or in-silico predictor scores",
     "Evidence": "[0,1] scores with various missingness patterns; possibly transition from AF to predictor scores",
     "Confidence": "MEDIUM",
     "Known Tool": "Transition zone — possibly a mix of AF and predictor outputs",
     "Leakage Risk": "MEDIUM — if predictor scores, circularity possible"},

    {"Feature Group": "AL_223 to AL_334",
     "Count": 112,
     "Inferred Identity": "In-silico predictor scores and annotation features",
     "Evidence": "Many binary + continuous pairs; some constant columns; [0,1] range but with different missingness patterns than AF features",
     "Confidence": "MEDIUM",
     "Known Tool": "Possibly SIFT, PolyPhen, FATHMM, MutationTaster, REVEL, BayesDel, CADD_phred normalized, etc.",
     "Leakage Risk": "HIGH — many of these may be ClinVar-trained predictors"},

    {"Feature Group": "EK_1",
     "Count": 1,
     "Inferred Identity": "CADD Phred-scaled score (truncated) OR another conservation/deleteriousness metric",
     "Evidence": "Range [0.856, 6.170], mean=5.26 — CADD phred typically 0-40+, so this may be log-transformed or truncated. Alternatively a GnomAD constraint metric (pLI, LOEUF).",
     "Confidence": "MEDIUM",
     "Known Tool": "CADD phred (compressed) or gene constraint score",
     "Leakage Risk": "LOW if CADD; MEDIUM if gene constraint"},

    {"Feature Group": "EK_2",
     "Count": 1,
     "Inferred Identity": "CADD raw score OR FATHMM score",
     "Evidence": "Range [-11.9, 6.17], continuous — matches CADD raw score range; also plausible as FATHMM",
     "Confidence": "MEDIUM",
     "Known Tool": "CADD raw or FATHMM",
     "Leakage Risk": "LOW"},

    {"Feature Group": "EK_3",
     "Count": 1,
     "Inferred Identity": "Another deleteriousness/conservation raw score",
     "Evidence": "Range [-11.37, 7.01], 45% missing — similar to EK_2 but more missing",
     "Confidence": "LOW",
     "Known Tool": "Unknown — possibly Eigen, DANN raw, or MutationAssessor",
     "Leakage Risk": "LOW to MEDIUM"},

    {"Feature Group": "EK_4",
     "Count": 1,
     "Inferred Identity": "REVEL score or BayesDel_addAF or ClinPred",
     "Evidence": "Range [0, 1], mean=0.897, highly right-skewed — matches REVEL or BayesDel distribution",
     "Confidence": "MEDIUM-HIGH",
     "Known Tool": "REVEL, BayesDel, or ClinPred",
     "Leakage Risk": "HIGH — ClinVar-trained meta-predictor"},

    {"Feature Group": "EK_5",
     "Count": 1,
     "Inferred Identity": "Another meta-predictor [0,1] score or phastCons",
     "Evidence": "Range [0, 1], mean=0.805 — could be phastCons (conservation) or MetaLR/MetaSVM",
     "Confidence": "LOW-MEDIUM",
     "Known Tool": "phastCons or MetaLR/MetaSVM",
     "Leakage Risk": "LOW if phastCons; HIGH if MetaSVM/MetaLR"},

    {"Feature Group": "EK_6",
     "Count": 1,
     "Inferred Identity": "REVEL or BayesDel_noAF or AlphaMissense",
     "Evidence": "Range [0, 1], mean=0.906, very high mean — matches REVEL or AlphaMissense for pathogenic-enriched dataset",
     "Confidence": "MEDIUM",
     "Known Tool": "REVEL, BayesDel_noAF, or AlphaMissense",
     "Leakage Risk": "LOW if AlphaMissense; HIGH if REVEL/BayesDel"},

    {"Feature Group": "EK_7",
     "Count": 1,
     "Inferred Identity": "phyloP (100-way or 470-way vertebrate)",
     "Evidence": "Range [-3.32, 10.0], mean=5.94 — EXACTLY matches phyloP score range and distribution",
     "Confidence": "VERY HIGH",
     "Known Tool": "phyloP",
     "Leakage Risk": "NONE — purely evolutionary conservation"},

    {"Feature Group": "EK_8",
     "Count": 1,
     "Inferred Identity": "Amino acid substitution matrix score (Grantham/Blosum) or SiPhy",
     "Evidence": "Range [-5.72, 0.756], mean=0.559 — unusual range, possibly normalized substitution score",
     "Confidence": "LOW",
     "Known Tool": "Unknown — possibly normalized Grantham distance or SiPhy omega",
     "Leakage Risk": "LOW"},

    {"Feature Group": "EK_9",
     "Count": 1,
     "Inferred Identity": "GERP++ RS score",
     "Evidence": "Range [-20, 11.93], mean=7.48 — EXACTLY matches GERP++ rejected substitutions range",
     "Confidence": "VERY HIGH",
     "Known Tool": "GERP++",
     "Leakage Risk": "NONE — purely evolutionary conservation"},

    {"Feature Group": "CAT_1",
     "Count": 1,
     "Inferred Identity": "gnomAD population with maximum allele frequency",
     "Evidence": "Values: gnomADe_EAS, gnomADg_NFE, etc. — clearly gnomAD sub-population labels",
     "Confidence": "VERY HIGH",
     "Known Tool": "gnomAD (popmax population)",
     "Leakage Risk": "LOW"},

    {"Feature Group": "CAT_2",
     "Count": 1,
     "Inferred Identity": "AllofUs population label",
     "Evidence": "Values: AllofUs_EUR, AllofUs_AFR, etc.",
     "Confidence": "VERY HIGH",
     "Known Tool": "AllofUs",
     "Leakage Risk": "LOW"},

    {"Feature Group": "CAT_3 + CAT_4 + CAT_5",
     "Count": 3,
     "Inferred Identity": "Reference genotype across databases (likely gnomAD, AllofUs, another)",
     "Evidence": "Values: C/C, T/T, G/G, A/A, ./. — homozygous reference genotype representations",
     "Confidence": "HIGH",
     "Known Tool": "Genotype fields from multiple databases",
     "Leakage Risk": "LOW"},

    {"Feature Group": "CAT_6",
     "Count": 1,
     "Inferred Identity": "Genomic region flag (low complexity, segmental duplication, decoy)",
     "Evidence": "Values: lcr, segdup, decoy&segdup — 97.7% missing (most variants not in these regions)",
     "Confidence": "VERY HIGH",
     "Known Tool": "UCSC/gnomAD region annotations",
     "Leakage Risk": "NONE"},

    {"Feature Group": "AA_1 + AA_2",
     "Count": 2,
     "Inferred Identity": "Reference and alternate amino acids of the missense variant",
     "Evidence": "Single amino acid letter codes — directly defines the missense substitution",
     "Confidence": "CERTAIN",
     "Known Tool": "Variant annotation (VEP/Ensembl/RefSeq)",
     "Leakage Risk": "NONE — legitimate variant feature"},
]

feature_overlap_df = pd.DataFrame(feature_overlap)
feature_overlap_df.to_csv(REPORT_DIR / "feature_overlap_audit.csv", index=False)
print(f"  Feature overlap audit: {len(feature_overlap_df)} feature groups analyzed")

# ============================================================
# E. STRATEGY COMPARISON
# ============================================================
print("\n[E] Strategy comparison...")

strategies = [
    {
        "Strategy": "S1: Generic Tabular ML",
        "Description": "XGBoost/LightGBM/CatBoost on all provided features",
        "Performance Potential": "HIGH — these models excel at tabular data with mixed types and missingness",
        "Implementation Difficulty": "LOW",
        "Leakage Risk": "LOW-MEDIUM — depends on EK features",
        "Report Value": "MEDIUM — expected baseline, not innovative",
        "Robustness Risk": "LOW — well-understood, reproducible",
        "Final Recommendation": "MUST DO — this is the baseline AND likely winner"
    },
    {
        "Strategy": "S2: External Predictor Scores",
        "Description": "Query AlphaMissense/REVEL/CADD/EVE externally and add as features",
        "Performance Potential": "POTENTIALLY HIGH — but likely redundant with embedded features",
        "Implementation Difficulty": "HIGH — needs variant identification (coordinates hidden)",
        "Leakage Risk": "VERY HIGH — violates competition rules + circularity",
        "Report Value": "NONE — would be disqualifying",
        "Robustness Risk": "HIGH — dependency on external services",
        "Final Recommendation": "AVOID — cannot identify variants + likely rule violation"
    },
    {
        "Strategy": "S3: Meta-Classifier Over Embedded Predictors",
        "Description": "Identify EK features as existing predictor scores, build meta-learner on top",
        "Performance Potential": "HIGH — this is essentially what REVEL/BayesDel themselves do",
        "Implementation Difficulty": "LOW-MEDIUM",
        "Leakage Risk": "MEDIUM — circularity present but unavoidable since features are given",
        "Report Value": "HIGH — shows understanding of feature provenance",
        "Robustness Risk": "LOW",
        "Final Recommendation": "STRONG — use EK features as meta-predictor inputs in ensemble"
    },
    {
        "Strategy": "S4: Custom Competition Architecture",
        "Description": "Global model + panel adapters + missingness-aware + calibration + SHAP",
        "Performance Potential": "HIGHEST — addresses all identified challenges",
        "Implementation Difficulty": "MEDIUM-HIGH",
        "Leakage Risk": "LOW — careful design",
        "Report Value": "VERY HIGH — shows methodological sophistication",
        "Robustness Risk": "MEDIUM — complexity adds failure modes",
        "Final Recommendation": "PRIMARY STRATEGY — this is the target architecture"
    },
    {
        "Strategy": "S5: Panel-Specific Models",
        "Description": "Train separate CFTR, PAH, KANSER models + global model",
        "Performance Potential": "VARIABLE — depends on panel size",
        "Implementation Difficulty": "MEDIUM",
        "Leakage Risk": "MEDIUM — panel identity as implicit feature",
        "Report Value": "HIGH — demonstrates disease-specific reasoning",
        "Robustness Risk": "HIGH — CFTR (n=111) too small for reliable independent model",
        "Final Recommendation": "USE AS SUPPLEMENT — panel-specific thresholds, not separate models"
    },
]

strategy_df = pd.DataFrame(strategies)
strategy_df.to_csv(REPORT_DIR / "strategy_comparison.csv", index=False)

# ============================================================
# F. RISK REGISTER
# ============================================================
print("\n[F] Risk register...")

risks = [
    {"Risk ID": "R01",
     "Risk": "EK features are ClinVar-trained meta-predictors (REVEL, BayesDel, ClinPred)",
     "Category": "Circularity",
     "Severity": "HIGH",
     "Probability": "HIGH",
     "Impact": "Model learns to rely on circular features; generalization to novel variants is poor; competition judges may penalize",
     "Mitigation": "Train models WITH and WITHOUT EK features; report both; discuss circularity in report; compare performance gap"},

    {"Risk ID": "R02",
     "Risk": "Labels derived from ClinVar, same source as training data for embedded predictors",
     "Category": "Circularity",
     "Severity": "HIGH",
     "Probability": "VERY HIGH",
     "Impact": "Artificial performance inflation; model does not generalize to truly novel variants",
     "Mitigation": "Acknowledge in report; show EK-excluded model performance as 'honest' baseline; discuss clinical implications"},

    {"Risk ID": "R03",
     "Risk": "CFTR panel too small (n=111) for reliable modeling",
     "Category": "Statistical",
     "Severity": "HIGH",
     "Probability": "CERTAIN",
     "Impact": "Overfitting, high CV variance, unreliable performance estimates",
     "Mitigation": "Use global model for CFTR; only calibrate threshold on panel; bootstrap CI; regularize heavily"},

    {"Risk ID": "R04",
     "Risk": "Distribution shift between MASTER and panel datasets",
     "Category": "Generalization",
     "Severity": "MEDIUM",
     "Probability": "HIGH",
     "Impact": "Global model may underperform on panels, especially PAH (57 shifted features)",
     "Mitigation": "Evaluate per-panel; panel-specific calibration; domain adaptation if severe"},

    {"Risk ID": "R05",
     "Risk": "54.9% mean row-level missingness; critical information may be lost",
     "Category": "Data Quality",
     "Severity": "MEDIUM",
     "Probability": "CERTAIN",
     "Impact": "Reduced effective sample size; imputation errors; biased models",
     "Mitigation": "Use GBDT native missing handling; add missingness indicator features; avoid aggressive imputation"},

    {"Risk ID": "R06",
     "Risk": "Feature anonymization prevents feature validation and biological interpretation",
     "Category": "Interpretability",
     "Severity": "MEDIUM",
     "Probability": "CERTAIN",
     "Impact": "Cannot verify feature provenance; cannot use domain knowledge for feature engineering; limited SHAP interpretation",
     "Mitigation": "Infer feature groups from patterns; use SHAP for relative importance; discuss limitations in report"},

    {"Risk ID": "R07",
     "Risk": "Variant_ID overlap between MASTER and panels causes train-test leakage",
     "Category": "Leakage",
     "Severity": "HIGH",
     "Probability": "HIGH",
     "Impact": "Overoptimistic performance; unfair advantage if competition uses overlapping test set",
     "Mitigation": "Use Variant_ID-aware splits; never leak MASTER variants into panel test sets"},

    {"Risk ID": "R08",
     "Risk": "Class imbalance (73.3% pathogenic) biases threshold and metrics",
     "Category": "Evaluation",
     "Severity": "MEDIUM",
     "Probability": "CERTAIN",
     "Impact": "Accuracy misleading; default threshold suboptimal; precision-recall trade-off critical",
     "Mitigation": "Use PR-AUC alongside ROC-AUC; class weights; calibrated probabilities; sensitivity-optimized threshold"},

    {"Risk ID": "R09",
     "Risk": "Differential missingness by label creates spurious predictive signal",
     "Category": "Bias",
     "Severity": "MEDIUM",
     "Probability": "HIGH (KANSER: 57% differential missingness)",
     "Impact": "Model may learn 'missing = pathogenic' rather than genuine biological signal",
     "Mitigation": "Add missingness indicators explicitly; analyze SHAP values for missingness features; report this risk"},

    {"Risk ID": "R10",
     "Risk": "Overfitting to competition training data if hyperparameter search is too aggressive",
     "Category": "Overfitting",
     "Severity": "MEDIUM",
     "Probability": "MEDIUM",
     "Impact": "Poor generalization on competition test set",
     "Mitigation": "Strict nested CV; conservative hyperparameter ranges; early stopping; ensemble for stability"},
]

risk_df = pd.DataFrame(risks)
risk_df.to_csv(REPORT_DIR / "risk_register.csv", index=False)
print(f"  Risk register: {len(risk_df)} risks documented")

# ============================================================
# MARKDOWN REPORTS
# ============================================================
print("\n[REPORTS] Generating markdown reports...")

def write_md(filename, content):
    with open(REPORT_DIR / filename, 'w', encoding='utf-8') as f:
        f.write(content)

# 1. Specialized Model Catalogue
write_md("specialized_model_catalogue.md", f"""# Specialized Model Catalogue for Missense Variant Pathogenicity Prediction

## Overview
This catalogue covers {len(models)} tools and models relevant to missense variant pathogenicity classification.
The tools are organized by category and evaluated for their applicability to the TEKNOFEST competition dataset.

## Important Distinction

**Pathogenicity prediction** and **deleteriousness prediction** are related but different:
- **Deleteriousness**: Whether a variant damages protein function (e.g., CADD, SIFT)
- **Pathogenicity**: Whether a variant causes disease in humans (e.g., REVEL, ClinPred)

Deleteriousness is necessary but not sufficient for pathogenicity. A variant can be deleterious
but benign if the gene is not associated with disease, or if compensatory mechanisms exist.

## Category 1: Classical Score-Based Predictors (2003-2013)

These tools use sequence conservation, protein structure, and biochemical properties to predict
whether a missense variant affects protein function.

| Model | Year | Target | Missense-Specific | ClinVar-Trained | Likely in Dataset |
|-------|------|--------|-------------------|-----------------|-------------------|
| SIFT | 2003 | Functional impact | Yes | No | Possibly (AL feature) |
| PolyPhen-2 | 2010 | Functional impact | Yes | HumVar uses disease data | Possibly (AL feature) |
| MutationTaster | 2010 | Pathogenicity | No | Yes (uses ClinVar/HGMD) | Possibly |
| MutationAssessor | 2011 | Functional impact | Yes | No | Possibly |
| PROVEAN | 2012 | Deleteriousness | No | No | Possibly |
| FATHMM | 2013 | Functional impact | Yes | Disease-weighted | Possibly |

**Key insight**: SIFT and PolyPhen-2 are foundational. They are almost certainly embedded as AL features
in the dataset given they are standard dbNSFP annotations.

## Category 2: Ensemble Meta-Predictors (2015-2019)

These tools combine multiple individual predictor scores to produce a more accurate pathogenicity assessment.
**This is conceptually identical to what our competition model does.**

| Model | Year | Target | Key Feature | ClinVar-Trained | Likely in Dataset |
|-------|------|--------|-------------|-----------------|-------------------|
| REVEL | 2016 | Pathogenicity | 13-feature ensemble | Yes (trained on disease mutations) | Very likely (EK_4 or EK_6) |
| BayesDel | 2019 | Pathogenicity | Bayesian ensemble | Yes | Possibly (EK feature) |
| MetaSVM/MetaLR | 2015 | Pathogenicity | SVM/LR meta-learner | Yes | Possibly |
| ClinPred | 2018 | Clinical pathogenicity | Clinical-focused | Directly on ClinVar | Possibly (HIGHEST circularity risk) |
| M-CAP | 2016 | Pathogenicity | Gradient boosting | Yes | Possibly |

**Critical warning**: If our competition labels come from ClinVar, and REVEL/BayesDel/ClinPred scores
are embedded as features, we have a **circularity problem**. The model is partly being asked to predict
labels using features that were trained on those same labels.

## Category 3: Deleteriousness/Annotation Models (2014-2016)

These models score general deleteriousness, not disease-specific pathogenicity.

| Model | Year | Target | ClinVar-Trained | Likely in Dataset |
|-------|------|--------|-----------------|-------------------|
| CADD | 2014/2019 | Deleteriousness | No (simulated vs fixed) | Very likely (EK_1 or EK_2) |
| DANN | 2015 | Deleteriousness | No | Possibly |
| Eigen | 2016 | Deleteriousness | No (unsupervised) | Possibly |

**Key insight**: CADD is the most widely used deleteriousness score. Its training does NOT use ClinVar labels,
making it low-circularity risk. EK_1 or EK_2 almost certainly encode CADD scores.

## Category 4: Missense-Specific ML Models (2017-2022)

| Model | Year | Target | Approach | ClinVar-Trained | Likely in Dataset |
|-------|------|--------|----------|-----------------|-------------------|
| VARITY | 2021 | Pathogenicity | Multi-source integration | Partially | Possibly |
| MVP/gMVP | 2021 | Pathogenicity | ML on conservation+structure | Partially | Possibly |
| MutPred2 | 2020 | Pathogenicity + mechanisms | Neural network | Yes | Possibly |

## Category 5: Protein Language Model / Deep Learning (2018-2023)

| Model | Year | Target | Approach | ClinVar-Trained | Likely in Dataset |
|-------|------|--------|----------|-----------------|-------------------|
| AlphaMissense | 2023 | Pathogenicity | AlphaFold-derived | No (primate data) | Possibly (EK_6?) |
| EVE | 2021 | Pathogenicity | Variational autoencoder | No (evolutionary) | Possibly |
| ESM1b/ESM2 | 2021-2023 | Functional impact | Protein LM | No (unsupervised) | Unlikely |
| PrimateAI-3D | 2023 | Pathogenicity | 3D CNN | No (primate variation) | Possibly |

**Key insight**: AlphaMissense, EVE, and ESM-based models are trained WITHOUT ClinVar labels,
making them the safest from a circularity standpoint. If present in the dataset, these are the
most trustworthy features.

## Category 6: Conservation Scores (Features, Not Predictors)

| Score | Likely EK Feature | Range Match | Confidence |
|-------|-------------------|-------------|------------|
| phyloP | **EK_7** | [-3.32, 10.0] matches phyloP exactly | VERY HIGH |
| GERP++ RS | **EK_9** | [-20, 11.93] matches GERP++ exactly | VERY HIGH |
| phastCons | Possibly EK_5 | [0, 1] matches phastCons range | MEDIUM |

**These are the safest features** — purely evolutionary, no label circularity whatsoever.
""")

# 2. Applicability
write_md("applicability_to_competition_data.md", f"""# Applicability to Competition Data

## Core Constraint
All feature names are **anonymized**. We have no genomic coordinates, no gene names (except panel identity),
and no way to look up external databases. The Variant_ID format (VAR_XXXXXX) does not correspond to
standard databases like ClinVar, dbSNP, or gnomAD.

## Can We Use These Models Directly?

**Answer: NO.**

None of the specialized models can be directly applied because:
1. We lack genomic coordinates (chr:pos:ref:alt)
2. We lack protein identifiers (UniProt ID, Ensembl transcript)
3. We cannot run external prediction tools
4. Competition rules likely prohibit external database lookups

## Are These Models Already Embedded?

**Answer: ALMOST CERTAINLY YES.**

The dataset structure strongly suggests it was generated from **dbNSFP** or a similar variant annotation
database. The evidence:

1. **AL features (triplet pattern)**: Population-specific allele frequencies from gnomAD and AllofUs
2. **EK features**: Pre-computed predictor scores matching known ranges:
   - EK_7 = phyloP (conservation, no circularity)
   - EK_9 = GERP++ (conservation, no circularity)
   - EK_4/EK_6 = REVEL, BayesDel, or AlphaMissense (possible circularity)
   - EK_1/EK_2 = CADD phred/raw (low circularity)
3. **CAT_1/CAT_2**: gnomAD and AllofUs population labels
4. **AA_1/AA_2**: Reference and alternate amino acids

## Applicability Matrix

{applicability_df.to_markdown(index=False)}

## What This Means for Our Strategy

1. **We ARE building a meta-predictor** — our model is conceptually identical to REVEL/BayesDel
2. The competition essentially asks: "Given these annotation features, classify pathogenicity"
3. Our model's advantage must come from:
   - Better handling of missingness
   - Better calibration
   - Better panel-specific adaptation
   - Better feature engineering (interaction terms, missingness indicators)
   - Ensemble diversity

## Key Questions for Competition Rules

The following should be clarified with organizers if possible:
1. Are we allowed to use all provided features, including EK features?
2. Is there a separate test set, or is evaluation on the panel datasets?
3. Is external data usage explicitly prohibited?
""")

# 3. Feature Overlap
write_md("feature_overlap_audit.md", f"""# Feature Overlap Audit

## Summary

The competition dataset features appear to be sourced from **dbNSFP** (a widely-used functional prediction
and annotation database for human nonsynonymous SNVs) combined with population allele frequency data
from gnomAD and AllofUs.

## Feature Group Identification

{feature_overlap_df[['Feature Group', 'Inferred Identity', 'Confidence', 'Leakage Risk']].to_markdown(index=False)}

## Feature Architecture

The dataset appears to follow this structure:

```
[Variant_ID] [AA_1] [AA_2]  ← Variant identity
[AL_1..AL_26]                ← gnomAD exome population allele frequencies
[CAT_1]                      ← gnomAD popmax population label
[AL_27..AL_38]               ← Rare/small population frequencies (high missing)
[AL_39..AL_95]               ← gnomAD genome + other database frequencies (triplet pattern)
[CAT_2]                      ← AllofUs popmax population label
[AL_96..AL_98]               ← Additional frequency annotations
[CAT_3, CAT_4]               ← Reference genotypes (homozygous ref alleles)
[EK_1..EK_3]                 ← Deleteriousness/pathogenicity scores (CADD-like)
[AL_99..AL_185]              ← AllofUs + other database frequencies (triplet pattern)
[CAT_5]                      ← Additional genotype field
[AL_186..AL_222]             ← Predictor scores or continued AF data
[CAT_6]                      ← Genomic region flag (lcr, segdup)
[AL_223..AL_334]             ← Additional predictor scores and annotations
[EK_4..EK_9]                 ← Meta-predictor scores + conservation scores
[AA_1, AA_2]                 ← Amino acid change
[Label]                      ← Target (0=Benign, 1=Pathogenic)
```

## Circularity Risk Assessment

| Circularity Level | Features | Risk |
|-------------------|----------|------|
| NO circularity | EK_7 (phyloP), EK_9 (GERP++), AL allele frequencies, CAT, AA | SAFE |
| LOW circularity | EK_1 (CADD phred?), EK_2 (CADD raw?), SIFT, PolyPhen (if embedded) | PROBABLY SAFE |
| MEDIUM circularity | EK_3, some AL predictor scores | MONITOR |
| HIGH circularity | EK_4 (REVEL?), EK_5, EK_6 (REVEL/BayesDel?), ClinPred (if embedded) | MUST REPORT |

## Recommendation

1. **Use ALL features** for best competition performance — they are provided, so fair to use
2. **Train a model WITHOUT EK_4/EK_5/EK_6** to show non-circular performance
3. **Document circularity risk** in the competition report
4. **Emphasize phyloP, GERP++, allele frequencies** as trustworthy features in the report
""")

# 4. Strategy Comparison
write_md("strategy_comparison.md", f"""# Strategy Comparison

## Strategy Overview

{strategy_df.to_markdown(index=False)}

## Detailed Strategy Analysis

### Strategy 1: Generic Tabular ML (MUST DO)

**What**: XGBoost/LightGBM/CatBoost ensemble on all provided features.

**Why it works**: The dataset is tabular with mixed types, high missingness, and moderate size —
exactly the domain where gradient boosted trees dominate. No need to reinvent the wheel.

**Expected ROC-AUC**: 0.92-0.97 (based on comparable missense classification benchmarks)

**Verdict**: ✓ BASELINE — do this first, then improve.

### Strategy 2: External Predictor Scores (AVOID)

**What**: Look up AlphaMissense/REVEL/CADD scores externally.

**Why it fails**:
1. We CANNOT identify variants (coordinates are hidden)
2. Even if we could, this likely violates competition rules
3. These scores are probably ALREADY in the dataset as EK features
4. Adding duplicate features provides zero marginal value

**Verdict**: ✗ NOT FEASIBLE — do not attempt.

### Strategy 3: Meta-Classifier Over Embedded Predictors (STRONG)

**What**: Recognize that EK features ARE existing predictor scores, and build a calibrated
meta-learner that combines them intelligently.

**Why it works**: This is literally what REVEL and BayesDel did — and they became state-of-the-art
by doing it well. We can do the same, with the advantage of panel-specific calibration.

**Key insight for report**: Our model is a **competition-specific meta-predictor** that combines
population frequencies, conservation scores, and existing predictor outputs — the same architecture
as published meta-predictors, but trained specifically for this competition's variant set.

**Verdict**: ✓ STRONG — emphasize this framing in the report.

### Strategy 4: Custom Competition Architecture (PRIMARY)

**What**: Full architecture with:
- Multiple GBDT base learners (XGBoost, LightGBM, CatBoost)
- Feature-group-aware design
- Missingness indicators
- Panel-specific calibration
- Sensitivity-optimized thresholding
- SHAP explainability

**Why it works**: Addresses every identified risk. The architecture shows methodological rigor
without being overly complex.

**Verdict**: ✓ PRIMARY STRATEGY — this is what we build.

### Strategy 5: Panel-Specific Models (SUPPLEMENT ONLY)

**What**: Separate models for CFTR, PAH, KANSER.

**Why it's risky**:
- CFTR has only 111 samples — cannot train a reliable model
- PAH has 5:1 class imbalance — difficult
- Panel-specific models fragment the limited data

**Better approach**: Global model + panel-specific threshold calibration.

**Verdict**: ✓ SUPPLEMENT — use panel thresholds, not separate models.

## Final Strategy Recommendation

**Implement Strategy 4 (Custom Competition Architecture) with Strategy 3 framing.**

1. Build robust GBDT ensemble on all features
2. Frame as "competition-specific meta-predictor" in report
3. Add panel-specific calibration layer
4. Show models with and without EK features
5. Provide SHAP analysis for interpretability
6. Optimize for sensitivity (clinical relevance)
""")

# 5. Recommended Modeling Direction
write_md("recommended_modeling_direction.md", f"""# Recommended Modeling Direction

## TL;DR

Build a **gradient boosted tree ensemble** (LightGBM + XGBoost + CatBoost) on all provided features,
with panel-specific calibration, missingness-aware feature engineering, and sensitivity-optimized thresholding.
Frame it as a **competition-specific meta-predictor for missense variant pathogenicity**.

## Why NOT a Specialized Model?

1. **We cannot run specialized models** — no coordinates, no protein sequences available
2. **Specialized models are already embedded** — the EK features ARE their outputs
3. **Our dataset IS a meta-prediction dataset** — we're given predictor outputs as features
4. **GBDT is the right tool** for tabular data with missingness and mixed types

## Why This Is Still "Specialized"

Our model IS specialized for missense pathogenicity because:
1. It operates on missense-specific annotations (predictor scores, AA properties, conservation)
2. It incorporates population allele frequencies (BA1/BS1 criteria from ACMG guidelines)
3. It uses panel-specific calibration (gene-specific pathogenicity patterns)
4. It addresses clinical risk asymmetry (sensitivity > specificity)

## Phase 3 Implementation Plan

### Step 1: Baseline (Day 1)
- LightGBM with default parameters on MASTER
- 5-fold stratified CV
- ROC-AUC, PR-AUC, F1
- No feature engineering

### Step 2: Feature Engineering (Day 2)
- Missingness indicator features (binary: is_missing for each feature)
- Missingness profile features (total missing count per row, per feature group)
- AA_1/AA_2 encoding (label encoding, Grantham-distance-inspired if identifiable)
- CAT encoding (target encoding with regularization)
- Feature group interaction terms (conservation * AF)

### Step 3: Model Suite (Day 3)
- LightGBM (primary)
- XGBoost (diversity)
- CatBoost (handles categoricals natively)
- ElasticNet logistic regression (interpretable baseline)

### Step 4: Ensemble (Day 4)
- Stacking: meta-learner (logistic regression) over base model predictions
- Blending: weighted average of calibrated probabilities
- Feature selection: remove features with zero SHAP importance

### Step 5: Panel Calibration (Day 5)
- Per-panel threshold optimization
- Per-panel Platt scaling
- Per-panel performance evaluation
- Bootstrap confidence intervals

### Step 6: Clinical Optimization (Day 6)
- Sensitivity-constrained threshold (e.g., 95% sensitivity target)
- Asymmetric loss exploration
- Calibration curve analysis
- Brier score optimization

### Step 7: Explainability (Day 7)
- SHAP global importance
- SHAP local explanations for misclassified variants
- Feature group contribution analysis
- Panel-level SHAP comparison

### Step 8: Final Validation and Report (Day 8)
- Final model selection
- Nested CV for unbiased performance estimate
- EK-included vs EK-excluded comparison
- Competition report writing
""")

# 6. Proposed Architecture
write_md("proposed_specialized_architecture.md", f"""# Proposed Competition-Specific Architecture

## Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │            RAW FEATURES (353 cols)           │
                    │  AL_1..AL_334 | CAT_1..6 | EK_1..9 | AA_1,2│
                    └───────────────┬─────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │         FEATURE ENGINEERING LAYER              │
                    │                                                │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
                    │  │ Missing  │ │ Category │ │ AA Property  │  │
                    │  │ Indicators│ │ Encoding │ │ Encoding     │  │
                    │  └──────────┘ └──────────┘ └──────────────┘  │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
                    │  │ Group    │ │Feature   │ │ Interaction  │  │
                    │  │ Profiles │ │Selection │ │ Features     │  │
                    │  └──────────┘ └──────────┘ └──────────────┘  │
                    └───────────────┬───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │           BASE LEARNER LAYER                  │
                    │                                                │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
                    │  │ LightGBM │ │ XGBoost  │ │  CatBoost    │  │
                    │  │ (5-fold) │ │ (5-fold) │ │  (5-fold)    │  │
                    │  └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
                    └───────┼────────────┼──────────────┼───────────┘
                            │            │              │
                    ┌───────▼────────────▼──────────────▼───────────┐
                    │         STACKING / BLENDING LAYER              │
                    │                                                │
                    │  Meta-learner: Logistic Regression / Ridge     │
                    │  Input: base model OOF predictions             │
                    │  Output: blended probability                   │
                    └───────────────┬───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │         CALIBRATION LAYER                     │
                    │                                                │
                    │  ┌──────────────────────────────────────────┐ │
                    │  │ Global: Platt scaling / Isotonic          │ │
                    │  │ Panel-specific: Per-panel Platt scaling   │ │
                    │  └──────────────────────────────────────────┘ │
                    └───────────────┬───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │     CLINICAL THRESHOLDING LAYER               │
                    │                                                │
                    │  ┌──────────────────────────────────────────┐ │
                    │  │ Global threshold (PR-AUC optimized)      │ │
                    │  │ Panel-specific thresholds                 │ │
                    │  │ Sensitivity-constrained (≥95% target)    │ │
                    │  └──────────────────────────────────────────┘ │
                    └───────────────┬───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────────┐
                    │     EXPLAINABILITY LAYER                      │
                    │                                                │
                    │  SHAP values (global + local)                  │
                    │  Feature group contributions                   │
                    │  Panel-level importance differences             │
                    └───────────────────────────────────────────────┘
```

## Layer Details

### 1. Feature Engineering Layer

#### Missingness Indicators
```python
for col in feature_cols:
    df[f'{{col}}_missing'] = df[col].isnull().astype(int)
df['total_missing'] = df[feature_cols].isnull().sum(axis=1)
df['af_group_missing'] = df[af_cols].isnull().sum(axis=1)
df['predictor_group_missing'] = df[predictor_cols].isnull().sum(axis=1)
```

#### Category Encoding
- CAT_1, CAT_2: Target encoding with leave-one-out regularization
- CAT_3, CAT_4, CAT_5: One-hot encoding (few categories)
- CAT_6: Binary (has_region_flag or not)
- AA_1, AA_2: One-hot or ordinal by amino acid property group

#### Amino Acid Property Features
- Charge change (positive→negative, etc.)
- Size change (small→large, etc.)
- Polarity change
- Hydrophobicity change (Kyte-Doolittle scale)

### 2. Base Learner Layer

#### LightGBM Configuration
```python
params_lgbm = {{
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'is_unbalance': True,
    'verbose': -1,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
}}
```

#### XGBoost Configuration
```python
params_xgb = {{
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'scale_pos_weight': benign_count / pathogenic_count,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
}}
```

#### CatBoost Configuration
```python
params_cat = {{
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'auto_class_weights': 'Balanced',
    'eval_metric': 'AUC',
    'early_stopping_rounds': 50,
    'cat_features': cat_feature_indices,
}}
```

### 3. Stacking Layer
```python
meta_model = LogisticRegressionCV(
    cv=5, scoring='roc_auc', max_iter=1000, class_weight='balanced'
)
# Fit on OOF predictions from base models
X_meta = np.column_stack([lgbm_oof, xgb_oof, cat_oof])
meta_model.fit(X_meta, y_train)
```

### 4. Calibration Layer
```python
from sklearn.calibration import CalibratedClassifierCV
calibrator = CalibratedClassifierCV(base_estimator=None, method='isotonic', cv=5)
# Per-panel calibration
for panel in ['CFTR', 'PAH', 'KANSER']:
    panel_data = ...
    panel_calibrator = CalibratedClassifierCV(method='sigmoid')
    panel_calibrator.fit(panel_probs, panel_labels)
```

### 5. Thresholding Layer
```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
# Find threshold for ≥95% sensitivity
target_sensitivity = 0.95
valid_thresholds = thresholds[recall[:-1] >= target_sensitivity]
optimal_threshold = valid_thresholds[-1]  # highest threshold meeting sensitivity target
```

### 6. B-Plan Architecture

If the full ensemble fails or overfits:

1. **Simplify**: Single LightGBM model with conservative hyperparameters
2. **Feature reduction**: Top 50-100 features by SHAP importance
3. **Remove high-missing features**: Only features with <50% missingness
4. **Remove EK features**: If circularity is suspected
5. **Global model only**: Skip panel-specific calibration for small panels
6. **Ridge logistic regression**: If GBDT overfits on small panels
""")

# 7. Executive Summary
write_md("phase_02_executive_summary.md", f"""# Phase 02: Executive Summary

## 1. Are there specialized models for missense pathogenicity prediction?

**YES — extensively.** At least {len(models)} models/tools exist, spanning classical score-based predictors
(SIFT, PolyPhen-2), ensemble meta-predictors (REVEL, BayesDel, ClinPred), deleteriousness
annotators (CADD), missense-specific ML models (VARITY, MutPred2), protein language models
(AlphaMissense, EVE, ESM), and conservation scores (phyloP, GERP++).

## 2. Which models are most relevant to this competition?

The most relevant are:
1. **REVEL** — gold-standard meta-predictor, likely embedded as EK_4 or EK_6
2. **CADD** — most widely used deleteriousness score, likely embedded as EK_1/EK_2
3. **AlphaMissense** — state-of-the-art (2023), possibly embedded as EK_6
4. **phyloP** — almost certainly EK_7 (range match is exact)
5. **GERP++** — almost certainly EK_9 (range match is exact)
6. **BayesDel** — another meta-predictor, possibly embedded
7. **gnomAD allele frequencies** — certainly the AL features (confirmed by CAT_1 labels)

## 3. Can we directly use them?

**NO.** We cannot run these tools because:
- Feature names are anonymized
- Genomic coordinates are not provided
- Variant_IDs don't map to standard databases
- Competition rules likely prohibit external lookups

## 4. Which ones are likely already represented inside the provided features?

**Most of them, with high confidence:**

| Feature | Likely Identity | Confidence |
|---------|----------------|------------|
| AL_1..AL_95 | gnomAD population allele frequencies | VERY HIGH |
| AL_99..AL_185 | AllofUs + other database frequencies | HIGH |
| CAT_1 | gnomAD popmax population | CERTAIN |
| CAT_2 | AllofUs population | CERTAIN |
| CAT_3/4/5 | Reference genotypes | HIGH |
| CAT_6 | Genomic region flags | CERTAIN |
| EK_1 | CADD phred (compressed?) or constraint score | MEDIUM |
| EK_2 | CADD raw or FATHMM | MEDIUM |
| EK_4 | REVEL or BayesDel (0-1 meta-predictor) | MEDIUM-HIGH |
| EK_6 | REVEL, BayesDel, or AlphaMissense | MEDIUM |
| EK_7 | phyloP conservation | VERY HIGH |
| EK_9 | GERP++ RS score | VERY HIGH |
| AA_1/AA_2 | Amino acid change | CERTAIN |

## 5. What is the safest competition strategy?

**Strategy 4: Custom Competition Architecture**
- GBDT ensemble (LightGBM + XGBoost + CatBoost) on ALL provided features
- Feature engineering: missingness indicators, AA encoding, category encoding
- Panel-specific calibration
- Sensitivity-optimized thresholding
- SHAP explainability

This is safe because:
- Uses only provided data
- Does not require external lookups
- Handles missingness natively
- Produces interpretable results

## 6. What is the strongest performance strategy?

Same as #5, with additions:
- Include ALL features including EK (these are the strongest predictors)
- Stacking ensemble for maximum accuracy
- Post-hoc calibration with isotonic regression
- Per-panel threshold optimization

**Expected ROC-AUC**: 0.93-0.97 on MASTER (based on comparable benchmarks)

## 7. What is the biggest leakage/circularity risk?

**EK_4 and EK_6 are likely ClinVar-trained meta-predictor scores (REVEL, BayesDel, or ClinPred).**

If the competition labels come from ClinVar (which is highly probable for a binary pathogenic/benign
classification), then these features create a circularity problem:
- The features were trained to predict the same labels we're given
- Using them inflates apparent performance
- The model may not generalize to truly novel variants

**Mitigation**: Report performance WITH and WITHOUT these features.

## 8. What should Phase 3 implement?

1. LightGBM baseline on all features → immediate performance benchmark
2. Feature engineering (missingness indicators, AA encoding, CAT encoding)
3. XGBoost and CatBoost base models
4. Stacking ensemble
5. Panel-specific calibration
6. EK-excluded model for circularity analysis
7. SHAP analysis
8. Clinical thresholding (sensitivity ≥ 95%)
9. Final report generation

## 9. What should be written in the project report?

1. **Literature review**: Describe REVEL, CADD, AlphaMissense, and their roles in variant classification
2. **Dataset understanding**: Show that features include allele frequencies, conservation scores,
   and meta-predictor outputs
3. **Circularity analysis**: Acknowledge and quantify the impact of ClinVar-trained features
4. **Model architecture**: Present as "competition-specific meta-predictor" — same conceptual
   framework as REVEL/BayesDel, applied to this specific dataset
5. **Panel analysis**: Show per-panel performance and challenges
6. **Clinical framing**: Discuss sensitivity vs specificity trade-off in clinical genetics
7. **ACMG connection**: Frame allele frequency features as computational BS1/BA1 criteria

## 10. What is the B-plan if specialized models cannot be used?

Since specialized models are already EMBEDDED as features, the B-plan is about **feature robustness**:

1. **If EK features are circulant**: Drop EK_4/EK_5/EK_6, use only AF + conservation + AA features
2. **If panel models fail**: Use global model with panel-specific thresholds
3. **If CFTR panel is too small**: Use leave-one-out CV or bootstrap; accept higher variance
4. **If missingness breaks models**: Reduce to features with <30% missing
5. **If ensemble is too complex**: Fall back to single LightGBM with early stopping
""")

print("  All markdown reports generated.")

# ============================================================
# WORD DOCUMENT
# ============================================================
print("\n[WORD] Generating Word document...")

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)

def add_table_from_df(doc, df, max_rows=None):
    if max_rows:
        df = df.head(max_rows)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Light Grid Accent 1'
    for i, col in enumerate(df.columns):
        table.rows[0].cells[i].text = str(col)
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            text = str(val)
            if len(text) > 200:
                text = text[:197] + "..."
            cells[i].text = text

# Title
doc.add_heading('TEKNOFEST Healthcare AI\nPhase 02: Specialized Model Research', 0)
doc.add_paragraph('Generated: 2026-05-24')
doc.add_paragraph('Missense Variant Pathogenicity Classification — Model & Strategy Research')
doc.add_page_break()

# TOC
doc.add_heading('Table of Contents', 1)
toc_items = [
    '1. Project Context from Phase 1',
    '2. Specialized Model Catalogue',
    '3. Applicability to Competition Data',
    '4. Feature Overlap Audit',
    '5. Strategy Comparison',
    '6. Risk Register',
    '7. Proposed Architecture',
    '8. Executive Summary and Recommendations',
]
for item in toc_items:
    doc.add_paragraph(item)
doc.add_page_break()

# 1. Context
doc.add_heading('1. Project Context from Phase 1', 1)
doc.add_paragraph('Phase 1 established that the dataset contains 4 CSV files (MASTER: 2931 variants, CFTR: 111, PAH: 372, KANSER: 388) with 353 anonymized columns.')
doc.add_paragraph('Key findings:')
doc.add_paragraph('- 334 AL features (allele frequencies in [0,1] with triplet pattern)', style='List Bullet')
doc.add_paragraph('- 6 CAT features (gnomAD/AllofUs population labels, genotypes, region flags)', style='List Bullet')
doc.add_paragraph('- 9 EK features (pre-computed scores: conservation, deleteriousness, meta-predictors)', style='List Bullet')
doc.add_paragraph('- 2 AA features (reference and alternate amino acids)', style='List Bullet')
doc.add_paragraph('- Binary target: 0=Benign, 1=Pathogenic (73.3% pathogenic in MASTER)', style='List Bullet')
doc.add_paragraph('- 54.9% mean row-level missingness', style='List Bullet')
doc.add_paragraph('- Feature names are fully anonymized', style='List Bullet')
doc.add_page_break()

# 2. Model Catalogue
doc.add_heading('2. Specialized Model Catalogue', 1)

doc.add_heading('2.1 Classical Predictors', 2)
doc.add_paragraph('SIFT (2003), PolyPhen-2 (2010), MutationTaster (2010), MutationAssessor (2011), PROVEAN (2012), FATHMM (2013)')
doc.add_paragraph('These tools predict functional impact using sequence conservation and protein structure. They are likely embedded as AL features in the dataset.')

doc.add_heading('2.2 Ensemble Meta-Predictors', 2)
doc.add_paragraph('REVEL (2016), BayesDel (2019), MetaSVM/MetaLR (2015), ClinPred (2018), M-CAP (2016)')
doc.add_paragraph('These combine multiple predictor outputs — conceptually identical to our competition model. CRITICAL: Most are trained on ClinVar labels, creating circularity risk.')

doc.add_heading('2.3 Deleteriousness Models', 2)
doc.add_paragraph('CADD (2014/2019), DANN (2015), Eigen (2016)')
doc.add_paragraph('CADD is the most important. Trained on simulated vs fixed variants (NOT ClinVar), making it low-circularity. Likely embedded as EK_1/EK_2.')

doc.add_heading('2.4 Deep Learning / Protein LM Models', 2)
doc.add_paragraph('AlphaMissense (2023), EVE (2021), ESM1b/ESM2 (2021-2023), PrimateAI-3D (2023)')
doc.add_paragraph('State-of-the-art approaches. AlphaMissense in particular is trained WITHOUT ClinVar labels, making it very safe from circularity. Possibly embedded as EK_6.')

doc.add_heading('2.5 Conservation Scores', 2)
doc.add_paragraph('phyloP (almost certainly EK_7), GERP++ (almost certainly EK_9), phastCons (possibly EK_5)')
doc.add_paragraph('These are the SAFEST features — purely evolutionary, zero circularity risk.')

catalogue_display = model_df[['Model', 'Year', 'Category', 'Prediction Target', 'Missense Specific', 'Leakage Risk', 'Recommendation']].copy()
doc.add_heading('Summary Table', 2)
add_table_from_df(doc, catalogue_display)
doc.add_page_break()

# 3. Applicability
doc.add_heading('3. Applicability to Competition Data', 1)
doc.add_paragraph('Core constraint: All feature names are anonymized. No genomic coordinates, no gene names, no external database access.')
doc.add_paragraph('')
doc.add_paragraph('Can we use specialized models directly? NO.')
doc.add_paragraph('Are they already embedded as features? ALMOST CERTAINLY YES.')
doc.add_paragraph('')
add_table_from_df(doc, applicability_df)
doc.add_page_break()

# 4. Feature Overlap
doc.add_heading('4. Feature Overlap Audit', 1)
doc.add_paragraph('The dataset features are consistent with dbNSFP annotations plus population allele frequencies from gnomAD and AllofUs.')
doc.add_paragraph('')

fo_display = feature_overlap_df[['Feature Group', 'Inferred Identity', 'Confidence', 'Leakage Risk']]
add_table_from_df(doc, fo_display)

doc.add_heading('EK Feature Identification', 2)
ek_table = pd.DataFrame([
    {"EK Feature": "EK_7", "Likely Identity": "phyloP", "Confidence": "VERY HIGH", "Risk": "NONE"},
    {"EK Feature": "EK_9", "Likely Identity": "GERP++ RS", "Confidence": "VERY HIGH", "Risk": "NONE"},
    {"EK Feature": "EK_1", "Likely Identity": "CADD phred (compressed)", "Confidence": "MEDIUM", "Risk": "LOW"},
    {"EK Feature": "EK_2", "Likely Identity": "CADD raw / FATHMM", "Confidence": "MEDIUM", "Risk": "LOW"},
    {"EK Feature": "EK_4", "Likely Identity": "REVEL / BayesDel", "Confidence": "MEDIUM-HIGH", "Risk": "HIGH"},
    {"EK Feature": "EK_6", "Likely Identity": "REVEL / AlphaMissense", "Confidence": "MEDIUM", "Risk": "HIGH if REVEL, LOW if AlphaMissense"},
])
add_table_from_df(doc, ek_table)
doc.add_page_break()

# 5. Strategy
doc.add_heading('5. Strategy Comparison', 1)
strat_display = strategy_df[['Strategy', 'Performance Potential', 'Implementation Difficulty', 'Leakage Risk', 'Final Recommendation']]
add_table_from_df(doc, strat_display)
doc.add_paragraph('')
doc.add_paragraph('FINAL RECOMMENDATION: Strategy 4 (Custom Competition Architecture) with Strategy 3 framing.')
doc.add_paragraph('Build a GBDT ensemble (LightGBM + XGBoost + CatBoost) as a competition-specific meta-predictor.')
doc.add_page_break()

# 6. Risk Register
doc.add_heading('6. Risk Register', 1)
risk_display = risk_df[['Risk ID', 'Risk', 'Severity', 'Probability', 'Mitigation']]
add_table_from_df(doc, risk_display)
doc.add_page_break()

# 7. Architecture
doc.add_heading('7. Proposed Architecture', 1)
doc.add_paragraph('The proposed architecture has 6 layers:')
doc.add_paragraph('1. Feature Engineering: missingness indicators, AA encoding, category encoding', style='List Bullet')
doc.add_paragraph('2. Base Learners: LightGBM + XGBoost + CatBoost (5-fold CV each)', style='List Bullet')
doc.add_paragraph('3. Stacking: Logistic regression meta-learner over OOF predictions', style='List Bullet')
doc.add_paragraph('4. Calibration: Platt scaling + panel-specific calibration', style='List Bullet')
doc.add_paragraph('5. Thresholding: Sensitivity-constrained (>=95% target)', style='List Bullet')
doc.add_paragraph('6. Explainability: SHAP values (global + local)', style='List Bullet')
doc.add_page_break()

# 8. Executive Summary
doc.add_heading('8. Executive Summary', 1)

doc.add_heading('Key Answers', 2)
answers = [
    ("Specialized models exist?", "YES — 25+ models for missense pathogenicity prediction"),
    ("Can we use them?", "NO — cannot run them; but they are EMBEDDED in our features"),
    ("Safest strategy?", "GBDT ensemble on all provided features with panel calibration"),
    ("Strongest strategy?", "Same + stacking + sensitivity-optimized thresholding"),
    ("Biggest risk?", "EK_4/EK_6 are ClinVar-trained meta-predictors (circularity)"),
]
for q, a in answers:
    doc.add_paragraph(f'{q} — {a}')

doc.add_heading('Phase 3 Implementation Plan', 2)
steps = [
    'Step 1: LightGBM baseline on all features',
    'Step 2: Feature engineering (missingness, AA, CAT encoding)',
    'Step 3: XGBoost + CatBoost base models',
    'Step 4: Stacking ensemble',
    'Step 5: Panel-specific calibration',
    'Step 6: EK-excluded model (circularity analysis)',
    'Step 7: SHAP analysis',
    'Step 8: Clinical thresholding (sensitivity >= 95%)',
    'Step 9: Final report generation',
]
for step in steps:
    doc.add_paragraph(step, style='List Bullet')

doc.add_heading('Top 5 Risks', 2)
doc.add_paragraph('1. ClinVar circularity via EK_4/EK_6 meta-predictor scores')
doc.add_paragraph('2. CFTR panel too small (n=111) for reliable modeling')
doc.add_paragraph('3. Distribution shift between MASTER and panels')
doc.add_paragraph('4. Differential missingness creates spurious signal (KANSER: 57%)')
doc.add_paragraph('5. Feature anonymization prevents biological validation')

word_path = REPORT_DIR / "Phase_02_Model_Research_Report.docx"
doc.save(str(word_path))
print(f"  Word document saved: {word_path}")

# ============================================================
# JUPYTER NOTEBOOK
# ============================================================
print("\n[NOTEBOOK] Generating Jupyter Notebook...")

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

cells = []

cells.append(new_markdown_cell("""# TEKNOFEST Healthcare AI — Phase 02: Specialized Model Research
## Missense Variant Pathogenicity Classification

**Objective**: Research specialized models, evaluate applicability, and define competition strategy.

This notebook provides an interactive companion to the Phase 02 research report.
"""))

cells.append(new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
print("Libraries loaded.")
"""))

cells.append(new_markdown_cell("## A. Project Context Recap"))

cells.append(new_code_cell("""from pathlib import Path

DATA_DIR = Path("EĞİTİM (TRAIN) SETLERİ 2")
master = pd.read_csv(DATA_DIR / "YARISMA_TRAIN_MASTER.csv")

print(f"MASTER: {master.shape[0]} rows x {master.shape[1]} cols")
print(f"Label distribution: {dict(master['Label'].value_counts())}")
print(f"Positive ratio: {master['Label'].mean():.3f}")
print(f"Mean row missingness: {master.isnull().mean(axis=1).mean()*100:.1f}%")
"""))

cells.append(new_markdown_cell("""## B. Feature Pattern Analysis

The AL features follow a clear triplet pattern:
1. **Continuous score** [0, ~0.9]: allele frequency from a specific population
2. **Binary flag** {0, 1}: whether data exists for this population
3. **Secondary flag**: filter status or related annotation
"""))

cells.append(new_code_cell("""# Demonstrate the triplet pattern
al_cols = [c for c in master.columns if c.startswith("AL_")]

print("Feature structure demonstration (AL_39 to AL_50):")
print("-" * 80)
for i in range(39, 51):
    col = f"AL_{i}"
    if col in master.columns:
        s = master[col].dropna()
        nu = s.nunique()
        is_binary = nu <= 3
        pattern = "BINARY FLAG" if is_binary else f"SCORE [0, {s.max():.3f}]"
        print(f"  {col:>8} | n={len(s):>5} | unique={nu:>5} | {pattern}")

print()
print("Interpretation:")
print("  AL_39: Population membership flag (few values: 0, 1, or special)")
print("  AL_40: Allele frequency score (continuous, unique per variant)")
print("  AL_41: Data availability flag (binary: 0 or 1)")
print("  AL_42: Filter/quality flag (binary: 0 or 1)")
print("  AL_43: Next allele frequency score...")
print("  ... Pattern repeats for each population in database")
"""))

cells.append(new_markdown_cell("""## C. EK Feature Deep Dive (Pre-computed Scores)

The EK features are the most informative and the most risky.
They appear to be pre-computed from external models.
"""))

cells.append(new_code_cell("""ek_cols = [c for c in master.columns if c.startswith("EK_")]

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
for idx, col in enumerate(ek_cols):
    ax = axes[idx // 3][idx % 3]
    s = master[col].dropna()

    for label, color, name in [(0, '#2196F3', 'Benign'), (1, '#F44336', 'Pathogenic')]:
        subset = master[master['Label'] == label][col].dropna()
        if len(subset) > 10:
            subset.plot(kind='kde', ax=ax, color=color, label=name, alpha=0.7)

    ax.set_title(f"{col}\\nrange=[{s.min():.2f}, {s.max():.2f}], miss={master[col].isnull().mean()*100:.1f}%")
    ax.legend(fontsize=8)

plt.suptitle("EK Feature Distributions by Class", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(new_code_cell("""# EK feature identification table
ek_info = []
for col in ek_cols:
    s = master[col].dropna()
    corr = master[['Label', col]].dropna().corr().iloc[0, 1]
    ek_info.append({
        "Feature": col,
        "Range": f"[{s.min():.3f}, {s.max():.3f}]",
        "Mean": f"{s.mean():.3f}",
        "Missing%": f"{master[col].isnull().mean()*100:.1f}%",
        "Corr w/Label": f"{corr:.4f}",
        "Likely Identity": {
            "EK_1": "CADD phred (compressed) or gene constraint",
            "EK_2": "CADD raw or FATHMM",
            "EK_3": "Unknown deleteriousness score",
            "EK_4": "REVEL, BayesDel, or ClinPred [0,1]",
            "EK_5": "phastCons or MetaLR/MetaSVM",
            "EK_6": "REVEL, BayesDel, or AlphaMissense [0,1]",
            "EK_7": "phyloP (EXACT range match)",
            "EK_8": "Grantham distance (normalized) or SiPhy",
            "EK_9": "GERP++ RS (EXACT range match)",
        }.get(col, "Unknown"),
        "Circularity Risk": {
            "EK_1": "LOW", "EK_2": "LOW", "EK_3": "LOW-MEDIUM",
            "EK_4": "HIGH", "EK_5": "LOW-MEDIUM", "EK_6": "MEDIUM-HIGH",
            "EK_7": "NONE", "EK_8": "LOW", "EK_9": "NONE",
        }.get(col, "Unknown")
    })

ek_df = pd.DataFrame(ek_info)
display(ek_df)
"""))

cells.append(new_markdown_cell("""## D. Specialized Model Catalogue

### Key Models for Missense Variant Pathogenicity

| Category | Models | Likely in Dataset | ClinVar-Trained | Circularity Risk |
|----------|--------|-------------------|-----------------|------------------|
| Classical | SIFT, PolyPhen-2 | Yes (AL features) | Partially | LOW-MEDIUM |
| Meta-predictors | REVEL, BayesDel, ClinPred | Yes (EK_4/EK_6) | YES | HIGH |
| Deleteriousness | CADD, DANN, Eigen | Yes (EK_1/EK_2) | NO | LOW |
| Deep Learning | AlphaMissense, EVE | Possibly (EK_6?) | NO | LOW |
| Conservation | phyloP, GERP++ | Yes (EK_7, EK_9) | NO | NONE |

### Critical Insight
Our competition model IS a meta-predictor. The dataset provides pre-computed outputs from
multiple specialized models, and we build a classifier on top of them.
This is architecturally identical to REVEL and BayesDel.
"""))

cells.append(new_markdown_cell("""## E. Strategy Comparison"""))

cells.append(new_code_cell("""strategies = pd.read_csv("reports/phase_02_specialized_model_research/strategy_comparison.csv")
display(strategies[['Strategy', 'Performance Potential', 'Leakage Risk', 'Final Recommendation']])
"""))

cells.append(new_markdown_cell("""## F. Risk Register"""))

cells.append(new_code_cell("""risks = pd.read_csv("reports/phase_02_specialized_model_research/risk_register.csv")
display(risks[['Risk ID', 'Risk', 'Severity', 'Probability']])
"""))

cells.append(new_markdown_cell("""## G. Feature Overlap Visualization"""))

cells.append(new_code_cell("""# Feature group architecture visualization
feature_groups = {
    'Population AF\\n(gnomAD exome)': (0, 26, '#4CAF50'),
    'Rare pop AF': (27, 38, '#81C784'),
    'Population AF\\n(gnomAD genome\\n+ others)': (39, 95, '#66BB6A'),
    'Additional AF\\nmetadata': (96, 98, '#A5D6A7'),
    'Population AF\\n(AllofUs +\\nothers)': (99, 185, '#43A047'),
    'Predictor\\nscores /\\ntransition zone': (186, 222, '#FFA726'),
    'In-silico\\npredictor\\nscores': (223, 334, '#FF7043'),
}

fig, ax = plt.subplots(figsize=(16, 3))
for label, (start, end, color) in feature_groups.items():
    ax.barh(0, end - start + 1, left=start, height=0.6, color=color, edgecolor='white', linewidth=0.5)
    mid = start + (end - start) / 2
    ax.text(mid, 0, label, ha='center', va='center', fontsize=7, fontweight='bold')

ax.set_xlim(0, 340)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("AL Feature Index")
ax.set_title("Inferred Feature Architecture (AL_1 to AL_334)", fontsize=14, fontweight='bold')
ax.set_yticks([])
plt.tight_layout()
plt.show()

print("\\nEK Feature Layer (separate from AL):")
print("  EK_1-EK_3: Deleteriousness scores (CADD-like)")
print("  EK_4-EK_6: Meta-predictor scores (REVEL/BayesDel-like)")
print("  EK_7: phyloP conservation score")
print("  EK_8: Unknown (possibly Grantham or SiPhy)")
print("  EK_9: GERP++ conservation score")
"""))

cells.append(new_markdown_cell("""## H. Circularity Analysis

The key question: if labels come from ClinVar, and some features (EK_4, EK_6) are ClinVar-trained
meta-predictors, how much does performance inflate?
"""))

cells.append(new_code_cell("""# Demonstrate circularity risk by comparing EK vs non-EK feature importance
from scipy.stats import mannwhitneyu

path_idx = master['Label'] == 1
ben_idx = master['Label'] == 0

# Compare EK features vs best AL features
comparison = []
for col in ['EK_4', 'EK_6', 'EK_7', 'EK_9', 'AL_26', 'AL_287', 'AL_215', 'AL_73']:
    sp = master.loc[path_idx, col].dropna()
    sb = master.loc[ben_idx, col].dropna()
    if len(sp) > 5 and len(sb) > 5:
        u, p = mannwhitneyu(sp, sb, alternative='two-sided')
        r = 1 - (2 * u) / (len(sp) * len(sb))
        comparison.append({
            "Feature": col,
            "Effect Size |r|": round(abs(r), 4),
            "p-value": f"{p:.2e}",
            "Likely Type": "Meta-predictor" if col in ['EK_4', 'EK_6'] else
                          "Conservation" if col in ['EK_7', 'EK_9'] else "Allele Freq",
            "Circularity Risk": "HIGH" if col in ['EK_4', 'EK_6'] else
                               "NONE" if col in ['EK_7', 'EK_9'] else "LOW"
        })

comp_df = pd.DataFrame(comparison)
display(comp_df)

print("\\nKey observation:")
print("If EK_4/EK_6 are much more predictive than EK_7/EK_9 and AF features,")
print("this suggests the model may be relying on ClinVar-circular features.")
print("We MUST train a model WITHOUT EK_4/EK_6 to measure honest performance.")
"""))

cells.append(new_markdown_cell("""## I. Proposed Architecture

```
RAW FEATURES → Feature Engineering → [LightGBM, XGBoost, CatBoost] → Stacking → Calibration → Threshold → SHAP
```

### Phase 3 Implementation Plan:
1. LightGBM baseline on all features
2. Feature engineering (missingness indicators, AA encoding)
3. XGBoost + CatBoost models
4. Stacking ensemble
5. Panel-specific calibration
6. EK-excluded circularity analysis
7. SHAP explainability
8. Clinical thresholding (sensitivity ≥ 95%)
9. Final report

### B-Plan:
- If ensemble fails: single LightGBM
- If EK features circular: drop EK_4/EK_5/EK_6
- If CFTR too small: global model + CFTR threshold only
- If too much missingness: features with <50% missing only
"""))

cells.append(new_markdown_cell("""## J. Final Recommendation

**PRIMARY STRATEGY**: Custom GBDT ensemble (LightGBM + XGBoost + CatBoost) framed as a
competition-specific meta-predictor, with panel calibration and sensitivity-optimized thresholding.

**CRITICAL ACTION**: Train models WITH and WITHOUT EK_4/EK_6 to quantify circularity impact.

**REPORT FRAMING**: Present as a meta-predictor that combines population frequencies, conservation
scores, and existing predictor outputs — the same architecture as REVEL/BayesDel, but competition-specific.
"""))

nb.cells = cells

nb_path = REPORT_DIR / "Phase_02_Model_Research_Notebook.ipynb"
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"  Notebook saved: {nb_path}")

# ============================================================
# FINAL
# ============================================================
print("\n" + "=" * 70)
print("ALL PHASE 02 OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\nOutput directory: {REPORT_DIR}")
print("\nGenerated files:")
for f in sorted(REPORT_DIR.glob("*")):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
