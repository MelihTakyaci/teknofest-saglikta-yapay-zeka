#!/usr/bin/env python3
"""Generate a single, simple Word document for Pipeline B results."""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

BASE = Path("/Users/melihtakyaci/Documents/TeknofestSagliktaYapayZekâVerisi")
FIG = BASE / "reports" / "phase_03_pipeline_b" / "figures"
OUT = BASE / "reports" / "phase_03_pipeline_b"

doc = Document()

# page margins
for section in doc.sections:
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

style = doc.styles["Normal"]
style.font.size = Pt(11)
style.font.name = "Calibri"
style.paragraph_format.space_after = Pt(4)

# ── helper ──
def add_table(doc, headers, rows, col_widths=None):
    t = doc.add_table(rows=1, cols=len(headers), style="Light Shading Accent 1")
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
    for row_data in rows:
        row = t.add_row()
        for i, val in enumerate(row_data):
            row.cells[i].text = str(val)
            for p in row.cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    return t

def add_fig(doc, name, width=5.8):
    p = FIG / name
    if p.exists():
        doc.add_picture(str(p), width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ── title ──
title = doc.add_heading("Phase 3 — Pipeline B: Missingness-Aware GBDT", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph("TEKNOFEST Healthcare AI — Missense Variant Classification")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(12)
sub.runs[0].font.color.rgb = RGBColor(100, 100, 100)

# ── 1. model config ──
doc.add_heading("1. Model Configuration", level=1)
add_table(doc, ["Parameter", "Value"], [
    ["Algorithm", "LightGBM (GBDT)"],
    ["Learning Rate", "0.05"],
    ["Num Leaves", "63"],
    ["Early Stopping", "100 rounds"],
    ["Class Balance", "is_unbalance=True"],
    ["CV Strategy", "5-fold Stratified"],
    ["Total Features", "655"],
    ["Training Samples", "2931"],
])

# ── 2. preprocessing ──
doc.add_heading("2. Preprocessing Steps", level=1)
steps = [
    "Drop Variant_ID (leakage prevention)",
    "Drop 57 constant columns (zero information)",
    "Add 286 per-feature missingness indicators",
    "Add 7 group-level missingness counts + row-level totals",
    "One-hot encode CAT_3, CAT_4, CAT_5 (genotype fields)",
    "Label encode CAT_1, CAT_2 (population labels)",
    "Binary flag for CAT_6 (region annotation)",
    "One-hot encode AA_1, AA_2 (amino acids)",
]
for s in steps:
    doc.add_paragraph(s, style="List Bullet")

# ── 3. overall performance ──
doc.add_heading("3. Overall OOF Performance", level=1)
add_table(doc, ["Metric", "Value"], [
    ["ROC-AUC", "0.8461"],
    ["PR-AUC", "0.9249"],
    ["Log Loss", "0.4127"],
    ["Brier Score", "0.1295"],
    ["F1 @ 0.5", "0.8733"],
    ["Sensitivity @ 0.5", "0.8660"],
    ["Specificity @ 0.5", "0.6777"],
    ["MCC @ 0.5", "0.5361"],
    ["Balanced Accuracy @ 0.5", "0.7719"],
    ["Calibration Error (ECE)", "0.0771"],
])

doc.add_paragraph("")
add_fig(doc, "roc_pr_curves.png")

# ── 4. per-fold stability ──
doc.add_heading("4. Per-Fold Stability", level=1)
add_table(doc,
    ["Fold", "ROC-AUC", "PR-AUC", "F1", "Sens", "Spec", "MCC", "Iters"],
    [
        ["1", "0.8609", "0.9350", "0.8759", "0.8698", "0.6815", "0.5449", "90"],
        ["2", "0.8423", "0.9265", "0.8737", "0.8767", "0.6410", "0.5210", "101"],
        ["3", "0.8519", "0.9318", "0.8791", "0.8791", "0.6667", "0.5457", "59"],
        ["4", "0.8554", "0.9266", "0.8737", "0.8605", "0.6987", "0.5456", "83"],
        ["5", "0.8222", "0.9035", "0.8640", "0.8438", "0.7006", "0.5252", "52"],
        ["Mean", "0.8465", "0.9247", "0.8733", "0.8660", "0.6777", "0.5365", "77"],
        ["Std", "0.0152", "0.0124", "0.0056", "0.0143", "0.0234", "0.0123", "20"],
    ])

doc.add_paragraph("")
add_fig(doc, "fold_stability.png")

# ── 5. threshold optimisation ──
doc.add_heading("5. Threshold Optimisation", level=1)
add_table(doc,
    ["Strategy", "Threshold", "Sensitivity", "Specificity", "F1", "MCC"],
    [
        ["Default", "0.500", "0.8660", "0.6777", "0.8733", "0.5361"],
        ["Best F1", "0.334", "0.9563", "0.4949", "0.8937", "0.5387"],
        ["Sens ≥ 95%", "0.359", "0.9502", "0.5141", "0.8935", "0.5421"],
        ["Youden's J", "0.596", "0.8288", "0.7289", "0.8600", "0.5287"],
        ["Best MCC", "0.370", "0.9474", "0.5205", "0.8930", "0.5417"],
    ])

doc.add_paragraph("")
add_fig(doc, "threshold_sweep.png")
doc.add_paragraph("")
add_fig(doc, "confusion_matrices.png")

# ── 6. feature importance ──
doc.add_heading("6. Top 30 Features", level=1)
add_table(doc,
    ["#", "Feature", "Category", "Gain"],
    [
        ["1", "EK_7", "Conservation (phyloP)", "5335"],
        ["2", "AL_327", "Allele annotation", "1341"],
        ["3", "AL_16", "Allele annotation", "1270"],
        ["4", "EK_9", "Conservation (GERP++)", "1085"],
        ["5", "EK_2", "Extra (CADD raw)", "1050"],
        ["6", "EK_6", "Extra (meta-predictor)", "916"],
        ["7", "AL_26", "Allele annotation", "744"],
        ["8", "AL_11", "Allele annotation", "742"],
        ["9", "EK_8", "Extra knowledge", "718"],
        ["10", "AL_7", "Allele annotation", "645"],
        ["11", "EK_1", "Extra (CADD phred)", "637"],
        ["12", "EK_5", "Extra knowledge", "611"],
        ["13", "AL_9", "Allele annotation", "610"],
        ["14", "EK_3", "Extra knowledge", "603"],
        ["15", "EK_4", "Extra (meta-predictor)", "588"],
        ["16", "AL_8", "Allele annotation", "586"],
        ["17", "AL_14", "Allele annotation", "572"],
        ["18", "AL_10", "Allele annotation", "568"],
        ["19", "AL_12", "Allele annotation", "524"],
        ["20", "AA_1_R", "Amino acid", "519"],
        ["21", "AL_287", "Allele annotation", "496"],
        ["22", "AL_215", "Allele annotation", "458"],
        ["23", "AL_331", "Allele annotation", "421"],
        ["24", "AL_315", "Allele annotation", "419"],
        ["25", "AL_319", "Allele annotation", "375"],
        ["26", "AL_18", "Allele annotation", "337"],
        ["27", "AL_311", "Allele annotation", "326"],
        ["28", "AL_186", "Allele annotation", "325"],
        ["29", "AL_1", "Allele annotation", "316"],
        ["30", "AL_15", "Allele annotation", "304"],
    ])

doc.add_paragraph("")
add_fig(doc, "feature_importance_top30.png")

# ── 7. feature category summary ──
doc.add_heading("7. Feature Category Summary", level=1)
add_table(doc,
    ["Category", "N", "Mean Gain", "Total Gain", "Top Feature"],
    [
        ["Allele annotation", "277", "84.1", "23292", "AL_327"],
        ["Extra knowledge", "9", "1282.4", "11541", "EK_7"],
        ["Amino acid", "51", "21.1", "1078", "AA_1_R"],
        ["Categorical", "23", "15.7", "361", "CAT_1_enc"],
        ["Missingness (agg)", "2", "144.8", "290", "total_miss_count"],
        ["Missingness (group)", "7", "10.5", "74", "miss_grp_al_39_95"],
        ["Missingness (per-feat)", "286", "0.0", "5", "miss_EK_3"],
    ])

p = doc.add_paragraph()
p.add_run("\nMissingness features contribution: ").bold = True
p.add_run("1.0% of total importance gain (368 / 36,641). "
           "Per-feature indicators were mostly unused; group-level and aggregate counts were more useful.")

# ── 8. panel performance ──
doc.add_heading("8. Per-Panel Performance", level=1)
add_table(doc,
    ["Panel", "N", "Shared", "ROC-AUC", "PR-AUC", "F1@0.5", "Sens", "Spec", "Opt Thr"],
    [
        ["CFTR", "111", "77", "0.9402", "0.9857", "0.9371", "0.9111", "0.8571", "0.413"],
        ["PAH", "372", "255", "0.7720", "0.9336", "0.8893", "0.8806", "0.5000", "0.205"],
        ["KANSER", "388", "246", "0.9141", "0.9547", "0.9055", "0.9478", "0.6750", "0.466"],
    ])

# ── 9. calibration ──
doc.add_heading("9. Calibration", level=1)
add_fig(doc, "calibration_curve.png", width=4.5)

# ── 10. prediction distribution ──
doc.add_heading("10. Prediction Distribution", level=1)
add_fig(doc, "prediction_distribution.png")

# ── 11. next steps ──
doc.add_heading("11. Next Steps", level=1)
nexts = [
    "Pipeline C: Drop EK_4/5/6, re-train to measure circularity impact",
    "Hyperparameter tuning via Bayesian optimisation",
    "XGBoost and CatBoost comparison runs",
    "Per-panel threshold calibration (especially PAH)",
    "Stacking ensemble (Pipeline E)",
]
for n in nexts:
    doc.add_paragraph(n, style="List Bullet")

# ── save ──
out_path = OUT / "Pipeline_B_Results.docx"
doc.save(str(out_path))
print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
