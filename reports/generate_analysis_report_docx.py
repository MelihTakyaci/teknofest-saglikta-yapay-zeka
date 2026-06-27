"""
Teknofest 2026 - Veri Analizi Raporu (DOCX)
Generates a Turkish Word document summarizing the full data analysis.

Usage:
    python reports/generate_analysis_report_docx.py
"""

import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "reports", "Teknofest_Veri_Analizi_Raporu.docx")


def set_cell_shading(cell, color_hex):
    shading = cell._element.get_or_add_tcPr()
    shd = shading.makeelement(qn("w:shd"), {
        qn("w:fill"): color_hex,
        qn("w:val"): "clear",
    })
    shading.append(shd)


def add_styled_table(doc, headers, rows, col_widths=None, header_color="1F4E79"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(cell, header_color)

    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)
            if r_idx % 2 == 1:
                set_cell_shading(cell, "D6E4F0")

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    return table


def build_document():
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15

    for level in range(1, 4):
        hs = doc.styles[f"Heading {level}"]
        hs.font.name = "Calibri"
        hs.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    # ── TITLE PAGE ──
    for _ in range(6):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("TEKNOFEST 2026\nSağlıkta Yapay Zeka Yarışması")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Veri Analizi ve Modelleme Strateji Raporu")
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run("Üniversite ve Üzeri Seviyesi\nTarih: 02.06.2026\nKategori: Genetik Varyant Patojenisite Sınıflandırma")
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0x59, 0x56, 0x59)

    doc.add_page_break()

    # ── TABLE OF CONTENTS ──
    doc.add_heading("İÇİNDEKİLER", level=1)
    toc_items = [
        "1. Yönetici Özeti",
        "2. Yarışma Tanımı ve Kurallar",
        "3. Veri Seti Genel Bakış",
        "4. Veri Kalitesi Analizi",
        "5. Eksik Veri Analizi",
        "6. Sınıf Dengesizliği Analizi",
        "7. Özellik (Feature) Analizi",
        "8. Dört Model Stratejisi",
        "9. Ön İşleme Hattı (Pipeline) Önerisi",
        "10. Temel Model (Baseline) Planı",
        "11. Risk Analizi",
        "12. Açık Sorular",
        "13. Sonraki Adımlar",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 1. YÖNETİCİ ÖZETİ
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("1. Yönetici Özeti", level=1)

    doc.add_paragraph(
        "Bu rapor, TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması kapsamında "
        "Üniversite ve Üzeri seviyesinde verilen genetik varyant veri setlerinin "
        "kapsamlı teknik analizini sunmaktadır."
    )

    doc.add_heading("Temel Bulgular", level=2)

    bullets = [
        ("Görev: ", "Genetik varyantları Patojenik (hastalık yapıcı) veya Benign (zararsız) olarak sınıflandırma. Değerlendirme metriği: F1 Skoru."),
        ("4 veri seti: ", "1 genel (MASTER, 2931 varyant) + 3 gen paneli (KANSER 388, PAH 372, CFTR 111). Tüm veri setleri aynı 353 sütunluk şemayı paylaşıyor."),
        ("Eksik veri çok yüksek: ", "MASTER'da hücrelerin %54.9'u eksik. Eksiklik rastgele değil — etiketle (label) ilişkili (MNAR)."),
        ("Sınıf dengesizliği: ", "Patojenik varyantlar %67-83 oranında baskın. PAH paneli 5:1 oranında en dengesiz. Ancak test setleri dengeli (1:1)."),
        ("En güçlü özellikler: ", "EK özellikleri (EK_7, EK_9, EK_4) en yüksek korelasyona sahip. Bunların bazıları döngüsel tahmin riski taşıyor (ClinVar ile eğitilmiş olabilir)."),
        ("Mevcut baseline: ", "LightGBM ile ROC-AUC 0.846, F1 0.860 elde edilmiş. PAH paneli en zayıf (ROC-AUC 0.772)."),
    ]
    for bold_part, normal_part in bullets:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(bold_part)
        run.bold = True
        p.add_run(normal_part)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 2. YARIŞMA TANIMI
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("2. Yarışma Tanımı ve Kurallar", level=1)

    doc.add_paragraph(
        "TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması, bu yıl Genetik ve Kardiyoloji "
        "alanlarını öncelikli odak olarak belirlemiştir. Üniversite ve Üzeri seviyesinde "
        "yarışmacılardan genetik varyantların patojenisite durumunu tahmin eden modeller "
        "geliştirmeleri beklenmektedir."
    )

    doc.add_heading("Yarışma Bilgileri", level=2)

    add_styled_table(doc,
        ["Madde", "Detay"],
        [
            ["Yarışma", "TEKNOFEST 2026 Sağlıkta Yapay Zeka"],
            ["Kategori", "Üniversite ve Üzeri Seviyesi"],
            ["Görev", "İkili sınıflandırma: Patojenik vs Benign"],
            ["Referans standart", "ACMG kriterleri, ClinVar/ClinGen veritabanları"],
            ["Sıralama metriği", "F1 Skoru (TP, FP, FN üzerinden)"],
            ["Puan dağılımı", "%90 yarışma görevi + %10 final sunumu"],
            ["Proje Detay Raporu", "Son teslim: 29.06.2026"],
            ["Finaller", "Ağustos-Eylül 2026, Şanlıurfa"],
        ],
        col_widths=[5, 12],
    )

    doc.add_paragraph()
    doc.add_heading("Eğitim Veri Seti Boyutları (Şartnamede Belirtilen)", level=2)

    add_styled_table(doc,
        ["Panel", "Patojenik", "Benign", "Toplam"],
        [
            ["Genel (MASTER)", "1500", "1500", "3000"],
            ["Kalıtsal Kanser", "200", "200", "400"],
            ["Fenilketonüri (PAH)", "200", "200", "400"],
            ["Kistik Fibrozis (CFTR)", "70", "70", "140"],
        ],
        col_widths=[5, 3, 3, 3],
    )

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Not: ")
    run.bold = True
    p.add_run(
        "Gerçek eğitim verisinde boyutlar şartnameden farklıdır (MASTER: 2931, KANSER: 388, "
        "PAH: 372, CFTR: 111). Ayrıca şartname dengeli dağılım belirtse de gerçek veri "
        "belirgin sınıf dengesizliği içermektedir."
    )

    doc.add_heading("Test Veri Seti Boyutları (Şartnamede Belirtilen)", level=2)

    add_styled_table(doc,
        ["Panel", "Patojenik", "Benign", "Toplam"],
        [
            ["Genel (MASTER)", "1000", "1000", "2000"],
            ["Kalıtsal Kanser", "100", "100", "200"],
            ["Fenilketonüri (PAH)", "100", "100", "200"],
            ["Kistik Fibrozis (CFTR)", "30", "30", "60"],
        ],
        col_widths=[5, 3, 3, 3],
    )

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("KRİTİK BULGU: ")
    run.bold = True
    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    p.add_run(
        "Test setleri dengeli (1:1) olarak belirtilmiş, ancak eğitim setleri dengesizdir "
        "(5:1'e kadar). Bu nedenle eğitim verisinde optimize edilen eşik değeri (threshold) "
        "test zamanında uygun olmayacaktır. Eşik değeri dengeli dağılım varsayımıyla "
        "yeniden kalibre edilmelidir."
    )

    doc.add_heading("Özellik Kategorileri (Şartnameden)", level=2)

    feature_cats = [
        ("Sekans ve Değişim Bilgisi", "Nükleotid değişimi, kodon değişimi, amino asit dönüşümü"),
        ("Yerel Sekans ve Çevresel Bağlam", "5 nükleotid genomik komşuluk, 5 amino asit proteomik komşuluk"),
        ("Biyokimyasal ve Yapısal Etkiler", "Hidrofobiklik, polarite, moleküler ağırlık, 3D yapı etkileri"),
        ("Evrimsel Korunmuşluk", "Filogenetik çeşitlilik, türler arası genomik benzerlik"),
        ("Popülasyon Verileri", "Minör Allel Frekansı (MAF) ve popülasyon sıklıkları"),
        ("In Silico Risk Skorları", "Hesaplamalı patojenisite tahmin skorları"),
    ]
    for title, desc in feature_cats:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(title + ": ")
        run.bold = True
        p.add_run(desc)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 3. VERİ SETİ GENEL BAKIŞ
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("3. Veri Seti Genel Bakış", level=1)

    doc.add_heading("Veri Seti Şeması", level=2)
    doc.add_paragraph(
        "Tüm 4 veri seti aynı 353 sütunluk şemayı paylaşmaktadır:"
    )

    add_styled_table(doc,
        ["Özellik Grubu", "Ön Ek", "Sayı", "Tür", "Açıklama"],
        [
            ["Varyant Kimliği", "Variant_ID", "1", "Metin", "Benzersiz varyant tanımlayıcı"],
            ["Algoritma Skorları", "AL_1 – AL_334", "334", "Sayısal", "Anonimleştirilmiş in-silico skorlar"],
            ["Kategorik", "CAT_1 – CAT_6", "6", "Metin", "Popülasyon, genotip, bölge etiketleri"],
            ["Ek Bilgi", "EK_1 – EK_9", "9", "Sayısal", "Meta-tahmin veya bileşik skorlar"],
            ["Amino Asit", "AA_1, AA_2", "2", "Metin", "Referans ve alternatif amino asit"],
            ["Etiket", "Label", "1", "Tamsayı", "0=Benign, 1=Patojenik"],
        ],
        col_widths=[3.5, 3.5, 1.5, 2, 6],
    )

    doc.add_paragraph()
    doc.add_heading("Veri Seti Boyutları", level=2)

    add_styled_table(doc,
        ["Veri Seti", "Satır", "Sütun", "Toplam Hücre", "Eksik Hücre", "Eksik %"],
        [
            ["MASTER", "2931", "353", "1,034,643", "568,464", "%54.9"],
            ["KANSER", "388", "353", "136,964", "78,309", "%57.2"],
            ["PAH", "372", "353", "131,316", "71,301", "%54.3"],
            ["CFTR", "111", "353", "39,183", "11,864", "%30.3"],
        ],
        col_widths=[2.5, 2, 2, 2.5, 2.5, 2],
    )

    doc.add_paragraph()
    doc.add_heading("Variant_ID Örtüşme Analizi", level=2)

    doc.add_paragraph(
        "Panel veri setleri birbirleriyle örtüşmez, ancak MASTER ile önemli örtüşme vardır. "
        "Bu durum, MASTER ile eğitim yapıp panellerde değerlendirirken sızıntı (leakage) riski oluşturur."
    )

    add_styled_table(doc,
        ["Panel Çifti", "Örtüşen Varyant Sayısı"],
        [
            ["MASTER ∩ PAH", "255"],
            ["MASTER ∩ KANSER", "246"],
            ["MASTER ∩ CFTR", "77"],
            ["CFTR ∩ PAH", "0"],
            ["CFTR ∩ KANSER", "0"],
            ["PAH ∩ KANSER", "0"],
        ],
        col_widths=[6, 5],
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 4. VERİ KALİTESİ ANALİZİ
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("4. Veri Kalitesi Analizi", level=1)

    doc.add_heading("Sabit (Constant) Sütunlar", level=2)
    doc.add_paragraph(
        "Sabit sütunlar sıfır bilgi taşır ve modelleme öncesinde mutlaka çıkarılmalıdır."
    )

    add_styled_table(doc,
        ["Veri Seti", "Sabit (1 değer)", "Yarı-sabit (2 değer)"],
        [
            ["MASTER", "57", "48"],
            ["CFTR", "70", "94"],
            ["PAH", "91", "21"],
            ["KANSER", "69", "91"],
        ],
        col_widths=[4, 4, 4],
    )

    doc.add_paragraph()
    doc.add_heading("Tekrar Eden Satırlar", level=2)

    add_styled_table(doc,
        ["Veri Seti", "Tekrar Eden Variant_ID", "Tekrar Eden Özellik Satırları"],
        [
            ["MASTER", "0", "114"],
            ["CFTR", "0", "0"],
            ["PAH", "0", "3"],
            ["KANSER", "0", "2"],
        ],
        col_widths=[4, 4, 5],
    )

    doc.add_paragraph()
    doc.add_paragraph(
        "MASTER'da 114 satır aynı özellik değerlerine sahip (farklı Variant_ID ile). "
        "Bu, benzer biyolojik profilli varyantlardan kaynaklanmaktadır ve genomik verilerde beklenen bir durumdur."
    )

    doc.add_heading("Şüpheli Değerler", level=2)
    suspicious = [
        ("AL_185", "Bazı satırlarda 264690.0 değeri alır — diğer AL özellikleri genellikle [0,1] aralığındayken bu değer bir sayım/pozisyon alanı olabilir."),
        ("CAT_6", "%97.7 eksik — yalnızca belirli genomik bölgeler için mevcut. İkili bayrak (flag) olarak kodlanmalıdır."),
        ("EK_2", "[-11.9, 6.17] aralığı — negatif değerler log-ölçek veya Z-skoru dönüşümü işaret eder."),
    ]
    for feat, desc in suspicious:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(feat + ": ")
        run.bold = True
        p.add_run(desc)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 5. EKSİK VERİ ANALİZİ
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("5. Eksik Veri Analizi", level=1)

    doc.add_paragraph(
        "Eksik veri bu veri setinin en kritik zorluğudur. MASTER'da hücrelerin %54.9'u eksiktir "
        "ve bu eksiklik rastgele değildir (MNAR — Missing Not At Random)."
    )

    doc.add_heading("Eksiklik Bantları (MASTER)", level=2)

    add_styled_table(doc,
        ["Eksiklik Aralığı", "Sütun Sayısı", "Yorum"],
        [
            ["%0 (tam)", "2", "Sadece Variant_ID ve Label"],
            ["%0-10", "0", "Hiçbir özellik bu aralıkta değil"],
            ["%10-25", "13", "En güvenilir özellikler"],
            ["%25-50", "173", "Kullanılabilir ama dikkatli olunmalı"],
            ["%50-75", "146", "Yüksek eksiklik — sinyal hâlâ olabilir"],
            ["%75-90", "6", "Çok yüksek eksiklik"],
            ["%90-100", "13", "Neredeyse boş sütunlar"],
        ],
        col_widths=[3.5, 3, 6],
    )

    doc.add_paragraph()
    doc.add_heading("Özellik Grubuna Göre Eksiklik (%)", level=2)

    add_styled_table(doc,
        ["Özellik Grubu", "MASTER", "CFTR", "PAH", "KANSER"],
        [
            ["AL özellikleri", "%56.9", "%31.4", "%56.5", "%59.5"],
            ["EK özellikleri", "%16.9", "%4.9", "%7.9", "%9.0"],
            ["CAT özellikleri", "%38.4", "%24.3", "%34.4", "%37.3"],
            ["AA özellikleri", "%11.9", "%0.0", "%2.4", "%5.7"],
        ],
        col_widths=[3.5, 2.5, 2.5, 2.5, 2.5],
    )

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Önemli bulgu: ")
    run.bold = True
    p.add_run(
        "EK ve AA özellikleri AL özelliklerine göre çok daha az eksik veriye sahiptir. "
        "Bu nedenle EK özellikleri model tarafından daha güvenilir şekilde kullanılabilir."
    )

    doc.add_heading("Etikete Göre Eksiklik (MASTER)", level=2)

    add_styled_table(doc,
        ["Etiket", "N", "Genel Eksiklik %"],
        [
            ["0 (Benign)", "782", "%41.2"],
            ["1 (Patojenik)", "2149", "%59.9"],
        ],
        col_widths=[4, 3, 4],
    )

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("KRİTİK: ")
    run.bold = True
    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    p.add_run(
        "Patojenik varyantlarda %59.9, Benign'de %41.2 eksiklik var. Bu fark, eksikliğin "
        "kendisinin etiketi tahmin etmek için bilgi taşıdığını gösterir. Sonuçlar:"
    )

    mnar_items = [
        "Ağaç tabanlı modeller (LightGBM) eksikliği doğal olarak kullanabilir",
        "İmpütasyon bu sinyali YOK EDER — ağaç modelleri için impütasyon yapılmamalı",
        "Eksiklik göstergesi (is_missing) özellikleri tahmine katkı sağlar",
    ]
    for item in mnar_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Satır Düzeyinde Eksiklik (MASTER)", level=2)

    add_styled_table(doc,
        ["Yüzdelik Dilim", "Eksik Sütun Sayısı (353 üzerinden)"],
        [
            ["Minimum", "0"],
            ["%25", "30"],
            ["Medyan (%50)", "212"],
            ["%75", "337"],
            ["Maksimum", "351"],
        ],
        col_widths=[5, 6],
    )

    doc.add_paragraph()
    doc.add_paragraph(
        "Medyan varyant 353 sütunun 212'sinde (%60) eksik veriye sahiptir. "
        "Dağılım iki modlu: bazı varyantlar çok iyi anotasyonlu, diğerleri neredeyse tamamen boş."
    )

    doc.add_heading("Önerilen Eksik Veri Stratejisi", level=2)

    strategies = [
        ("YAPILMALI — Doğal GBDT NaN desteği: ", "LightGBM/XGBoost eksik değerleri kendi içinde öğrenir. Bu en iyi yaklaşımdır."),
        ("YAPILMALI — Eksiklik göstergeleri: ", "Her özellik için is_missing bayrağı ekleyin. Eksiklik bilgi taşıyor."),
        ("ÖNERİLİR — Satır düzeyinde eksiklik sayısı: ", "Her satır için toplam eksik sütun sayısını yeni özellik olarak ekleyin."),
        ("YAPILMAMALI — İmpütasyon (ağaç modelleri için): ", "Medyan/KNN impütasyonu sinyali yok eder. Sadece doğrusal modeller için kullanın."),
        ("YAPILMAMALI — SMOTE: ", "Sentetik genomik varyantlar biyolojik olarak anlamsızdır."),
    ]
    for bold_part, normal_part in strategies:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(bold_part)
        run.bold = True
        p.add_run(normal_part)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 6. SINIF DENGESİZLİĞİ
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("6. Sınıf Dengesizliği Analizi", level=1)

    doc.add_heading("Sınıf Dağılımı", level=2)

    add_styled_table(doc,
        ["Veri Seti", "Patojenik (1)", "Benign (0)", "Toplam", "Patojenik %", "Oran"],
        [
            ["MASTER", "2149", "782", "2931", "%73.3", "2.75:1"],
            ["KANSER", "268", "120", "388", "%69.1", "2.23:1"],
            ["PAH", "310", "62", "372", "%83.3", "5.00:1"],
            ["CFTR", "90", "21", "111", "%81.1", "4.29:1"],
        ],
        col_widths=[2.5, 2.5, 2, 2, 2.5, 2],
    )

    doc.add_paragraph()
    doc.add_heading("Ciddiyet Değerlendirmesi", level=2)

    add_styled_table(doc,
        ["Veri Seti", "Ciddiyet", "Açıklama"],
        [
            ["MASTER", "ORTA", "2.75:1 — sınıf ağırlıkları ile yönetilebilir"],
            ["KANSER", "ORTA", "2.23:1 — en az dengesiz panel"],
            ["PAH", "CİDDİ", "5.00:1 — sadece 62 Benign örnek"],
            ["CFTR", "CİDDİ + KÜÇÜK", "4.29:1, sadece 21 Benign — yüksek varyans"],
        ],
        col_widths=[3, 3, 7],
    )

    doc.add_paragraph()
    doc.add_heading("Eğitim-Test Dengesizlik Uyumsuzluğu", level=2)

    p = doc.add_paragraph()
    run = p.add_run("ÖNEMLİ: ")
    run.bold = True
    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    p.add_run(
        "Şartname test setlerinin dengeli (1:1) olacağını belirtmektedir. "
        "Eğitim setleri ise dengesizdir (5:1'e kadar). Bu durum şu sonuçları doğurur:"
    )

    mismatch_items = [
        "Eğitim verisinde optimize edilen eşik değeri test için çok agresif olacaktır",
        "Eşik değeri dengeli dağılım varsayımıyla yeniden kalibre edilmelidir",
        "F1 skoru davranışı dengesiz ve dengeli veride farklıdır",
    ]
    for item in mismatch_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Önerilen Dengesizlik Stratejileri", level=2)

    add_styled_table(doc,
        ["Strateji", "Öncelik", "Detay"],
        [
            ["Sınıf ağırlıkları", "ZORUNLU", "LightGBM: is_unbalance=True"],
            ["Katmanlı K-Fold", "ZORUNLU", "Her katlamada etiket oranını koru"],
            ["Eşik kalibrasyonu", "YÜKSEK", "Dengeli test dağılımı için F1 optimize et"],
            ["SMOTE KULLANMA", "ZORUNLU", "Sentetik varyantlar biyolojik olarak anlamsız"],
            ["Alt örnekleme YAPMA", "ÖNERİLİR", "Zaten sınırlı veri; Benign kaybetmek zararlı"],
        ],
        col_widths=[4, 3, 7],
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 7. ÖZELLİK ANALİZİ
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("7. Özellik (Feature) Analizi", level=1)

    doc.add_heading("Etiketle En Yüksek Korelasyona Sahip 15 Özellik", level=2)

    add_styled_table(doc,
        ["Sıra", "Özellik", "|Korelasyon|", "Kategori"],
        [
            ["1", "EK_7", "0.380", "Ek bilgi"],
            ["2", "EK_9", "0.322", "Ek bilgi"],
            ["3", "EK_4", "0.320", "Ek bilgi"],
            ["4", "EK_2", "0.316", "Ek bilgi"],
            ["5", "EK_6", "0.302", "Ek bilgi"],
            ["6", "EK_3", "0.296", "Ek bilgi"],
            ["7", "EK_8", "0.232", "Ek bilgi"],
            ["8", "AL_83", "0.212", "Algoritma skoru"],
            ["9", "AL_92", "0.173", "Algoritma skoru"],
            ["10", "AL_77", "0.165", "Algoritma skoru"],
            ["11", "EK_5", "0.154", "Ek bilgi"],
            ["12", "AL_34", "0.125", "Algoritma skoru"],
            ["13", "AL_26", "0.120", "Algoritma skoru"],
            ["14", "AL_32", "0.116", "Algoritma skoru"],
            ["15", "AL_95", "0.107", "Algoritma skoru"],
        ],
        col_widths=[1.5, 3, 3, 4],
    )

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Gözlem: ")
    run.bold = True
    p.add_run(
        "EK özellikleri baskın — ilk 11 özelliğin 8'i EK grubundandır. Bu özellikler "
        "muhtemelen çoklu araçlardan bilgiyi birleştiren meta-tahmin skorlarıdır."
    )

    doc.add_heading("EK Özelliklerinin Etikete Göre Dağılımı", level=2)

    add_styled_table(doc,
        ["Özellik", "Benign Ort.", "Patojenik Ort.", "Ayrışma", "Olası Yorum"],
        [
            ["EK_7", "3.85", "6.64", "Güçlü", "Bileşik hasar skoru"],
            ["EK_9", "5.31", "8.21", "Güçlü", "Toplam patojenisite skoru"],
            ["EK_4", "0.75", "0.95", "Güçlü", "Meta-tahmin [0,1], ör. REVEL"],
            ["EK_2", "3.29", "4.85", "Güçlü", "Korunmuşluk/hasar skoru"],
            ["EK_6", "0.77", "0.95", "Güçlü", "Meta-tahmin [0,1], EK_4'e benzer"],
            ["EK_3", "1.90", "3.52", "Güçlü", "Korunmuşluk skoru"],
            ["EK_8", "0.42", "0.61", "Orta", "Olası phastCons benzeri"],
            ["EK_1", "5.18", "5.29", "Zayıf", "Anotasyon kalite skoru"],
            ["EK_5", "0.72", "0.83", "Orta", "Korunmuşluk olasılığı"],
        ],
        col_widths=[2, 2.5, 2.5, 2, 4.5],
    )

    doc.add_paragraph()
    doc.add_heading("Veri Sızıntısı (Leakage) Risk Değerlendirmesi", level=2)

    add_styled_table(doc,
        ["Özellik(ler)", "Risk Seviyesi", "Açıklama"],
        [
            ["EK_4, EK_6", "YÜKSEK", "[0,1] aralığında meta-tahminler — ClinVar ile eğitilmiş olabilir (döngüsel tahmin)"],
            ["EK_7, EK_9", "ORTA", "Güçlü tahmin ediciler ama geniş aralık — bileşik skorlar"],
            ["Variant_ID", "YÜKSEK", "Özelliklerden mutlaka çıkarılmalı — kimlik kodlar, biyoloji değil"],
            ["AL_185", "DÜŞÜK", "264690 değeri — muhtemelen allel sayısı"],
        ],
        col_widths=[3, 3, 8],
    )

    doc.add_paragraph()
    doc.add_heading("Kategorik Özellik Özeti", level=2)

    add_styled_table(doc,
        ["Sütun", "Benzersiz Değer", "Eksik %", "Olası Anlam", "Önerilen Kodlama"],
        [
            ["CAT_1", "30", "%37.0", "Popülasyon/veritabanı", "Label encoding"],
            ["CAT_2", "7", "%59.2", "AllofUs popülasyon grubu", "One-hot / label"],
            ["CAT_3", "5", "%12.2", "Referans genotip", "One-hot"],
            ["CAT_4", "5", "%12.2", "Alternatif genotip", "One-hot"],
            ["CAT_5", "5", "%12.2", "Başka genotip alanı", "One-hot"],
            ["CAT_6", "3", "%97.7", "Yapısal bölge tipi", "İkili bayrak"],
            ["AA_1", "24", "%11.9", "Referans amino asit", "One-hot"],
            ["AA_2", "25", "%11.9", "Alternatif amino asit", "One-hot"],
        ],
        col_widths=[2, 2.5, 2, 3.5, 3],
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 8. DÖRT MODEL STRATEJİSİ
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("8. Dört Model Stratejisi", level=1)

    doc.add_heading("Neden 4 Model Gerekiyor?", level=2)

    doc.add_paragraph(
        "Yarışma şartnamesi 4 ayrı test seti tanımlamaktadır (Bölüm 3.2 ve 7.7). "
        "Her test seti bağımsız olarak F1 Skoru ile değerlendirilecektir. "
        "Dolayısıyla 4 ayrı tahmin çıktısı üretmek zorunludur."
    )

    reasons = [
        ("Dağılım kayması: ", "Paneller arasında özellik dağılımları önemli ölçüde farklıdır (KS testi p<0.001, 57 özellikte)."),
        ("Eksiklik desenleri farklı: ", "CFTR %30.3 eksik iken KANSER %57.2 eksik."),
        ("Sınıf dengesizliği değişken: ", "PAH 5:1 iken KANSER 2.23:1."),
        ("Gene özgü biyoloji: ", "Farklı genler farklı patojenisite mekanizmalarına sahip."),
        ("Variant_ID örtüşmesi: ", "Panel varyantları kısmen MASTER ile örtüşür — dikkatli veri yönetimi gerekir."),
    ]
    for bold_part, normal_part in reasons:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(bold_part)
        run.bold = True
        p.add_run(normal_part)

    doc.add_heading("Model Analiz Tablosu", level=2)

    add_styled_table(doc,
        ["Model", "Amaç", "Eğitim Verisi", "Zorluk", "Önerilen Algoritma"],
        [
            ["Model 1", "Genel varyant sınıflandırma", "MASTER (n=2931)", "Yüksek eksik veri", "LightGBM ensemble"],
            ["Model 2", "Kalıtsal Kanser paneli", "KANSER (n=388)", "Farklı eksiklik deseni", "MASTER'dan transfer + kalibrasyon"],
            ["Model 3", "Fenilketonüri (PAH)", "PAH (n=372)", "5:1 dengesizlik, 62 Benign", "MASTER + panel kalibrasyonu"],
            ["Model 4", "Kistik Fibrozis (CFTR)", "CFTR (n=111)", "Çok küçük veri, 21 Benign", "Global model + eşik kalibrasyonu"],
        ],
        col_widths=[2, 3.5, 3, 3, 4],
    )

    doc.add_paragraph()
    doc.add_heading("Önerilen Mimari: Global Model + Panel Kalibrasyonu", level=2)

    doc.add_paragraph(
        "En güvenli yaklaşım, MASTER üzerinde eğitilmiş tek bir global GBDT ensemble modeli "
        "ve her panel için ayrı eşik değeri kalibrasyonudur."
    )

    arch_items = [
        "MASTER'da eğitilmiş global LightGBM + XGBoost + CatBoost ensemble",
        "Her varyant için P(patojenik) olasılığı üretir",
        "Panel bazlı eşik kalibrasyonu: MASTER t=0.50, KANSER/PAH/CFTR ayrı optimize",
        "Eğer global model panellerde yetersiz kalırsa: panel-spesifik ince ayar (fine-tune)",
    ]
    for item in arch_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Neden 4 bağımsız model değil? ")
    run.bold = True
    p.add_run(
        "(1) Veri verimliliği — MASTER'ın 2931 örneği genel desenleri öğretir. "
        "(2) Aşırı öğrenme riski — 111 örnekle bağımsız model güvenilir değil. "
        "(3) Basitlik — tek model hata ayıklama ve açıklama açısından kolay. "
        "(4) Ortak şema — 4 veri seti aynı 353 sütunu paylaşıyor."
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 9. ÖN İŞLEME HATTI
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("9. Ön İşleme Hattı (Pipeline) Önerisi", level=1)

    steps = [
        ("1. Veri Yükle", "4 CSV dosyasını oku, şema tutarlılığını doğrula (353 sütun)"),
        ("2. Bilgi taşımayan sütunları çıkar", "Variant_ID (sızıntı riski) + 57 sabit sütun (MASTER)"),
        ("3. Eksiklik özellik mühendisliği", "Her özellik için is_missing bayrağı + satır düzeyinde toplam eksik sayısı + grup düzeyinde eksik sayıları"),
        ("4. Kategorik kodlama", "CAT_1: label encode, CAT_2: one-hot/label, CAT_3/4/5: one-hot, CAT_6: ikili bayrak, AA_1/AA_2: one-hot"),
        ("5. İmpütasyon YAPMA", "LightGBM/XGBoost NaN'ı doğal olarak yönetir — impütasyon sinyali yok eder"),
        ("6. Sınıf ağırlıkları", "is_unbalance=True (LightGBM), scale_pos_weight (XGBoost)"),
        ("7. Doğrulama ayrımı", "5-Fold katmanlı çapraz doğrulama, sabit tohum değeri (seed=42)"),
    ]
    for title, desc in steps:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(title + ": ")
        run.bold = True
        p.add_run(desc)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 10. TEMEL MODEL PLANI
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("10. Temel Model (Baseline) Planı", level=1)

    doc.add_heading("Temel Modeller", level=2)

    add_styled_table(doc,
        ["Model", "Amaç", "Neden"],
        [
            ["LightGBM (varsayılan)", "Birincil baseline", "Hızlı, eksik değer desteği, kategorik destek"],
            ["XGBoost (varsayılan)", "Çeşitlilik baseline", "Farklı ağaç oluşturma algoritması"],
            ["Lojistik Regresyon", "Yorumlanabilir referans", "Doğrusal ayrılabilirlik tabanı"],
        ],
        col_widths=[4, 3.5, 6],
    )

    doc.add_paragraph()
    doc.add_heading("Mevcut Baseline Sonuçları (Pipeline B)", level=2)

    add_styled_table(doc,
        ["Metrik", "Değer"],
        [
            ["ROC-AUC (OOF)", "0.846"],
            ["PR-AUC (OOF)", "0.925"],
            ["F1 Skoru (Youden eşiği)", "0.860"],
            ["Duyarlılık @ %95 hedef", "0.950"],
            ["CFTR panel ROC-AUC", "0.940"],
            ["KANSER panel ROC-AUC", "0.914"],
            ["PAH panel ROC-AUC", "0.772"],
        ],
        col_widths=[6, 4],
    )

    doc.add_paragraph()
    doc.add_paragraph(
        "PAH paneli en zayıf performansı göstermektedir — aşırı dengesizlik (5:1) ve yüksek "
        "eksik veri ile tutarlıdır."
    )

    doc.add_heading("Güçlü Model Adayları", level=2)

    strong = [
        ("LightGBM (ayarlı)", "Baseline sonrası Bayesian hiperparametre optimizasyonu"),
        ("CatBoost", "Kategorik özellikleri doğal olarak yönetir"),
        ("Yığınlama Ensemble", "LightGBM + XGBoost + CatBoost → lojistik regresyon meta-öğrenici"),
        ("TabNet", "Yalnızca ağaç modelleri platoya ulaşırsa"),
    ]
    for name, desc in strong:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(name + ": ")
        run.bold = True
        p.add_run(desc)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 11. RİSK ANALİZİ
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("11. Risk Analizi", level=1)

    add_styled_table(doc,
        ["#", "Risk", "Ciddiyet", "Etki", "Azaltma"],
        [
            ["1", "Aşırı eksik veri (%54.9)", "YÜKSEK", "Azaltılmış bilgi", "GBDT doğal NaN + eksiklik özellikleri"],
            ["2", "MNAR eksiklik (etiketle ilişkili)", "YÜKSEK", "İmpütasyon yanlılığı", "İmpütasyon YAPMA, eksikliği sinyal olarak kullan"],
            ["3", "CFTR çok küçük (n=111)", "YÜKSEK", "Güvenilir olmayan tahminler", "Global model + eşik kalibrasyonu"],
            ["4", "EK_4/EK_6 döngüsel tahmin", "YÜKSEK", "Yapay yüksek performans", "Ablasyon deneyi (Pipeline C)"],
            ["5", "Eğitim-test dengesizlik uyumsuzluğu", "YÜKSEK", "Yanlış eşik değeri", "Dengeli dağılım için eşik kalibrasyonu"],
            ["6", "PAH 5:1 dengesizlik", "CİDDİ", "Düşük panel performansı", "Sınıf ağırlıkları + eşik optimizasyonu"],
            ["7", "57 sabit sütun", "ORTA", "Gürültü", "Modelleme öncesi çıkar"],
            ["8", "Dağılım kayması MASTER→PAH", "ORTA", "Genelleme hatası", "Panel kalibrasyonu"],
            ["9", "Metrik değişikliği hakkı", "ORTA", "Strateji güncellemesi gerekebilir", "Birden fazla metrik optimize et"],
            ["10", "Kod tekrar üretilebilirlik", "ORTA", "Jüri talep edebilir", "Sabit tohum, dokümantasyon"],
        ],
        col_widths=[0.8, 4, 2, 2.5, 5],
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 12. AÇIK SORULAR
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("12. Açık Sorular", level=1)

    doc.add_heading("Model Geliştirme Öncesi Yanıtlanması Gereken Sorular", level=2)

    questions = [
        "F1 skoru her panel için ayrı mı hesaplanacak, yoksa tüm paneller üzerinden toplu mı?",
        "EK_4 ve EK_6 ClinVar verisi ile mi eğitilmiş? Evetse, dahil etmek döngüsel tahmin yaratır.",
        "Test setleri şartnamede belirtildiği gibi tam 1:1 dengeli mi olacak?",
        "Takımlar panel başına ayrı model mi sunabilir, yoksa tek birleşik model mi gerekli?",
        "Final sırasında çıkarım (inference) için donanım/süre kısıtlaması var mı?",
        "Anonimleştirilmiş özellik isimlerinin biyolojik karşılıklarını gösteren ek dokümantasyon var mı?",
    ]
    for i, q in enumerate(questions, 1):
        doc.add_paragraph(f"{i}. {q}")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════
    # 13. SONRAKİ ADIMLAR
    # ═══════════════════════════════════════════════════════════
    doc.add_heading("13. Sonraki Adımlar", level=1)

    doc.add_heading("Öncelik Sırasına Göre Yapılacaklar", level=2)

    next_steps = [
        ("1. EK ablasyonu (Pipeline C)", "EK_4/EK_5/EK_6 olmadan model eğitin. Performans farkı küçükse (<0.02 AUC) güvenlik için çıkarın."),
        ("2. F1 optimizasyonu dengeli veri için", "Test setleri 1:1 dengeli olduğundan, eşik değerlerini dengeli sınıf dağılımı varsayarak yeniden optimize edin."),
        ("3. Ensemble oluşturma", "XGBoost ve CatBoost'u LightGBM'in yanına ekleyin. Lojistik regresyon meta-öğreniciyle yığınlama (stacking) yapın."),
        ("4. Panel bazlı kalibrasyon", "Her panel için F1'i maksimize eden eşik değerini bulun."),
        ("5. Hiperparametre optimizasyonu", "Optuna ile Bayesian arama: num_leaves, min_child_samples, learning_rate, feature_fraction."),
        ("6. Hata analizi", "OOF tahminlerinden yanlış sınıflandırılan varyantları inceleyin. Hatalar belirli panellerde/eksiklik profillerinde mi yoğunlaşıyor?"),
        ("7. Dokümantasyon", "Proje Detay Raporu yazmaya başlayın. Son teslim: 29.06.2026."),
    ]
    for title, desc in next_steps:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(title + ": ")
        run.bold = True
        p.add_run(desc)

    doc.add_paragraph()
    doc.add_heading("Önerilen Zaman Çizelgesi", level=2)

    add_styled_table(doc,
        ["Hafta", "Görev"],
        [
            ["Hafta 1", "EK ablasyonu (Pipeline C), ensemble eğitimi (XGBoost, CatBoost)"],
            ["Hafta 2", "Yığınlama ensemble, panel eşik kalibrasyonu"],
            ["Hafta 3", "Hiperparametre optimizasyonu, hata analizi, özellik mühendisliği deneyleri"],
            ["Hafta 4", "Final model seçimi, sağlamlık testi, dokümantasyon"],
        ],
        col_widths=[3, 11],
    )

    doc.add_paragraph()
    doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("— Rapor Sonu —")
    run.italic = True
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    # ── SAVE ──
    doc.save(OUTPUT_PATH)
    print(f"Rapor kaydedildi: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_document()
