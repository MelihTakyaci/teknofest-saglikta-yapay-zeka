# Teknofest Data Analysis — Team Summary

**Yarışma**: TEKNOFEST 2026 Sağlıkta Yapay Zeka — Üniversite ve Üzeri Seviyesi
**Tarih**: 2026-06-02

---

## Sorun Ne?

Genetik varyantları **Patojenik** (hastalık yapıcı) veya **Benign** (zararsız) olarak sınıflandırmamız gerekiyor. Yarışma **F1 Skoru** ile değerlendiriliyor.

---

## Veri Seti Kullanılabilir mi?

**EVET**, ancak ciddi sorunlar var:

| Sorun | Durum |
|-------|-------|
| Eksik veri (missing data) | **ÇOK YÜKSEK** — Verilerin %55'i eksik |
| Sınıf dengesizliği | **ORTA-CİDDİ** — Patojenik varyantlar %67-83 oranında |
| Küçük panel boyutları | **CİDDİ** — CFTR sadece 111 örnek (21 Benign) |
| Olası veri sızıntısı (leakage) | **ORTA** — EK_4/EK_6 ClinVar'dan türetilmiş olabilir |
| Test seti dengeleme farkı | **ÖNEMLİ** — Eğitim dengesiz, test dengeli (1:1) |

---

## Eksik Veri Durumu

- MASTER'da **medyan varyant %60'ı eksik** (353 sütundan 212'si)
- Patojenik varyantlarda (%59.9) Benign'e göre (%41.2) **daha fazla eksik veri** var
- Eksiklik rastgele değil (MNAR) — **modele sinyal olarak verilmeli**
- Çözüm: **İmpütasyon YAPMAYIN**, LightGBM'in doğal NaN desteğini kullanın + eksiklik göstergesi (indicator) özellikler ekleyin

---

## Sınıf Dengesizliği

| Panel | Patojenik | Benign | Oran |
|-------|-----------|--------|------|
| MASTER | 2149 | 782 | 2.75:1 |
| KANSER | 268 | 120 | 2.23:1 |
| PAH | 310 | 62 | **5.00:1** |
| CFTR | 90 | 21 | **4.29:1** |

Test seti ise **dengeli (1:1)**. Bu yüzden eşik değeri (threshold) eğitim ve test için farklı optimize edilmeli.

---

## Neden 4 Model Gerekiyor?

Yarışma 4 ayrı test seti ile değerlendirme yapacak:

| Model | Amaç | Eğitim Verisi | Ana Zorluk |
|-------|------|---------------|-----------|
| Model 1 | Genel varyant sınıflandırma | MASTER (n=2931) | Yüksek eksik veri |
| Model 2 | Kalıtsal Kanser paneli | KANSER (n=388) | Farklı eksiklik deseni |
| Model 3 | Fenilketonüri (PAH) paneli | PAH (n=372) | 5:1 dengesizlik |
| Model 4 | Kistik Fibrozis (CFTR) paneli | CFTR (n=111) | Çok küçük veri |

**Önerilen yaklaşım**: MASTER üzerinde eğitilmiş **tek global model** + her panel için **ayrı eşik değeri kalibrasyonu**. CFTR gibi küçük paneller için bağımsız model eğitmek riskli.

---

## En Güçlü Özellikler

EK özellikleri (özellikle EK_7, EK_9, EK_4, EK_2) en güçlü tahmin ediciler. Ancak EK_4 ve EK_6'nın ClinVar'dan türetilmiş olma riski var (döngüsel tahmin / circularity).

Temel aksiyon: EK_4/5/6 olmadan da model eğitip performans farkını ölçün.

---

## Önerilen Sonraki Adımlar

1. **EK ablasyonu**: EK_4/5/6 olmadan model eğitin, performans farkını ölçün
2. **F1 optimizasyonu**: Dengeli test seti için eşik değerlerini yeniden kalibre edin
3. **Ensemble**: LightGBM + XGBoost + CatBoost yığınlama (stacking) deneyin
4. **Panel kalibrasyonu**: Her panel için ayrı eşik değeri belirleyin
5. **Hiperparametre optimizasyonu**: Optuna ile Bayesian arama
6. **Hata analizi**: Yanlış sınıflandırılan varyantları inceleyin
7. **Dokümantasyon**: Proje Detay Raporu yazmaya başlayın (son teslim: 29.06.2026)

---

## Mevcut Baseline Sonucu

Pipeline B (LightGBM, missingness-aware):

| Metrik | Değer |
|--------|-------|
| ROC-AUC | 0.846 |
| F1 Score | 0.860 |
| CFTR ROC-AUC | 0.940 |
| KANSER ROC-AUC | 0.914 |
| PAH ROC-AUC | 0.772 |

PAH en zayıf panel — 5:1 dengesizlik ve yüksek eksik veri nedeniyle.

---

*Detaylı teknik rapor için: `reports/teknofest_data_analysis_report.md`*
