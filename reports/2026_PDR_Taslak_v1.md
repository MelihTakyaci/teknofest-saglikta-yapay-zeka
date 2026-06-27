# 2026 TEKNOFEST Sağlıkta Yapay Zeka Yarışması — Proje Detay Raporu (Taslak v2)

**Kategori:** Üniversite ve Üzeri Seviyesi
**Görev:** Genetik varyantların Patojenik / Benign olarak ikili sınıflandırılması
**Birincil Metrik:** F1 Skoru
**Birleşik Operasyon Noktası:** Yığınlanmış Topluluk (Stacked Ensemble), eşik = 0.305
**Taslak Tarihi:** 27.06.2026
**Kaynak:** Faz 1–6 analiz, modelleme ve birleştirme raporları (`reports/` dizini)

> Tüm panel ve kohort metrikleri, iç tutarlılığı güvence altına almak için **tek bir operasyon noktasında** — yığınlanmış topluluğun kalibre edilmiş 0.305 eşiğinde — raporlanmıştır. Sayısal değerler `reports/phase_06_unification/phase_06_unified_results.json` dosyasından doğrudan alınmıştır.

---

## 1. Özet (Yönetici Özeti)

Bu çalışma, ACMG kriterleri ve ClinVar/ClinGen veri tabanları temel alınarak etiketlenmiş genetik varyantların patojenisitesini, anonimleştirilmiş 353 sütunluk bir öznitelik şeması üzerinden tahmin eden, klinik güvenliği önceleyen bir makine öğrenmesi sistemi sunmaktadır. Sistem; genel (MASTER) ve üç gen-spesifik panel (Kalıtsal Kanser, Fenilketonüri/PAH, Kistik Fibrozis/CFTR) üzerinden değerlendirilmektedir.

Yöntemsel çekirdek, MASTER üzerinde eğitilen **küresel bir gradyan-artırmalı ağaç topluluğudur** (LightGBM + XGBoost + CatBoost), bir **Lojistik Regresyon meta-öğrenicisi** ile yığınlanmıştır. Karar eşiği, dengeli (1:1) test dağılımı ve **2:1 klinik maliyet** varsayımı altında **yalnızca örneklem-dışı (out-of-fold) tahminler** üzerinde kalibre edilerek **0.305** olarak sabitlenmiştir. Tüm metrikler bu tek operasyon noktasında raporlanır.

Başlıca bulgular (eşik = 0.305):

- **Topluluk performansı:** Yığınlanmış model 5-katlı çapraz doğrulamada **ROC-AUC 0.8581** değerine ulaşarak tekil LightGBM temel modeline (0.8461) üstünlük sağlar; MASTER üzerinde dengeli F1 **0.7974**'tür.
- **Panel performansı (sızıntı-farkında):** CFTR **AUC 0.9545 / Dengeli F1 0.8811**, KANSER **AUC 0.9232 / Dengeli F1 0.8423**, PAH **AUC 0.7935 / Dengeli F1 0.7567**.
- **Veri sızıntısı savunması:** Baskın EK öznitelikleri (EK_4/5/6/7) çıkarıldığında ROC-AUC yalnızca **0.0112** düşer (0.8461 → 0.8349); bu, döngüsel sızıntıya karşı **düşük bağımlılığı** kanıtlar.
- **Ayrılmış bağımsız kohort:** Eğitimde hiç görülmemiş 293 varyant üzerinde topluluk **ROC-AUC 0.8934** ve **dengeli F1 0.8282** elde eder.
- **Açıklanabilirlik:** SHAP analizleri EK_7'yi baskın belirleyici olarak gösterir; hata analizi, kaçırılan patojenik varyantların (FN) anormal derecede düşük EK_7 sinyali taşıdığını ortaya koyar.

---

## 2. Problem Tanımı ve Veri Analizi

### 2.1 Problem Tanımı

Görev, her bir genetik varyantın **Patojenik (1)** veya **Benign (0)** olarak sınıflandırılmasıdır; değerlendirme TP/FP/FN üzerinden hesaplanan **F1 Skoru** ile yapılır. Yer gösterimsel koordinatlar kaldırılmış, öznitelik adları anonimleştirilmiş ve veri erişimi NDA ile sınırlandırılmıştır.

### 2.2 Veri Setleri ve Şema

| Veri Seti | Satır | Sütun | Amaç |
|-----------|-------|-------|------|
| MASTER (Genel) | 2931 | 353 | Genel varyant sınıflandırma |
| KANSER (Kalıtsal Kanser) | 388 | 353 | Gen paneli |
| PAH (Fenilketonüri) | 372 | 353 | Gen paneli |
| CFTR (Kistik Fibrozis) | 111 | 353 | Gen paneli |

Öznitelik grupları: `AL_1–AL_334` (334 sayısal skor), `EK_1–EK_9` (9 birleşik/meta-tahmin skoru), `CAT_1–CAT_6` (6 kategorik), `AA_1–AA_2` (2 amino asit), `Variant_ID` ve `Label`.

### 2.3 Veri Kalitesi: Eksiklik ve Dengesizlik

- **Aşırı eksik veri:** MASTER hücrelerinin **%54.9'u eksiktir**. Eksiklik rastgele değildir (MNAR): patojenik varyantlarda eksiklik **%59.9**, benign varyantlarda **%41.2**'dir; dolayısıyla eksikliğin kendisi etiket için bilgi taşır.
- **Sınıf dengesizliği:** Eğitim setlerinde patojenik sınıf baskındır (MASTER 2.75:1, KANSER 2.23:1, PAH 5.00:1, CFTR 4.29:1).
- **Eğitim/Test dağılım uyumsuzluğu:** Test setleri dengeli (1:1) olduğundan eşik, eğitim dağılımına göre değil dengeli dağılıma göre kalibre edilmiştir (bkz. Bölüm 3.3).
- **Sabit sütunlar:** MASTER'da 57 sütun tek değer taşır ve elenir.

---

## 3. Yöntem ve Mimari

### 3.1 Ön İşleme ve Eksiklik-Farkında Özellik Mühendisliği

MNAR yapısı nedeniyle **impütasyon uygulanmamış**, eksiklik bir sinyal olarak modellenmiştir:

- `Variant_ID` ve sabit sütunların elenmesi.
- Her sayısal öznitelik için ikili **eksiklik göstergesi** (`miss_*`).
- **Satır düzeyi** toplam eksiklik sayısı/oranı ve **grup düzeyi** (AL alt-blokları, EK) eksiklik sayaçları.
- Kategorik kodlama: CAT_3/4/5 one-hot, CAT_1/2 etiket kodlama, CAT_6 ikili bayrak; AA_1/AA_2 one-hot.
- Ağaç tabanlı modeller eksik değerleri doğal olarak işlediğinden NaN değerleri korunmuştur. SMOTE veya yapay örnekleme **kullanılmamıştır**.

İşlem hattı 655 öznitelikten oluşan bir matris üretir ve tüm fazlarda dondurulmuş kodlayıcı durumu (`preprocessor_state`) ile tutarlı biçimde uygulanır.

### 3.2 Model Mimarisi: Küresel Topluluk + Yığınlama

Küçük panellerde bağımsız model eğitmenin yüksek varyans riski nedeniyle, MASTER üzerinde eğitilen tek bir küresel model benimsenmiştir. Topluluk üç temel öğreniciden oluşur: **LightGBM** (`is_unbalance=True`), **XGBoost** (`scale_pos_weight`) ve **CatBoost** (`auto_class_weights='Balanced'`). Üç modelin örneklem-dışı (OOF) olasılıkları bir **Lojistik Regresyon meta-öğrenicisi** ile yığınlanır. Meta-öğrenici katsayıları (LightGBM 1.47, XGBoost 1.29, CatBoost 2.92) CatBoost'un en yüksek ağırlığa sahip olduğunu gösterir. Tüm aşamalarda sabit tohum (seed=42) ve özdeş 5-katlı StratifiedKFold bölünmeleri kullanılarak yeniden üretilebilirlik ve sızıntısız panel değerlendirmesi korunmuştur.

### 3.3 2:1 Klinik Maliyetli Örneklem-Dışı Eşik Kalibrasyonu ve Operasyon Noktasının Birleştirilmesi

Eğitim setleri dengesiz, test setleri dengeli (1:1) olduğundan eşik **yalnızca OOF tahminleri** üzerinde ve **1:1 öncül** varsayımıyla kalibre edilmiştir. Klinik bağlamda patojenik bir varyantı kaçırmanın (yanlış negatif) maliyeti yanlış alarmdan yüksek olduğundan, maliyet oranı **YN:YP = 2:1** alınmış ve beklenen maliyeti (`2·YN + 1·YP`) en aza indiren eşik seçilmiştir.

Raporun iç tutarlılığını güvence altına almak için, **dağıtılan nihai model olan yığınlanmış topluluğun kalibre eşiği 0.305** tüm panellerde ve kohortlarda **tek operasyon noktası** olarak benimsenmiştir. Önceki ara değerlendirmelerde tekil modeller için kullanılan farklı eşikler (ör. 0.365) nihai raporlamadan çıkarılmıştır; ROC-AUC ve PR-AUC zaten eşikten bağımsızdır.

---

## 4. Model Performansı ve Doğrulama

### 4.1 Birleşik Operasyon Noktası ve Topluluk Performansı

5-katlı çapraz doğrulamada OOF ROC-AUC değerleri:

| Model | OOF ROC-AUC |
|-------|-------------|
| LightGBM | 0.8461 |
| XGBoost | 0.8491 |
| CatBoost | 0.8538 |
| **Yığınlanmış Topluluk** | **0.8581** |

Yığınlanmış topluluk MASTER üzerinde 0.305 eşiğinde: ROC-AUC **0.8581**, PR-AUC 0.9302, dengeli F1 **0.7974**, duyarlılık 0.9055, özgüllük 0.6343, MCC 0.5602.

**Panel bazlı performans (sızıntı-farkında, yığınlanmış topluluk, eşik = 0.305):** Paylaşılan varyantlar için örneklem-dışı (OOF) olasılıklar, panele özgü (paylaşılmayan) varyantlar için dondurulmuş topluluk kullanılmıştır.

| Panel | n | ROC-AUC | Dengeli F1 | Duyarlılık | Özgüllük | MCC |
|-------|---|---------|------------|------------|----------|-----|
| CFTR | 111 | 0.9545 | 0.8811 | 0.9000 | 0.8571 | 0.6912 |
| KANSER | 388 | 0.9232 | 0.8423 | 0.9701 | 0.6667 | 0.7029 |
| PAH | 372 | 0.7935 | 0.7567 | 0.9129 | 0.5000 | 0.4242 |

Yığınlama, tekil LightGBM'e kıyasla her panelde ROC-AUC iyileşmesi sağlamıştır (CFTR 0.9402→0.9545, KANSER 0.9141→0.9232, PAH 0.7720→0.7935). Yüksek duyarlılık değerleri, seçilen 2:1 klinik maliyet politikasının doğrudan ve amaçlanan sonucudur.

### 4.2 Yanlış Pozitif ve Yanlış Negatif Sonuçların Klinik Analizi

0.305 operasyon noktasında MASTER örneklem-dışı tahminleri üzerinde sistematik hata profillemesi yapılmıştır. Toplam **286 yanlış pozitif (YP)** ve **203 yanlış negatif (YN)** gözlenmiştir (YN:YP = 0.71). YP sayısının YN'yi aşması, modelin 2:1 maliyet politikası gereği **bilinçli olarak duyarlılık lehine** yanlılaştırıldığını ve klinik açıdan daha kritik olan kaçırılan-patojenik hatasını en aza indirdiğini doğrular.

Hataların başlıca **biyolojik sürücüsü, baskın EK_7 sinyalindeki çelişkidir.** Doğru sınıflandırılan varyantlarda ortalama EK_7 değeri 6.093 iken, kaçırılan patojenik varyantlarda (YN) bu değer **3.131'e** kadar düşmektedir; yani model, atipik biçimde düşük birleşik hasar skoru taşıyan gerçek patojenik varyantları kaçırma eğilimindedir. Yanlış pozitiflerde ise EK_7 ortalaması (5.793) doğru sınıfa yakındır; bu, yükselmiş hasar skoru taşıyan benign varyantların yanlış alarma yol açtığını gösterir.

**İstatistiksel sürücü olarak eksiklik:** Hatalar, satır düzeyi eksiklik bandlarına göre çift-modlu (bimodal) dağılır. En yüksek mutlak hata ve yanlış negatif yoğunluğu, **%75–100 eksiklik bandında** görülür (n=1321; 214 hata, 102 YN); seyrek anote edilmiş varyantlarda sinyal yetersizliği patojenik kaçırmalarını artırmaktadır. Buna karşın iyi anote edilmiş %0–25 bandında hata oranı yüksektir (0.203) ancak YP/YN dengelidir (97 YP / 90 YN); bu bandın hataları, eksiklikten değil sınır-değer (borderline) varyantların doğasından kaynaklanır. PAH panelinin zorluğu MASTER'a örtüşen varyantlarda değil (örtüşen hata oranı 0.1216, örtüşmeyene göre 0.1712'den düşük), panele özgü varyantlardaki düşük özgüllükte (0.5000) tezahür etmektedir.

### 4.3 EK Ablasyonu ile Veri Sızıntısı Savunması (Pipeline C)

EK öznitelikleri en güçlü tahmin edicilerdir ancak ClinVar üzerinde eğitilmiş meta-tahmin ediciler olma ve döngüsel sızıntı içerme riski taşırlar. Bu riski ölçmek için temel LightGBM öğrenicisinde EK_4/5/6/7 (ve ilgili eksiklik göstergeleri) çıkarılmıştır:

| Yapılandırma | OOF ROC-AUC |
|--------------|-------------|
| Temel (tüm öznitelikler) | 0.8461 |
| Ablasyon (EK_4/5/6/7 yok) | 0.8349 |
| **Fark (Δ)** | **−0.0112** |

0.02'nin altındaki bu düşüş, modelin baskın EK öznitelikilerine **düşük bağımlılık** gösterdiğini ve olası etiket sızıntısına karşı dayanıklı olduğunu kanıtlar.

### 4.4 Ayrılmış Bağımsız Kohort Testi (Dışsal Geçerlilik Simülasyonu)

Anonimleştirme nedeniyle gerçek anlamda bağımsız bir dış veri seti oluşturulamadığından, **kesin biçimde izole edilmiş bir ayrılmış kohort** ile dışsal geçerlilik simülasyonu yapılmıştır. Temel modeller yalnızca MASTER üzerinde eğitildiğinden, panel dosyalarında bulunup MASTER'da bulunmayan **293 varyant** eğitimde hiç görülmemiştir; izolasyon `Variant_ID` üzerinden kod içinde doğrulanmıştır (0 sızıntı). Dondurulmuş `inference_pipeline.pkl` bu kohorta 0.305 eşiğinde uygulanmıştır (yeniden eğitim yoktur):

| Kohort | n | ROC-AUC | PR-AUC | Dengeli F1 | Duyarlılık | Özgüllük | MCC |
|--------|---|---------|--------|------------|------------|----------|-----|
| **Birleşik** | 293 | **0.8934** | 0.8557 | **0.8282** | 0.9538 | 0.6503 | 0.6177 |
| CFTR | 34 | 0.9619 | 0.9640 | 0.8235 | 0.8235 | 0.8235 | 0.6471 |
| KANSER | 142 | 0.9416 | 0.8848 | 0.8589 | 0.9778 | 0.7010 | 0.6319 |
| PAH | 117 | 0.8187 | 0.8272 | 0.7825 | 0.9706 | 0.4898 | 0.5463 |

Görülmemiş varyantlar üzerinde elde edilen ROC-AUC 0.8934 değeri, modelin eğitim örneklerini ezberlemek yerine **genelleştirilebilir bir sinyal** öğrendiğine dair kanıt sunar. Bu sonucun, bağımsız bir dış veri kaynağı değil, aynı anonim öznitelik çıkarım hattından gelen bir **ayrılmış kohort** üzerinde elde edildiği özellikle vurgulanır (bkz. Bölüm 6).

---

## 5. Açıklanabilirlik (SHAP)

Modelin karar mekanizması dondurulmuş LightGBM üzerinde **SHAP TreeExplainer** ile incelenmiş; hem küresel hem yerel açıklamalar üretilmiştir.

- **Küresel önem (ortalama |SHAP|) ilk öznitelikler:** EK_7, AL_327, AA_1_R, EK_2, AL_1, AL_16, EK_6, EK_9, AL_311, AL_319.
- EK_7'nin baskın belirleyici olması, bu öznitelğin birleşik bir patojenisite skoru olduğu yönündeki bulgularla ve Bölüm 4.2'deki hata analiziyle (düşük EK_7 → kaçırılan patojenik) tutarlıdır.
- **Yerel açıklamalar:** En yüksek güvenli YP ve YN vakaları için öznitelik katkı grafikleri üretilmiş ve `reports/phase_04_optimization/figures/` dizinine kaydedilmiştir (`shap_summary_beeswarm.png`, `shap_summary_bar.png`, `shap_local_false_positive.png`, `shap_local_false_negative.png`).

---

## 6. Tartışma ve Kısıtlar

**Güçlü yönler.** Sistem; eksiklik-farkında özellik mühendisliği, dengeli dağılıma ve klinik maliyete duyarlı örneklem-dışı eşik kalibrasyonu, çok modelli yığınlama, kanıtlanmış sızıntı izolasyonu ve tek bir tutarlı operasyon noktası (0.305) ile uçtan uca yeniden üretilebilir (seed=42) bir mimariye sahiptir. EK ablasyonu, sistematik hata analizi ve ayrılmış kohort testi sonuçların savunulabilirliğini güçlendirir.

**Kısıtlar ve dürüst çerçeveleme:**

- **Dışsal geçerlilik simülasyonu gerçek anlamda bağımsız değildir.** Ayrılmış kohort aynı anonim öznitelik çıkarım hattından gelir; eğitim/değerlendirme izolasyonu kesindir ancak bu bir *örneklem-dışı* tahmindir, bağımsız bir dış veri kaynağı değildir.
- **Ayrılmış kohort ROC-AUC'sinin (0.8934) iç OOF değerinden (0.8581) yüksek olması**, modelin "eğitimden daha iyi" olduğu biçiminde yorumlanmamalıdır; bunun nedeni görülmemiş CFTR ve KANSER varyantlarının daha ayrıştırılabilir olmasıdır. Gerçekten zor olan PAH paneli, ayrılmış kohortta da 0.8187 ile en düşük değeri verir.
- **EK döngüselliği tamamen dışlanamaz.** Ablasyon düşük bağımlılık gösterse de anonim öznitelilerin kaynağı doğrulanamadığından belirsizlik korunur.
- **CFTR örneklem büyüklüğü küçüktür** (n=111; ayrılmış alt küme n=34); metrikleri yüksek varyans taşır.
- **PAH paneli** düşük özgüllük (0.5000) ile en zayıf performansı verir; bu, 5:1 dengesizlik ve yüksek eksiklikten kaynaklanır ve gelecekteki iyileştirmelerin öncelikli hedefidir.

**Gelecek çalışmalar.** Optuna ile hiperparametre optimizasyonu, panel-bazlı (leave-one-panel-out) eşik kalibrasyonu ve yanlış sınıflandırılan varyantların daha derin hata analizi mevcut altyapı üzerinde doğrudan uygulanabilir.

---

*Bu taslak `reports/gap_analysis_report.md`, `reports/phase_04_optimization/`, `reports/phase_05_external_validation.md` ve `reports/phase_06_unification/phase_06_unified_results.json` dosyalarındaki metriklere dayanılarak hazırlanmıştır.*
