# Yapay Zekâ Destekli Otonom Kan Hücresi Analiz Sistemi

Periferik kan yayması mikroskop görüntülerinden hücreleri otomatik tespit eden, sayan ve patolojik olarak sınıflandıran derin öğrenme tabanlı bir yardımcı tanı sistemi. Nesne tespiti (YOLOv8) ile görüntü sınıflandırmayı (EfficientNet-B3) birleştiren **hibrit bir pipeline** kullanır; Sıtma, Orak Hücreli Anemi ve Akut Lenfoblastik Lösemi dâhil **4 hastalık/durum sınıfı** ile **11 beyaz kan hücresi tipi** olmak üzere toplam **15 sınıfı** otonom olarak tarar.

> **Not:** Bu sistem araştırma ve eğitim amaçlıdır; klinik teşhis için onaylanmış bir tıbbi cihaz değildir. Çıktılar uzman hekim değerlendirmesinin yerine geçmez.

## Özellikler

- Tam saha mikroskop görüntüsünden tek tek hücrelerin otomatik tespiti ve sayımı
- Tespit edilen her hücrenin 15 sınıflı patolojik sınıflandırması
- Çoklu hücre oylamasıyla görüntü düzeyinde nihai tanı ve klinik yorum
- Grad-CAM ile karar açıklanabilirliği (modelin hangi bölgeye baktığının görselleştirilmesi)
- Gradio tabanlı etkileşimli demo arayüzü
- MLflow ile deney takibi

## Sistem Mimarisi

Pipeline iki aşamalı çalışır:

```
Ham mikroskop görüntüsü
        │
        ▼
[1] YOLOv8 ile hücre tespiti  (3 sınıf: RBC · WBC · Platelet)
        │
        ▼
   Her hücrenin kırpılması → 224×224 boyutlandırma
        │
        ▼
[2] EfficientNet-B3 ile 15 sınıflı sınıflandırma
        │
        ▼
   Çoklu hücre oylaması
        │
        ▼
   Final tanı + klinik yorum
```

Sınıflandırma katmanı, alt görevlere ayrılmış **uzman modellerden** oluşur (grup sınıflandırıcı + içerik / şekil / sayım uzmanları), böylece her hücre tipi/hastalık kendi en güçlü modeliyle değerlendirilir.

## Sınıflar (15)

**Hastalık / durum sınıfları (4):** Sıtma (Malaria) · Orak Hücreli Anemi (Sickle Cell) · Akut Lenfoblastik Lösemi (ALL) · Sağlıklı (Normal)

**Beyaz kan hücresi tipleri (11):** Lenfosit (LY) · Bazofil (BA) · Metamyelosit (MMY) · Promyelosit (PMY) · Myelosit (MY) · Nötrofil (SNE) · Platelet (PLT) · Eozinofil (EO) · Band Nötrofil (BNE) · Monosit (MO) · Eritroblast (ERB)

## Veri Setleri

| Veri Seti | Kaynak | Görüntü | Sınıf |
|---|---|---|---|
| NIH Malaria | Kaggle (iarunava) | 27.558 | 2 (Parazitli / Normal) |
| C-NMC Leukemia | Kaggle (andrewmvd) | ~12.000 | 2 (ALL / Normal) |
| Sickle Cell Uganda | Kaggle (florencetushabe) | ~600 | 2 (Orak / Normal) |
| PBC Peripheral | Kaggle (paultimothymooney) | ~17.000 | 8 WBC tipi |
| Blood11 | Kaggle | 26.534 | 11 WBC tipi |
| BCCD | (YOLOv8 hücre tespiti için) | — | 3 (RBC / WBC / Platelet) |

## Sonuçlar

| Metrik | Değer |
|---|---|
| YOLOv8 hücre tespiti — mAP50 | **0,917** |
| Genel sınıflandırma doğruluğu (15 sınıf) | **%98,28** |
| Hastalık alt kümesi doğruluğu (4 sınıf) | %98,28 |
| Blood11 alt kümesi doğruluğu | %95,66 |
| Makro F1-Skoru (hastalık) | %99 |
| Makro Precision / Recall (hastalık) | %99 / %98 |
| Orak Hücreli Anemi & Lösemi — bireysel F1 | %100 |

Karşılaştırma: EfficientNet-B3 (%98,28) aynı koşullarda ResNet-50'yi (%95,36) geçmiştir.

## Teknolojiler

`Python` · `PyTorch` · `Ultralytics (YOLOv8)` · `timm (EfficientNet-B3)` · `OpenCV` · `scikit-learn` · `Grad-CAM (pytorch-grad-cam)` · `Gradio` · `MLflow` · `Roboflow` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Eğitim Yapılandırması

- YOLOv8: 50 epoch, 640×640 giriş çözünürlüğü, batch size 16, BCCD veri seti üzerinde
- EfficientNet-B3: ImageNet ağırlıklarıyla transfer öğrenme, kırpılmış 224×224 hücre görüntüleri üzerinde

## Kurulum

```bash
git clone https://github.com/kpsyusuf/<repo-adi>.git
cd <repo-adi>

pip install -r requirements.txt
# veya doğrudan:
pip install ultralytics timm torch torchvision opencv-python-headless \
            scikit-learn pytorch-grad-cam gradio mlflow pandas numpy \
            matplotlib seaborn pyyaml
```

## Kullanım

> Aşağıdaki yollar/komutlar örnek niteliğindedir; kendi dosya adlarına göre güncelleyin.

```bash
# Etkileşimli demo arayüzünü başlat
python app.py        # Gradio arayüzü
```

Yapılandırma dosyaları `config/` altında bulunur (`config.yaml`, `diseases.yaml`, `paths.yaml`).

## Proje Yapısı

```
.
├── blood_cell_ai_setup.ipynb     # Uçtan uca eğitim/çıkarım notebook'u
├── config/
│   ├── config.yaml
│   ├── diseases.yaml
│   └── paths.yaml
├── weights/                      # Eğitilmiş model ağırlıkları (repo'da değil — ayrıca indirilir)
│   ├── yolov8_cells/
│   ├── group_classifier/
│   ├── content_expert/           # malaria
│   ├── shape_expert/             # sickle cell
│   └── count_expert/             # blood11 / pbc / leukemia
└── README.md
```

> **Veri ve ağırlıklar:** Eğitim veri setleri ve model ağırlıkları boyutları nedeniyle bu depoya dâhil edilmemiştir. Veri setlerine yukarıdaki "Veri Setleri" tablosundaki kaynaklardan, ağırlıklara ise [buraya bir indirme linki ekleyin] üzerinden ulaşabilirsiniz.

## Yazar

**Yusuf Kasap** — Muş Alparslan Üniversitesi, Yazılım Mühendisliği
kspyusuf.00@gmail.com · [github.com/kpsyusuf](https://github.com/kpsyusuf)
