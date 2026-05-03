# DRONEQUBE Tarımsal Görüntü İşleme Görev Havuzu

Bu repository, DRONEQUBE "Aday Görev Havuzu" kapsamında **Görev 3 (Yeni Ekim Tespiti)** ve **Görev 4 (Yabancı Ot Tespiti)** için geliştirilen derin öğrenme ve görüntü işleme çözümlerini içermektedir.

Ayrıca iki görev birleştirilerek ekstra bir modül olarak **Tarımsal Risk Analizi** (mahsulün yabancı ota olan uzaklığına göre risk haritalandırması) geliştirilmiştir.

## 📌 Proje Kapsamı ve Çıktılar
* **Görev 3:** Mısır, fasulye ve pırasa fidelerinin tespiti ve DBSCAN ile tarladaki konumlarının kümelenmesi (mAP@50: %79.2).
* **Görev 4:** Farklı veri setleri birleştirilerek eğitilmiş genel yabancı ot (weed) tespiti ve yoğunluk analizi (mAP@50: %85.7).
* **Ekstra:** Tespit edilen kültür bitkileri ile yabancı otlar arasındaki mesafeye dayalı entegre risk analizi.

Detaylı teknik mimari, veri seti seçim gerekçeleri ve mühendislik varsayımları için proje dizinindeki `docs/Droneqube_Teknik_Rapor.pdf` dosyasını inceleyebilirsiniz.

## 📂 Klasör Yapısı (Modüler Mimari)
\`\`\`text
Weed_Crop_Detection/
├── crop/                           # Görev 3: Mahsul eğitim ve analiz modülleri
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   └── analyze.py
├── weed/                           # Görev 4: Yabancı Ot eğitim ve analiz modülleri
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   └── analyze.py
├── docs/                           # Teknik Rapor (PDF)
├── sample_data/                    # Test görüntüleri ve eğitilmiş model ağırlıkları
├── field_combined_analyze.py       # Entegre tarımsal risk analizi scripti
├── main.py                         # Ana çalıştırıcı
├── finetune_train.py               # Model fine-tune scripti
├── resume_train.py                 # Eğitime kaldığı yerden devam etme
├── check_dataset.py                # Veri seti bütünlük kontrolü
└── environment.yml                 # Kurulum gereksinimleri (Conda)
\`\`\`

## ⚙️ Kurulum (Kurumsal Kullanım İçin)

Projedeki bağımlılıkları yüklemek ve izole bir ortam oluşturmak için Conda kullanılması önerilir:

\`\`\`bash
# Repoyu klonlayın
git clone https://github.com/Yaren04/Weed_Crop_Detection.git
cd Weed_Crop_Detection

# Ortamı oluşturun ve aktif edin
conda env create -f environment.yml
conda activate crop_detection
\`\`\`

## 🚀 Kullanım ve Test (Tak-Çalıştır)

Repoda bulunan test görüntüleri ile kodları anında çalıştırabilirsiniz. Model yolları dinamik hale getirilmiş olup, sistem ağırlıkları (`runs/detect/` veya belirlenen dizin altında) otomatik olarak bulmaktadır.

**1. Görev 3 (Ekin Tespiti ve Kümeleme) Analizi:**
\`\`\`bash
python crop/analyze.py --image tarla.jpg
\`\`\`

**2. Görev 4 (Yabancı Ot ve Yoğunluk) Analizi:**
\`\`\`bash
python weed/analyze.py --image tarla.jpg
\`\`\`

**3. Birleşik Tarımsal Risk Analizi:**
\`\`\`bash
python field_combined_analyze.py --image tarla.jpg 
\`\`\`

**4. Modelleri Sıfırdan Eğitmek İçin (Eğitim Pipeline'ları):**
\`\`\`bash
python crop/pipeline.py
python weed/pipeline.py --skip-extract
\`\`\`

*Tüm analiz scriptlerinin çıktıları (maskelenmiş görseller, ısı haritaları, DBSCAN kümeleri ve koordinatları/alanları içeren CSV dosyaları) ilgili çıktı klasörlerine kaydedilir.*

---
**Geliştirici:** Yaren | **Tarih:** Mayıs 2026
