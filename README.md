# Akademik Makalelerin Araştırma Alanlarına Göre Kümelenmesi

Bu proje, ArXiv veritabanından toplanan akademik makaleleri içeriklerine göre otomatik olarak kümelere ayırmak ve görselleştirmek için **PySpark** kullanır.

## 🎯 Proje Amacı

- ArXiv'den akademik makaleleri toplamak
- Makaleleri içeriklerine göre otomatik kümelere ayırmak  
- Kümeleri görselleştirmek ve analiz etmek
- Araştırma alanlarını keşfetmek
- Büyük veri işleme için PySpark'ın gücünden yararlanmak

## 🛠️ Teknolojiler

- **PySpark**: Büyük veri işleme ve makine öğrenmesi
- **ArXiv API**: Akademik makale verisi toplama
- **Python**: Ana programlama dili
- **Matplotlib/Plotly/Seaborn**: Görselleştirme
- **NLTK**: Doğal dil işleme
- **Pandas**: Veri manipülasyonu
- **WordCloud**: Kelime bulutları

## 📁 Proje Yapısı

```
BigData/
├── src/                           # Kaynak kodlar
│   ├── arxiv_data_collector.py   # ArXiv veri toplama
│   └── spark_clustering.py       # PySpark kümeleme
├── data/                         # Veri dosyaları
├── visualizations/              # Görselleştirmeler  
├── main.py                      # Ana çalıştırma scripti
├── requirements.txt             # Python bağımlılıkları
└── README.md                    # Bu dosya
```

## 🚀 Kurulum

### 1. Gereksinimleri Yükleyin

```bash
# Python paketlerini yükle
pip install -r requirements.txt
```

4. **Gerekli dizinleri oluşturun:**
```bash
mkdir -p data visualizations logs
```

### 🧪 Kurulum Testi

```bash
# Virtual environment'ı aktifleştirin
source academic_env/bin/activate
```

## 📈 Kullanım

### ⚠️ Önemli: Her kullanımdan önce

```bash
# Virtual environment'ı aktifleştirin
source academic_env/bin/activate
```

### 📊 Tam Pipeline

Tüm süreci (veri toplama + kümeleme) çalıştırmak için:

```bash
# Tam pipeline (1000 makale ile)
python main.py --full-pipeline --max-results 1000

# Sadece veri toplama
python main.py --collect-data --max-results 500

# Sadece kümeleme (varolan veri ile)
python main.py --cluster --data-file data/arxiv_papers.csv
```

### ⚙️ Gelişmiş Parametreler

```bash
# Büyük veri seti ile (5000 makale)
python main.py --full-pipeline --max-results 5000 --vocab-size 10000

# Özel veri dosyası ile kümeleme
python main.py --cluster --data-file my_data.csv --vocab-size 3000
```

## 📊 Özellikler

### 🔍 Veri Toplama
- ArXiv API'sinden otomatik veri toplama
- Çoklu kategori desteği
- Metin temizleme ve ön işleme
- Duplicate detection

### 🧠 Makine Öğrenmesi (PySpark)
- **TF-IDF** özellik çıkarma
- **K-means** kümeleme algoritması
- Otomatik optimal k bulma (Elbow method + Silhouette analysis)
- **Silhouette Score** ile model değerlendirme

### 📊 Görselleştirme
- Küme boyutları (pasta grafiği)
- Kategori dağılımları (bar grafiği)
- Küme-kategori ilişki matrisi (heatmap)
- Her küme için kelime bulutları
- Optimal k analizi grafikleri

### 📋 Analiz Çıktıları
- Detaylı küme analizi raporu
- Her küme için anahtar kelimeler
- Kategori dağılımları
- Örnek makale başlıkları

## 🔧 ArXiv Kategoriler

Projede varsayılan olarak şu ArXiv kategorileri kullanılır:

- `cs.AI` - Artificial Intelligence
- `cs.ML` - Machine Learning  
- `cs.CV` - Computer Vision
- `cs.CL` - Natural Language Processing
- `cs.LG` - Learning
- `stat.ML` - Machine Learning (Statistics)
- `physics.data-an` - Data Analysis
- `q-bio.QM` - Quantitative Methods
- `econ.EM` - Econometrics
- `math.ST` - Statistics Theory

## 📁 Çıktı Dosyaları

Proje çalıştıktan sonra şu dosyalar oluşur:

```
├── data/
│   ├── arxiv_papers.csv          # Ham ArXiv verileri
│   └── clustered_papers.csv      # Kümelenmiş veriler
├── visualizations/
│   ├── cluster_sizes.html        # İnteraktif pasta grafiği  
│   ├── category_distribution.png # Kategori dağılımı
│   ├── cluster_category_heatmap.png # Küme-kategori ilişkisi
│   ├── cluster_wordclouds.png    # Kelime bulutları
│   └── optimal_k_analysis.png    # Optimal k analizi
└── akademik_makaleler_raporu.txt # Detaylı rapor
```

## 🎯 Örnek Kullanım Senaryoları

### 1. Araştırma Alanları Keşfetme
```bash
python main.py --full-pipeline --max-results 2000
```

### 2. Belirli Kategorilere Odaklanma
Kod içinde `categories` listesini değiştirerek istediğiniz kategorileri seçebilirsiniz.

### 3. Büyük Veri Analizi  
```bash
python main.py --full-pipeline --max-results 10000 --vocab-size 15000
```

## 🐛 Sorun Giderme

### Virtual Environment Sorunu
```bash
# Virtual environment'ı yeniden oluşturun
rm -rf academic_env
python3 -m venv academic_env
source academic_env/bin/activate
pip install -r requirements.txt
```

### Java Hatası
```bash
# Java 8+ yüklü olduğundan emin olun
sudo apt install openjdk-11-jdk  # Ubuntu için
```

### Memory Hatası
```bash
# Spark memory ayarları (büyük veri setleri için)
export SPARK_DRIVER_MEMORY=4g
export SPARK_EXECUTOR_MEMORY=4g
```

### ArXiv API Rate Limiting
Eğer çok hızlı istek gönderiyorsanız, `arxiv_data_collector.py` dosyasında `delay` parametresini artırın.

Sorunlarınız için aşağıdaki kontrol listesini takip edin:

- [ ] Python 3.7+ yüklü mü?
- [ ] Java 8+ yüklü mü?
- [ ] Virtual environment aktif mi?
- [ ] requirements.txt yüklendi mi?
- [ ] İnternet bağlantısı var mı? (ArXiv API için)# BigDataProject
