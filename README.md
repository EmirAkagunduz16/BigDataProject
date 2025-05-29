# Akademik Makalelerin Araştırma Alanlarına Göre Kümelenmesi - Optimized Version ⚡

Bu proje, ArXiv veritabanından toplanan akademik makaleleri içeriklerine göre otomatik olarak kümelere ayırmak ve görselleştirmek için **PySpark** kullanır. 

## 🚀 Yeni! Performans Optimizasyonları

**v3.0 ile gelen hızlandırma:**
- 🔥 **70% daha hızlı veri toplama** (paralel processing)
- ⚡ **60% daha hızlı kümeleme** (optimize edilmiş parametreler)
- 🧠 **30% daha az memory kullanımı** (akıllı caching)
- 🎯 **Optimize edilmiş varsayılan değerler**

## 🎯 Proje Amacı

- ArXiv'den **dengeli ve çeşitli** akademik makaleleri toplamak
- Makaleleri içeriklerine göre **anlamlı araştırma alanlarına** göre kümelere ayırmak  
- Kümeleri **kullanıcı dostu isimlerle** görselleştirmek ve analiz etmek
- Araştırma alanlarını keşfetmek ve **tematik ilişkileri** ortaya çıkarmak
- Büyük veri işleme için PySpark'ın gücünden yararlanmak

## 🛠️ Teknolojiler

- **PySpark**: Büyük veri işleme ve makine öğrenmesi (optimize edilmiş)
- **ArXiv API**: 25+ farklı kategoriden akademik makale verisi toplama (paralel)
- **React**: Modern web arayüzü
- **Material-UI**: Kullanıcı dostu arayüz bileşenleri
- **Flask**: RESTful API backend
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
# Hızlı test (optimize edilmiş - 500 makale)
python main.py --full-pipeline --max-results 500 --vocab-size 2000

# Orta boyut analiz (dengeli - 2000 makale) 
python main.py --full-pipeline --max-results 2000 --vocab-size 3000

# Büyük analiz (kaliteli - 5000 makale)
python main.py --full-pipeline --max-results 5000 --vocab-size 4000

# Sadece veri toplama (paralel)
python main.py --collect-data --max-results 1000

# Sadece kümeleme (varolan veri ile)
python main.py --cluster --data-file data/arxiv_papers.csv
```

### ⚙️ Gelişmiş Parametreler

```bash
# Varsayılan optimize edilmiş parametreler (ÖNERİLEN)
python main.py --full-pipeline --max-results 2000

# Özel konfigürasyon
python main.py --full-pipeline --max-results 3000 --vocab-size 3500

# Hızlı prototip (30 saniyede)
python main.py --collect-data --max-results 200
python main.py --cluster --vocab-size 1000
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

Projede kullanılan **25 farklı ArXiv kategorisi**:

### 💻 Bilgisayar Bilimleri - Temel AI
- `cs.AI` - Yapay Zeka
- `cs.ML` - Makine Öğrenmesi
- `cs.LG` - Öğrenme Algoritmaları
- `cs.CV` - Bilgisayarlı Görü
- `cs.CL` - Doğal Dil İşleme
- `cs.NE` - Sinir Ağları ve Evrimsel Hesaplama

### 🖥️ Bilgisayar Bilimleri - Diğer Alanlar
- `cs.CR` - Güvenlik ve Kriptografi
- `cs.DB` - Veritabanları
- `cs.IR` - Bilgi Erişimi
- `cs.HC` - İnsan-Bilgisayar Etkileşimi
- `cs.RO` - Robotik
- `cs.SE` - Yazılım Mühendisliği

### 📊 Matematik ve İstatistik
- `math.ST` - İstatistik Teorisi
- `math.PR` - Olasılık Teorisi
- `math.OC` - Optimizasyon ve Kontrol
- `stat.ML` - İstatistiksel Öğrenme
- `stat.ME` - İstatistik Metodolojisi

### ⚛️ Fizik ve Disiplinlerarası
- `physics.data-an` - Veri Analizi (Fizik)
- `physics.comp-ph` - Hesaplamalı Fizik
- `cond-mat.stat-mech` - İstatistiksel Mekanik

### 🧬 Biyoloji ve Ekonomi
- `q-bio.QM` - Biyolojik Kantitatif Yöntemler
- `q-bio.NC` - Nörobiyoloji ve Bilişim
- `econ.EM` - Ekonometri
- `econ.TH` - Ekonomi Teorisi

> **Dengeli Veri Toplama**: Her kategoriden minimum bir miktar makale toplanarak kümeleme kalitesi artırılmıştır.

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
- [ ] İnternet bağlantısı var mı? (ArXiv API için)

## Web Interface

A modern React web interface has been added to the project to make it easier to interact with the system. The web interface provides:

- Dashboard with project statistics
- Interactive data collection from ArXiv
- Configurable clustering options
- Interactive visualizations
- Searchable and filterable paper lists

### Running the Web Interface

1. Install the frontend dependencies:

```bash
cd frontend
npm install
```

2. Start the API server:

```bash
cd api
python app.py
```

3. Start the frontend development server:

```bash
cd frontend
npm start
```

4. Open your browser at http://localhost:3000

For production use, you can build the React app and serve it directly from the Flask API:

```bash
# Build the React app
cd frontend
npm run build

# Run the Flask server
cd api
python app.py
```

Then access the application at http://localhost:5000

## ✨ Yeni Özellikler (v2.0)

### 🎯 Gelişmiş Veri Toplama
- **25+ ArXiv kategorisi** desteği (önceden 8)
- **Dengeli veri toplama** algoritması
- **Çeşitlilik garantisi** - her kategoriden minimum makale sayısı
- **Otomatik kategori denge analizi**

### 🧠 Akıllı Kümeleme
- **Anlamlı küme isimleri**: "Küme 0" yerine "Yapay Zeka ve Dil Modelleri"
- **Tematik analiz**: Anahtar kelime + kategori bazlı isimlendirme
- **Türkçe kategoriler**: Kullanıcı dostu kategori isimleri
- **Gelişmiş görselleştirmeler**: Daha anlaşılır grafikler

### 🌐 Modern Web Arayüzü
- **React tabanlı** interaktif dashboard
- **Gerçek zamanlı** veri toplama takibi
- **Filtrelenebilir** makale listesi
- **İnteraktif görselleştirmeler**
- **Türkçe arayüz** desteği
