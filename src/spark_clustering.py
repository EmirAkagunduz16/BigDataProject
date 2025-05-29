import os
from typing import List

# Spark kütüphaneleri - Büyük veri işleme için
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, HashingTF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf, regexp_replace, lower, trim, split, size, concat_ws
from pyspark.ml.evaluation import ClusteringEvaluator

# Veri analizi ve görselleştirme kütüphaneleri
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Doğal dil işleme kütüphaneleri
import nltk
from nltk.corpus import stopwords

class SparkTextClustering:
    """
    Spark Tabanlı Metin Kümeleme Sınıfı
    
    Bu sınıf akademik makalelerin büyük ölçekli kümelenmesi için 
    Apache Spark teknolojisini kullanır.
    """
    
    def __init__(self, app_name: str = "AcademicPaperClustering"):
        """
        Spark Kümeleme Sistemini Başlat
        
        Args:
            app_name (str): Spark uygulamasının adı
        """
        # Spark oturumunu optimize edilmiş ayarlarla başlat
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.default.parallelism", "2") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.executor.memory", "1g") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.cores", "1") \
            .config("spark.driver.maxResultSize", "1g") \
            .getOrCreate()
        
        # Log seviyesini uyarı düzeyine ayarla (çok fazla log vermemesi için)
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # İngilizce stop words listesini indir ve hazırla
        try:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Proje dizin yapısını oluştur
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.base_dir)
        self.data_dir = os.path.join(self.project_root, 'data')
        self.viz_dir = os.path.join(self.project_root, 'visualizations')
        
        # Gerekli dizinleri oluştur
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def load_data(self, filepath: str = None):
        """
        CSV dosyasından ArXiv makale verilerini yükle
        
        Args:
            filepath (str): CSV dosya yolu (None ise varsayılan yol kullanılır)
            
        Returns:
            pyspark.sql.DataFrame: Yüklenen veri seti
        """
        # Varsayılan dosya yolunu belirle
        filepath = os.path.join(self.data_dir, 'arxiv_papers.csv')
        
        try:
            # Önce pandas ile oku (küçük/orta boyut dosyalar için)
            pandas_df = pd.read_csv(filepath)
            
            # Zorunlu sütunları kontrol et
            essential_columns = ['id', 'title', 'summary', 'primary_category', 'authors', 'published']
            existing_columns = [col for col in essential_columns if col in pandas_df.columns]
            
            # Sadece mevcut sütunları seç
            pandas_df = pandas_df[existing_columns]
            
            # Veri boyutuna göre optimum partition sayısını hesapla
            row_count = len(pandas_df)
            optimal_partitions = max(1, min(8, row_count // 1000))
            
            # Pandas DataFrame'i Spark DataFrame'e dönüştür ve önbelleğe al
            self.df = self.spark.createDataFrame(pandas_df).repartition(optimal_partitions)
            self.df.cache()  # Hızlı erişim için belleğe al
            
            return self.df
            
        except Exception as e:
            raise
    
    def preprocess_text(self, input_cols: List[str] = ['title', 'summary'], output_col: str = 'combined_text'):
        """
        Metin verilerini kümeleme için ön işle
        
        Bu fonksiyon şu adımları gerçekleştirir:
        1. Birden fazla sütunu birleştir
        2. Küçük harfe dönüştür
        3. Özel karakterleri temizle
        4. Fazla boşlukları kaldır
        
        Args:
            input_cols (List[str]): Birleştirilecek sütun adları
            output_col (str): Çıktı sütunu adı
            
        Returns:
            pyspark.sql.DataFrame: İşlenmiş veri seti
        """
        # Belirtilen sütunları boşlukla ayırarak birleştir
        self.df = self.df.withColumn(output_col, concat_ws(" ", *input_cols))
        
        # Tüm metni küçük harfe dönüştür
        self.df = self.df.withColumn(output_col, lower(col(output_col)))
        
        # Harf ve boşluk dışındaki karakterleri kaldır
        self.df = self.df.withColumn(output_col, regexp_replace(col(output_col), r'[^a-zA-Z\s]', ' '))
        
        # Birden fazla boşluğu tek boşlukla değiştir
        self.df = self.df.withColumn(output_col, regexp_replace(col(output_col), r'\s+', ' '))
        
        # Baştan ve sondan boşlukları kaldır
        self.df = self.df.withColumn(output_col, trim(col(output_col)))
        
        return self.df
    
    def create_features(self, text_col: str = 'combined_text', vocab_size: int = 5000, min_df: int = 1):
        """
        Metinlerden TF-IDF özellik vektörleri oluştur
        
        Bu fonksiyon makine öğrenmesi için sayısal özellikler üretir:
        1. Metni kelimelere böl (tokenization)
        2. Stop words'leri kaldır
        3. TF (Term Frequency) hesapla
        4. IDF (Inverse Document Frequency) hesapla
        5. TF-IDF matrisini oluştur
        
        Args:
            text_col (str): İşlenecek metin sütunu
            vocab_size (int): Maksimum kelime sayısı
            min_df (int): Minimum doküman frekansı
            
        Returns:
            pyspark.sql.DataFrame: TF-IDF özellikleri içeren veri seti
        """
        # Metni kelimelere böl (minimum 3 karakter)
        tokenizer = RegexTokenizer(inputCol=text_col, outputCol="words", pattern="\\W", minTokenLength=3)
        
        # Sadece temel stop words kullanıyoruz, domain-specific kelimeleri tutuyoruz
        # Akademik metinlerde önemli olan teknik terimler korunur
        basic_stop_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them'
        ]
        
        # Stop words'leri kaldır
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=basic_stop_words)
        
        # Hash tabanlı TF (Term Frequency) hesapla
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=vocab_size)
        
        # IDF (Inverse Document Frequency) hesapla
        idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=max(1, min_df))
        
        # Tüm adımları içeren pipeline oluştur
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
        
        # Pipeline'ı eğit ve veriyi dönüştür
        self.feature_model = pipeline.fit(self.df)
        self.df_features = self.feature_model.transform(self.df)
        self.df_features.cache()  # Performans için önbelleğe al
        
        # Gereksiz ara sütunları temizle (bellek tasarrufu)
        self.df_features = self.df_features.drop("words", "rawFeatures")
        
        return self.df_features
    
    def perform_clustering(self, k: int = 5, max_iterations: int = 100, seed: int = 123):
        """
        K-Means kümeleme algoritmasını uygula ve aykırı değerleri tespit et
        
        Bu fonksiyon şu adımları gerçekleştirir:
        1. K-Means modelini eğit
        2. Veriyi kümelere ata
        3. Her noktanın küme merkezine uzaklığını hesapla
        4. Aykırı değerleri tespit et ve özel kümeye ata
        5. Kümeleme kalitesini değerlendir
        
        Args:
            k (int): Küme sayısı
            max_iterations (int): Maksimum iterasyon sayısı
            seed (int): Rastgelelik için tohum değeri
            
        Returns:
            pyspark.sql.DataFrame: Küme atamaları içeren veri seti
        """
        # K-Means modelini yapılandır ve eğit
        kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k,
                       maxIter=max_iterations, seed=seed, tol=1e-4, initMode="k-means||")
        
        # Modeli eğit ve veriyi dönüştür
        self.kmeans_model = kmeans.fit(self.df_features)
        self.df_clustered = self.kmeans_model.transform(self.df_features)
        
        # Aykırı değer tespiti için küme merkezlerine uzaklık hesapla
        from pyspark.sql.functions import udf
        from pyspark.sql.types import DoubleType
        import numpy as np
        
        # Küme merkezlerini al
        centers = self.kmeans_model.clusterCenters()
        
        # Her noktanın atandığı küme merkezine uzaklığını hesaplayan UDF
        def calculate_distance_to_center(features, prediction):
            """Bir noktanın küme merkezine Öklid uzaklığını hesapla"""
            if features is None or prediction is None:
                return float('inf')
            try:
                center = centers[int(prediction)]
                features_array = np.array(features.toArray())
                distance = np.linalg.norm(features_array - center)
                return float(distance)
            except:
                return float('inf')
        
        # UDF'yi kaydet
        distance_udf = udf(calculate_distance_to_center, DoubleType())
        
        # Uzaklık sütununu ekle
        self.df_clustered = self.df_clustered.withColumn(
            "distance_to_center", 
            distance_udf(col("features"), col("cluster"))
        )
        
        # Aykırı değer tespiti için istatistikleri hesapla - Konservatif yaklaşım
        df_stats = self.df_clustered.select("distance_to_center").toPandas()
        distance_q75 = df_stats['distance_to_center'].quantile(0.75)  # 3. çeyrek
        distance_q25 = df_stats['distance_to_center'].quantile(0.25)  # 1. çeyrek
        iqr = distance_q75 - distance_q25  # Çeyrekler arası fark
        
        # IQR yöntemi ile aykırı değer eşiği (daha dengeli)
        outlier_threshold = distance_q75 + 1.5 * iqr
        
        # Alternatif: Yüzdelik tabanlı yaklaşım
        distance_p90 = df_stats['distance_to_center'].quantile(0.90)
        outlier_threshold = min(outlier_threshold, distance_p90)
        
        # Aykırı değerleri işaretle ve özel "Karma/Diğer" kümeye ata
        max_cluster_id = k
        self.df_clustered = self.df_clustered.withColumn(
            "is_outlier", 
            col("distance_to_center") > outlier_threshold
        )
        
        # Aykırı değer yüzdesini kontrol et ve çok fazlaysa ayarla
        outlier_count = self.df_clustered.filter(col("is_outlier") == True).count()
        total_count = self.df_clustered.count()
        outlier_percentage = (outlier_count / total_count) * 100
        
        # Eğer aykırı değerler %15'ten fazlaysa, daha sıkı eşik kullan
        if outlier_percentage > 15:
            distance_p95 = df_stats['distance_to_center'].quantile(0.95)
            outlier_threshold = distance_p95
            self.df_clustered = self.df_clustered.withColumn(
                "is_outlier", 
                col("distance_to_center") > outlier_threshold
            )
        
        # Aykırı değerleri yeni kümeye ata
        from pyspark.sql.functions import when
        self.df_clustered = self.df_clustered.withColumn(
            "cluster",
            when(col("is_outlier"), max_cluster_id).otherwise(col("cluster"))
        )
        
        # Performans için önbelleğe al
        self.df_clustered.cache()
        
        # Silhouette skoru ile kümeleme kalitesini değerlendir
        evaluator = ClusteringEvaluator(predictionCol="cluster", featuresCol="features",
                                      metricName="silhouette", distanceMeasure="squaredEuclidean")
        
        self.silhouette_score = evaluator.evaluate(self.df_clustered)
        
        # Küme denge metriklerini hesapla
        self._calculate_cluster_balance()
        
        return self.df_clustered
    
    def _calculate_cluster_balance(self):
        """
        Küme denge ve kalite metriklerini hesapla
        
        Bu fonksiyon kümelerin ne kadar dengeli dağıldığını analiz eder:
        - Küme boyutları arasındaki fark
        - Varyasyon katsayısı (CV)
        - Denge skoru (0-1 arası)
        """
        # Her kümedeki makale sayısını al
        cluster_counts = self.df_clustered.groupBy("cluster").count().collect()
        sizes = [row['count'] for row in cluster_counts]
        total_papers = sum(sizes)
        
        # Temel denge metrikleri
        max_size = max(sizes)          # En büyük küme boyutu
        min_size = min(sizes)          # En küçük küme boyutu
        mean_size = total_papers / len(sizes)  # Ortalama küme boyutu
        
        # Varyasyon katsayısını hesapla (CV) - düşük değer daha dengeli
        import numpy as np
        cv = np.std(sizes) / mean_size if mean_size > 0 else float('inf')
        
        # Denge skoru (0-1 arası, 1 mükemmel dengeli)
        balance_score = min_size / max_size if max_size > 0 else 0
        
        # Sonuçları kaydet
        self.cluster_balance = {
            'sizes': sizes,                    # Küme boyutları listesi
            'max_size': max_size,              # Maksimum küme boyutu
            'min_size': min_size,              # Minimum küme boyutu
            'mean_size': mean_size,            # Ortalama küme boyutu
            'coefficient_variation': cv,       # Varyasyon katsayısı
            'balance_score': balance_score,    # Denge skoru
            'is_balanced': cv < 0.5 and balance_score > 0.3  # Dengeli mi?
        }
    
    def find_optimal_k(self, k_range: range = range(2, 11), iterations: int = 50):
        """
        Optimum küme sayısını (k) bul
        
        Bu fonksiyon farklı k değerleri için kümeleme yapar ve
        kalite metriklerini karşılaştırarak en iyi k'yı seçer.
        
        Kullanılan metrikler:
        - Elbow yöntemi (maliyet azalması)
        - Silhouette skoru (küme kalitesi)
        - Denge skoru (küme boyut dengesi)
        - Kompozit skor (kalite + denge)
        
        Args:
            k_range (range): Test edilecek k değerleri
            iterations (int): Her k için iterasyon sayısı
            
        Returns:
            tuple: (optimal_k, costs, silhouette_scores)
        """
        # Metrik listelerini başlat
        costs = []              # Maliyet (WSSSE - Within Sum of Squared Errors)
        silhouette_scores = []  # Silhouette skoru
        balance_scores = []     # Denge skoru
        composite_scores = []   # Kompozit skor
        
        # Her k değeri için kümeleme yap
        for k in k_range:
            # K-Means modelini oluştur ve eğit
            kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, maxIter=iterations, seed=42)
            model = kmeans.fit(self.df_features)
            predictions = model.transform(self.df_features)
            
            # Maliyet (düşük daha iyi)
            cost = model.summary.trainingCost
            costs.append(cost)
            
            # Silhouette skoru (yüksek daha iyi, -1 ile 1 arası)
            evaluator = ClusteringEvaluator(predictionCol="cluster", featuresCol="features", metricName="silhouette")
            silhouette = evaluator.evaluate(predictions)
            silhouette_scores.append(silhouette)
            
            # Bu k için denge skorunu hesapla
            cluster_counts = predictions.groupBy("cluster").count().collect()
            sizes = [row['count'] for row in cluster_counts]
            max_size = max(sizes)
            min_size = min(sizes)
            balance_score = min_size / max_size if max_size > 0 else 0
            balance_scores.append(balance_score)
            
            # Kompozit skor: silhouette ve denge skorunun ağırlıklı kombinasyonu
            # Silhouette skorunu 0-1 aralığına normalize et (-1,1 aralığından)
            normalized_silhouette = (silhouette + 1) / 2
            composite_score = 0.7 * normalized_silhouette + 0.3 * balance_score
            composite_scores.append(composite_score)
        
        # Sonuçları görselleştir
        self._plot_elbow_curve(list(k_range), costs, silhouette_scores, balance_scores, composite_scores)
        
        # Kompozit skoruna göre optimal k'yı seç (kalite + denge)
        optimal_k = list(k_range)[np.argmax(composite_scores)]
        
        # Sonuçları yazdır
        print(f"K optimizasyon sonuçları:")
        for i, k in enumerate(k_range):
            print(f"K={k}: Silhouette={silhouette_scores[i]:.3f}, Denge={balance_scores[i]:.3f}, Kompozit={composite_scores[i]:.3f}")
        print(f"Seçilen optimal K: {optimal_k}")
        
        return optimal_k, costs, silhouette_scores
    
    def _plot_elbow_curve(self, k_values: List[int], costs: List[float], silhouette_scores: List[float], 
                         balance_scores: List[float], composite_scores: List[float]):
        """
        K optimizasyon sonuçlarını görselleştir
        
        4 farklı grafik oluşturur:
        1. Elbow Curve (Maliyet vs K)
        2. Silhouette Skoru vs K  
        3. Denge Skoru vs K
        4. Kompozit Skor vs K (en önemli)
        
        Args:
            k_values: K değerleri listesi
            costs: Maliyet değerleri
            silhouette_scores: Silhouette skorları
            balance_scores: Denge skorları  
            composite_scores: Kompozit skorlar
        """
        # 2x2 subplot düzeni oluştur
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Maliyet grafiği (Elbow Method)
        ax1.plot(k_values, costs, 'bo-')
        ax1.set_xlabel('K Değeri')
        ax1.set_ylabel('Maliyet (WSSSE)')
        ax1.set_title('Optimal K için Elbow Yöntemi')
        ax1.grid(True)
        
        # 2. Silhouette skoru grafiği
        ax2.plot(k_values, silhouette_scores, 'ro-')
        ax2.set_xlabel('K Değeri')
        ax2.set_ylabel('Silhouette Skoru')
        ax2.set_title('Silhouette Skoru vs K')
        ax2.grid(True)
        
        # 3. Denge skoru grafiği
        ax3.plot(k_values, balance_scores, 'go-')
        ax3.set_xlabel('K Değeri')
        ax3.set_ylabel('Denge Skoru')
        ax3.set_title('Küme Dengesi vs K')
        ax3.grid(True)
        
        # 4. Kompozit skor grafiği (en önemli)
        ax4.plot(k_values, composite_scores, 'mo-')
        ax4.set_xlabel('K Değeri')
        ax4.set_ylabel('Kompozit Skor')
        ax4.set_title('Kompozit Skor (Kalite + Denge)')
        ax4.grid(True)
        
        # Optimal k'yı vurgula
        optimal_k = k_values[np.argmax(composite_scores)]
        ax4.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax4.legend()
        
        # Grafikleri düzenle ve kaydet
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'optimal_k_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_cluster_names(self):
        """
        Kümeler için anlamlı Türkçe isimler üret
        
        Bu fonksiyon her kümenin içeriğini analiz ederek:
        1. En sık geçen kelimeleri belirler
        2. Dominant kategorileri tespit eder
        3. Kategori temalarını Türkçe karşılıklarla eşleştirir
        4. Akıllı isimlendirme kuralları uygular
        
        Returns:
            dict: Her küme için isim, tema ve anahtar kelimeler
        """
        cluster_names = {}
        
        for cluster_id, info in self.cluster_analysis.items():
            # Aykırı değer kümesini tespit et (en yüksek cluster_id)
            max_cluster_id = max(self.cluster_analysis.keys())
            if cluster_id == max_cluster_id and info.get('percentage', 0) < 15:  # Muhtemelen aykırı değer kümesi
                cluster_names[cluster_id] = {
                    'name': 'Karma/Diğer Alanlar', 
                    'theme': 'Karma', 
                    'keywords': 'Çeşitli Konular'
                }
                continue
            
            # En önemli kelimeleri ve kategorileri al
            top_words = list(info['top_words'].keys())[:3]
            top_categories = list(info['top_categories'].keys())[:2]
            
            # ArXiv kategorilerinin Türkçe karşılıkları
            category_themes = {
                'cs.AI': 'Yapay Zeka', 'cs.ML': 'Makine Öğrenmesi', 'cs.LG': 'Öğrenme Algoritmaları',
                'cs.CV': 'Bilgisayarlı Görü', 'cs.CL': 'Doğal Dil İşleme', 'cs.NE': 'Sinir Ağları',
                'cs.IR': 'Bilgi Erişimi', 'cs.RO': 'Robotik', 'cs.CR': 'Güvenlik ve Kriptografi',
                'cs.DB': 'Veritabanları', 'cs.SE': 'Yazılım Mühendisliği', 'cs.DS': 'Veri Yapıları ve Algoritmalar',
                'math.ST': 'İstatistik Teorisi', 'math.PR': 'Olasılık Teorisi', 'math.OC': 'Optimizasyon',
                'stat.ML': 'İstatistiksel Öğrenme', 'stat.ME': 'İstatistik Metodolojisi', 'stat.AP': 'Uygulamalı İstatistik',
                'physics.data-an': 'Veri Analizi (Fizik)', 'physics.comp-ph': 'Hesaplamalı Fizik',
                'cond-mat.stat-mech': 'İstatistiksel Mekanik', 'cond-mat.soft': 'Yumuşak Madde Fiziği',
                'q-bio.QM': 'Biyolojik Yöntemler', 'q-bio.NC': 'Sinirbilim', 'q-bio.CB': 'Hücre Biyolojisi',
                'physics.bio-ph': 'Biyofizik', 'physics.med-ph': 'Tıp Fiziği', 'physics.soc-ph': 'Sosyal Fizik',
                'econ.EM': 'Ekonometri', 'econ.TH': 'Ekonomi Teorisi'
            }
            
            # Ana kategori ve temayı belirle
            main_category = top_categories[0] if top_categories else 'Genel'
            theme = category_themes.get(main_category, main_category)
            keywords = ', '.join(top_words[:2]).title()
            
            # Akıllı isimlendirme kuralları
            # Çeşitli kategorilerin varlığı karma kümeyi işaret eder
            if len(set(top_categories)) >= 3:  # 2'den fazla farklı ana kategori
                cluster_name = f"Karma Araştırma Alanları ({keywords})"
            elif 'learning' in top_words or 'model' in top_words:
                if 'cs.CV' in top_categories:
                    cluster_name = f"{theme} ve Görü Modelleri"
                elif 'cs.CL' in top_categories or 'cs.NLP' in top_categories:
                    cluster_name = f"{theme} ve Dil Modelleri"
                else:
                    cluster_name = f"{theme} ve Öğrenme"
            elif 'data' in top_words or 'analysis' in top_words:
                cluster_name = f"Veri Analizi ({keywords})"
            elif 'neural' in top_words or 'network' in top_words:
                cluster_name = f"Sinir Ağları ({theme})"
            elif any(word in top_words for word in ['optimization', 'algorithm']):
                cluster_name = f"Algoritmalar ve Optimizasyon"
            elif any(word in top_words for word in ['statistical', 'probability']):
                cluster_name = f"İstatistiksel Yöntemler"
            else:
                cluster_name = f"{theme} ({keywords})"
            
            # Sonuçları kaydet
            cluster_names[cluster_id] = {'name': cluster_name, 'theme': theme, 'keywords': keywords}
        
        return cluster_names

    def analyze_clusters(self, top_words: int = 10):
        """
        Kümelerin detaylı analizini gerçekleştir
        
        Bu fonksiyon her küme için şu analizleri yapar:
        1. En sık kullanılan kelimeleri tespit eder
        2. Kategorilerin dağılımını hesaplar
        3. Küme homojenliğini değerlendirir
        4. Örnek makale başlıklarını seçer
        5. Küme kalite metriklerini hesaplar
        
        Args:
            top_words (int): Her küme için gösterilecek kelime sayısı
            
        Returns:
            dict: Detaylı küme analiz sonuçları
        """
        # Küme istatistiklerini al
        cluster_stats = self.df_clustered.groupBy("cluster").count().orderBy("cluster")
        available_columns = self.df_clustered.columns
        
        # Analiz için gerekli sütunları seç
        select_columns = ["cluster", "title", "summary", "primary_category"]
        if "combined_text" in available_columns:
            select_columns.append("combined_text")
        if "filtered_words" in available_columns:
            select_columns.append("filtered_words")
        if "distance_to_center" in available_columns:
            select_columns.append("distance_to_center")
        if "is_outlier" in available_columns:
            select_columns.append("is_outlier")
        
        # Spark DataFrame'i Pandas'a dönüştür (analiz için)
        df_pandas = self.df_clustered.select(*select_columns).toPandas()
        
        # Her küme için analiz sonuçlarını saklayacak sözlük
        self.cluster_analysis = {}
        
        # Her kümeyi ayrı ayrı analiz et
        for cluster_id in sorted(df_pandas['cluster'].unique()):
            cluster_data = df_pandas[df_pandas['cluster'] == cluster_id]
            
            # Kelime frekansı analizi için veri hazırla
            all_words = []
            
            # Farklı metin kaynaklarından kelimeleri topla
            if "filtered_words" in df_pandas.columns:
                # Önceden işlenmiş kelimeler varsa onları kullan
                for words_list in cluster_data['filtered_words']:
                    if isinstance(words_list, list):
                        all_words.extend(words_list)
                    elif isinstance(words_list, str):
                        words = words_list.split()
                        filtered = [w for w in words if len(w) > 3 and w.isalpha()]
                        all_words.extend(filtered)
            elif "combined_text" in df_pandas.columns:
                # Birleştirilmiş metin varsa onu işle
                for text in cluster_data['combined_text']:
                    if isinstance(text, str):
                        words = text.lower().split()
                        filtered = [w for w in words if len(w) > 3 and w.isalpha()]
                        all_words.extend(filtered)
            else:
                # Son çare olarak başlık ve özeti birleştir
                for idx, row in cluster_data.iterrows():
                    text = f"{row['title']} {row['summary']}".lower()
                    words = text.split()
                    filtered = [w for w in words if len(w) > 3 and w.isalpha()]
                    all_words.extend(filtered)
            
            # Kelime ve kategori frekanslarını hesapla
            word_freq = pd.Series(all_words).value_counts().head(top_words) if all_words else pd.Series(dtype=int)
            category_freq = cluster_data['primary_category'].value_counts().head(5)
            sample_titles = cluster_data['title'].head(3).tolist()
            
            # Küme homojenliğini hesapla (kategorilerin ne kadar benzer olduğu)
            category_counts = cluster_data['primary_category'].value_counts()
            dominant_category = category_counts.index[0] if len(category_counts) > 0 else 'Unknown'
            dominant_count = category_counts.iloc[0] if len(category_counts) > 0 else 0
            homogeneity = (dominant_count / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0
            
            # Kümeleme kalitesi metrikleri
            avg_distance = cluster_data['distance_to_center'].mean() if 'distance_to_center' in cluster_data.columns else 0
            outlier_count = cluster_data['is_outlier'].sum() if 'is_outlier' in cluster_data.columns else 0
            
            # Sonuçları kaydet
            self.cluster_analysis[cluster_id] = {
                'size': int(len(cluster_data)),                    # Küme boyutu
                'top_words': word_freq.to_dict(),                  # En sık kelimeler
                'top_categories': {k: int(v) for k, v in category_freq.to_dict().items()},  # En yaygın kategoriler
                'sample_titles': sample_titles,                    # Örnek başlıklar
                'percentage': float(len(cluster_data) / len(df_pandas) * 100),  # Küme yüzdesi
                'dominant_category': dominant_category,            # Baskın kategori
                'homogeneity': float(homogeneity),                # Homojenlik yüzdesi
                'avg_distance': float(avg_distance),              # Ortalama merkez uzaklığı
                'outlier_count': int(outlier_count),              # Aykırı değer sayısı
                'category_diversity': int(len(category_counts))    # Kategori çeşitliliği
            }
        
        # Küme isimlerini üret
        self.cluster_names = self._generate_cluster_names()
        
        return self.cluster_analysis
    
    def create_visualizations(self):
        """
        Kümeleme sonuçları için kapsamlı görselleştirmeler oluştur
        
        Bu fonksiyon şu görselleştirmeleri yapar:
        1. Küme dağılımı pasta grafiği
        2. Kategori dağılımı bar grafiği
        3. Küme-kategori ilişki ısı haritası
        4. Kelime bulutları
        
        Tüm grafikler Türkçe etiketler ve açıklamalar içerir.
        """
        # Görselleştirme için gerekli verileri hazırla
        df_pandas = self.df_clustered.select("cluster", "title", "summary", "primary_category").toPandas()
        
        # Küme isimlerini veri setine ekle
        df_pandas['cluster_name'] = df_pandas['cluster'].map(
            lambda x: self.cluster_names[x]['name'] if hasattr(self, 'cluster_names') else f"Küme {x}"
        )
        
        # Pasta grafiği için veri hazırla
        cluster_labels = []
        cluster_sizes = []
        cluster_info_for_analysis = {}
        
        for cluster_id, info in self.cluster_analysis.items():
            if hasattr(self, 'cluster_names'):
                label = f"{self.cluster_names[cluster_id]['name']}\n({info['size']} makale)"
                cluster_info_for_analysis[cluster_id] = {
                    'name': self.cluster_names[cluster_id]['name'],
                    'size': info['size'],
                    'percentage': info['percentage']
                }
            else:
                label = f"Küme {cluster_id}\n({info['size']} makale)"
                cluster_info_for_analysis[cluster_id] = {
                    'name': f"Küme {cluster_id}",
                    'size': info['size'],
                    'percentage': info['percentage']
                }
            cluster_labels.append(label)
            cluster_sizes.append(info['size'])

        # 1. Küme Dağılımı Pasta Grafiği
        fig = px.pie(values=cluster_sizes, names=cluster_labels, title="Araştırma Alanlarının Küme Dağılımı",
                    width=800, height=600)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        # En büyük küme hakkında analiz notu ekle
        largest_cluster = max(cluster_info_for_analysis.values(), key=lambda x: x['size'])
        largest_cluster_text = f"En büyük küme olan \"{largest_cluster['name']}\" makalelerin %{largest_cluster['percentage']:.1f}'ini içermektedir."
        
        fig.add_annotation(text=f"<b>Analiz:</b> {largest_cluster_text}",
                          xref="paper", yref="paper", x=0.5, y=-0.1, showarrow=False,
                          font=dict(size=12), align="center")
        
        # Grafiği kaydet ve göster
        fig.write_html(os.path.join(self.viz_dir, 'cluster_sizes.html'))
        fig.show()
        
        # 2. Kategori Dağılımı Bar Grafiği
        plt.figure(figsize=(16, 12))
        category_counts = df_pandas['primary_category'].value_counts().head(25)
        
        # ArXiv kategorilerinin Türkçe karşılıkları
        category_turkish = {
            'cs.AI': 'Yapay Zeka', 'cs.ML': 'Makine Öğrenmesi', 'cs.LG': 'Öğrenme Algoritmaları',
            'cs.CV': 'Bilgisayarlı Görü', 'cs.CL': 'Doğal Dil İşleme', 'cs.NE': 'Sinir Ağları',
            'cs.IR': 'Bilgi Erişimi', 'cs.RO': 'Robotik', 'cs.HC': 'İnsan-Bilgisayar Etkileşimi',
            'cs.CR': 'Güvenlik & Kriptografi', 'cs.DB': 'Veritabanları', 'cs.SE': 'Yazılım Mühendisliği',
            'cs.DS': 'Veri Yapıları ve Algoritmalar', 'cs.DC': 'Dağıtık Hesaplama', 'cs.SY': 'Sistemler ve Kontrol',
            'math.ST': 'İstatistik Teorisi', 'math.PR': 'Olasılık Teorisi', 'math.OC': 'Optimizasyon',
            'math.NA': 'Sayısal Analiz', 'stat.ML': 'İstatistiksel Öğrenme', 'stat.ME': 'İstatistik Metodolojisi',
            'stat.TH': 'İstatistik Teorisi', 'stat.AP': 'Uygulamalı İstatistik',
            'physics.data-an': 'Veri Analizi (Fizik)', 'physics.comp-ph': 'Hesaplamalı Fizik',
            'cond-mat.stat-mech': 'İstatistiksel Mekanik', 'cond-mat.soft': 'Yumuşak Madde Fiziği',
            'q-bio.QM': 'Biyolojik Yöntemler', 'econ.EM': 'Ekonometri', 'econ.TH': 'Ekonomi Teorisi'
        }
        
        # Türkçe etiketleri oluştur
        turkish_labels = [category_turkish.get(cat, cat) for cat in category_counts.index]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(category_counts)))
        
        # Bar grafiği oluştur
        bars = plt.bar(range(len(category_counts)), category_counts.values, color=colors)
        
        # Grafik özelliklerini ayarla
        plt.xticks(range(len(category_counts)), turkish_labels, rotation=45, ha='right')
        plt.ylabel('Makale Sayısı', fontsize=14, fontweight='bold')
        plt.title('ArXiv Kategorilerinin Dağılımı', fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Her barın üstüne değer yazısı ekle
        for i, (bar, value) in enumerate(zip(bars, category_counts.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(category_counts.values)*0.01, 
                    str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Analiz notu ekle
        top_3_categories = [category_turkish.get(cat, cat) for cat in category_counts.head(3).index]
        insight_text = f"En yaygın kategoriler: {', '.join(top_3_categories)}"
        
        plt.figtext(0.5, 0.02, f"Analiz: {insight_text}", ha='center', fontsize=12, 
                   style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # Grafiği kaydet ve göster
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(os.path.join(self.viz_dir, 'category_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Küme-Kategori İlişki Isı Haritası
        cluster_category = pd.crosstab(df_pandas['cluster_name'], df_pandas['primary_category'])
        top_categories = df_pandas['primary_category'].value_counts().head(15).index
        cluster_category_filtered = cluster_category[top_categories]
        cluster_category_filtered.columns = [category_turkish.get(col, col) for col in cluster_category_filtered.columns]
        
        plt.figure(figsize=(20, 12))
        
        # Isı haritası oluştur
        sns.heatmap(cluster_category_filtered, annot=True, fmt='d', cmap='YlOrRd',
                   cbar_kws={'label': 'Makale Sayısı', 'shrink': 0.8},
                   linewidths=0.5, square=False)
        
        plt.title('Araştırma Alanları ve ArXiv Kategorileri İlişkisi\n(En Yaygın 15 Kategori)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('ArXiv Kategorisi', fontsize=14, fontweight='bold')
        plt.ylabel('Araştırma Alanı', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Baskın kümeleri tespit et ve analiz notu ekle
        dominant_clusters = []
        for idx, row in cluster_category_filtered.iterrows():
            if row.sum() > 0:
                dominant_cat = row.idxmax()
                dominant_count = row.max()
                total_in_cluster = row.sum()
                dominance_percentage = (dominant_count / total_in_cluster) * 100
                
                if dominance_percentage > 50:
                    dominant_clusters.append(f"{idx} -> {dominant_cat} (%{dominance_percentage:.0f})")
        
        insight_text = f"Özelleşmiş kümeler: {'; '.join(dominant_clusters[:3])}" if dominant_clusters else "Tüm kümeler çok kategorili yapıda - genel araştırma alanları tespit edildi"
            
        plt.figtext(0.5, 0.02, f"Analiz: {insight_text}", ha='center', fontsize=12,
                   style='italic', wrap=True, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        # Grafiği kaydet ve göster
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(os.path.join(self.viz_dir, 'cluster_category_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Kelime Bulutlarını oluştur
        self._create_wordclouds(df_pandas)
    
    def _create_wordclouds(self, df_pandas: pd.DataFrame):
        """
        Her küme için kelime bulutu görselleştirmesi oluşturur.
        
        Bu metod her kümenin en önemli kelimelerini görsel olarak temsil eden
        kelime bulutları oluşturur. Kelime boyutları TF-IDF skorlarına göre ayarlanır.
        
        Args:
            df_pandas (pd.DataFrame): Kümeleme sonuçları içeren Pandas DataFrame
        """
        # Küme sayısına göre ızgara düzenini hesapla
        n_clusters = len(self.cluster_analysis)
        cols = 3  # Her satırda 3 kelime bulutu
        rows = (n_clusters + cols - 1) // cols  # Gerekli satır sayısı
        
        # Çoklu subplot oluştur (büyük boyutta netlik için)
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        
        # Tek satır durumunda axes'i 2D yapmak için reshape
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Her küme için kelime bulutu oluştur
        for i, (cluster_id, info) in enumerate(self.cluster_analysis.items()):
            # Izgara pozisyonunu hesapla
            row = i // cols
            col = i % cols
            
            # WordCloud objesi oluştur ve kelime sıklıklarından üret
            # Viridis renk paleti kullanarak profesyonel görünüm sağla
            wordcloud = WordCloud(width=500, height=400, background_color='white',
                                max_words=60, colormap='viridis').generate_from_frequencies(info['top_words'])
            
            # Küme başlığını oluştur (isim varsa kullan, yoksa sadece numara)
            title = f'{self.cluster_names[cluster_id]["name"]}\n({info["size"]} makale, %{info["percentage"]:.1f})' if hasattr(self, 'cluster_names') else f'Küme {cluster_id}\n({info["size"]} makale)'
            
            # Kelime bulutunu subplot'a yerleştir
            axes[row, col].imshow(wordcloud, interpolation='bilinear')
            axes[row, col].set_title(title, fontsize=12, fontweight='bold')
            axes[row, col].axis('off')  # Koordinat eksenlerini gizle
        
        # Boş subplot'ları gizle (küme sayısı ızgara boyutundan azsa)
        for i in range(n_clusters, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        # Genel başlık ekle ve düzeni ayarla
        plt.suptitle('Araştırma Alanları - Anahtar Kelime Bulutları', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Yüksek çözünürlükte kaydet ve görüntüle
        plt.savefig(os.path.join(self.viz_dir, 'cluster_wordclouds.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, output_path: str = None):
        """
        Kümeleme sonuçlarını CSV formatında diske kaydeder.
        
        Bu metod kümeleme analizinin sonuçlarını, küme isimleri ve temaları ile birlikte
        CSV dosyası olarak kaydeder. Sonuçlar hem orijinal veri hem de kümeleme 
        bilgilerini içerir.
        
        Args:
            output_path (str, optional): Çıktı dosyasının kaydedileceği yol. 
                                       Belirtilmezse varsayılan data_dir kullanılır.
        
        Returns:
            pd.DataFrame: Kümeleme sonuçlarını içeren Pandas DataFrame
        """
        # Çıktı yolunu belirle (varsayılan: data_dir/clustered_papers.csv)
        if output_path is None:
            output_path = os.path.join(self.data_dir, 'clustered_papers.csv')
        
        # Mevcut sütunları kontrol et ve temel sütunları seç
        available_columns = self.df_clustered.columns
        base_columns = ["id", "title", "summary", "authors", "published", "primary_category", "cluster"]
        select_columns = [col for col in base_columns if col in available_columns]
        
        # Birleştirilmiş metin sütunu varsa dahil et
        if "combined_text" in available_columns:
            select_columns.append("combined_text")
        
        # Spark DataFrame'den Pandas DataFrame'e dönüştür
        result_df = self.df_clustered.select(*select_columns).toPandas()
        
        # Küme isim ve tema bilgilerini ekle (varsa)
        if hasattr(self, 'cluster_names'):
            # Küme ID'lerini anlamlı isimlere dönüştür
            result_df['cluster_name'] = result_df['cluster'].map(lambda x: self.cluster_names[x]['name'])
            result_df['cluster_theme'] = result_df['cluster'].map(lambda x: self.cluster_names[x]['theme'])
            result_df['cluster_keywords'] = result_df['cluster'].map(lambda x: self.cluster_names[x]['keywords'])
        else:
            # Küme isimleri yoksa varsayılan isimlendirme
            result_df['cluster_name'] = result_df['cluster'].map(lambda x: f"Küme {x}")
            result_df['cluster_theme'] = ""
            result_df['cluster_keywords'] = ""
        
        # CSV formatında kaydet (UTF-8 encoding ile Türkçe karakter desteği)
        result_df.to_csv(output_path, index=False)
        
        # Kayıt işlemi tamamlandı, DataFrame'i döndür
        print(f"Kümeleme sonuçları kaydedildi: {output_path}")
        print(f"Toplam makale sayısı: {len(result_df)}")
        print(f"Küme sayısı: {result_df['cluster'].nunique()}")
        
        return result_df
    
    def stop_spark(self):
        """
        Spark oturumunu güvenli bir şekilde sonlandırır.
        
        Bu metod Spark kaynaklarını temizler ve bellek sızıntılarını önler.
        İşlem tamamlandıktan sonra mutlaka çağrılmalıdır.
        """
        # Spark oturumunu sonlandır ve tüm kaynakları serbest bırak
        self.spark.stop()
        print("Spark oturumu başarıyla sonlandırıldı.")

def main():
    """
    Ana çalıştırılabilir fonksiyon - Tam akademik makale kümeleme iş akışını gerçekleştirir.
    
    Bu fonksiyon ArXiv akademik makalelerinin tamamlanmış bir kümeleme analizini 
    gerçekleştirir. İş akışı şu adımlardan oluşur:
    
    1. Spark oturumu başlatma ve veri yükleme
    2. Metin ön işleme (temizleme, normalleştirme)
    3. TF-IDF özellik çıkarımı
    4. Optimal küme sayısı belirleme (K-Means)
    5. Kümeleme algoritması uygulama
    6. Küme analizi ve adlandırma
    7. Görselleştirme oluşturma
    8. Sonuçları CSV formatında kaydetme
    9. Kaynakları temizleme
    """
    # SparkTextClustering sınıfından örnek oluştur
    print("=== ArXiv Akademik Makale Kümeleme Sistemi ===")
    print("Spark Text Clustering başlatılıyor...")
    
    clustering = SparkTextClustering()
    
    try:
        # 1. Adım: Veri yükleme ve doğrulama
        print("\n1. Veri yükleme işlemi başlatılıyor...")
        df = clustering.load_data()
        print(f"   ✓ {df.count()} makale başarıyla yüklendi")
        
        # 2. Adım: Metin ön işleme
        print("\n2. Metin ön işleme başlatılıyor...")
        df = clustering.preprocess_text()
        print("   ✓ Metin temizleme ve normalleştirme tamamlandı")
        
        # 3. Adım: TF-IDF özellik çıkarımı
        print("\n3. TF-IDF özellik çıkarımı başlatılıyor...")
        df_features = clustering.create_features(vocab_size=5000, min_df=1)
        print("   ✓ 5000 boyutlu TF-IDF özellik vektörleri oluşturuldu")
        
        # 4. Adım: Optimal küme sayısı belirleme
        print("\n4. Optimal küme sayısı araştırılıyor...")
        optimal_k, costs, silhouette_scores = clustering.find_optimal_k(range(3, 12), 50)
        print(f"   ✓ Optimal küme sayısı: {optimal_k}")
        
        # 5. Adım: K-Means kümeleme algoritması
        print(f"\n5. K-Means kümeleme (k={optimal_k}) başlatılıyor...")
        df_clustered = clustering.perform_clustering(k=optimal_k, max_iterations=50)
        print(f"   ✓ {optimal_k} kümeye ayrılma işlemi tamamlandı")
        
        # 6. Adım: Küme analizi ve akıllı adlandırma
        print("\n6. Küme analizi ve adlandırma işlemi...")
        cluster_analysis = clustering.analyze_clusters(top_words=15)
        print("   ✓ Küme karakteristikleri analiz edildi")
        print("   ✓ Otomatik küme isimlendirmesi tamamlandı")
        
        # 7. Adım: Görselleştirme oluşturma
        print("\n7. Görselleştirmeler oluşturuluyor...")
        clustering.create_visualizations()
        print("   ✓ Pasta grafiği, bar grafik, ısı haritası ve kelime bulutları oluşturuldu")
        
        # 8. Adım: Sonuçları kaydetme
        print("\n8. Sonuçlar kaydediliyor...")
        result_df = clustering.save_results()
        print("   ✓ Kümeleme sonuçları CSV formatında kaydedildi")
        
        # İşlem başarılı tamamlandı
        print("\n=== KÜMELEME ANALİZİ TAMAMLANDI ===")
        print(f"Toplam işlenen makale: {len(result_df)}")
        print(f"Oluşturulan küme sayısı: {result_df['cluster'].nunique()}")
        print("Tüm görselleştirmeler 'visualizations/' klasörüne kaydedildi")
        print("Kümeleme sonuçları 'data/clustered_papers.csv' dosyasına kaydedildi")
        
    except Exception as e:
        # Hata durumunda ayrıntılı bilgi ver
        print(f"\n❌ HATA: Kümeleme işlemi sırasında bir hata oluştu: {str(e)}")
        print("Lütfen veri dosyalarını ve sistem gereksinimlerini kontrol edin.")
        
    finally:
        # Kaynakları temizle (hata olsa da olmasa da)
        print("\n9. Sistem kaynakları temizleniyor...")
        clustering.stop_spark()
        print("   ✓ Spark oturumu sonlandırıldı")
        print("\nİşlem tamamlandı.")


# Programın ana giriş noktası
if __name__ == "__main__":
    main() 