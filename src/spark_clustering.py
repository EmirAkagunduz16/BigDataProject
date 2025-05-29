"""
PySpark ile Akademik Makale Kümeleme Modülü
Bu modül makaleleri PySpark kullanarak kümeler ve analiz eder.
"""

import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, HashingTF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf, regexp_replace, lower, trim, split, size, concat_ws
from pyspark.ml.evaluation import ClusteringEvaluator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

from typing import List
import nltk
from nltk.corpus import stopwords

class SparkTextClustering:
    """PySpark ile akademik makale kümeleme sınıfı"""
    
    def __init__(self, app_name: str = "AcademicPaperClustering"):
        """
        Args:
            app_name: Spark uygulaması adı
        """
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
            .config("spark.default.parallelism", "4") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.cores", "2") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # NLTK stopwords'ü indir
        try:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Set up project directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.base_dir)
        self.data_dir = os.path.join(self.project_root, 'data')
        self.viz_dir = os.path.join(self.project_root, 'visualizations')
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        print(f"⚡ Optimized Spark Session başlatıldı - Version: {self.spark.version}")
        print(f"📁 Project root: {self.project_root}")
        print(f"🎨 Visualizations dir: {self.viz_dir}")
    
    def load_data(self, filepath: str = None):
        """CSV dosyasından veri yükler - Optimized"""
        if filepath is None:
            filepath = os.path.join(self.data_dir, 'arxiv_papers.csv')
            
        print(f"📊 Veri yükleniyor: {filepath}")
        
        try:
            # Pandas ile oku, daha hızlı
            pandas_df = pd.read_csv(filepath)
            
            # Küçük dataset için gerekli sütunları seç
            essential_columns = ['id', 'title', 'summary', 'primary_category', 'authors', 'published']
            existing_columns = [col for col in essential_columns if col in pandas_df.columns]
            
            if len(existing_columns) < len(essential_columns):
                print(f"⚠️  Bazı sütunlar eksik, mevcut: {existing_columns}")
            
            pandas_df = pandas_df[existing_columns]
            
            # Spark DataFrame'e çevir, optimal partition sayısı ile
            row_count = len(pandas_df)
            optimal_partitions = max(1, min(8, row_count // 1000))  # Her partition ~1000 row
            
            self.df = self.spark.createDataFrame(pandas_df).repartition(optimal_partitions)
            self.df.cache()  # Memory'de sakla
            
            print(f"✅ Yüklenen veri boyutu: {row_count} satır, {len(existing_columns)} sütun")
            print(f"⚡ Optimal partitions: {optimal_partitions}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {str(e)}")
            raise
    
    def preprocess_text(self, input_cols: List[str] = ['title', 'summary'], 
                       output_col: str = 'combined_text'):
        """Metinleri ön işleme tabi tutar"""
        print("Metin ön işleme başlıyor...")
        
        # Metin sütunlarını birleştir
        self.df = self.df.withColumn(
            output_col,
            concat_ws(" ", *input_cols)
        )
        
        # Metni temizle
        # Küçük harfe çevir
        self.df = self.df.withColumn(
            output_col,
            lower(col(output_col))
        )
        
        # Özel karakterleri kaldır, sadece harfler ve boşluklar kalsın
        self.df = self.df.withColumn(
            output_col,
            regexp_replace(col(output_col), r'[^a-zA-Z\s]', ' ')
        )
        
        # Fazla boşlukları kaldır
        self.df = self.df.withColumn(
            output_col,
            regexp_replace(col(output_col), r'\s+', ' ')
        )
        
        # Başındaki ve sonundaki boşlukları kaldır
        self.df = self.df.withColumn(
            output_col,
            trim(col(output_col))
        )
        
        print("Metin ön işleme tamamlandı")
        return self.df
    
    def create_features(self, text_col: str = 'combined_text', 
                       vocab_size: int = 5000, min_df: int = 2):
        """TF-IDF özellik vektörleri oluşturur - Optimized"""
        print("🔍 Hızlandırılmış TF-IDF özellik çıkarma başlıyor...")
        
        # Pipeline oluştur - Optimize edilmiş
        # 1. Tokenization
        tokenizer = RegexTokenizer(
            inputCol=text_col, 
            outputCol="words", 
            pattern="\\W",
            minTokenLength=3  # En az 3 karakter - gürültüyü azaltır
        )
        
        # 2. Stop words removal - Kompakt liste
        # Sadece en yaygın stop words (performans için)
        essential_stop_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
            # Akademik terimler - sadece en yaygınları
            'paper', 'study', 'research', 'method', 'result', 'data', 'model', 'algorithm', 'analysis',
            'approach', 'system', 'show', 'present', 'propose', 'based', 'using', 'used'
        ]
        
        remover = StopWordsRemover(
            inputCol="words", 
            outputCol="filtered_words",
            stopWords=essential_stop_words
        )
        
        # 3. TF (Term Frequency) - Optimize edilmiş
        hashingTF = HashingTF(
            inputCol="filtered_words", 
            outputCol="rawFeatures", 
            numFeatures=vocab_size  # Varsayılan olarak daha küçük
        )
        
        # 4. IDF (Inverse Document Frequency)
        idf = IDF(
            inputCol="rawFeatures", 
            outputCol="features",
            minDocFreq=max(2, min_df)  # En az 2 dokümanda geçmeli
        )
        
        # Pipeline'ı oluştur ve fit et
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
        
        print(f"⚡ Pipeline fit ediliyor (vocab_size={vocab_size})...")
        self.feature_model = pipeline.fit(self.df)
        
        # Transform et
        print("🔄 Feature transformation...")
        self.df_features = self.feature_model.transform(self.df)
        self.df_features.cache()  # Memory'de sakla
        
        # Gereksiz intermediate columns'ları temizle
        self.df_features = self.df_features.drop("words", "filtered_words", "rawFeatures")
        
        print("✅ Optimized TF-IDF özellik çıkarma tamamlandı")
        return self.df_features
    
    def perform_clustering(self, k: int = 5, max_iterations: int = 50, seed: int = 42):
        """K-means kümeleme yapar - Optimized"""
        print(f"⚡ Hızlandırılmış K-means kümeleme başlıyor (k={k})...")
        
        # K-means model oluştur - Optimize edilmiş parametreler
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=k,
            maxIter=max_iterations,  # 100'den 50'ye düşürüldü
            seed=seed,
            tol=1e-3,  # Convergence tolerance - biraz gevşetildi
            initMode="k-means||"  # Daha hızlı initialization
        )
        
        print("🔄 Model training...")
        # Model'i fit et
        self.kmeans_model = kmeans.fit(self.df_features)
        
        print("🔄 Predictions...")
        # Predictions yap
        self.df_clustered = self.kmeans_model.transform(self.df_features)
        self.df_clustered.cache()  # Cache for multiple uses
        
        # Kümeleme sonuçlarını değerlendir
        print("📊 Silhouette score hesaplanıyor...")
        evaluator = ClusteringEvaluator(
            predictionCol="cluster",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        
        self.silhouette_score = evaluator.evaluate(self.df_clustered)
        
        print(f"✅ K-means kümeleme tamamlandı")
        print(f"📈 Silhouette Score: {self.silhouette_score:.4f}")
        print(f"🎯 Optimal iteration count: {self.kmeans_model.summary.totalIterations}")
        
        return self.df_clustered
    
    def find_optimal_k(self, k_range: range = range(2, 11), iterations: int = 50):
        """Optimal k değerini bulur"""
        print("Optimal k değeri aranıyor...")
        
        costs = []
        silhouette_scores = []
        
        for k in k_range:
            print(f"k={k} test ediliyor...")
            
            kmeans = KMeans(
                featuresCol="features",
                predictionCol="cluster",
                k=k,
                maxIter=iterations,
                seed=42
            )
            
            model = kmeans.fit(self.df_features)
            predictions = model.transform(self.df_features)
            
            # Cost (WSSSE - Within Set Sum of Squared Errors)
            cost = model.summary.trainingCost
            costs.append(cost)
            
            # Silhouette Score
            evaluator = ClusteringEvaluator(
                predictionCol="cluster",
                featuresCol="features",
                metricName="silhouette"
            )
            silhouette = evaluator.evaluate(predictions)
            silhouette_scores.append(silhouette)
            
            print(f"k={k}: Cost={cost:.2f}, Silhouette={silhouette:.4f}")
        
        # Sonuçları görselleştir
        self._plot_elbow_curve(list(k_range), costs, silhouette_scores)
        
        # En iyi k'yı seç (en yüksek silhouette score)
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]
        print(f"Önerilen optimal k değeri: {optimal_k}")
        
        return optimal_k, costs, silhouette_scores
    
    def _plot_elbow_curve(self, k_values: List[int], costs: List[float], 
                         silhouette_scores: List[float]):
        """Elbow curve ve silhouette score grafiğini çizer"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(k_values, costs, 'bo-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Cost (WSSSE)')
        ax1.set_title('Elbow Method For Optimal k')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(k_values, silhouette_scores, 'ro-')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score vs k')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'optimal_k_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_cluster_names(self):
        """Cluster'ları anahtar kelimelerine göre anlamlı isimler verir"""
        cluster_names = {}
        
        for cluster_id, info in self.cluster_analysis.items():
            top_words = list(info['top_words'].keys())[:3]
            top_categories = list(info['top_categories'].keys())[:2]
            
            # Kategori bazlı isimlendirme
            category_themes = {
                'cs.AI': 'Yapay Zeka',
                'cs.ML': 'Makine Öğrenmesi', 
                'cs.LG': 'Öğrenme Algoritmaları',
                'cs.CV': 'Bilgisayarlı Görü',
                'cs.CL': 'Doğal Dil İşleme',
                'cs.NE': 'Sinir Ağları',
                'cs.CR': 'Güvenlik ve Kriptografi',
                'cs.DB': 'Veritabanları',
                'cs.IR': 'Bilgi Erişimi',
                'cs.HC': 'İnsan-Bilgisayar Etkileşimi',
                'cs.RO': 'Robotik',
                'cs.SE': 'Yazılım Mühendisliği',
                'math.ST': 'İstatistik Teorisi',
                'math.PR': 'Olasılık Teorisi',
                'math.OC': 'Optimizasyon',
                'stat.ML': 'İstatistiksel Öğrenme',
                'stat.ME': 'İstatistik Metodolojisi',
                'physics.data-an': 'Veri Analizi (Fizik)',
                'physics.comp-ph': 'Hesaplamalı Fizik',
                'cond-mat.stat-mech': 'İstatistiksel Mekanik',
                'q-bio.QM': 'Biyolojik Yöntemler',
                'q-bio.NC': 'Nörobiyoloji',
                'econ.EM': 'Ekonometri',
                'econ.TH': 'Ekonomi Teorisi'
            }
            
            # Ana tema belirle
            main_category = top_categories[0] if top_categories else 'Genel'
            theme = category_themes.get(main_category, main_category)
            
            # Anahtar kelimeler bazlı alt tema
            keywords = ', '.join(top_words[:2]).title()
            
            # Anlamlı isim oluştur
            if 'learning' in top_words or 'model' in top_words:
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
            
            cluster_names[cluster_id] = {
                'name': cluster_name,
                'theme': theme,
                'keywords': keywords
            }
        
        return cluster_names

    def analyze_clusters(self, top_words: int = 10):
        """Kümeleri analiz eder ve anahtar kelimeleri bulur"""
        print("Küme analizi başlıyor...")
        
        # Küme istatistikleri
        cluster_stats = self.df_clustered.groupBy("cluster").count().orderBy("cluster")
        print("\nKüme boyutları:")
        cluster_stats.show()
        
        # Pandas'a çevir daha kolay analiz için
        df_pandas = self.df_clustered.select(
            "cluster", "title", "summary", "primary_category", "combined_text", "filtered_words"
        ).toPandas()
        
        self.cluster_analysis = {}
        
        for cluster_id in sorted(df_pandas['cluster'].unique()):
            cluster_data = df_pandas[df_pandas['cluster'] == cluster_id]
            
            # En yaygın kelimeler
            all_words = []
            for words_list in cluster_data['filtered_words']:
                if isinstance(words_list, list):
                    all_words.extend(words_list)
            
            word_freq = pd.Series(all_words).value_counts().head(top_words)
            
            # En yaygın kategoriler
            category_freq = cluster_data['primary_category'].value_counts().head(5)
            
            # Örnek makaleler
            sample_titles = cluster_data['title'].head(3).tolist()
            
            self.cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'top_words': word_freq.to_dict(),
                'top_categories': category_freq.to_dict(),
                'sample_titles': sample_titles,
                'percentage': len(cluster_data) / len(df_pandas) * 100
            }
        
        # Anlamlı cluster isimleri oluştur
        self.cluster_names = self._generate_cluster_names()
        
        # Sonuçları yazdır
        for cluster_id, info in self.cluster_analysis.items():
            cluster_name = self.cluster_names[cluster_id]['name']
            print(f"\n--- {cluster_name.upper()} (Küme {cluster_id}) ---")
            print(f"Boyut: {info['size']} makale ({info['percentage']:.1f}%)")
            print(f"Ana tema: {self.cluster_names[cluster_id]['theme']}")
            print(f"Anahtar kelimeler: {list(word_freq.head(5).index)}")
            print(f"Ana kategoriler: {list(category_freq.head(3).index)}")
        
        return self.cluster_analysis
    
    def create_visualizations(self):
        """Kümeleme sonuçlarını görselleştirir"""
        print("Görselleştirmeler oluşturuluyor...")
        
        # Pandas DataFrame'e çevir
        df_pandas = self.df_clustered.select(
            "cluster", "title", "summary", "primary_category"
        ).toPandas()
        
        # Cluster isimlerini DataFrame'e ekle
        df_pandas['cluster_name'] = df_pandas['cluster'].map(
            lambda x: self.cluster_names[x]['name'] if hasattr(self, 'cluster_names') else f"Küme {x}"
        )
        
        # 1. Küme boyutları pasta grafiği (anlamlı isimlerle)
        cluster_labels = []
        cluster_sizes = []
        cluster_info_for_analysis = {}  # Analysis için veri
        
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

        fig = px.pie(
            values=cluster_sizes,
            names=cluster_labels,
            title="Araştırma Alanlarının Küme Dağılımı",
            width=800,
            height=600
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        # En büyük kümeyi bul analiz için
        largest_cluster = max(cluster_info_for_analysis.values(), key=lambda x: x['size'])
        largest_cluster_text = f"En büyük küme olan \"{largest_cluster['name']}\" makalelerin %{largest_cluster['percentage']:.1f}'ini içermektedir."
        
        # HTML'e analiz metni ekle
        fig.add_annotation(
            text=f"<b>Analiz:</b> {largest_cluster_text}",
            xref="paper", yref="paper",
            x=0.5, y=-0.1, showarrow=False,
            font=dict(size=12),
            align="center"
        )
        
        fig.write_html(os.path.join(self.viz_dir, 'cluster_sizes.html'))
        fig.show()
        
        # 2. Geliştirilmiş kategori dağılımı
        plt.figure(figsize=(16, 12))
        category_counts = df_pandas['primary_category'].value_counts().head(25)
        
        # Kategori isimlerini Türkçeleştir
        category_turkish = {
            'cs.AI': 'Yapay Zeka',
            'cs.ML': 'Makine Öğrenmesi', 
            'cs.LG': 'Öğrenme Algoritmaları',
            'cs.CV': 'Bilgisayarlı Görü',
            'cs.CL': 'Doğal Dil İşleme',
            'cs.NE': 'Sinir Ağları',
            'cs.CR': 'Güvenlik & Kriptografi',
            'cs.DB': 'Veritabanları',
            'cs.IR': 'Bilgi Erişimi',
            'cs.HC': 'İnsan-Bilgisayar Etkileşimi',
            'cs.RO': 'Robotik',
            'cs.SE': 'Yazılım Mühendisliği',
            'math.ST': 'İstatistik Teorisi',
            'math.PR': 'Olasılık Teorisi',
            'math.OC': 'Optimizasyon',
            'stat.ML': 'İstatistiksel Öğrenme',
            'stat.ME': 'İstatistik Metodolojisi',
            'physics.data-an': 'Veri Analizi (Fizik)',
            'physics.comp-ph': 'Hesaplamalı Fizik',
            'cond-mat.stat-mech': 'İstatistiksel Mekanik',
            'q-bio.QM': 'Biyolojik Yöntemler',
            'q-bio.NC': 'Nörobiyoloji',
            'econ.EM': 'Ekonometri',
            'econ.TH': 'Ekonomi Teorisi'
        }
        
        turkish_labels = [category_turkish.get(cat, cat) for cat in category_counts.index]
        
        # Renk gradyenti oluştur
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(category_counts)))
        
        # Dikey bar chart (daha modern görünüm)
        bars = plt.bar(range(len(category_counts)), category_counts.values, color=colors)
        
        # X ekseni etiketlerini düzenle
        plt.xticks(range(len(category_counts)), turkish_labels, rotation=45, ha='right')
        plt.ylabel('Makale Sayısı', fontsize=14, fontweight='bold')
        plt.title('ArXiv Kategorilerinin Dağılımı', fontsize=16, fontweight='bold', pad=20)
        
        # Grid ekle
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Değerleri çubukların üzerine ekle
        for i, (bar, value) in enumerate(zip(bars, category_counts.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(category_counts.values)*0.01, 
                    str(value), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # En yaygın kategorileri bul
        top_3_categories = [category_turkish.get(cat, cat) for cat in category_counts.head(3).index]
        insight_text = f"En yaygın kategoriler: {', '.join(top_3_categories)}"
        
        plt.figtext(0.5, 0.02, f"Analiz: {insight_text}", ha='center', fontsize=12, 
                   style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Alt kısım için yer bırak
        plt.savefig(os.path.join(self.viz_dir, 'category_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Geliştirilmiş küme-kategori ilişkisi heatmap
        cluster_category = pd.crosstab(df_pandas['cluster_name'], df_pandas['primary_category'])
        
        # Sadece en yaygın kategorileri göster (görselleştirmeyi basitleştirmek için)
        top_categories = df_pandas['primary_category'].value_counts().head(15).index
        cluster_category_filtered = cluster_category[top_categories]
        
        # Kategori isimlerini Türkçeleştir
        cluster_category_filtered.columns = [category_turkish.get(col, col) for col in cluster_category_filtered.columns]
        
        plt.figure(figsize=(20, 12))
        
        # Heatmap'i çiz
        sns.heatmap(cluster_category_filtered, 
                   annot=True, 
                   fmt='d', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Makale Sayısı', 'shrink': 0.8},
                   linewidths=0.5,
                   square=False)
        
        plt.title('Araştırma Alanları ve ArXiv Kategorileri İlişkisi\n(En Yaygın 15 Kategori)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('ArXiv Kategorisi', fontsize=14, fontweight='bold')
        plt.ylabel('Araştırma Alanı', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Genel insight hesapla
        dominant_clusters = []
        for idx, row in cluster_category_filtered.iterrows():
            if row.sum() > 0:  # Sadece makale içeren kümeler
                dominant_cat = row.idxmax()
                dominant_count = row.max()
                total_in_cluster = row.sum()
                dominance_percentage = (dominant_count / total_in_cluster) * 100
                
                if dominance_percentage > 50:  # %50'den fazla tek kategori
                    dominant_clusters.append(f"{idx} -> {dominant_cat} (%{dominance_percentage:.0f})")
        
        if dominant_clusters:
            insight_text = f"Özelleşmiş kümeler: {'; '.join(dominant_clusters[:3])}"
        else:
            insight_text = "Tüm kümeler çok kategorili yapıda - genel araştırma alanları tespit edildi"
            
        plt.figtext(0.5, 0.02, f"Analiz: {insight_text}", ha='center', fontsize=12,
                   style='italic', wrap=True, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Alt kısım için yer bırak
        plt.savefig(os.path.join(self.viz_dir, 'cluster_category_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Her küme için kelime bulutu (anlamlı isimlerle)
        self._create_wordclouds(df_pandas)
    
    def _create_wordclouds(self, df_pandas: pd.DataFrame):
        """Her küme için kelime bulutu oluşturur"""
        n_clusters = len(self.cluster_analysis)
        cols = 3
        rows = (n_clusters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (cluster_id, info) in enumerate(self.cluster_analysis.items()):
            row = i // cols
            col = i % cols
            
            # Kelime bulutunu oluştur
            wordcloud = WordCloud(
                width=500, height=400,
                background_color='white',
                max_words=60,
                colormap='viridis'
            ).generate_from_frequencies(info['top_words'])
            
            # Anlamlı küme ismi kullan
            if hasattr(self, 'cluster_names'):
                title = f'{self.cluster_names[cluster_id]["name"]}\n({info["size"]} makale, %{info["percentage"]:.1f})'
            else:
                title = f'Küme {cluster_id}\n({info["size"]} makale)'
            
            axes[row, col].imshow(wordcloud, interpolation='bilinear')
            axes[row, col].set_title(title, fontsize=12, fontweight='bold')
            axes[row, col].axis('off')
        
        # Boş subplotları gizle
        for i in range(n_clusters, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Araştırma Alanları - Anahtar Kelime Bulutları', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'cluster_wordclouds.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, output_path: str = None):
        """Kümeleme sonuçlarını kaydet"""
        if output_path is None:
            output_path = os.path.join(self.data_dir, 'clustered_papers.csv')
        
        print(f"Sonuçlar kaydediliyor: {output_path}")
        
        # Spark DataFrame'i Pandas'a çevir
        result_df = self.df_clustered.select(
            "id", "title", "summary", "authors", "published", 
            "primary_category", "cluster", "combined_text"
        ).toPandas()
        
        # Cluster isimlerini ekle
        if hasattr(self, 'cluster_names'):
            result_df['cluster_name'] = result_df['cluster'].map(
                lambda x: self.cluster_names[x]['name']
            )
            result_df['cluster_theme'] = result_df['cluster'].map(
                lambda x: self.cluster_names[x]['theme']
            )
            result_df['cluster_keywords'] = result_df['cluster'].map(
                lambda x: self.cluster_names[x]['keywords']
            )
        else:
            result_df['cluster_name'] = result_df['cluster'].map(lambda x: f"Küme {x}")
            result_df['cluster_theme'] = ""
            result_df['cluster_keywords'] = ""
        
        result_df.to_csv(output_path, index=False)
        print(f"Sonuçlar {output_path} dosyasına kaydedildi")
        
        return result_df
    
    def stop_spark(self):
        """Spark session'ı durdur"""
        self.spark.stop()
        print("Spark session durduruldu")

def main():
    """Ana fonksiyon"""
    # Clustering sınıfını başlat
    clustering = SparkTextClustering()
    
    try:
        # 1. Veri yükle
        df = clustering.load_data()
        
        # 2. Metin ön işleme
        df = clustering.preprocess_text()
        
        # 3. Özellik çıkarma
        df_features = clustering.create_features(vocab_size=5000, min_df=2)
        
        # 4. Optimal k bulma
        optimal_k, costs, silhouette_scores = clustering.find_optimal_k(
            k_range=range(3, 12), iterations=50
        )
        
        # 5. En iyi k ile kümeleme
        df_clustered = clustering.perform_clustering(k=optimal_k, max_iterations=50)
        
        # 6. Küme analizi
        cluster_analysis = clustering.analyze_clusters(top_words=15)
        
        # 7. Görselleştirmeler
        clustering.create_visualizations()
        
        # 8. Sonuçları kaydet
        result_df = clustering.save_results()
        
        print("\n=== KÜMELEME TAMAMLANDI ===")
        print(f"Toplam makale: {len(result_df)}")
        print(f"Küme sayısı: {optimal_k}")
        print(f"Silhouette Score: {clustering.silhouette_score:.4f}")
        
    finally:
        # Spark'ı durdur
        clustering.stop_spark()

if __name__ == "__main__":
    main() 