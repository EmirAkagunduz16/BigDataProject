"""
PySpark ile Akademik Makale Kümeleme Modülü
Bu modül makaleleri PySpark kullanarak kümeler ve analiz eder.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, HashingTF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf, regexp_replace, lower, trim, split, size, concat_ws
from pyspark.sql.types import StringType, IntegerType, ArrayType, DoubleType
from pyspark.ml.evaluation import ClusteringEvaluator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

import re
from typing import List, Dict, Tuple
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
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # NLTK stopwords'ü indir
        try:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        print(f"Spark Session başlatıldı - Version: {self.spark.version}")
    
    def load_data(self, filepath: str):
        """CSV dosyasından veri yükler"""
        print(f"Veri yükleniyor: {filepath}")
        
        # Pandas ile oku, sonra Spark DataFrame'e çevir
        pandas_df = pd.read_csv(filepath)
        
        # Spark DataFrame'e çevir
        self.df = self.spark.createDataFrame(pandas_df)
        
        print(f"Yüklenen veri boyutu: {self.df.count()} satır, {len(self.df.columns)} sütun")
        return self.df
    
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
                       vocab_size: int = 10000, min_df: int = 2):
        """TF-IDF özellik vektörleri oluşturur"""
        print("TF-IDF özellik çıkarma başlıyor...")
        
        # Pipeline oluştur
        # 1. Tokenization
        tokenizer = RegexTokenizer(
            inputCol=text_col, 
            outputCol="words", 
            pattern="\\W"
        )
        
        # 2. Stop words removal
        # İngilizce stop words listesi
        english_stop_words = StopWordsRemover.loadDefaultStopWords("english")
        
        # Akademik yazım için ek stop words
        additional_stop_words = [
            'paper', 'study', 'research', 'analysis', 'method', 'approach',
            'result', 'conclusion', 'introduction', 'abstract', 'figure',
            'table', 'section', 'chapter', 'algorithm', 'model', 'system',
            'data', 'et', 'al', 'also', 'however', 'therefore', 'furthermore',
            'moreover', 'thus', 'hence', 'consequently', 'respectively',
            'example', 'case', 'cases', 'problem', 'problems', 'solution',
            'solutions', 'work', 'works', 'related', 'previous', 'existing',
            'proposed', 'novel', 'new', 'different', 'various', 'several',
            'many', 'much', 'most', 'more', 'less', 'first', 'second',
            'third', 'last', 'final', 'initial', 'main', 'general', 'specific',
            'particular', 'important', 'significant', 'relevant', 'similar',
            'different', 'same', 'other', 'another', 'such', 'based', 'using',
            'used', 'show', 'shows', 'shown', 'present', 'presents',
            'presented', 'describe', 'describes', 'described', 'discuss',
            'discusses', 'discussed', 'propose', 'proposes', 'proposed'
        ]
        
        all_stop_words = list(set(english_stop_words + additional_stop_words))
        
        remover = StopWordsRemover(
            inputCol="words", 
            outputCol="filtered_words",
            stopWords=all_stop_words
        )
        
        # 3. TF (Term Frequency)
        hashingTF = HashingTF(
            inputCol="filtered_words", 
            outputCol="rawFeatures", 
            numFeatures=vocab_size
        )
        
        # 4. IDF (Inverse Document Frequency)
        idf = IDF(
            inputCol="rawFeatures", 
            outputCol="features",
            minDocFreq=min_df
        )
        
        # Pipeline'ı oluştur ve fit et
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf])
        self.feature_model = pipeline.fit(self.df)
        
        # Transform et
        self.df_features = self.feature_model.transform(self.df)
        
        print("TF-IDF özellik çıkarma tamamlandı")
        return self.df_features
    
    def perform_clustering(self, k: int = 8, max_iterations: int = 100, seed: int = 42):
        """K-means kümeleme yapar"""
        print(f"K-means kümeleme başlıyor (k={k})...")
        
        # K-means model oluştur
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=k,
            maxIter=max_iterations,
            seed=seed
        )
        
        # Model'i fit et
        self.kmeans_model = kmeans.fit(self.df_features)
        
        # Predictions yap
        self.df_clustered = self.kmeans_model.transform(self.df_features)
        
        # Kümeleme sonuçlarını değerlendir
        evaluator = ClusteringEvaluator(
            predictionCol="cluster",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean"
        )
        
        self.silhouette_score = evaluator.evaluate(self.df_clustered)
        
        print(f"K-means kümeleme tamamlandı")
        print(f"Silhouette Score: {self.silhouette_score:.4f}")
        
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
        plt.savefig('visualizations/optimal_k_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
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
            
            print(f"\n--- KÜME {cluster_id} ---")
            print(f"Boyut: {len(cluster_data)} makale ({len(cluster_data)/len(df_pandas)*100:.1f}%)")
            print(f"En yaygın kelimeler: {list(word_freq.head(5).index)}")
            print(f"En yaygın kategoriler: {list(category_freq.head(3).index)}")
        
        return self.cluster_analysis
    
    def create_visualizations(self):
        """Kümeleme sonuçlarını görselleştirir"""
        print("Görselleştirmeler oluşturuluyor...")
        
        # Pandas DataFrame'e çevir
        df_pandas = self.df_clustered.select(
            "cluster", "title", "summary", "primary_category"
        ).toPandas()
        
        # 1. Küme boyutları pasta grafiği
        fig = px.pie(
            values=[info['size'] for info in self.cluster_analysis.values()],
            names=[f"Küme {k}" for k in self.cluster_analysis.keys()],
            title="Kümelerin Boyut Dağılımı"
        )
        fig.write_html("visualizations/cluster_sizes.html")
        fig.show()
        
        # 2. Kategori dağılımı
        plt.figure(figsize=(12, 8))
        category_counts = df_pandas['primary_category'].value_counts().head(15)
        sns.barplot(x=category_counts.values, y=category_counts.index)
        plt.title('En Yaygın ArXiv Kategorileri')
        plt.xlabel('Makale Sayısı')
        plt.tight_layout()
        plt.savefig('visualizations/category_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Küme-kategori ilişkisi
        cluster_category = pd.crosstab(df_pandas['cluster'], df_pandas['primary_category'])
        
        plt.figure(figsize=(15, 10))
        sns.heatmap(cluster_category, annot=True, fmt='d', cmap='Blues')
        plt.title('Küme-Kategori İlişkisi')
        plt.xlabel('ArXiv Kategorisi')
        plt.ylabel('Küme')
        plt.tight_layout()
        plt.savefig('visualizations/cluster_category_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Her küme için kelime bulutu
        self._create_wordclouds(df_pandas)
    
    def _create_wordclouds(self, df_pandas: pd.DataFrame):
        """Her küme için kelime bulutu oluşturur"""
        n_clusters = len(self.cluster_analysis)
        cols = 3
        rows = (n_clusters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (cluster_id, info) in enumerate(self.cluster_analysis.items()):
            row = i // cols
            col = i % cols
            
            # Kelime bulutunu oluştur
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                max_words=50
            ).generate_from_frequencies(info['top_words'])
            
            axes[row, col].imshow(wordcloud, interpolation='bilinear')
            axes[row, col].set_title(f'Küme {cluster_id}\n({info["size"]} makale)')
            axes[row, col].axis('off')
        
        # Boş subplotları gizle
        for i in range(n_clusters, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/cluster_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, output_path: str = "data/clustered_papers.csv"):
        """Kümeleme sonuçlarını kaydet"""
        print(f"Sonuçlar kaydediliyor: {output_path}")
        
        # Spark DataFrame'i Pandas'a çevir ve kaydet
        result_df = self.df_clustered.select(
            "id", "title", "summary", "authors", "published", 
            "primary_category", "cluster", "combined_text"
        ).toPandas()
        
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
        df = clustering.load_data("data/arxiv_papers.csv")
        
        # 2. Metin ön işleme
        df = clustering.preprocess_text()
        
        # 3. Özellik çıkarma
        df_features = clustering.create_features(vocab_size=5000, min_df=2)
        
        # 4. Optimal k bulma
        optimal_k, costs, silhouette_scores = clustering.find_optimal_k(
            k_range=range(3, 12), iterations=50
        )
        
        # 5. En iyi k ile kümeleme
        df_clustered = clustering.perform_clustering(k=optimal_k, max_iterations=100)
        
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