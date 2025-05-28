"""
Akademik Makalelerin Araştırma Alanlarına Göre Kümelenmesi - Ana Script
Bu script veri toplama, işleme, kümeleme ve görselleştirme süreçlerini yönetir.
"""

import os
import sys
import argparse
from datetime import datetime

# Kendi modüllerimizi import et
sys.path.append('src')
from arxiv_data_collector import ArXivDataCollector
from spark_clustering import SparkTextClustering

def ensure_directories():
    """Gerekli dizinleri oluştur"""
    directories = ['data', 'visualizations', 'logs']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    print("Dizinler kontrol edildi/oluşturuldu")

def collect_data(categories=None, max_results=500, output_file="data/arxiv_papers.csv"):
    """ArXiv'den veri topla"""
    print("\n" + "="*50)
    print("VERİ TOPLAMA AŞAMASI")
    print("="*50)
    
    if categories is None:
        categories = [
            'cs.AI',      # Artificial Intelligence
            'cs.ML',      # Machine Learning  
            'cs.CV',      # Computer Vision
            'cs.CL',      # Computation and Language (NLP)
            'cs.LG',      # Learning
            'stat.ML',    # Machine Learning (Statistics)
            'physics.data-an',  # Data Analysis, Statistics and Probability
            'q-bio.QM',   # Quantitative Methods in Biology
            'econ.EM',    # Econometrics
            'math.ST'     # Statistics Theory
        ]
    
    collector = ArXivDataCollector(max_results=max_results, delay=0.5)
    
    print(f"Kategoriler: {categories}")
    print(f"Maksimum makale sayısı: {max_results}")
    
    # Veri topla
    df = collector.collect_papers_by_category(categories)
    
    # Temizle
    df_clean = collector.preprocess_dataframe(df)
    
    # Kaydet
    collector.save_data(df_clean, output_file)
    
    # İstatistikler
    print(f"\nToplanan makale sayısı: {len(df_clean)}")
    print(f"Benzersiz kategori sayısı: {df_clean['primary_category'].nunique()}")
    print("\nEn yaygın kategoriler:")
    print(df_clean['primary_category'].value_counts().head(10))
    
    return output_file

def perform_clustering(data_file="data/arxiv_papers.csv", k_range=None, vocab_size=5000):
    """PySpark ile kümeleme yap"""
    print("\n" + "="*50)
    print("KÜMELEME AŞAMASI")
    print("="*50)
    
    if k_range is None:
        k_range = range(3, 11)
    
    clustering = SparkTextClustering()
    
    try:
        # 1. Veri yükle
        print("1. Veri yükleniyor...")
        df = clustering.load_data(data_file)
        
        # 2. Metin ön işleme
        print("2. Metin ön işleme...")
        df = clustering.preprocess_text(['title', 'summary'])
        
        # 3. Özellik çıkarma (TF-IDF)
        print("3. TF-IDF özellik çıkarma...")
        df_features = clustering.create_features(
            vocab_size=vocab_size, 
            min_df=2
        )
        
        # 4. Optimal k bulma
        print("4. Optimal k değeri bulunuyor...")
        optimal_k, costs, silhouette_scores = clustering.find_optimal_k(
            k_range=k_range, 
            iterations=50
        )
        
        # 5. En iyi k ile kümeleme
        print(f"5. K-means kümeleme (k={optimal_k})...")
        df_clustered = clustering.perform_clustering(
            k=optimal_k, 
            max_iterations=100
        )
        
        # 6. Küme analizi
        print("6. Küme analizi...")
        cluster_analysis = clustering.analyze_clusters(top_words=15)
        
        # 7. Görselleştirmeler
        print("7. Görselleştirmeler oluşturuluyor...")
        clustering.create_visualizations()
        
        # 8. Sonuçları kaydet
        print("8. Sonuçlar kaydediliyor...")
        result_df = clustering.save_results()
        
        # Özet bilgiler
        print("\n" + "="*50)
        print("KÜMELEME SONUÇLARI")
        print("="*50)
        print(f"Toplam makale: {len(result_df)}")
        print(f"Küme sayısı: {optimal_k}")
        print(f"Silhouette Score: {clustering.silhouette_score:.4f}")
        
        # Her küme için özet
        for cluster_id, info in cluster_analysis.items():
            print(f"\nKüme {cluster_id}:")
            print(f"  - Boyut: {info['size']} makale ({info['percentage']:.1f}%)")
            print(f"  - Ana kelimeler: {list(info['top_words'].keys())[:5]}")
            print(f"  - Ana kategoriler: {list(info['top_categories'].keys())[:3]}")
        
        return result_df, cluster_analysis
        
    finally:
        clustering.stop_spark()

def generate_report(cluster_analysis, output_file="akademik_makaleler_raporu.txt"):
    """Analiz raporu oluştur"""
    print(f"\nRapor oluşturuluyor: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("AKADEMIK MAKALELERIN KÜMELEME ANALİZİ RAPORU\n")
        f.write("="*60 + "\n")
        f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("GENEL BİLGİLER\n")
        f.write("-"*30 + "\n")
        total_papers = sum(info['size'] for info in cluster_analysis.values())
        f.write(f"Toplam Makale Sayısı: {total_papers}\n")
        f.write(f"Küme Sayısı: {len(cluster_analysis)}\n\n")
        
        for cluster_id, info in cluster_analysis.items():
            f.write(f"KÜME {cluster_id}\n")
            f.write("-"*20 + "\n")
            f.write(f"Boyut: {info['size']} makale ({info['percentage']:.1f}%)\n")
            f.write(f"Ana Anahtar Kelimeler: {', '.join(list(info['top_words'].keys())[:10])}\n")
            f.write(f"Ana Kategoriler: {', '.join(list(info['top_categories'].keys())[:5])}\n")
            f.write("Örnek Makaleler:\n")
            for i, title in enumerate(info['sample_titles'], 1):
                f.write(f"  {i}. {title}\n")
            f.write("\n")
    
    print(f"Rapor {output_file} dosyasına kaydedildi")

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Akademik Makalelerin Kümelenmesi')
    parser.add_argument('--collect-data', action='store_true', 
                       help='ArXiv\'den veri topla')
    parser.add_argument('--cluster', action='store_true', 
                       help='Kümeleme analizi yap')
    parser.add_argument('--full-pipeline', action='store_true', 
                       help='Tüm pipeline\'ı çalıştır')
    parser.add_argument('--max-results', type=int, default=1000,
                       help='Maksimum makale sayısı (default: 1000)')
    parser.add_argument('--vocab-size', type=int, default=5000,
                       help='TF-IDF kelime dağarcığı boyutu (default: 5000)')
    parser.add_argument('--data-file', type=str, default='data/arxiv_papers.csv',
                       help='Veri dosyası yolu')
    
    args = parser.parse_args()
    
    # Eğer hiçbir argüman verilmemişse full pipeline çalıştır
    if not (args.collect_data or args.cluster or args.full_pipeline):
        args.full_pipeline = True
    
    print("AKADEMIK MAKALELERIN KÜMELEME ANALİZİ")
    print("="*60)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dizinleri oluştur
    ensure_directories()
    
    cluster_analysis = None
    
    if args.full_pipeline or args.collect_data:
        # Veri toplama
        data_file = collect_data(
            max_results=args.max_results,
            output_file=args.data_file
        )
    else:
        data_file = args.data_file
    
    if args.full_pipeline or args.cluster:
        # Kümeleme
        result_df, cluster_analysis = perform_clustering(
            data_file=data_file,
            vocab_size=args.vocab_size
        )
    
    if cluster_analysis:
        # Rapor oluştur
        generate_report(cluster_analysis)
    
    print(f"\nTamamlanma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Analiz tamamlandı!")
    print("\nÇıktılar:")
    print("- data/arxiv_papers.csv: Ham veri")
    print("- data/clustered_papers.csv: Kümelenmiş veri")
    print("- visualizations/: Görselleştirmeler")
    print("- akademik_makaleler_raporu.txt: Analiz raporu")

if __name__ == "__main__":
    main() 