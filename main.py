import os
import sys
import argparse
from datetime import datetime

sys.path.append('src')
from arxiv_data_collector import ArXivDataCollector
from spark_clustering import SparkTextClustering

def ensure_directories():
    directories = ['data', 'visualizations', 'logs']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)

def collect_data(categories=None, max_results=3000, output_file="data/arxiv_papers.csv", use_primary_only=False):
    if categories is None:
        categories = [
            # Bilgisayar Bilimleri - AI & ML (8 kategori)
            'cs.AI', 'cs.ML', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE', 'cs.IR', 'cs.RO',
            
            # Bilgisayar Bilimleri - Sistemler (5 kategori)  
            'cs.CR', 'cs.DB', 'cs.SE', 'cs.DS', 'cs.DC',
            
            # Matematik & İstatistik (6 kategori)
            'math.ST', 'math.PR', 'math.OC', 'math.NA', 'stat.ML', 'stat.ME',
            
            # Fizik & Mühendislik (4 kategori)
            'physics.data-an', 'physics.comp-ph', 'cond-mat.stat-mech', 'cond-mat.soft',
            
            # Biyoloji & Tıp (4 kategori)
            'q-bio.QM', 'q-bio.MN', 'q-bio.CB', 'q-bio.BM',
            
            # Ekonomi & Sosyal Bilimler (3 kategori)
            'econ.EM', 'econ.TH', 'econ.GN'
        ]
    
    collector = ArXivDataCollector(max_results=max_results, delay=0.1, max_workers=4)
    
    if use_primary_only:
        df = collector.collect_papers_by_primary_category(categories)
    else:
        df = collector.collect_papers_by_category(categories)
        df_clean = collector.preprocess_dataframe(df)
        df = df_clean[df_clean['primary_category'].isin(categories)]
    
    df_clean = collector.preprocess_dataframe(df)
    collector.save_data(df_clean, output_file)
    return output_file

def perform_clustering(data_file="data/arxiv_papers.csv", k_range=None, vocab_size=3000):
    clustering = SparkTextClustering()
    try:
        df = clustering.load_data(data_file)
        df = clustering.preprocess_text(['title', 'summary'])
        df_features = clustering.create_features(vocab_size=vocab_size, min_df=2)
        
        optimal_k = 5 if not k_range else clustering.find_optimal_k(range(k_range[0], k_range[1] + 1), 30)[0]
        
        df_clustered = clustering.perform_clustering(k=optimal_k, max_iterations=50)
        cluster_analysis = clustering.analyze_clusters(top_words=10)
        clustering.create_visualizations()
        result_df = clustering.save_results()
        return result_df
    except Exception as e:
        raise
    finally:
        clustering.stop_spark()

def generate_report(cluster_analysis, output_file="akademik_makaleler_raporu.txt"):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("AKADEMIK MAKALELERIN KÜMELEME ANALİZİ RAPORU\n")
        f.write("="*60 + "\n")
        f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_papers = sum(info['size'] for info in cluster_analysis.values())
        f.write(f"Toplam Makale: {total_papers}, Küme Sayısı: {len(cluster_analysis)}\n\n")
        
        for cluster_id, info in cluster_analysis.items():
            f.write(f"KÜME {cluster_id} - {info['size']} makale\n")
            f.write(f"Kelimeler: {', '.join(list(info['top_words'].keys())[:5])}\n")
            f.write(f"Kategoriler: {', '.join(list(info['top_categories'].keys())[:3])}\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect-data', action='store_true')
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--full-pipeline', action='store_true')
    parser.add_argument('--max-results', type=int, default=5000)
    parser.add_argument('--vocab-size', type=int, default=3000)
    parser.add_argument('--data-file', type=str, default='data/arxiv_papers.csv')
    parser.add_argument('--use-primary-only', action='store_true')
    
    args = parser.parse_args()
    
    if not (args.collect_data or args.cluster or args.full_pipeline):
        args.full_pipeline = True
    
    ensure_directories()
    
    if args.full_pipeline or args.collect_data:
        data_file = collect_data(max_results=args.max_results, output_file=args.data_file, use_primary_only=args.use_primary_only)
    else:
        data_file = args.data_file
    
    if args.full_pipeline or args.cluster:
        perform_clustering(data_file=data_file, vocab_size=args.vocab_size)

if __name__ == "__main__":
    main() 