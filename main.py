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

def collect_data(categories=None, max_results=3000, output_file="data/arxiv_papers.csv", use_primary_only=False):
    """ArXiv'den veri topla"""
    print("\n" + "="*50)
    print("VERİ TOPLAMA AŞAMASI")
    print("="*50)
    
    if categories is None:
        categories = [
            # Computer Science - Core AI & ML
            'cs.AI',      # Artificial Intelligence
            'cs.ML',      # Machine Learning  
            'cs.LG',      # Learning
            'cs.CV',      # Computer Vision
            'cs.CL',      # Computational Linguistics
            'cs.NE',      # Neural and Evolutionary Computing
            'cs.IR',      # Information Retrieval
            
            # Computer Science - Systems & Theory
            'cs.CR',      # Cryptography and Security
            'cs.DB',      # Databases
            'cs.DC',      # Distributed Computing
            'cs.DS',      # Data Structures and Algorithms
            'cs.HC',      # Human-Computer Interaction
            'cs.RO',      # Robotics
            'cs.SE',      # Software Engineering
            'cs.SY',      # Systems and Control
            'cs.PL',      # Programming Languages
            'cs.OS',      # Operating Systems
            'cs.AR',      # Architecture
            'cs.CC',      # Computational Complexity
            'cs.DM',      # Discrete Mathematics
            'cs.FL',      # Formal Languages and Automata
            'cs.GT',      # Computer Science and Game Theory
            'cs.IT',      # Information Theory
            'cs.LO',      # Logic in Computer Science
            'cs.MA',      # Multiagent Systems
            'cs.MM',      # Multimedia
            'cs.MS',      # Mathematical Software
            'cs.NA',      # Numerical Analysis
            'cs.NI',      # Networking and Internet Architecture
            'cs.OH',      # Other Computer Science
            'cs.PF',      # Performance
            'cs.SC',      # Symbolic Computation
            'cs.SD',      # Sound
            
            # Mathematics & Statistics - Extended
            'math.ST',    # Statistics Theory
            'math.PR',    # Probability
            'math.OC',    # Optimization and Control
            'math.NA',    # Numerical Analysis
            'math.AP',    # Analysis of PDEs
            'math.CO',    # Combinatorics
            'math.DG',    # Differential Geometry
            'math.DS',    # Dynamical Systems
            'math.FA',    # Functional Analysis
            'math.GM',    # General Mathematics
            'math.GR',    # Group Theory
            'math.GT',    # Geometric Topology
            'math.HO',    # History and Overview
            'math.IT',    # Information Theory
            'math.KT',    # K-Theory and Homology
            'math.LO',    # Logic
            'math.MG',    # Metric Geometry
            'math.MP',    # Mathematical Physics
            'math.NT',    # Number Theory
            'math.QA',    # Quantum Algebra
            'math.RA',    # Rings and Algebras
            'math.RT',    # Representation Theory
            'math.SG',    # Symplectic Geometry
            'math.SP',    # Spectral Theory
            'stat.ML',    # Machine Learning (Statistics)
            'stat.ME',    # Methodology
            'stat.TH',    # Statistics Theory
            'stat.AP',    # Applications
            'stat.CO',    # Computation
            'stat.OT',    # Other Statistics
            
            # Physics & Interdisciplinary - Extended
            'physics.data-an',    # Data Analysis
            'physics.comp-ph',    # Computational Physics
            'physics.soc-ph',     # Physics and Society
            'physics.bio-ph',     # Biological Physics
            'physics.chem-ph',    # Chemical Physics
            'physics.class-ph',   # Classical Physics
            'physics.flu-dyn',    # Fluid Dynamics
            'physics.gen-ph',     # General Physics
            'physics.geo-ph',     # Geophysics
            'physics.hist-ph',    # History and Philosophy of Physics
            'physics.ins-det',    # Instrumentation and Detectors
            'physics.med-ph',     # Medical Physics
            'physics.optics',     # Optics
            'physics.plasm-ph',   # Plasma Physics
            'physics.pop-ph',     # Popular Physics
            'physics.space-ph',   # Space Physics
            'cond-mat.dis-nn',    # Disordered Systems and Neural Networks
            'cond-mat.mes-hall',  # Mesoscale and Nanoscale Physics
            'cond-mat.mtrl-sci',  # Materials Science
            'cond-mat.other',     # Other Condensed Matter
            'cond-mat.quant-gas', # Quantum Gases
            'cond-mat.soft',      # Soft Condensed Matter
            'cond-mat.stat-mech', # Statistical Mechanics
            'cond-mat.str-el',    # Strongly Correlated Electrons
            'cond-mat.supr-con',  # Superconductivity
            
            # Biology & Life Sciences - Extended
            'q-bio.BM',    # Biomolecules
            'q-bio.CB',    # Cell Behavior
            'q-bio.GN',    # Genomics
            'q-bio.MN',    # Molecular Networks
            'q-bio.NC',    # Neurons and Cognition
            'q-bio.OT',    # Other Quantitative Biology
            'q-bio.PE',    # Populations and Evolution
            'q-bio.QM',    # Quantitative Methods
            'q-bio.SC',    # Subcellular Processes
            'q-bio.TO',    # Tissues and Organs
            
            # Economics & Finance - Extended
            'econ.EM',     # Econometrics
            'econ.GN',     # General Economics
            'econ.TH',     # Theoretical Economics
            'q-fin.CP',    # Computational Finance
            'q-fin.EC',    # Economics
            'q-fin.GN',    # General Finance
            'q-fin.MF',    # Mathematical Finance
            'q-fin.PM',    # Portfolio Management
            'q-fin.PR',    # Pricing of Securities
            'q-fin.RM',    # Risk Management
            'q-fin.ST',    # Statistical Finance
            'q-fin.TR',    # Trading and Market Microstructure
            
            # Engineering & Applied Sciences
            'nlin.AO',     # Adaptation and Self-Organizing Systems
            'nlin.CD',     # Chaotic Dynamics
            'nlin.CG',     # Cellular Automata and Lattice Gases
            'nlin.PS',     # Pattern Formation and Solitons
            'nlin.SI',     # Exactly Solvable and Integrable Systems
            
            # Additional interdisciplinary fields
            'astro-ph.CO', # Cosmology and Nongalactic Astrophysics
            'astro-ph.EP', # Earth and Planetary Astrophysics
            'astro-ph.GA', # Astrophysics of Galaxies
            'astro-ph.HE', # High Energy Astrophysical Phenomena
            'astro-ph.IM', # Instrumentation and Methods for Astrophysics
            'astro-ph.SR', # Solar and Stellar Astrophysics
        ]
    
    collector = ArXivDataCollector(max_results=max_results, delay=0.3)
    
    print(f"Kategoriler ({len(categories)} adet): {categories}")
    print(f"Maksimum toplam makale sayısı: {max_results}")
    print(f"Kategori başına yaklaşık: {max_results // len(categories)} makale")
    
    if use_primary_only:
        print("Primary category filtreleme kullanılıyor...")
        # Yeni metodu kullan
        df = collector.collect_papers_by_primary_category(categories)
    else:
        print("Standart kategori arama kullanılıyor...")
        # Veri topla
        df = collector.collect_papers_by_category(categories)
        
        # Temizle
        df_clean = collector.preprocess_dataframe(df)
        
        # Sadece istenen primary kategorileri filtrele
        print(f"Primary kategori filtreleme uygulanıyor...")
        original_count = len(df_clean)
        df = df_clean[df_clean['primary_category'].isin(categories)]
        print(f"Filtreleme sonrası: {len(df)} makale kaldı (önceki: {original_count})")
    
    # Son temizlik işlemleri
    df_clean = collector.preprocess_dataframe(df)
    
    # Kaydet
    collector.save_data(df_clean, output_file)
    
    # İstatistikler
    print(f"\nToplanan makale sayısı: {len(df_clean)}")
    print(f"Benzersiz kategori sayısı: {df_clean['primary_category'].nunique()}")
    
    # Kategori dengesi analizi
    category_dist = df_clean['primary_category'].value_counts()
    print(f"\nKategori Denge Analizi:")
    print(f"En az makale: {category_dist.min()}")
    print(f"En fazla makale: {category_dist.max()}")
    print(f"Ortalama: {category_dist.mean():.1f}")
    print(f"Standart sapma: {category_dist.std():.1f}")
    
    print("\nEn yaygın kategoriler:")
    print(category_dist.head(10))
    
    # Eksik kategorileri kontrol et
    missing_categories = set(categories) - set(df_clean['primary_category'].unique())
    if missing_categories:
        print(f"\nEksik kategoriler: {missing_categories}")
    
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
    parser.add_argument('--max-results', type=int, default=5000,
                       help='Maksimum makale sayısı (default: 5000)')
    parser.add_argument('--vocab-size', type=int, default=5000,
                       help='TF-IDF kelime dağarcığı boyutu (default: 5000)')
    parser.add_argument('--data-file', type=str, default='data/arxiv_papers.csv',
                       help='Veri dosyası yolu')
    parser.add_argument('--use-primary-only', action='store_true',
                       help='Sadece primary category makaleleri topla (daha kesin)')
    
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
            output_file=args.data_file,
            use_primary_only=args.use_primary_only
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