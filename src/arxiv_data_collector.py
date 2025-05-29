"""
ArXiv Veri Toplama Modülü - Optimized Version
Bu modül ArXiv API'sini kullanarak akademik makaleleri hızlı bir şekilde toplar.
"""

import arxiv
import pandas as pd
import time
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ArXivDataCollector:
    """ArXiv'den akademik makale verisi toplayan sınıf - Optimized"""
    
    def __init__(self, max_results: int = 1000, delay: float = 0.1, max_workers: int = 4):
        """
        Args:
            max_results: Toplanacak maksimum makale sayısı
            delay: API istekleri arasındaki bekleme süresi (saniye) - azaltıldı
            max_workers: Paralel thread sayısı
        """
        self.max_results = max_results
        self.delay = delay  # 1.0'dan 0.1'e düşürüldü
        self.max_workers = max_workers
        self.client = arxiv.Client(
            page_size=1000,  # Daha büyük page size
            delay_seconds=delay,  # ArXiv client'ın kendi rate limiting'i
            num_retries=3
        )
        self._lock = threading.Lock()
    
    def _collect_category_batch(self, category: str, target_count: int) -> List[dict]:
        """Tek kategori için makale toplar - paralel işlem için"""
        print(f"🔄 Kategori işleniyor: {category}")
        
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=target_count * 2,  # Fazladan topla, sonra filtrele
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        count = 0
        
        try:
            for paper in self.client.results(search):
                if count >= target_count:
                    break
                    
                paper_data = {
                    'id': paper.entry_id,
                    'title': paper.title,
                    'summary': paper.summary,
                    'authors': ', '.join([author.name for author in paper.authors]),
                    'published': paper.published,
                    'updated': paper.updated,
                    'categories': ', '.join(paper.categories),
                    'primary_category': paper.primary_category,
                    'pdf_url': paper.pdf_url,
                    'doi': paper.doi
                }
                papers.append(paper_data)
                count += 1
                
                # Minimal delay
                if self.delay > 0:
                    time.sleep(self.delay)
        
        except Exception as e:
            print(f"❌ {category} kategorisinde hata: {str(e)}")
            
        print(f"✅ {category}: {len(papers)} makale toplandı")
        return papers
    
    def collect_papers_by_category(self, categories: List[str]) -> pd.DataFrame:
        """
        Belirtilen kategorilerdeki makaleleri paralel olarak toplar - HIZLANDIRILMIŞ
        """
        print(f"🚀 {len(categories)} kategori paralel olarak işleniyor...")
        
        all_papers = []
        target_per_category = max(10, self.max_results // len(categories))
        
        # Paralel işlem ile kategorileri topla
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(categories))) as executor:
            # Her kategori için task oluştur
            future_to_category = {
                executor.submit(self._collect_category_batch, category, target_per_category): category 
                for category in categories
            }
            
            # Sonuçları topla
            for future in as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    papers = future.result(timeout=120)  # 2 dakika timeout
                    all_papers.extend(papers)
                except Exception as e:
                    print(f"❌ {category} kategorisi başarısız: {str(e)}")
        
        df = pd.DataFrame(all_papers)
        print(f"🎉 Toplam {len(df)} makale toplandı")
        return df
    
    def _collect_primary_category_batch(self, category: str, target_count: int) -> List[dict]:
        """Primary kategori için optimized batch collection"""
        print(f"🔄 Primary kategori işleniyor: {category}")
        
        # Daha spesifik sorgu kullan
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=target_count * 3,  # Primary olmayanlara karşı buffer
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        
        try:
            for paper in self.client.results(search):
                # Sadece primary category eşleşenleri al
                if paper.primary_category == category:
                    paper_data = {
                        'id': paper.entry_id,
                        'title': paper.title,
                        'summary': paper.summary,
                        'authors': ', '.join([author.name for author in paper.authors]),
                        'published': paper.published,
                        'updated': paper.updated,
                        'categories': ', '.join(paper.categories),
                        'primary_category': paper.primary_category,
                        'pdf_url': paper.pdf_url,
                        'doi': paper.doi
                    }
                    papers.append(paper_data)
                    
                    # Hedef sayıya ulaştık mı?
                    if len(papers) >= target_count:
                        break
                
                # Minimal delay
                if self.delay > 0:
                    time.sleep(self.delay)
                    
        except Exception as e:
            print(f"❌ {category} primary kategorisinde hata: {str(e)}")
        
        print(f"✅ {category} primary: {len(papers)} makale toplandı")
        return papers
    
    def collect_papers_by_primary_category(self, categories: List[str]) -> pd.DataFrame:
        """
        Primary kategorilerdeki makaleleri paralel olarak toplar - HIZLANDIRILMIŞ
        """
        print(f"🚀 {len(categories)} primary kategori paralel olarak işleniyor...")
        
        all_papers = []
        target_per_category = max(10, self.max_results // len(categories))
        
        # Paralel işlem
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(categories))) as executor:
            future_to_category = {
                executor.submit(self._collect_primary_category_batch, category, target_per_category): category 
                for category in categories
            }
            
            for future in as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    papers = future.result(timeout=120)
                    all_papers.extend(papers)
                except Exception as e:
                    print(f"❌ {category} primary kategorisi başarısız: {str(e)}")
        
        df = pd.DataFrame(all_papers)
        print(f"🎉 Toplam {len(df)} primary kategori makale toplandı")
        return df
    
    def collect_papers_by_keywords(self, keywords: List[str]) -> pd.DataFrame:
        """
        Anahtar kelimeler ile makale toplar
        
        Args:
            keywords: Aranacak anahtar kelimeler
        
        Returns:
            DataFrame: Toplanan makale verileri
        """
        all_papers = []
        
        for keyword in keywords:
            print(f"Anahtar kelime işleniyor: {keyword}")
            
            # ArXiv sorgusu oluştur
            search = arxiv.Search(
                query=f'all:"{keyword}"',
                max_results=self.max_results // len(keywords),
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers_for_keyword = []
            
            for paper in self.client.results(search):
                paper_data = {
                    'id': paper.entry_id,
                    'title': paper.title,
                    'summary': paper.summary,
                    'authors': ', '.join([author.name for author in paper.authors]),
                    'published': paper.published,
                    'updated': paper.updated,
                    'categories': ', '.join(paper.categories),
                    'primary_category': paper.primary_category,
                    'pdf_url': paper.pdf_url,
                    'doi': paper.doi,
                    'search_keyword': keyword
                }
                papers_for_keyword.append(paper_data)
                
                # Rate limiting
                time.sleep(self.delay)
            
            all_papers.extend(papers_for_keyword)
            print(f"'{keyword}' için {len(papers_for_keyword)} makale toplandı")
        
        df = pd.DataFrame(all_papers)
        print(f"Toplam {len(df)} makale toplandı")
        return df
    
    def clean_text(self, text: str) -> str:
        """Metni temizler"""
        if not isinstance(text, str):
            return ""
        
        # HTML etiketlerini kaldır
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fazla boşlukları kaldır
        text = re.sub(r'\s+', ' ', text)
        
        # Başındaki ve sonundaki boşlukları kaldır
        text = text.strip()
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame'i ön işlemden geçirir"""
        df_clean = df.copy()
        
        # Metin sütunlarını temizle
        text_columns = ['title', 'summary']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.clean_text)
        
        # Duplicate'leri kaldır
        df_clean = df_clean.drop_duplicates(subset=['title'], keep='first')
        
        # Boş title veya summary olan satırları kaldır
        df_clean = df_clean.dropna(subset=['title', 'summary'])
        df_clean = df_clean[df_clean['title'].str.strip() != '']
        df_clean = df_clean[df_clean['summary'].str.strip() != '']
        
        # Çok kısa özetleri filtrele (minimum 100 karakter)
        df_clean = df_clean[df_clean['summary'].str.len() >= 100]
        
        print(f"Temizlik sonrası {len(df_clean)} makale kaldı")
        return df_clean
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """Veriyi CSV olarak kaydet"""
        df.to_csv(filepath, index=False)
        print(f"Veri {filepath} dosyasına kaydedildi")

def main():
    """Ana fonksiyon - örnek kullanım"""
    # Veri toplayıcı oluştur
    collector = ArXivDataCollector(max_results=5000, delay=0.1)  # Daha fazla veri, daha hızlı
    
    # Genişletilmiş ve dengeli kategoriler
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
    
    print("ArXiv'den dengeli veri toplama başlıyor...")
    print(f"Toplam {len(categories)} kategori, her kategoriden yaklaşık {collector.max_results // len(categories)} makale")
    
    # Kategorilere göre veri topla
    df = collector.collect_papers_by_category(categories)
    
    # Veriyi temizle
    df_clean = collector.preprocess_dataframe(df)
    
    # Kaydet
    collector.save_data(df_clean, "../data/arxiv_papers.csv")
    
    # Temel istatistikler
    print("\n=== VERİ İSTATİSTİKLERİ ===")
    print(f"Toplam makale sayısı: {len(df_clean)}")
    print(f"Benzersiz kategori sayısı: {df_clean['primary_category'].nunique()}")
    print("\nKategori dağılımı:")
    category_dist = df_clean['primary_category'].value_counts()
    print(category_dist.head(15))
    
    # Kategoriler arası denge kontrolü
    print(f"\nEn az makale sayısı: {category_dist.min()}")
    print(f"En fazla makale sayısı: {category_dist.max()}")
    print(f"Ortalama makale sayısı: {category_dist.mean():.1f}")
    print(f"Standart sapma: {category_dist.std():.1f}")

if __name__ == "__main__":
    main() 