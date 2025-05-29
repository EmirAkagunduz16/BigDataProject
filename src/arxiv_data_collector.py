"""
ArXiv Veri Toplama Modülü - Optimized Version with Progress Tracking
Bu modül ArXiv API'sini kullanarak akademik makaleleri hızlı bir şekilde toplar.
"""

import arxiv
import pandas as pd
import time
import re
from typing import List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class ArXivDataCollector:
    """ArXiv'den akademik makale verisi toplayan sınıf - Optimized with Progress Tracking"""
    
    def __init__(self, max_results: int = 1000, delay: float = 0.1, max_workers: int = 4, 
                 progress_callback: Optional[Callable[[str, int], None]] = None):
        """
        Args:
            max_results: Toplanacak maksimum makale sayısı
            delay: API istekleri arasındaki bekleme süresi (saniye) - azaltıldı
            max_workers: Paralel thread sayısı
            progress_callback: İlerleme bildirimi için callback fonksiyonu
        """
        self.max_results = max_results
        self.delay = delay
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.client = arxiv.Client(page_size=1000, delay_seconds=delay, num_retries=3)
        self._lock = threading.Lock()
        self._collected_count = 0
        self._total_target = 0
        
        # Define relevant categories to focus on
        self.relevant_categories = {
            # Computer Science
            'cs.AI', 'cs.ML', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE', 'cs.IR', 
            'cs.RO', 'cs.CR', 'cs.DB', 'cs.SE', 'cs.DS', 'cs.DC', 'cs.HC',
            'cs.SY', 'cs.CC', 'cs.CG', 'cs.DM', 'cs.FL', 'cs.GT', 'cs.HCI',
            # Mathematics
            'math.ST', 'math.PR', 'math.OC', 'math.NA', 'math.CO', 'math.IT',
            # Statistics
            'stat.ML', 'stat.ME', 'stat.TH', 'stat.AP', 'stat.CO',
            # Relevant Physics (computational/data-focused)
            'physics.data-an', 'physics.comp-ph',
            # Economics (quantitative)
            'econ.EM', 'econ.TH'
        }
    
    def _update_progress(self, message: str):
        """İlerleme durumunu günceller"""
        if self.progress_callback:
            percentage = min(90, int((self._collected_count / self._total_target) * 70) + 30) if self._total_target > 0 else 30
            self.progress_callback(message, percentage)
    
    def _collect_category_batch(self, category: str, target_count: int) -> List[dict]:
        """Tek kategori için makale toplar - paralel işlem için"""
        search = arxiv.Search(query=f"cat:{category}", max_results=target_count * 2,
                            sort_by=arxiv.SortCriterion.SubmittedDate,
                            sort_order=arxiv.SortOrder.Descending)
        
        papers = []
        count = 0
        
        try:
            self._update_progress(f"Collecting from category: {category}")
            
            for paper in self.client.results(search):
                if count >= target_count:
                    break
                
                # CATEGORY FILTER - Only accept if primary category is relevant
                if paper.primary_category in self.relevant_categories:
                    paper_data = {
                        'id': paper.entry_id, 'title': paper.title, 'summary': paper.summary,
                        'authors': ', '.join([author.name for author in paper.authors]),
                        'published': paper.published, 'updated': paper.updated,
                        'categories': ', '.join(paper.categories),
                        'primary_category': paper.primary_category,
                        'pdf_url': paper.pdf_url, 'doi': paper.doi
                    }
                    papers.append(paper_data)
                    count += 1
                    
                    with self._lock:
                        self._collected_count += 1
                        if self._collected_count % 50 == 0:  # Her 50 makalede bir güncelle
                            self._update_progress(f"Collected {self._collected_count} papers...")
                
                if self.delay > 0:
                    time.sleep(self.delay)
        except Exception as e:
            self._update_progress(f"Error in category {category}: {str(e)}")
            
        return papers
    
    def collect_papers_by_category(self, categories: List[str]) -> pd.DataFrame:
        """
        Belirtilen kategorilerdeki makaleleri paralel olarak toplar - HIZLANDIRILMIŞ
        """
        self._collected_count = 0
        self._total_target = self.max_results
        all_papers = []
        target_per_category = max(10, self.max_results // len(categories))
        
        self._update_progress(f"Starting collection from {len(categories)} categories")
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(categories))) as executor:
            future_to_category = {executor.submit(self._collect_category_batch, category, target_per_category): category 
                                for category in categories}
            
            completed = 0
            for future in as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    papers = future.result(timeout=180)  # 3 dakika timeout
                    all_papers.extend(papers)
                    completed += 1
                    progress = 30 + int((completed / len(categories)) * 50)
                    self._update_progress(f"Completed category {category}, total papers: {len(all_papers)}", progress)
                except Exception as e:
                    self._update_progress(f"Failed category {category}: {str(e)}")
        
        return pd.DataFrame(all_papers)
    
    def _collect_primary_category_batch(self, category: str, target_count: int) -> List[dict]:
        """Primary kategori için optimized batch collection"""
        search = arxiv.Search(query=f"cat:{category}", max_results=min(target_count * 3, 500),
                            sort_by=arxiv.SortCriterion.SubmittedDate,
                            sort_order=arxiv.SortOrder.Descending)
        
        papers = []
        processed = 0
        start_time = time.time()
        
        try:
            self._update_progress(f"Processing primary category: {category}")
            
            for paper in self.client.results(search):
                # Timeout kontrolü - maksimum 2 dakika per kategori
                if time.time() - start_time > 120:
                    self._update_progress(f"Timeout reached for {category} after {processed} papers, found {len(papers)} matching")
                    break
                
                processed += 1
                
                # Her 50 işlemde bir progress güncelle
                if processed % 50 == 0:
                    elapsed = time.time() - start_time
                    self._update_progress(f"Processed {processed} papers in {category} ({elapsed:.1f}s), found {len(papers)} matching...")
                
                # PRIMARY CATEGORY FILTER - Only accept if primary category is relevant
                if paper.primary_category in self.relevant_categories:
                    paper_data = {
                        'id': paper.entry_id, 'title': paper.title, 'summary': paper.summary,
                        'authors': ', '.join([author.name for author in paper.authors]),
                        'published': paper.published, 'updated': paper.updated,
                        'categories': ', '.join(paper.categories),
                        'primary_category': paper.primary_category,
                        'pdf_url': paper.pdf_url, 'doi': paper.doi
                    }
                    papers.append(paper_data)
                    
                    with self._lock:
                        self._collected_count += 1
                    
                    # Target count check
                    if len(papers) >= target_count:
                        self._update_progress(f"Reached target for {category}: {len(papers)} papers")
                        break
                
                if self.delay > 0:
                    time.sleep(self.delay)
                    
        except Exception as e:
            self._update_progress(f"Error in category {category}: {str(e)}")
            
        self._update_progress(f"Completed {category}: collected {len(papers)} papers from {processed} processed")
        return papers
    
    def collect_papers_by_primary_category(self, categories: List[str]) -> pd.DataFrame:
        """
        Primary kategorilerdeki makaleleri paralel olarak toplar - HIZLANDIRILMIŞ
        """
        self._collected_count = 0
        self._total_target = self.max_results
        all_papers = []
        target_per_category = max(10, self.max_results // len(categories))
        
        self._update_progress(f"Starting primary category collection from {len(categories)} categories", 30)
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(categories))) as executor:
            future_to_category = {executor.submit(self._collect_primary_category_batch, category, target_per_category): category 
                                for category in categories}
            
            completed = 0
            for future in as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    papers = future.result(timeout=300)  # 5 dakika timeout
                    all_papers.extend(papers)
                    completed += 1
                    progress = 30 + int((completed / len(categories)) * 50)
                    self._update_progress(f"Completed {category}: {len(papers)} papers, total: {len(all_papers)}", progress)
                except Exception as e:
                    self._update_progress(f"Failed category {category}: {str(e)}")
                    completed += 1
        
        self._update_progress(f"Collection completed: {len(all_papers)} papers total", 80)
        return pd.DataFrame(all_papers)
    
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
            search = arxiv.Search(query=f'all:"{keyword}"',
                                max_results=self.max_results // len(keywords),
                                sort_by=arxiv.SortCriterion.Relevance)
            
            papers_for_keyword = []
            
            for paper in self.client.results(search):
                paper_data = {
                    'id': paper.entry_id, 'title': paper.title, 'summary': paper.summary,
                    'authors': ', '.join([author.name for author in paper.authors]),
                    'published': paper.published, 'updated': paper.updated,
                    'categories': ', '.join(paper.categories),
                    'primary_category': paper.primary_category,
                    'pdf_url': paper.pdf_url, 'doi': paper.doi,
                    'search_keyword': keyword
                }
                papers_for_keyword.append(paper_data)
                time.sleep(self.delay)
            
            all_papers.extend(papers_for_keyword)
        
        return pd.DataFrame(all_papers)
    
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
        
        return df_clean
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """Veriyi CSV olarak kaydet"""
        df.to_csv(filepath, index=False)

def main():
    """Ana fonksiyon - örnek kullanım"""
    collector = ArXivDataCollector(max_results=1000, delay=0.1, max_workers=4)
    
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