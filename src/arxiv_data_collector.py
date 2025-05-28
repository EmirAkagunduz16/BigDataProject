"""
ArXiv Veri Toplama Modülü
Bu modül ArXiv API'sini kullanarak akademik makaleleri toplar.
"""

import arxiv
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class ArXivDataCollector:
    """ArXiv'den akademik makale verisi toplayan sınıf"""
    
    def __init__(self, max_results: int = 1000, delay: float = 1.0):
        """
        Args:
            max_results: Toplanacak maksimum makale sayısı
            delay: API istekleri arasındaki bekleme süresi (saniye)
        """
        self.max_results = max_results
        self.delay = delay
        self.client = arxiv.Client()
    
    def collect_papers_by_category(self, categories: List[str]) -> pd.DataFrame:
        """
        Belirtilen kategorilerdeki makaleleri toplar
        
        Args:
            categories: ArXiv kategori listesi (örn: ['cs.AI', 'cs.ML', 'physics.gen-ph'])
        
        Returns:
            DataFrame: Toplanan makale verileri
        """
        all_papers = []
        
        for category in categories:
            print(f"Kategori işleniyor: {category}")
            
            # ArXiv sorgusu oluştur
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=self.max_results // len(categories),
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers_in_category = []
            
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
                    'doi': paper.doi
                }
                papers_in_category.append(paper_data)
                
                # Rate limiting
                time.sleep(self.delay)
            
            all_papers.extend(papers_in_category)
            print(f"{category} kategorisinden {len(papers_in_category)} makale toplandı")
        
        df = pd.DataFrame(all_papers)
        print(f"Toplam {len(df)} makale toplandı")
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
    collector = ArXivDataCollector(max_results=500, delay=0.5)
    
    # Popüler kategoriler
    categories = [
        'cs.AI',      # Artificial Intelligence
        'cs.ML',      # Machine Learning
        'cs.CV',      # Computer Vision
        'cs.NLP',     # Natural Language Processing
        'physics.gen-ph',  # General Physics
        'math.ST',    # Statistics Theory
        'q-bio.QM',   # Quantitative Methods
        'econ.EM'     # Econometrics
    ]
    
    print("ArXiv'den veri toplama başlıyor...")
    
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
    print(df_clean['primary_category'].value_counts().head(10))

if __name__ == "__main__":
    main() 