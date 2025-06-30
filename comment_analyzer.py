import json
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
from datetime import datetime
import torch

"""
Türkçe BERT modelini yükler (önce local, yoksa internetten), yorumları analiz eder, fine-tune ve model kaydetme işlemlerini yapar.
"""
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class CommentAnalyzer:
    def __init__(self):
        """Analizör başlatılır ve modeller yüklenir"""
        print("AI modelleri yükleniyor...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        print("AI modelleri hazır!")
        
    def load_model(self):
        """ABSA modelini yükler"""
        try:
            model_path = "absa_model_tr"
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        except Exception as e:
            print(f"Model yüklenemedi: {e}")
            raise
    
    def clean_text(self, text):
        """Metni temizler"""
        if not isinstance(text, str):
            return ""
        text = text.replace("<br>", " ").replace("<br/>", " ")
        return " ".join(text.split())
    
    def extract_aspects(self, text):
        """Metinden aspect'leri çıkarır"""
        aspects = []
        keywords = {
            'ses': ['ses', 'audio', 'mikrofon', 'gürültü', 'kalite', 'sessiz', 'yüksek', 'düşük', 'çınlama'],
            'görüntü': ['görüntü', 'video', 'kalite', 'çözünürlük', 'piksel', 'net', 'bulanık', 'hassas', 'kamera'],
            'içerik': ['içerik', 'konu', 'anlatım', 'açıklama', 'bilgi', 'eğitici', 'faydalı', 'öğretici'],
            'teknik': ['teknik', 'hata', 'problem', 'sorun', 'bug', 'çalışmıyor', 'bozuk', 'arıza'],
            'reklam': ['reklam', 'sponsor', 'promosyon', 'tanıtım', 'ürün'],
            'süre': ['süre', 'uzunluk', 'kısa', 'uzun', 'dakika', 'saat']
        }
        
        text_lower = text.lower()
        for aspect, words in keywords.items():
            if any(word in text_lower for word in words):
                aspects.append(aspect)
        return aspects
    
    def analyze_sentiment(self, text, aspects):
        """ABSA modeli ile duygu analizi yapar"""
        try:
            if len(aspects) == 0:
                aspects = ['genel']  # Aspect bulunamadıysa genel analiz yap
                
            results = []
            for aspect in aspects:
                combined_text = f"{aspect} hakkında: {text}"
                inputs = self.tokenizer(
                    combined_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=1).item()
                    sentiment = self.id2label[prediction]
                    score = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
                    results.append({
                        "aspect": aspect,
                        "sentiment": sentiment,
                        "score": score
                    })
            
            # En baskın duyguyu döndür
            if len(results) > 0:
                # Pozitif ve negatif sayısına göre karar ver
                pos_count = sum(1 for r in results if r["sentiment"] == "POSITIVE")
                neg_count = sum(1 for r in results if r["sentiment"] == "NEGATIVE")
                
                if pos_count > neg_count:
                    dominant = "POSITIVE"
                elif neg_count > pos_count:
                    dominant = "NEGATIVE"
                else:
                    dominant = "NEUTRAL"
                
                # Ortalama skor hesapla
                avg_score = sum(r["score"] for r in results) / len(results)
                
                return {
                    "label": dominant,
                    "score": avg_score,
                    "aspects": results
                }
            
            return {"label": "NEUTRAL", "score": 0.5, "aspects": []}
            
        except Exception as e:
            print(f"Model analizi hatası: {e}")
            return {"label": "NEUTRAL", "score": 0.5, "aspects": []}
    
    def analyze_comments(self, comments_data):
        """Tüm yorumları analiz eder"""
        print("Yorumlar analiz ediliyor...")
        
        all_comments = []
        
        # comments_data'nın formatını kontrol et
        if isinstance(comments_data, list):
            comments_list = comments_data
        elif isinstance(comments_data, dict) and 'comments' in comments_data:
            comments_list = comments_data['comments']
        else:
            raise ValueError("comments_data geçersiz format: liste veya 'comments' anahtarlı dictionary olmalı")
        
        # Tüm yorumları düzleştir
        for i, comment in enumerate(comments_list):
            all_comments.append({
                'id': f"comment_{i}_{comment.get('published_at', 'unknown')}",
                'text': comment['text'],
                'author': comment['author'],
                'like_count': comment['like_count'],
                'is_top_level': not comment.get('is_reply', False),
                'parent_id': comment.get('parent_id', None)
            })
        
        # Analiz sonuçları
        analyzed_comments = []
        
        for i, comment in enumerate(all_comments):
            if i % 50 == 0:
                print(f"Analiz ediliyor: {i+1}/{len(all_comments)}")
            
            # Metni temizle
            cleaned_text = self.clean_text(comment['text'])
            
            # Aspect'leri çıkar
            aspects = self.extract_aspects(cleaned_text)
            
            # Sentiment analizi
            sentiment = self.analyze_sentiment(cleaned_text, aspects)
            
            analyzed_comment = {
                **comment,
                'sentiment': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'aspects': sentiment['aspects'],
                'word_count': len(cleaned_text.split())
            }
            
            analyzed_comments.append(analyzed_comment)
        
        return analyzed_comments
    
    def generate_statistics(self, analyzed_comments):
        """Analiz istatistikleri oluşturur"""
        print("İstatistikler hesaplanıyor...")
        
        df = pd.DataFrame(analyzed_comments)
        
        stats = {
            'total_comments': len(df),
            'top_level_comments': len(df[df['is_top_level'] == True]),
            'replies': len(df[df['is_top_level'] == False]),
            'total_likes': df['like_count'].sum(),
            'avg_likes': df['like_count'].mean(),
            'avg_word_count': df['word_count'].mean()
        }
        
        # Sentiment dağılımı
        sentiment_counts = df['sentiment'].value_counts()
        stats['sentiment_distribution'] = {
            'positive': sentiment_counts.get('POSITIVE', 0),
            'negative': sentiment_counts.get('NEGATIVE', 0),
            'neutral': sentiment_counts.get('NEUTRAL', 0)
        }
        
        # Aspect dağılımı
        all_aspects = []
        for comment in analyzed_comments:
            for aspect_data in comment.get('aspects', []):
                all_aspects.append(f"{aspect_data['aspect']}_{aspect_data['sentiment']}")
        
        aspect_counts = Counter(all_aspects)
        stats['aspect_distribution'] = dict(aspect_counts)
        
        return stats
    
    def save_results(self, analyzed_comments, filename=None):
        """Analiz sonuçlarını kaydeder"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comment_analysis_{timestamp}.json"
        
        data = {
            'analyzed_at': datetime.now().isoformat(),
            'total_comments': len(analyzed_comments),
            'analyzed_comments': analyzed_comments
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Analiz sonuçları {filename} dosyasına kaydedildi.")
        return filename

def main():
    """Test fonksiyonu"""
    print("=" * 60)
    print("ABSA MODEL TEST")
    print("=" * 60)
    
    # Test yorumları
    test_comments = [
        "Bu video gerçekten harika! Ses kalitesi çok iyi.",
        "Ses kalitesi çok kötü, hiç anlaşılmıyor.",
        "Görüntü net değil, daha iyi olabilirdi.",
        "İçerik çok faydalı, teşekkürler.",
        "Bu video berbat, hiç beğenmedim."
    ]
    
    # Analizör oluştur
    analyzer = CommentAnalyzer()
    
    print("\nTest Sonuçları:")
    print("-" * 60)
    
    for i, comment in enumerate(test_comments, 1):
        aspects = analyzer.extract_aspects(comment)
        result = analyzer.analyze_sentiment(comment, aspects)
        
        print(f"{i}. {comment[:40]}...")
        print(f"   Aspects: {aspects}")
        print(f"   Sentiment: {result['label']} (Skor: {result['score']:.3f})")
        for aspect_result in result['aspects']:
            print(f"   - {aspect_result['aspect'].upper()}: {aspect_result['sentiment']}")
        print()

if __name__ == "__main__":
    main() 