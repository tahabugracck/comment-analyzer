import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from datetime import datetime
import os

"""
Sonuçları grafik ve word cloud olarak kaydeder.
"""

# Yardımcı fonksiyonlar 
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Ana görselleştirme fonksiyonu 
def visualize_from_json(json_file):
    print(f"\nGörselleştirme başlıyor: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    comments = data['analyzed_comments']
    df = pd.DataFrame(comments)

    # 1. Duygu Dağılımı (Sütun Grafik & Pasta Grafik) 
    plt.figure(figsize=(8, 5))
    sentiment_counts = df['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('Duygu Dağılımı (Sütun Grafik)')
    plt.xlabel('Duygu')
    plt.ylabel('Yorum Sayısı')
    plt.savefig('sentiment_bar.png', dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Duygu Dağılımı (Pasta Grafik)')
    plt.savefig('sentiment_pie.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Aspect-Sentiment Dağılımı (Sütun Grafik) 
    all_aspects = []
    for comment in comments:
        for aspect_data in comment.get('aspects', []):
            all_aspects.append({
                'aspect': aspect_data['aspect'],
                'sentiment': aspect_data['sentiment']
            })
    
    if all_aspects:
        aspect_df = pd.DataFrame(all_aspects)
        aspect_sentiment = pd.crosstab(aspect_df['aspect'], aspect_df['sentiment'])
        
        plt.figure(figsize=(12, 6))
        aspect_sentiment.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'])
        plt.title('Aspect-Bazlı Duygu Analizi')
        plt.xlabel('Aspect')
        plt.ylabel('Yorum Sayısı')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig('topic_bar.png', dpi=200, bbox_inches='tight')
        plt.close()

    # 3. Kelime Bulutu 
    all_text = ' '.join([clean_text(t) for t in df['text'].tolist()])
    if len(all_text) > 10:
        wordcloud = WordCloud(width=1000, height=500, background_color='white', max_words=200, colormap='viridis').generate(all_text)
        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Kelime Bulutu')
        plt.savefig('wordcloud_viz.png', dpi=200, bbox_inches='tight')
        plt.close()

    # 4. Duygu vs Beğeni İlişkisi (Scatter Plot) 
    plt.figure(figsize=(10, 6))
    sentiment_colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
    colors = [sentiment_colors.get(sent, 'blue') for sent in df['sentiment']]
    
    plt.scatter(df['like_count'], range(len(df)), c=colors, alpha=0.6, s=50)
    plt.xlabel('Beğeni Sayısı')
    plt.ylabel('Yorum Sırası')
    plt.title('Duygu vs Beğeni İlişkisi')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=sent) 
                       for sent, color in sentiment_colors.items()], 
              title='Duygu')
    plt.grid(True, alpha=0.3)
    plt.savefig('sentiment_likes_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 5. Duygu Yoğunluğu Analizi 
    plt.figure(figsize=(10, 6))
    sentiment_stats = df.groupby('sentiment')['like_count'].agg(['mean', 'count']).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ortalama beğeni sayısı
    ax1.bar(sentiment_stats['sentiment'], sentiment_stats['mean'], color=['green', 'red', 'gray'])
    ax1.set_title('Duygu Kategorisine Göre Ortalama Beğeni')
    ax1.set_ylabel('Ortalama Beğeni Sayısı')
    ax1.set_xlabel('Duygu')
    
    # Yorum sayısı
    ax2.bar(sentiment_stats['sentiment'], sentiment_stats['count'], color=['green', 'red', 'gray'])
    ax2.set_title('Duygu Kategorisine Göre Yorum Sayısı')
    ax2.set_ylabel('Yorum Sayısı')
    ax2.set_xlabel('Duygu')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    print("Görseller kaydedildi: sentiment_bar.png, sentiment_pie.png, topic_bar.png, wordcloud_viz.png, sentiment_likes_scatter.png, sentiment_analysis.png")

def create_visualizations(df, temp_dir):
    """Analiz sonuçlarını görselleştirir"""
    
    # 1. Sentiment Dağılımı (Pasta)
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=[colors[s] for s in sentiment_counts.index])
    plt.title('Duygu Analizi Dağılımı')
    plt.savefig(os.path.join(temp_dir, 'sentiment_pie.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Sentiment Dağılımı (Çubuk)
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=[colors[s] for s in sentiment_counts.index])
    plt.title('Duygu Analizi Dağılımı')
    plt.xlabel('Duygu')
    plt.ylabel('Yorum Sayısı')
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, 'sentiment_bar.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Aspect-Sentiment Dağılımı
    all_aspects = []
    for comment in df.to_dict('records'):
        for aspect_data in comment.get('aspects', []):
            all_aspects.append({
                'aspect': aspect_data['aspect'],
                'sentiment': aspect_data['sentiment']
            })
    
    if all_aspects:
        aspect_df = pd.DataFrame(all_aspects)
        aspect_sentiment = pd.crosstab(aspect_df['aspect'], aspect_df['sentiment'])
        
        plt.figure(figsize=(12, 6))
        aspect_sentiment.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'])
        plt.title('Aspect-Bazlı Duygu Analizi')
        plt.xlabel('Aspect')
        plt.ylabel('Yorum Sayısı')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir, 'topic_bar.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    # 4. Word Cloud
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.replace("<br>", " ").replace("<br/>", " ")
        return " ".join(text.split())
    
    # Tüm yorumları birleştir
    all_text = " ".join([clean_text(text) for text in df['text']])
    
    # Word cloud oluştur
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='viridis',
        collocations=False,
        max_words=100
    ).generate(all_text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(temp_dir, 'wordcloud_viz.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 5. Beğeni-Sentiment İlişkisi
    plt.figure(figsize=(10, 6))
    colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
    for sentiment in df['sentiment'].unique():
        sentiment_data = df[df['sentiment'] == sentiment]
        plt.scatter(
            sentiment_data.index, 
            sentiment_data['like_count'],
            label=sentiment,
            alpha=0.6,
            color=colors[sentiment]
        )
    plt.title('Yorum Beğenileri ve Duygu Analizi İlişkisi')
    plt.xlabel('Yorum Sırası')
    plt.ylabel('Beğeni Sayısı')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, 'sentiment_likes_scatter.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 6. Duygu Yoğunluğu Analizi
    sentiment_stats = df.groupby('sentiment')['like_count'].agg(['mean', 'count']).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ortalama beğeni sayısı
    ax1.bar(sentiment_stats['sentiment'], sentiment_stats['mean'], color=['green', 'red', 'gray'])
    ax1.set_title('Duygu Kategorisine Göre Ortalama Beğeni')
    ax1.set_ylabel('Ortalama Beğeni Sayısı')
    ax1.set_xlabel('Duygu')
    
    # Yorum sayısı
    ax2.bar(sentiment_stats['sentiment'], sentiment_stats['count'], color=['green', 'red', 'gray'])
    ax2.set_title('Duygu Kategorisine Göre Yorum Sayısı')
    ax2.set_ylabel('Yorum Sayısı')
    ax2.set_xlabel('Duygu')
    
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, 'sentiment_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Görselleştirmeler tamamlandı!")

if __name__ == "__main__":
    # Son analiz dosyasını kullan
    visualize_from_json('reanalyzed_comments_20250629_105422.json') 