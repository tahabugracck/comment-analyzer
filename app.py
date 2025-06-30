from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
from youtube_comment_analyzer import YouTubeCommentAnalyzer
from comment_analyzer import CommentAnalyzer
from auto_summary import generate_auto_summary
from db_manager import DBManager

"""
YouTube Yorum Analiz Sistemi - Flask Web Uygulaması
Bu dosya ana web uygulamasını içerir. Kullanıcılar YouTube video linkini
girer ve AI destekli analiz sonuçlarını görür.
"""


app = Flask(__name__)

UPLOAD_FOLDER = os.path.abspath('.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_visualizations(df, temp_dir):

    print("Görselleştirmeler oluşturuluyor...")
    
    # 1. Duygu dağılımı (sütun grafik)
    plt.figure(figsize=(8, 5))
    sentiment_counts = df['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
    plt.title('Duygu Dağılımı (Sütun Grafik)')
    plt.xlabel('Duygu')
    plt.ylabel('Yorum Sayısı')
    plt.savefig(os.path.join(temp_dir, 'sentiment_bar.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 2. Duygu dağılımı (pasta grafik)
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Duygu Dağılımı (Pasta Grafik)')
    plt.savefig(os.path.join(temp_dir, 'sentiment_pie.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 3. Aspect dağılımı
    all_aspects = []
    for comment in df.to_dict('records'):
        for aspect_data in comment.get('aspects', []):
            all_aspects.append(aspect_data['aspect'])
    aspect_counts = pd.Series(all_aspects).value_counts()
    
    if not aspect_counts.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=aspect_counts.index, y=aspect_counts.values, palette='magma')
        plt.title('Aspect Dağılımı')
        plt.xlabel('Aspect')
        plt.ylabel('Yorumda Geçme Sayısı')
        plt.savefig(os.path.join(temp_dir, 'aspect_bar.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    # 4. Kelime bulutu
    def clean_text(text):
        """Metni temizler - HTML etiketlerini ve linkleri kaldırır"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    all_text = ' '.join([clean_text(t) for t in df['text'].tolist()])
    if len(all_text) > 10:
        wordcloud = WordCloud(width=1000, height=500, background_color='white', 
                            max_words=200, colormap='viridis').generate(all_text)
        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Kelime Bulutu')
        plt.savefig(os.path.join(temp_dir, 'wordcloud_viz.png'), dpi=200, bbox_inches='tight')
        plt.close()
    
    # 5. Duygu vs Beğeni İlişkisi (Scatter Plot)
    plt.figure(figsize=(10, 6))
    sentiment_colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
    colors = [sentiment_colors.get(sent, 'blue') for sent in df['sentiment']]
    
    plt.scatter(df['like_count'], range(len(df)), c=colors, alpha=0.6, s=50)
    plt.xlabel('Beğeni Sayısı')
    plt.ylabel('Yorum Sırası')
    plt.title('Duygu vs Beğeni İlişkisi')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=8, label=sent) 
                       for sent, color in sentiment_colors.items()], 
              title='Duygu')
    plt.grid(True, alpha=0.3)
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

def generate_summary_for_web(analyzed_comments):
    """
    Web için otomatik özet oluşturur
    """
    df = pd.DataFrame(analyzed_comments)
    total = len(df)
    
    # Duygu istatistikleri
    pos = (df['sentiment'] == 'POSITIVE').sum()
    neg = (df['sentiment'] == 'NEGATIVE').sum()
    neu = (df['sentiment'] == 'NEUTRAL').sum()
    
    pos_pct = (pos / total) * 100 if total else 0
    neg_pct = (neg / total) * 100 if total else 0
    neu_pct = (neu / total) * 100 if total else 0
    
    # Aspect analizi
    all_aspects = []
    for comment in analyzed_comments:
        for aspect_data in comment.get('aspects', []):
            all_aspects.append(f"{aspect_data['aspect']}_{aspect_data['sentiment']}")
    
    aspect_counts = Counter(all_aspects)
    
    # Aspect listesi oluştur
    aspect_lines = []
    for aspect_sentiment, count in aspect_counts.most_common():
        aspect, sentiment = aspect_sentiment.split('_')
        pct = (count / total) * 100
        aspect_lines.append(f"- <b>{aspect.title()}</b>: {sentiment} (%{pct:.1f})")
    
    # Akıllı öneriler
    suggestions = []
    
    # Aspect bazlı öneriler
    aspect_sentiments = {}
    for aspect_sentiment in aspect_counts:
        aspect, sentiment = aspect_sentiment.split('_')
        if aspect not in aspect_sentiments:
            aspect_sentiments[aspect] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        aspect_sentiments[aspect][sentiment] += aspect_counts[aspect_sentiment]
    
    for aspect, sentiments in aspect_sentiments.items():
        total_aspect = sum(sentiments.values())
        neg_pct = (sentiments['NEGATIVE'] / total_aspect) * 100 if total_aspect else 0
        pos_pct = (sentiments['POSITIVE'] / total_aspect) * 100 if total_aspect else 0
        
        if neg_pct > 30:
            suggestions.append(f"⚠️ <b>{aspect.title()}</b> konusunda %{neg_pct:.1f} oranında olumsuz yorum var. İyileştirme yapılabilir.")
        elif pos_pct > 70:
            suggestions.append(f"🎉 <b>{aspect.title()}</b> konusunda izleyiciler çok memnun! (%{pos_pct:.1f} olumlu)")
    
    if pos_pct > 70:
        suggestions.append("🎉 Genel olarak izleyiciler videoyu çok beğenmiş!")
    elif pos_pct > 40:
        suggestions.append("👍 Olumlu yorumlar ağırlıkta.")
    elif neg_pct > 30:
        suggestions.append("⚠️ Olumsuz yorum oranı yüksek, dikkat!")
    
    if not suggestions:
        suggestions.append("Kullanıcılar genel olarak kararsız görünüyor. Daha fazla etkileşim için içerik çeşitlendirilebilir.")
    
    # HTML özeti oluştur
    summary = f"""
    <h3>Genel İstatistikler</h3>
    <ul>
    <li><b>Toplam Yorum Sayısı</b>: {total}</li>
    <li><b>Olumlu Yorumlar</b>: {pos} (%{pos_pct:.1f})</li>
    <li><b>Olumsuz Yorumlar</b>: {neg} (%{neg_pct:.1f})</li>
    <li><b>Nötr Yorumlar</b>: {neu} (%{neu_pct:.1f})</li>
    </ul>
    <h3>Aspect-Bazlı Analiz</h3>
    <ul>
    {''.join([f'<li>{line}</li>' for line in aspect_lines])}
    </ul>
    <h3>Otomatik Öneriler</h3>
    <ul>
    {''.join([f'<li>{s}</li>' for s in suggestions])}
    </ul>
    """
    
    return summary

# Ana sayfa - GET ve POST isteklerini karşılar
@app.route('/', methods=['GET', 'POST'])
def index():
    """Ana sayfa - kullanıcı video linkini girer"""
    if request.method == 'POST':
        video_url = request.form.get('video_url')
        max_comments = request.form.get('max_comments', '300')
        
        if not video_url:
            return render_template('index.html', error="Lütfen bir YouTube video linki girin.")
        
        return redirect(url_for('analyze', video_url=video_url, max_comments=max_comments))
    
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    """Video analizi yapar ve sonuçları gösterir"""
    video_url = request.args.get('video_url')
    max_comments = request.args.get('max_comments', '300')
    
    if not video_url:
        return redirect(url_for('index'))

    print(f"🎬 Analiz başlıyor: {video_url}")
    
    try:
        print("YouTube yorumları çekiliyor...")
        yt_analyzer = YouTubeCommentAnalyzer(os.getenv('YOUTUBE_API_KEY', 'AIzaSyCvHN8J_8KdOExxEcOraNX5ZzMoXbz2l8w'))
        result = yt_analyzer.analyze_comments(video_url, max_results=int(max_comments))
        
        if not result:
            return render_template('index.html', error="Yorumlar çekilemedi veya video bulunamadı.")

        print("AI analizi başlıyor...")
        analyzer = CommentAnalyzer()
        analyzed_comments = analyzer.analyze_comments(result)

        temp_id = str(uuid.uuid4())
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'static', temp_id)
        os.makedirs(temp_dir, exist_ok=True)

        print("Görselleştirmeler oluşturuluyor...")
        df = pd.DataFrame(analyzed_comments)
        create_visualizations(df, temp_dir)

        print("Özet oluşturuluyor...")
        summary_md = generate_summary_for_web(analyzed_comments)

        print("Veritabanına kaydediliyor...")
        db = DBManager()
        video_id = result['video_info'].get('id', video_url)
        db.save_comments(video_id, analyzed_comments)
        db.save_summary(video_id, summary_md)
        db.close()

        print("Analiz tamamlandı!")
        return render_template('result.html',
            video_info=result['video_info'],
            summary=summary_md,
            temp_id=temp_id,
            sentiment_bar=f'/static/{temp_id}/sentiment_bar.png',
            sentiment_pie=f'/static/{temp_id}/sentiment_pie.png',
            aspect_bar=f'/static/{temp_id}/aspect_bar.png',
            wordcloud_viz=f'/static/{temp_id}/wordcloud_viz.png',
            sentiment_likes_scatter=f'/static/{temp_id}/sentiment_likes_scatter.png',
            sentiment_analysis=f'/static/{temp_id}/sentiment_analysis.png',
            top_comments=df.nlargest(5, 'like_count').to_dict('records')
        )
        
    except Exception as e:
        print(f"Hata: {e}")
        return render_template('index.html', error=f"Analiz sırasında hata oluştu: {str(e)}")

@app.route('/static/<temp_id>/<filename>')
def static_files(temp_id, filename):
    """Geçici görselleri servis eder"""
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'static', temp_id), filename)

if __name__ == '__main__':
    os.makedirs(os.path.join('static'), exist_ok=True)
    
    print("🌸 YouTube Yorum Analiz Sistemi başlatılıyor...")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 