import json
import pandas as pd
from collections import Counter
from datetime import datetime

"""
YouTube Yorum Analiz Sistemi - Otomatik Özet Oluşturucu
"""

def generate_auto_summary(json_file, output_file=None):
    print(f"\nOtomatik özet hazırlanıyor: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    comments = data['analyzed_comments']
    df = pd.DataFrame(comments)

    # Genel istatistikler
    total = len(df)
    pos = (df['sentiment'] == 'POSITIVE').sum()
    neg = (df['sentiment'] == 'NEGATIVE').sum()
    neu = (df['sentiment'] == 'NEUTRAL').sum()
    pos_pct = (pos / total) * 100 if total else 0
    neg_pct = (neg / total) * 100 if total else 0
    neu_pct = (neu / total) * 100 if total else 0

    # Aspect dağılımı
    all_aspects = []
    for comment in comments:
        for aspect_data in comment.get('aspects', []):
            all_aspects.append(f"{aspect_data['aspect']}_{aspect_data['sentiment']}")
    
    aspect_counts = Counter(all_aspects)
    aspect_lines = []
    for aspect_sentiment, count in aspect_counts.most_common():
        aspect, sentiment = aspect_sentiment.split('_')
        pct = (count / total) * 100
        aspect_lines.append(f"- **{aspect.title()}**: {sentiment} (%{pct:.1f})")

    # En popüler yorumlar
    top_comments = df.nlargest(5, 'like_count')[['text', 'like_count', 'sentiment']].to_dict('records')
    top_lines = []
    for i, c in enumerate(top_comments, 1):
        top_lines.append(f"{i}. **{c['like_count']} beğeni** - {c['text'][:100]}... ({c['sentiment']})")

    # Otomatik öneriler
    suggestions = []
    
    # Aspect bazlı öneriler
    aspect_sentiments = {}
    for aspect_sentiment in aspect_counts:
        aspect, sentiment = aspect_sentiment.split('_')
        if aspect not in aspect_sentiments:
            aspect_sentiments[aspect] = {'OLUMLU': 0, 'OLUMSUZ': 0, 'NÖTR': 0}
        aspect_sentiments[aspect][sentiment] += aspect_counts[aspect_sentiment]
    
    for aspect, sentiments in aspect_sentiments.items():
        total_aspect = sum(sentiments.values())
        neg_pct = (sentiments['OLUMSUZ'] / total_aspect) * 100 if total_aspect else 0
        pos_pct = (sentiments['OLUMLU'] / total_aspect) * 100 if total_aspect else 0
        
        if neg_pct > 30:
            suggestions.append(f"**{aspect.title()}** konusunda %{neg_pct:.1f} oranında olumsuz yorum var. İyileştirme yapılabilir.")
        elif pos_pct > 70:
            suggestions.append(f"**{aspect.title()}** konusunda izleyiciler çok memnun! (%{pos_pct:.1f} olumlu)")
    
    if pos_pct > 70:
        suggestions.append("Genel olarak izleyiciler videoyu çok beğenmiş!")
    elif pos_pct > 40:
        suggestions.append("Olumlu yorumlar ağırlıkta.")
    elif neg_pct > 30:
        suggestions.append("Olumsuz yorum oranı yüksek, dikkat!")
    
    if not suggestions:
        suggestions.append("Kullanıcılar genel olarak kararsız görünüyor. Daha fazla etkileşim için içerik çeşitlendirilebilir.")

    # Markdown rapor
    report = f"""
# YouTube Video Yorum Analizi Otomatik Özeti

## Genel İstatistikler
- **Toplam Yorum Sayısı**: {total}
- **Olumlu Yorumlar**: {pos} (%{pos_pct:.1f})
- **Olumsuz Yorumlar**: {neg} (%{neg_pct:.1f})
- **Nötr Yorumlar**: {neu} (%{neu_pct:.1f})

## Aspect-Bazlı Analiz
"""
    report += '\n'.join(aspect_lines)
    report += f"""

## En Popüler Yorumlar
"""
    report += '\n'.join(top_lines)
    report += f"""

## Otomatik Öneriler
"""
    report += '\n'.join(suggestions)
    report += f"""

*Rapor oluşturulma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Kaydet
    if not output_file:
        output_file = f"auto_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Otomatik özet kaydedildi: {output_file}")

if __name__ == "__main__":
    generate_auto_summary('reanalyzed_comments_20250629_105422.json') 