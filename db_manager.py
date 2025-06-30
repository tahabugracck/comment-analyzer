import sqlite3
import json
from datetime import datetime

"""
YouTube Yorum Analiz Sistemi - Veritabanı Yöneticisi
"""

class DBManager:
    def __init__(self, db_path='analysis_results.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()
        # Yorumlar tablosu
        c.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                id TEXT PRIMARY KEY,
                video_id TEXT,
                author TEXT,
                text TEXT,
                like_count INTEGER,
                is_top_level INTEGER,
                parent_id TEXT,
                sentiment TEXT,
                sentiment_score REAL,
                aspects TEXT,
                word_count INTEGER,
                analyzed_at TEXT
            )
        ''')
        # Özetler tablosu
        c.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                summary TEXT,
                created_at TEXT
            )
        ''')
        self.conn.commit()

    def save_comments(self, video_id, comments):
        c = self.conn.cursor()
        for comment in comments:
            c.execute('''
                INSERT OR REPLACE INTO comments (
                    id, video_id, author, text, like_count, is_top_level, parent_id, sentiment, sentiment_score, aspects, word_count, analyzed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                comment['id'],
                video_id,
                comment.get('author', ''),
                comment.get('text', ''),
                int(comment.get('like_count', 0)),
                int(comment.get('is_top_level', True)),
                comment.get('parent_id'),
                comment.get('sentiment', ''),
                float(comment.get('sentiment_score', 0)),
                json.dumps(comment.get('aspects', []), ensure_ascii=False),
                int(comment.get('word_count', 0)),
                comment.get('reanalyzed_at', datetime.now().isoformat())
            ))
        self.conn.commit()
        print(f"{len(comments)} yorum veri tabanına kaydedildi.")

    def save_summary(self, video_id, summary):
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO summaries (video_id, summary, created_at) VALUES (?, ?, ?)
        ''', (video_id, summary, datetime.now().isoformat()))
        self.conn.commit()
        print("Özet veri tabanına kaydedildi.")

    def get_comments(self, video_id):
        c = self.conn.cursor()
        c.execute('''
            SELECT id, video_id, author, text, like_count, is_top_level, parent_id, 
                   sentiment, sentiment_score, aspects, word_count, analyzed_at
            FROM comments WHERE video_id = ?
        ''', (video_id,))
        rows = c.fetchall()
        
        comments = []
        for row in rows:
            comments.append({
                'id': row[0],
                'video_id': row[1],
                'author': row[2],
                'text': row[3],
                'like_count': row[4],
                'is_top_level': bool(row[5]),
                'parent_id': row[6],
                'sentiment': row[7],
                'sentiment_score': row[8],
                'aspects': json.loads(row[9]) if row[9] else [],
                'word_count': row[10],
                'analyzed_at': row[11]
            })
        return comments

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    # Test: Son analiz dosyasını kaydet
    with open('youtube_comments_trwo3t1qMDo.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    video_id = data.get('video_info', {}).get('id', 'test_video')
    comments = data['analyzed_comments']
    db = DBManager()
    db.save_comments(video_id, comments)
    # Özet eklemek için örnek
    with open('auto_summary_20250629_110811.md', 'r', encoding='utf-8') as f:
        summary = f.read()
    db.save_summary(video_id, summary)
    db.close() 