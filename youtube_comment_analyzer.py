import os
import json
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

"""
Youtube API entegrasyonu. Video ve yorumları çeker.
"""

class YouTubeCommentAnalyzer:
    """
    Youtube yorumlarını çekmek için kullanılan sınıf. (youtube data api v3)
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key) # YouTube API servisini oluştur
        print("API bağlantısı başarılı.")

    def video_id_from_url(self, video_url):
        """
        Youtube url'sinden vide id'sini çıkarır. (video_url(str))
        """
        if 'v=' in video_url:
            video_id = video_url.split('v=')[1]
        elif 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1]
        else:
            return None
        
        # URL parametrelerini temizle
        if '&' in video_id:
            video_id = video_id.split('&')[0]
        
        return video_id
    
    def get_video_info(self, video_id):
        """
        Video hakkında temel bilgileri çeker.
        """
        try:
            video_response = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            ).execute()

            if not video_response['items']:
                return None
            
            video = video_response['items'][0]
            snippet = video['snippet']
            stats = video['statistics']

            # Video bilgilerini düzenle
            video_info = {
                'id': video_id,
                'title': snippet['title'],
                'channel': snippet['channelTitle'],
                'published_at': snippet['publishedAt'][:10],  # Sadece tarih
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0))
            }

            return video_info
        
        except HttpError as e:
            print(f"Video bilgileri alınamadı: {e}")
            return None
        
    def get_all_comments(self, video_id, max_results=100):
        """
        Videodaki tüm yorumları çeker. (ana yorum + yorumlara gelen yanıtlar)
        """

        all_comments = []
        next_page_token = None
        
        print(f"{max_results} yorum çekiliyor...")
        
        try:
            while len(all_comments) < max_results:
                # Ana yorumları çek
                comments_response = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=min(100, max_results - len(all_comments)),
                    pageToken=next_page_token,
                    order='relevance'  # En popüler yorumlar önce
                ).execute()
                
                # Her ana yorum için
                for item in comments_response['items']:
                    if len(all_comments) >= max_results:
                        break
                    
                    # Ana yorum bilgileri
                    main_comment = item['snippet']['topLevelComment']['snippet']
                    comment_data = {
                        'text': main_comment['textDisplay'],
                        'author': main_comment['authorDisplayName'],
                        'like_count': main_comment['likeCount'],
                        'published_at': main_comment['publishedAt'],
                        'is_reply': False,
                        'parent_id': None
                    }
                    all_comments.append(comment_data)
                    
                    # Yanıtları da ekle
                    if 'replies' in item and len(all_comments) < max_results:
                        for reply in item['replies']['comments']:
                            if len(all_comments) >= max_results:
                                break
                            
                            reply_snippet = reply['snippet']
                            reply_data = {
                                'text': reply_snippet['textDisplay'],
                                'author': reply_snippet['authorDisplayName'],
                                'like_count': reply_snippet['likeCount'],
                                'published_at': reply_snippet['publishedAt'],
                                'is_reply': True,
                                'parent_id': main_comment['authorDisplayName']
                            }
                            all_comments.append(reply_data)
                
                # Sonraki sayfa var mı?
                next_page_token = comments_response.get('nextPageToken')
                if not next_page_token:
                    break
                
                print(f"{len(all_comments)} yorum çekildi...")
            
            print(f"Toplam {len(all_comments)} yorum başarıyla çekildi.")
            return all_comments
            
        except HttpError as e:
            print(f"Yorumlar çekilemedi: {e}")
            return []
        
    def save_comments_to_json(self, comments, video_info):
        """
        Yorumları JSON'a kaydetme. (video id'sine göre)(fine-tune için)
        """
        video_id = video_info.get('id', 'unknown_video')
        filename = f"youtube_comments_{video_id}.json"

        data = {
            'video_info': video_info,
            'comments': comments,
            'total_comments': len(comments),
            'extracted_at': datetime.now().isoformat()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Yorumlar {filename} dosyasına kaydedildi (video_id ile eşleşiyor)")
        return filename
    
    def analyze_comments(self, video_url, max_results=100):

        print(f"\nVideo analizi başlıyor: {video_url}")
        
        video_id = self.video_id_from_url(video_url)
        if not video_id:
            print("Geçersiz YouTube URL'si.")
            return None
        
        video_info = self.get_video_info(video_id)
        if not video_info:
            print("Video bulunamadı veya erişim izni yok.")
            return None
        
        print(f"Video: {video_info['title']}")
        print(f"Kanal: {video_info['channel']}")
        print(f"Görüntülenme: {video_info['view_count']:,}")
        print(f"Beğeni: {video_info['like_count']:,}")
        print(f"Yorum: {video_info['comment_count']:,}")
        
        comments = self.get_all_comments(video_id, max_results)
        if not comments:
            print("Hiç yorum bulunamadı.")
            return None
        
        filename = self.save_comments_to_json(comments, video_info)
        
        return {
            'video_info': video_info,
            'comments': comments,
            'filename': filename,
            'total_comments': len(comments)
        }

# ** Test fonksiyonu
if __name__ == "__main__":
    api_key = os.getenv('API_KEY')
    if not api_key:
        print("YOUTUBE_API_KEY environment variable'ı ayarlanmamış.")
        exit(1)
    
    analyzer = YouTubeCommentAnalyzer(api_key)
    
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    result = analyzer.analyze_comments(test_url, max_results=50)
    
    if result:
        print(f"\nAnaliz tamamlandı.")
        print(f"Dosya: {result['filename']}")
        print(f"Toplam yorum: {result['total_comments']}")
    else:
        print("\nAnaliz başarısız.")         
