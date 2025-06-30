# YouTube Yorum Analiz Sistemi

Yapay zeka destekli bir YouTube yorum analiz platformudur.

## Proje Hakkında

Bu proje, YouTube videolarındaki yorumları otomatik olarak analiz eden bir web uygulamasıdır. Özellikle **Türkçe yorumlar** için fine-tune edilmiş BERT tabanlı Aspect-Based Sentiment Analysis (ABSA) modeli kullanarak, yorumların hem genel duygusal tonunu hem de belirli konulara/özelliklere dair duygu durumlarını tespit eder. Sonuçlar, detaylı ve görsel raporlar halinde sunulur.

### Temel Özellikler

- **YouTube API Entegrasyonu**: Videolardan yorumları otomatik olarak çeker.
- **ABSA (Aspect-Based Sentiment Analysis)**:
  - Fine-tune edilmiş Türkçe BERT modeli
  - Genel duygu analizi (Pozitif / Negatif / Nötr)
  - Belirli aspektlere göre duygu tespiti (örneğin: ses kalitesi, içerik, görüntü vb.)
- **Görselleştirme**:
  - Duygu dağılımı (pasta ve sütun grafik)
  - Aspect bazlı analiz grafikleri
  - Duygu-beğeni ilişkisi (scatter plot)
  - Kelime bulutu
  - Zaman bazlı duygu yoğunluğu grafiği
- **Otomatik Özet**: Yorumlara dair akıllı içgörüler ve özet öneriler.
- **Kalıcı Veri Saklama**: SQLite veritabanı desteği.
- **Modern Web Arayüzü**: Kullanıcı dostu, responsive tasarım.
- **Gerçek Zamanlı İşlem Takibi**: Yükleme animasyonları ve işlem durum takibi.

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır. Detaylar için `LICENSE` dosyasına göz atabilirsiniz.

## Uygulama İçi Görseller

<p align="center">
  <img src="https://github.com/user-attachments/assets/a838c52e-c708-4376-8e49-61a0c7951d83" width="45%" />
  <img src="https://github.com/user-attachments/assets/5bc883ec-3933-476f-8380-b7768a03c579" width="45%" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/43aaba69-82f3-4393-8fa1-ceeefa4f0ea4" width="45%" />
  <img src="https://github.com/user-attachments/assets/251925a2-c2db-4959-99a7-48d2aa45a063" width="45%" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/07837896-0bd0-47fb-9802-18c1d6ef9c3b" width="45%" />
  <img src="https://github.com/user-attachments/assets/045fb9bb-c2a5-4d3e-8b3c-a11eca07b155" width="45%" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/0bdb8127-90eb-4b3a-a563-3a244e19b531" width="45%" />
</p>




## Katkıda Bulunanlar & Teşekkürler

- [Hugging Face](https://huggingface.co/) – AI modelleri
- [YouTube Data API](https://developers.google.com/youtube/v3) – Yorum verisi sağlama
- [Flask](https://flask.palletsprojects.com/) – Web framework
- [Bootstrap](https://getbootstrap.com/) – Arayüz tasarımı

