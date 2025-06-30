import streamlit as st
import json
import os

# Aspect ve duygu seçenekleri
aspects = ["ses", "görüntü", "içerik", "reklam", "sunum", "genel"]
sentiments = ["positive", "negative", "neutral"]

# Yorumları yükle
with open("youtube_comments_trwo3t1qMDo.json", "r", encoding="utf-8") as f:
    data = json.load(f)["comments"]

# Etiketli yorumları kayıt dosyası
save_path = "youtube_comments_trwo3t1qMDo_absa.json"
if os.path.exists(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        labeled = json.load(f)
else:
    labeled = []

st.title("ABSA Etiketleme Aracı")

for idx, comment in enumerate(data):
    if idx < len(labeled):
        continue  # Zaten etiketlenmişse geç

    st.subheader(f"Yorum {idx+1}")
    st.write(comment["text"])

    aspect_sentiments = []
    for aspect in aspects:
        if st.checkbox(f"{aspect} geçiyor mu?", key=f"{idx}-{aspect}"):
            sentiment = st.selectbox(
                f"{aspect} için duygu seç", sentiments, key=f"{idx}-{aspect}-s"
            )
            aspect_sentiments.append({"aspect": aspect, "sentiment": sentiment})

    if st.button("Kaydet", key=f"save-{idx}"):
        labeled.append({
            "text": comment["text"],
            "aspects": aspect_sentiments
        })
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(labeled, f, indent=2, ensure_ascii=False)
        st.success("Kaydedildi!")
        st.experimental_rerun()
    break
