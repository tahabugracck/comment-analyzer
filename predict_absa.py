import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Modeli ve tokenizer'ı yükle
model_path = "absa_model_tr"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Label encoder gibi string etiketler
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# Örnek yorumlar
yorumlar = [
    {
        "text": "Abi ses çok kötüydü ama içerik efsaneydi",
        "aspects": ["ses", "içerik"]
    },
    {
        "text": "Reklamlar çok fazla, artık baydı!",
        "aspects": ["reklam"]
    }
]

for yorum in yorumlar:
    text = yorum["text"]
    for aspect in yorum["aspects"]:
        combined_text = f"{aspect} hakkında: {text}"
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            sentiment = id2label[prediction]
            print(f"[{aspect.upper()}] → {sentiment.upper()} | '{text}'")
