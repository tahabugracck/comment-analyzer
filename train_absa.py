import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

"""
Veriyi yükle ve dönüştür
"""
def load_and_flatten(file_paths):
    all_data = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            comments = json.load(f)
        for item in comments:
            text = item["text"]
            for asp in item["aspects"]:
                all_data.append({
                    "text": text,
                    "aspect": asp["aspect"],
                    "sentiment": asp["sentiment"]
                })
    return all_data

file_paths = [
    "youtube_comments_trwo3t1qMDo_absa.json",
    "youtube_comments_lCCW0KupGDs_absa.json"
]
data = load_and_flatten(file_paths)

"""
Etiketleri sayısallaştır
"""
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform([item["sentiment"] for item in data])

"""
Dataset ve Tokenizer
"""
tokenizer = BertTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")

class ABSADataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_len=128):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['aspect']} hakkında: {item['text']}"
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        label = self.labels[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label)
        }

"""Eğitim ve doğrulama böl"""
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.1, random_state=42)

train_dataset = ABSADataset(train_data, train_labels, tokenizer)
val_dataset = ABSADataset(val_data, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


"""
Model Tanımı
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(
    "savasy/bert-base-turkish-sentiment-cased",
    num_labels=3,
    ignore_mismatched_sizes=True
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)


"""Eğitim Döngüsü"""
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} loss: {total_loss:.4f}")


"""
Modeli Kaydet
"""
model.save_pretrained("absa_model_tr")
tokenizer.save_pretrained("absa_model_tr")
print("Eğitim tamamlandı, model 'absa_model_tr' klasörüne kaydedildi.")
