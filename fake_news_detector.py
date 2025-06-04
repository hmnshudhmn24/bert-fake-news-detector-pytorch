
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import argparse
from flask import Flask, request, jsonify
import os

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df[['title', 'label']] if 'title' in df.columns else df[['text', 'label']]
    df.dropna(inplace=True)

    texts = df['title'].tolist() if 'title' in df.columns else df['text'].tolist()
    labels = df['label'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

    train_dataset = NewsDataset(X_train, y_train, tokenizer)
    val_dataset = NewsDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    torch.save(model.state_dict(), 'bert_fake_news_model.pt')
    print("✅ Model saved as 'bert_fake_news_model.pt'")

    # Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("📊 Classification Report:")
    print(classification_report(true_labels, predictions))
    print(f"✅ Accuracy: {accuracy_score(true_labels, predictions)}")

def serve_model():
    app = Flask(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('bert_fake_news_model.pt', map_location=device))
    model.to(device)
    model.eval()

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()

        return jsonify({'prediction': 'FAKE' if pred_label == 1 else 'REAL', 'confidence': confidence})

    app.run(port=5000, debug=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'serve'], required=True)
    parser.add_argument('--dataset_path', type=str, default='fake_or_real_news.csv')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args.dataset_path)
    elif args.mode == 'serve':
        serve_model()
