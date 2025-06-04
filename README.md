# 📰 BERT Fake News Detector

## 🧠 Project Description

This project utilizes a fine-tuned BERT model to classify news articles as real or fake. It highlights the power of natural language processing (NLP) in identifying misinformation and combating fake news. The model is trained using a labeled fake news dataset and deployed through a simple Flask API.

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/bert-fake-news-detector.git
cd bert-fake-news-detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the Dataset

Download a labeled fake news dataset such as the [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle. Place the CSV files in the root directory.

### 4️⃣ Train the Model

```bash
python fake_news_detector.py --mode train --dataset_path fake_or_real_news.csv
```

### 5️⃣ Run the Flask API

```bash
python fake_news_detector.py --mode serve
```

## 🧰 Tech Stack

- **PyTorch** – For training the classification model.
- **Transformers (HuggingFace)** – BERT model and tokenizer.
- **Pandas** – For data preprocessing.
- **Flask** – To serve the model via a RESTful API.

## 🌟 Key Features

- 🔍 Real vs Fake news prediction using deep NLP.
- 📦 Simple API server with Flask for real-time inference.
