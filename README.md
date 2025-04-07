# 🧠 NLP Project | Automated Customer Reviews 

## 📌 Project Goal

This project aims to build an NLP-powered system that automatically processes customer reviews from Amazon products. The full vision includes:

-  Classifying customer sentiment.
-  Clustering product categories.
-  Generating product summary articles using generative AI.

---

## 🎯 Problem Statement

With thousands of product reviews available across platforms, manually analyzing customer sentiment is inefficient. This project automates the process using Natural Language Processing models to extract structured insights that support product development and marketing.

---

## Task 1: Review Classification

### 🎯 Objective

Classify Amazon customer reviews into:
- **Positive**
- **Neutral**
- **Negative**

### 🗂️ Dataset

- **Source**: [Kaggle - Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- **File Used**: `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`
- **Description:** A sample of 5,000 Amazon product reviews including Kindle, Fire TV Stick, etc., collected between Sept 2017 – Oct 2018.

### 💡 Sentiment Mapping

| Star Rating | Sentiment Label |
|-------------|-----------------|
| 1 - 2       | Negative (0)     |
| 3           | Neutral (1)      |
| 4 - 5       | Positive (2)     |

---

## 🤖 Model

- **Base Model**: `distilbert-base-uncased` (lightweight Transformer model).
- **Framework**: Hugging Face Transformers
- **Fine-tuning**: On labeled Amazon reviews using PyTorch Trainer API
- **Upsampling**: Applied to balance class distribution

### ⚙️ Training Config

- Epochs: 3
- Batch size: 16
- Optimizer: AdamW
- Evaluation strategy: `epoch`

---

## 📊 Evaluation Results

| Metric       | Score |
|--------------|-------|
| Accuracy     | 94.9% |
| Precision    | 94.3% |
| Recall       | 94.9% |
| F1 Score     | 94.5% |

### 📌 Per-Class Performance

| Class     | Precision | Recall | F1 Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.73      | 0.48   | 0.58     | 23      |
| Neutral   | 0.41      | 0.31   | 0.35     | 35      |
| Positive  | 0.97      | 0.98   | 0.98     | 942     |

> ⚖️ Upsampling helped improve Neutral/Negative performance, though they remain underrepresented.

---

## 📦 Load Pretrained Model 

Due to the large size of the fine-tuned model, it has been uploaded to Google Drive.
You can download it from the link below and use it without retraining:

📁 **[Download from Google Drive](https://drive.google.com/drive/folders/1cePeMMkRisuMEuedgYQoZKSgNb-8uJWK?usp=sharing)**

---
## 🔧 How to Run

1. Clone the repo:
```bash
git clone https://github.com/RanwahSadik/Automated-Customers-Review.git
cd Automated-Customers-Review
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Train the model:
```
python Project_NLP.py
```
4. Use for prediction:
```
from transformers import pipeline
classifier = pipeline("text-classification", model="./ReviewClassificationModelBalanced", tokenizer="./ReviewClassificationModelBalanced")
classifier("The Kindle is lightweight and easy to use!")
```
