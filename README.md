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
- **File Used**: `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv`
- **Description:** A sample of +28,000 Amazon product reviews including Kindle, Fire Tablet, etc., collected between Feb 2019 – Apr 2019.

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

| Metric       | Score       |
|--------------|-------------|
| Accuracy     | 96.3%       |
| Precision    | 96.35%      |
| Recall       | 96.33%      |
| F1 Score     | 96.34%      |
| Eval Loss    | 0.29        |

### 📌 Per-Class Performance

| Class     | Precision | Recall | F1 Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.85      | 0.88   | 0.87     | 318     |
| Neutral   | 0.64      | 0.64   | 0.64     | 233     |
| Positive  | 0.99      | 0.98   | 0.98     | 5116    |

> ⚖️ Upsampling helped improve Neutral/Negative performance, though they remain underrepresented.

---

## 📦 Load Pretrained Model 

Due to the large size of the fine-tuned model, it has been uploaded to Google Drive.
You can download it from the link below and use it without retraining:

📁 **[Download from Google Drive](https://drive.google.com/drive/folders/16bP9oJPsyk1RTDdW2EfxxuHz-UEg0G1Q?usp=sharing)**

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
---

## Task 2: Product Category Clustering

### 🎯 Objective

Group all product reviews into **4–6 broader meta-categories** to simplify analysis and visualization across product types.

---

### 🗃️ Dataset

- **File Used**: `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv`
- **Columns Explored**:  
  - `name`  
  - `categories`  
  - `primaryCategories`

---

### 🔍 Methodology

1. **Text Preparation**
   - Combined `categories` and `primaryCategories` into a new column: `all_categories`
   - Cleaned and normalized text for embedding

2. **Embeddings**
   - Used **Sentence-BERT (`all-MiniLM-L6-v2`)** to generate vector representations of the category texts

3. **Clustering**
   - Applied **KMeans** clustering with `n_clusters=5` on the category embeddings
   - Mapped each cluster to a meaningful **Meta-Category**

4. **Dimensionality Reduction**
   - Used PCA for potential visualization and cluster interpretability

---

#### 📦 Cluster Mapping:

| Cluster | Meta Category                         |
|---------|----------------------------------------|
| 0       | Household & Camera Batteries          |
| 1       | Fire Tablets & Electronics            |
| 2       | Household & Camera Batteries (Mixed)  |
| 3       | E-Readers & Tablets                   |
| 4       | Kids Tablets & Office Tech Essentials |

> 🔁 These categories help streamline downstream analysis like dashboards or product performance by type.

---

### 📦 Load Pretrained Embedding & Clustering Model

Due to the large size of the Sentence-BERT embeddings and KMeans clustering model, they have been uploaded to Google Drive for convenience.

📁 **[Download the pretrained model from Google Drive](https://drive.google.com/drive/folders/1BDEWGKUw28HCjTPd7ssR7BhlnBgYVGv1?usp=drive_link)**

You can load the model from the folder and skip retraining if you want faster category mapping.

---
## Task 3: Review Summarization Using Generative AI

### 🎯 Objective

Automatically generate short blog-style summaries for each product category that:

- Recommend the **Top 3 Products** and describe their strengths
- Highlight **Top Complaints** for each of those products
- Identify the **Worst Product** in the category and explain why it should be avoided

---

### 🗃️ Dataset

- **File Used**: The data that produce in Task 2
- **Filtered Columns**: `name`, `meta_category`, `reviews.text`, `reviews.rating`

---

### 🤖 Model

- **Summarization Model**: `facebook/bart-large-cnn`
- **Framework**: Hugging Face Transformers
- **Tokenizer & Pipeline**: Used to summarize reviews and extract complaint highlights

---

### 🧪 Methodology

1. **Preprocessing**:
   - Removed missing values
   - Converted text to string and cleaned whitespace
   - Grouped by `meta_category` for category-wise analysis

2. **Top Product Selection**:
   - Products ranked by **average rating** and **review count**
   - Top 3 products selected per category
   - Worst product selected based on lowest average rating 

3. **Summarization Tasks**:
   - Used BART to generate summaries for each top product
   - Extracted and summarized complaints from reviews < 3⭐
   - Generated a clean Markdown-style blog post for each category

---

### 📝 Output Structure

The blog post output includes:

- ✅ **Top 3 Products** with a short summary
- ⚠️ **Top Complaints** for each product
- ❌ **Worst Product to Avoid** with complaint reasons

---

### 🧪 Example Output Snippet

```markdown
## 🛍️ Product Insights: Kids Tablets & Office Tech Essentials
### ⭐ Top Products:
▶ Expanding Accordion File Folder Plastic Portable Document Organizer Letter Size This folder also expands on the bottom. Will allow A LOT more than when the bottom does not expand. Bought a second. I recommended this to my friends. Exactly what I needed.

▶ Certified Refurbished Amazon Echo Amazon Echo Plus is a fun product it requires time to explore its skills. It copes well with accents, however the Echo have to be placed as center as possible to get the best listening results. If you have pets, you can count on it being knocked over.

▶ AmazonBasics USB 3.0 Cable - A-Male to B-Male - 6 Feet (1.8 Meters) This is a good length cable for those that don't want tight connections because a cable is too short. Amazon sells them in several lengths so you just get the right size. The connectors at both end are gold plated, and the cable is 9 feet long.

### ⚠️ Top Complaints:
Expanding Accordion File Folder Plastic Portable Document Organizer Letter Size: No major complaints.

Certified Refurbished Amazon Echo: No major complaints.

AmazonBasics USB 3.0 Cable - A-Male to B-Male - 6 Feet (1.8 Meters): No major complaints.

### ❌ Worst Product to Avoid:
Amazon Kindle Replacement Power Adapter (Fits Latest Generation Kindle and Kindle DX) For shipment in the U.S only — 2.8⭐ rating
Top issues: I purchased my Kindle2 in March of 09 - and it came with this adapter and cord. It is now October 09, and the small part of the adapter for this cord has stopped working. While plugged into a wall socket using Amazon's adapter, it does not charge at all.
