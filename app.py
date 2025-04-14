import pandas as pd
import random
import streamlit as st
from streamlit_option_menu import option_menu
import os
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from joblib import load 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import torch
import matplotlib.ticker as mtick
import textwrap
import plotly.express as px



# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    df_path = r"C:\Users\R\OneDrive\Desktop\Ironhack\Week6\CustomerReviewsModels\final_amazon_reviews.csv"
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame(columns=["name", "meta_category", "reviews.text", "reviews.rating"])
    df = df.dropna()
    df["reviews.text"] = df["reviews.text"].astype(str)
    return df
sentiment_model_path = r"C:\Users\R\OneDrive\Desktop\Ironhack\Week6\CustomerReviewsModels\ReviewClassificationModelBalanced"  

sentiment_tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_path)

df = load_data()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.markdown("""
    <style>
    body { 
        font-family: 'Poppins', sans-serif; 
        background: #E3F2FD; /* Light blue background */
        color: #0D47A1; /* Dark blue text */
    }
    .header { 
        text-align: center; 
        padding: 40px; 
        font-size: 3.5em; 
        font-weight: 700; 
        color: #2196F3; /* Blue color */
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    }
    .footer { 
        text-align: center; 
        padding: 20px; 
        font-size: 14px; 
        color: #0D47A1; 
        margin-top: 40px; 
    }
    .section-header { 
        font-size: 1.8em; 
        font-weight: bold; 
        color: #1565C0; 
        margin-top: 40px;
    }
    .section-content { 
        font-size: 1.1em; 
        color: #0D47A1; 
        margin-top: 15px; 
    }
    .review-box { 
        padding: 20px; 
        background-color: #FFFFFF; /* White background */
        border-radius: 10px; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
        margin-top: 25px;
    }
    .review-box strong { 
        color: #0D47A1; /* Dark blue for product name */
    }
    .review-box .rating { 
        color: #2196F3; /* Blue rating */
    }
    .section-content p { 
        font-size: 1.1rem;
    }
    .card { 
        background-color: #FFFFFF; /* White card background */
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
        padding: 40px 25px; 
        text-align: center;
        transition: all 0.3s ease-in-out;
    }
    .card:hover { 
        transform: scale(1.05); 
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); 
    }
    .card .icon { 
        font-size: 2.5rem; 
        margin-bottom: 20px;
    }
    .card h4 { 
        font-weight: 600; 
        font-size: 1.2rem; 
        color: #0D47A1; /* Blue text for card title */
    }
    .card p { 
        font-size: 1rem; 
        color: #1565C0; /* Lighter blue for card description */
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
st.markdown("""
<div class="header">
    <strong>Amazon Market Insights</strong>  
    <div style="font-size: 18px; color: #1565C0;">Make data-driven decisions with AI-powered review analysis!</div>
</div>
""", unsafe_allow_html=True)

# ---------------------- Navigation Menu ----------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "summarization", "Review Classificatin", "Find Category"],
    icons=["house", "bar-chart", "emoji-smile", "search"],
    default_index=0,
    orientation="horizontal",
)


# ---------------------- Home Page ----------------------
def home_page():
    st.markdown("""
        <style>
        .home-container {
            background: linear-gradient(to right, #e3f2fd, #ffffff);
            padding: 40px 20px;
            border-radius: 15px;
        }
        .feature-card {
            background-color: #ffffff;
            border-radius: 20px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
            padding: 30px;
            transition: transform 0.3s ease-in-out;
            text-align: center;
        }
        .feature-card:hover {
            transform: scale(1.05);
        }
        .feature-icon {
            font-size: 3rem;
            color: #2196F3;
            margin-bottom: 15px;
        }
        .review-box {
            padding: 20px;
            background-color: #F8FAFC;
            border-left: 5px solid #2196F3;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .review-box strong {
            font-size: 1.1rem;
            color: #0D47A1;
        }
        .rating {
            font-size: 1rem;
            color: #FFC107;
        }
        </style>
    """, unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h4>Explore Insights</h4>
            <p>Dive into customer opinions and patterns across product categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h4>Summarize Feedback</h4>
            <p>Get smart summaries of lengthy reviews in seconds.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <h4>AI Sentiment Analysis</h4>
            <p>Understand how customers feel: positive, neutral, or negative.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center; margin-top: 40px;">
            <h3 style="color:#1565C0;">✨ Random Product Reviews</h3>
            <p style="color:#0D47A1;">Here's a quick peek at some real feedback from customers.</p>
        </div>
    """, unsafe_allow_html=True)

    random_reviews = df.sample(7)
    for index, review in random_reviews.iterrows():
        product_name = review["name"]
        review_text = review["reviews.text"]
        rating = review["reviews.rating"]

        st.markdown(f"""
        <div class="review-box">
            <strong>{product_name}</strong> — <span class="rating">{round(rating, 2)}⭐</span>
            <div style="margin-top: 10px;">"{review_text}"</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 30px; font-size: 0.9rem; text-align: center; color: #888;">Amazon Review Analyzer | Powered by NLP & ❤️</div>', unsafe_allow_html=True)


# ---------------------- Topic Tagging ----------------------
def tag_review_topic(text):
    text = text.lower()

    if any(w in text for w in ["delivery", "shipping", "arrived", "delivered", "late", "fast", "on time", "delay", "delays", "ship", "courier", "package"]):
        return "Delivery"
    elif any(w in text for w in ["price", "expensive", "cheap", "value", "cost", "affordable", "worth", "overpriced", "underpriced", "deal", "pricing"]):
        return "Price"
    elif any(w in text for w in ["quality", "durable", "broke", "broken", "breaks", "material", "poor", "flimsy", "well-made", "solid", "defective", "sturdy", "cracked", "cheaply made"]):
        return "Quality"
    elif any(w in text for w in ["support", "service", "help", "customer", "return", "refund", "response", "rude", "friendly", "staff", "agent", "assistance", "warranty", "exchange"]):
        return "Customer Service"
    elif any(w in text for w in ["easy", "difficult", "install", "setup", "installation", "instructions", "manual", "user-friendly", "complicated", "confusing", "plug and play", "effortless"]):
        return "Ease of Use"
    elif any(w in text for w in ["design", "style", "look", "looks", "color", "aesthetic", "appearance", "elegant", "ugly", "sleek", "modern", "cool looking"]):
        return "Design"
    elif any(w in text for w in ["size", "fit", "too big", "too small", "perfect size", "bulky", "compact", "dimensions", "weight", "lightweight", "heavy", "oversized", "tiny"]):
        return "Size & Fit"
    return "Other"
df["review_topic"] = df["reviews.text"].apply(tag_review_topic) 

# ---------------------- summarization page ----------------------

def summarization_page():
    global df 
    st.markdown("## 🕵️ Product Review Insights")
    st.markdown("Get a high-level summary of product performance, user sentiment, and key feedback topics within each category.")

    category_options = ["🔽 Select a Category"] + list(df["meta_category"].unique()) + ["Add New Review"]
    selected_category = st.selectbox("🔎 Choose a Product Category", category_options)

    if selected_category == "🔽 Select a Category":
        st.info("Please choose a product category to explore insights.")
        return

    if selected_category == "Add New Review":
        st.markdown("### ✍️ Submit a New Product Review")

        user_name = st.text_input("Product Name")
        user_category = st.selectbox("Select Category", df["meta_category"].unique())
        user_review = st.text_area("Your Review")
        user_rating = st.slider("Rating", 1, 5, 5)

        if st.button("Submit Review"):
            if user_name and user_review:
                new_review = {
                    "name": user_name,
                    "meta_category": user_category,
                    "reviews.text": user_review,
                    "reviews.rating": user_rating,
                    "review_topic": tag_review_topic(user_review)
                }

                df = pd.concat([df, pd.DataFrame([new_review])], ignore_index=True)
                st.success("✅ Your review was added successfully!")
            else:
                st.warning("⚠️ Please fill in both the product name and the review.")
        return  w

    category_df = df[df["meta_category"] == selected_category]
    st.markdown(f"**📊 Total Reviews Analyzed:** {len(category_df)}")

    blog_post = generate_blog_post(selected_category, category_df)

    with st.expander("📝 Full Product Summary Report"):
        st.markdown(blog_post)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            x="reviews.rating",
            data=category_df,
            palette="Blues_d",
            edgecolor="black"
        )
        ax.set_title("Distribution of Customer Ratings", fontsize=14)
        ax.set_xlabel("Rating", fontsize=12)
        ax.set_ylabel("Number of Reviews", fontsize=12)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', padding=3, fontsize=10)
        st.pyplot(fig)

    with col2:
        st.markdown("### 🧩 Topics Mentioned in Reviews")
        topic_counts = category_df["review_topic"].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            y='Topic',
            x='Count',
            data=topic_counts,
            palette='viridis',
            edgecolor="black"
        )
        ax.set_title("Most Common Review Topics", fontsize=14)
        ax.set_xlabel("Number of Mentions", fontsize=12)
        ax.set_ylabel("")
        for i, v in enumerate(topic_counts['Count']):
            ax.text(v + 2, i, str(v), color='black', va='center', fontsize=10)
        st.pyplot(fig)

    st.markdown("### 🏆 Summary of Top Rated Products")
    top_products, _ = get_top_and_worst_products(category_df)
    st.table(top_products[['name', 'avg_rating', 'review_count']].rename(columns={
        'name': 'Product Name',
        'avg_rating': 'Average Rating',
        'review_count': 'Review Count'
    }))

    st.markdown("---")
 




def get_top_and_worst_products(category_df, top_n=3):
    grouped = category_df.groupby('name').agg(
        avg_rating=('reviews.rating', 'mean'),
        review_count=('reviews.text', 'count')
    ).reset_index()

    top = grouped.sort_values(by=['avg_rating', 'review_count'], ascending=False).head(top_n)
    worst_candidates = grouped[grouped['avg_rating'] < 4]
    worst = (
        worst_candidates.sort_values(by='avg_rating', ascending=True).head(1)
        if not worst_candidates.empty
        else pd.DataFrame()  # return empty 
    )

    return top, worst

def extract_complaints(df, product_name, threshold=3):
    complaints = df[(df['name'] == product_name) & (df['reviews.rating'] < threshold)]
    complaint_texts = complaints['reviews.text'].tolist()

    if not complaint_texts:
        return ["No major complaints."], []

    combined_text = " ".join(complaint_texts)[:2000]  
    summary_output = summarizer(combined_text, max_length=60, min_length=20, do_sample=False)
    complaint_summary = summary_output[0]['summary_text']

    return complaint_texts[:2], [complaint_summary]

def summarize_top_products(top_products, full_df):
    summary_text = ""
    for _, row in top_products.iterrows():
        product_df = full_df[full_df['name'] == row['name']]
        combined_text = " ".join(product_df['reviews.text'].tolist())[:2000]  
        summary = summarizer(combined_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        
        summary_text += f"\n▶ **{row['name']}**\n{summary}\n"
    return summary_text

def generate_blog_post(category_name, category_df):
    top, worst = get_top_and_worst_products(category_df)

    top_section = "### ⭐ Top Products:\n"
    top_summary = summarize_top_products(top, category_df)

    complaint_section = "### ⚠️ Top Complaints:\n"
    for product_name in top['name']:
        texts, _ = extract_complaints(category_df, product_name)
        formatted_complaints = '; '.join(texts) if texts else 'No major complaints.'
        complaint_section += f"- **{product_name}**: {formatted_complaints}\n\n"

    if worst.empty:
        worst_section = "### ❌ Worst Product to Avoid:\nNo significant underperforming products in this category."
    else:
        worst_name = worst['name'].values[0]
        worst_rating = worst['avg_rating'].values[0]
        _, worst_keywords = extract_complaints(category_df, worst_name)
        worst_issues = ', '.join(worst_keywords) if worst_keywords else 'Not enough data'
        worst_section = f"""### ❌ Worst Product to Avoid:
**{worst_name}** — {round(worst_rating, 2)}⭐ rating  
Top issues: {worst_issues}
"""

    article = f"""## 🛍️ Product Insights: **{category_name}**

---

{top_section}
{top_summary}

---

{complaint_section}

---

{worst_section}
"""

    return article



# ---------------------- Review Classifiactin page ----------------------


def classify_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
    
    labels = ["Negative", "Neutral", "Positive"]  
    return labels[predicted_class], probs[0].tolist()

def review_sentiment_page():
    st.subheader("🎭 Review Sentiment Classifier")

    review_input = st.text_area("Enter a customer review:")

    if 'history' not in st.session_state:
        st.session_state.history = []

    if review_input:
        sentiment, probs = classify_sentiment(review_input)
        
        review_result = {
            "Review": review_input,
            "Sentiment": sentiment,
            "Confidence (Positive)": probs[2] * 100,
            "Confidence (Neutral)": probs[1] * 100,
            "Confidence (Negative)": probs[0] * 100
        }
        
        st.session_state.history.insert(0, review_result)  
        
        N = 5
        st.session_state.history = st.session_state.history[:N]

        st.markdown(f"### Prediction: {sentiment}")
        st.markdown(f"Confidence: Positive {probs[2]*100:.1f}% | Neutral {probs[1]*100:.1f}% | Negative {probs[0]*100:.1f}%")
        
        labels = ["Negative", "Neutral", "Positive"]
        prob_df = pd.DataFrame({
            "Sentiment": labels,
            "Confidence": [p * 100 for p in probs]
        })
        
        fig = px.bar(prob_df, x="Sentiment", y="Confidence", color="Sentiment",
                     color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"},
                     text_auto=".1f")
        fig.update_layout(title="Confidence Levels (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔼 Upload a CSV with Reviews")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'reviews.text' in df.columns:
                results = []
                for review in df['reviews.text']:
                    sentiment, probs = classify_sentiment(review)
                    results.append({
                        "Review": review,
                        "Sentiment": sentiment,
                        "Confidence (Positive)": probs[2] * 100,
                        "Confidence (Neutral)": probs[1] * 100,
                        "Confidence (Negative)": probs[0] * 100
                    })
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)  
                
                sentiment_counts = results_df['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                sentiment_fig = px.bar(sentiment_counts, x='Sentiment', y='Count', color='Sentiment',
                                       color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"},
                                       text_auto=".1f", title="Sentiment Distribution of Uploaded Reviews")
                st.plotly_chart(sentiment_fig)
            else:
                st.error("CSV file does not contain 'reviews.text' column.")
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
       



# ---------------------- find product Page ----------------------

embedding_model = SentenceTransformer(r"C:\Users\R\OneDrive\Desktop\Ironhack\Week6\ClusteringCategory\sentence_transformer_model")
kmeans_model = load(r"C:\Users\R\OneDrive\Desktop\Ironhack\Week6\ClusteringCategory\kmeans_model.joblib")

cluster_to_meta = {
    0: "Household & Camera Batteries",
    1: "Fire Tablets & Electronics",
    2: "Household & Camera Batteries",
    3: "E-Readers & Tablets",
    4: "Kids Tablets & Office Tech Essentials"
}

def find_category_page():
    st.subheader("🔍 Predict Product Category")

    user_input = st.text_input("Enter a product name or short description:")

    if user_input:
        with st.spinner("Processing your request..."):
            embedding = embedding_model.encode([user_input])
            cluster_label = kmeans_model.predict(embedding)[0]
            predicted_category = cluster_to_meta.get(cluster_label, "Unknown")

            st.markdown(f"""
                <div class="card" style="margin-top: 15px;">
                    <h3>Predicted Category:</h3>
                    <p>{predicted_category}</p>
                </div>
            """, unsafe_allow_html=True)

            df = load_data()
            category_df = df[df["meta_category"] == predicted_category]

            st.markdown('<div style="font-size: 1.5rem; margin-top: 30px; font-weight: bold;">🔥 Top Products in this Category</div>', unsafe_allow_html=True)
            top_products = category_df.groupby("name").agg(
                avg_rating=("reviews.rating", "mean"),
                count=("reviews.text", "count")
            ).sort_values(by="avg_rating", ascending=False).head(3)

            for product, row in top_products.iterrows():
                st.markdown(f"""
                    <div style="background-color:#f9f9f9; padding: 15px; margin-top: 15px; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">
                        <strong style="font-size: 1.2rem;">{product}</strong> 
                        — {round(row['avg_rating'], 2)}⭐ ({row['count']} reviews)
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="font-size: 1.5rem; margin-top: 30px; font-weight: bold;">📝 Sample Reviews</div>', unsafe_allow_html=True)
            sample_reviews = category_df.sample(min(3, len(category_df)))
            for _, row in sample_reviews.iterrows():
                st.markdown(f"""
                    <div style="background-color:#f9f9f9; padding: 15px; margin-top: 15px; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">
                        <div style="font-size: 1.1rem; font-weight: bold;">{row['name']}</div> 
                        — <span style="color: #ff9900;">{row['reviews.rating']}⭐</span>
                        <div style="margin-top: 5px; font-style: italic;">"{row['reviews.text']}"</div>
                    </div>
                """, unsafe_allow_html=True)

               
       




# ---------------------- Render Selected Page ----------------------
if selected == "Home":
    home_page()
elif selected == "summarization":
    summarization_page()
elif selected == "Review Classificatin":
    review_sentiment_page()
elif selected == "Find Category":
    find_category_page()