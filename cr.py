# app.py
import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# 🌟 Page setup
st.set_page_config(page_title="Customer Sentiment Analysis", layout="wide")

# 🎨 Custom Title
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color:green;'>📊 Customer Review Sentiment Analysis</h1>
        <h3 style='color:gray;'>Powered by DistilBERT (Hugging Face Transformers)</h3>
        <p style='font-size:16px; color:#555;'>Analyze customer feedback (Single & Bulk) with AI 🚀</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ✅ Load pre-trained model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_model()

# ===========================================================
# 📂 Step 1: Bulk Review Analysis (CSV Upload)
# ===========================================================
st.subheader("📂 Step 1: Bulk Review Analysis (Upload CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with a column named 'review'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" in df.columns:
        with st.spinner("Analyzing reviews... ⏳"):
            results = sentiment_pipeline(df["review"].tolist())

        df["Sentiment"] = [res["label"] for res in results]
        df["Confidence"] = [res["score"] for res in results]

        st.success("✅ Analysis complete!")

        # Show results table
        st.subheader("📊 Results Table")
        st.dataframe(df)

        # 📌 Summary
        st.subheader("📈 Summary Statistics")
        positive = (df["Sentiment"] == "POSITIVE").sum()
        negative = (df["Sentiment"] == "NEGATIVE").sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Positive Reviews", positive)
        with col2:
            st.metric("❌ Negative Reviews", negative)

        # 📊 Visualization
        st.subheader("📊 Sentiment Distribution Chart")

        fig, ax = plt.subplots()
        counts = df["Sentiment"].value_counts()
        counts.plot(kind="bar", color=["green", "red"], ax=ax)
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    else:
        st.error("⚠️ CSV must contain a 'review' column.")

# ===========================================================
# 📝 Step 2: Single Review Analysis
# ===========================================================
st.subheader("📝 Step 2: Single Review Analysis")
review_text = st.text_area("💬 Enter your review here:")

if st.button("✨ Analyze Review"):
    if review_text.strip():
        with st.spinner("Analyzing review... ⏳"):
            result = sentiment_pipeline(review_text)[0]

        label = result["label"]
        score = result["score"]

        st.success("✅ Analysis complete!")

        st.subheader("Result")
        st.write(f"**Review:** {review_text}")
        st.write(f"➡ **Sentiment:** {label} (Confidence: {score:.2f})")

        if label == "POSITIVE":
            st.markdown("🌟 This is a **Positive** review. Customers are happy.")
        else:
            st.markdown("⚠️ This is a **Negative** review. Customer is dissatisfied.")
             
            
            