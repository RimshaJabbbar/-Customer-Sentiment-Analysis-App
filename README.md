# -Customer-Sentiment-Analysis-App
AI-powered Streamlit app for Customer Sentiment Analysis using Hugging Face DistilBERT, supporting single &amp; bulk review analysis with visual insights.

This repository contains an AI-powered Customer Sentiment Analysis Web Application built with Streamlit and Hugging Face Transformers (DistilBERT). The app allows users to analyze customer reviews by classifying them into Positive or Negative sentiments with confidence scores.

It supports both bulk review analysis (via CSV upload) and single review analysis (manual input), making it a versatile tool for businesses, analysts, and AI enthusiasts. The app also includes summary metrics and visualizations to help understand customer feedback trends more effectively.

✨ Key Features:

📂 Bulk Review Analysis → Upload CSV files with a review column for batch processing.

📝 Single Review Analysis → Enter a custom review and get instant predictions.

📊 Data Visualization → Sentiment distribution chart (Positive vs Negative) using Matplotlib.

✅ Summary Stats → Auto-calculates positive/negative review counts with interactive metrics.

🚀 Pre-trained AI Model → Hugging Face distilbert-base-uncased-finetuned-sst-2-english.

🛠 Tech Stack / Libraries Used:

Streamlit → For building the interactive web interface

Transformers (Hugging Face) → For loading the DistilBERT sentiment analysis pipeline

Pandas → For handling CSV file uploads and data processing

Matplotlib → For generating sentiment distribution bar charts

This project is ideal for:

Businesses analyzing customer feedback at scale

Data analysts looking for sentiment insights

Developers exploring NLP and Hugging Face Transformers with Streamlit
