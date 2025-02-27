
🚀 Stock Analysis with LLM & Yahoo Finance API

A Python-based stock analysis tool that integrates Local LLM (Large Language Model) and Yahoo Finance API to extract insights, trends, and news for a given stock.


📌 Features

✅ LLM-powered Entity Extraction (Company Name, Ticker, Date Range)

✅ Stock Trend Analysis (50-day & 200-day SMA, RSI, MACD)

✅ Stock Insights via LLM (Summarized performance for any year)

✅ Automated Stock News Fetching (Yahoo Finance API)

✅ Dynamic Stock Chart Generation (Price, RSI, MACD plots)

✅ Streamlit UI for Interactive Analysis



🔧 Installation

1️⃣ Clone the Repository

git clone https://github.com/syedayaan817/Stock-Advisor-Project.git
cd Stock-Advisor-Project

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App
before you do this, run the llm_utils.py script 
streamlit run main.py


📊 Usage Guide

💡 Query Format

✅ "Amazon stock trend from January 2022-February 2024
✅ "Summarize Tesla’s stock performance from 2020 to 2021"


📌 What Happens?

1. LLM extracts Company Name, Ticker, and Date Range from user query.


2. Yahoo Finance API fetches historical stock data.


3. ML-Based Trend Analysis is performed (SMA, RSI, MACD).


4. LLM generates stock insights (5-line summary).


5. Stock news & charts are displayed in Streamlit UI.



🛠 Project Structure

📂 stock-analysis-llm
│── main.py                # Streamlit App Entry Point
│── autogen_agent.py        # LLM-Based Entity Extraction
│── tools.py                # Stock Analysis, Charting, News Fetching
│── llm_utils.py            # Local LLM Response Generation
│── requirements.txt        # Dependencies
│── README.md               # Project Documentation



🚀 API Integrations

Yahoo Finance API (yfinance, yahooquery)

Local LLM (Hugging Face Transformers)

Streamlit (UI for Stock Analysis)





👤 Author

syedayaan817





📜 License

This project is licensed under the MIT License.




