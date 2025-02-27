import yfinance as yf
import yahooquery as yq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import spacy
from datetime import datetime
import os
from llm_utils import generate_local_llm_response  # ✅ Import LLM function

# Load the SpaCy language model
nlp = spacy.load("en_core_web_sm")
save_dir = "charts"

# ✅ Function to extract company name, ticker, and dates using LLM
def extract_entities(user_input):
    """
    Uses an LLM to extract stock-related entities (company name, start date, end date).
    If the ticker is missing, it is retrieved using Yahoo Finance search.
    """

    # ✅ LLM Prompt
    prompt = f"""
    Extract only these details from the following query:
    - Company Name
    - Start Date and End Date (YYYY-MM-DD format)
       Query: "{user_input}"
       **IMPORTANT INSTRUCTIONS**:
       1. **Do not assume or replace the company name.** **The company name **must exactly match** what is present in the query.
       2.**Do not make up a company name.** **If unsure, leave it blank
       3.**Do not include example names** like "Tesla,Apple,Microsoft" in the response. Return only the extracted details.
       4.**Return the response in the following format**(no extra text):

    Response format:
    - Company Name: <Company Name>
    - Start Date: <YYYY-MM-DD>
    - End Date: <YYYY-MM-DD>
    """

    response = generate_local_llm_response(prompt)  # Get LLM output
    print(f"🔹 Raw LLM Output:\n{response}")  # Debugging

    # ✅ Extracting Only Relevant Details
    company_name, start_date, end_date = None, None, None
    extracting = False  # Flag to detect real data

    for line in response.split("\n"):
        line = line.strip()
        
        # Ignore placeholder values and format instructions
        if "<Company Name>" in line or "<YYYY-MM-DD>" in line:
            continue  

        # Start extracting real data when we see a valid company name
        if line.startswith("Company Name:") or line.startswith("- Company Name:") and not extracting:
            extracting = True  # We are now in the real extracted data
            company_name = line.split(":", 1)[-1].strip()

        elif extracting and line.startswith("Start Date:") or line.startswith("- Start Date:"):
            start_date = line.split(":", 1)[-1].strip()
        elif extracting and line.startswith("End Date:") or line.startswith("- End Date:"):
            end_date = line.split(":", 1)[-1].strip()

    # ✅ Improved Ticker Extraction
    ticker = None
    if company_name:
        ticker = get_ticker_from_name(company_name)  # Fetch ticker from Yahoo Finance
        if not ticker:
            print(f"⚠️ Warning: No ticker found for '{company_name}'. Trying variations...")

            # Try alternative variations (e.g., Tesla Inc., NVIDIA Corporation)
            search_variations = [company_name, f"{company_name} Inc.", f"{company_name} Corporation"]
            for variation in search_variations:
                ticker = get_ticker_from_name(variation)
                if ticker:
                    print(f"✅ Found ticker '{ticker}' for '{variation}'")
                    break

    print(f"✅ Extracted - Company: {company_name}, Ticker: {ticker}, Date Range: {start_date} to {end_date}")

    return company_name, ticker, start_date, end_date
# Function to fetch stock data using Yahoo Finance API
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    if df.empty:
        return None

    # Compute technical indicators
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["RSI_14"] = compute_rsi(df["Close"], window=14)
    df["MACD"], df["MACD_Signal"] = compute_macd(df["Close"])
    print(df)
    return df

# Function to compute the Relative Strength Index (RSI)
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

# Function to compute the Moving Average Convergence Divergence (MACD)
def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    ema_short = series.ewm(span=short_window, adjust=False).mean()
    ema_long = series.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal

# Function to get ticker symbol from company name using Yahoo Finance search
def get_ticker_from_name(company_name):
    try:
        search_results = yq.search(company_name)
        if isinstance(search_results, dict) and "quotes" in search_results and search_results["quotes"]:
            return search_results["quotes"][0].get("symbol")
        return None
    except Exception as e:
        print(f"Error fetching ticker: {e}")
        return None
def stock_insights(company_name, year1, year2 ):
    """Generate LLM-Based Stock Insights and clean the response properly."""
    
    # ✅ Generate LLM query
    query = f"Summarize {company_name}'s stock performance from {year1} to {year2}, compare its performance in the past year in 5 lines."
    
    # ✅ Get LLM response
    llm_response = generate_local_llm_response(query).strip()

    # ✅ Clean and extract relevant insights
    response_lines = llm_response.split("\n")
    extracted_insights = []
    
    for line in response_lines:
        line = line.strip()
        
        # ✅ Exclude prompt repetition or unwanted text
        if not line.lower().startswith(("summarize", "compare", "year")) and line:
            extracted_insights.append(line)
            if len(extracted_insights) == 5:
                break

    # ✅ Ensure final cleaned response is structured
    cleaned_response = extracted_insights
    
    if not cleaned_response:
        cleaned_response = "⚠️ AI insights could not be extracted properly."
  # Debugging
    return cleaned_response
# Function to fetch stock news using Yahoo Finance API
def fetch_stock_news(ticker, year):
    search_results = yq.search(ticker).get("news", [])
    news_articles = [f"- {item.get('title', 'No title')}" for item in search_results]
    return news_articles if news_articles else ["No news found."]

# Function to analyze stock trends
def analyze_trend(df, ticker,user_input):
    latest = df.iloc[-1]
    close_price = latest["Close"]
    sma_50 = latest["SMA_50"]
    sma_200 = latest["SMA_200"]
    rsi_14 = latest["RSI_14"]
    macd = latest["MACD"]
    macd_signal = latest["MACD_Signal"]

    analysis = f"\n📊 Stock Analysis Report for {ticker} ({df.index[-1].date()})\n\n"
    analysis += f"\n📈 Current Closing Price: ${close_price:.2f}\n"
    analysis += f"\n📉 50-day SMA: $ {sma_50:.2f}, 200-day SMA: $ {sma_200:.2f}\n"

    if close_price > sma_50:
        analysis += "\n✅ Short-term Uptrend (Price > 50-day SMA)\n"
    else:
        analysis += "\n❌ Short-term Downtrend (Price < 50-day SMA)\n"

    if close_price > sma_200:
        analysis += "\n✅ Long-term Bullish (Price > 200-day SMA)\n"
    else:
        analysis += "\n❌ Long-term Bearish (Price < 200-day SMA)\n"

    if macd > macd_signal:
        analysis += "\n📊 MACD: Bullish Crossover (Uptrend Expected) ✅\n"
    else:
        analysis += "\n📊 MACD: Bearish Crossover (Downtrend Possible) ❌\n"

    if rsi_14 > 70:
        analysis += "\n⚠️ RSI: Overbought (Possible Price Drop) ⚠️\n"
    elif rsi_14 < 30:
        analysis += "\n💰 RSI: Oversold (Buying Opportunity) 💰\n"
    else:
        analysis += "\n📊 RSI: Neutral (Stable Market)\n"
    return analysis
def plot_stock_data(df, ticker,company_name):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

    # Price Chart + Moving Averages
    axes[0].plot(df.index, df["Close"], label="Close Price", color="black", linewidth=2)
    axes[0].plot(df.index, df["SMA_50"], label="50-day SMA", linestyle="dashed", color="blue", linewidth=1.5)
    axes[0].plot(df.index, df["SMA_200"], label="200-day SMA", linestyle="dashed", color="red", linewidth=1.5)
    axes[0].set_title(f"{company_name} Stock Price & Moving Averages")
    axes[0].legend()
    axes[0].grid(True)
    # *2️⃣ RSI Indicator*
    axes[1].plot(df.index, df["RSI_14"], label="RSI (14)", color="purple", linewidth=1.5)
    axes[1].axhline(y=70, linestyle="dashed", color="red", label="Overbought (70)")
    axes[1].axhline(y=30, linestyle="dashed", color="green", label="Oversold (30)")
    axes[1].set_title("Relative Strength Index (RSI)")
    axes[1].legend()
    axes[1].grid(True)

    # *3️⃣ MACD & Signal Line*
    axes[2].plot(df.index, df["MACD"], label="MACD", color="blue", linewidth=1.5)
    axes[2].plot(df.index, df["MACD_Signal"], label="MACD Signal", linestyle="dashed", color="red", linewidth=1.5)
    axes[2].axhline(y=0, linestyle="dashed", color="gray", linewidth=1)
    axes[2].set_title("MACD & Signal Line")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()


    plt.tight_layout()
    
    chart_path = os.path.join(save_dir, f"{ticker}_chart.png")
    plt.savefig(chart_path)
    plt.close()
    return chart_path