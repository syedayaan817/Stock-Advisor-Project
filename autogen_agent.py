import autogen
from tools import extract_entities, fetch_stock_data, analyze_trend, plot_stock_data, fetch_stock_news,stock_insights
from llm_utils import generate_local_llm_response

# Define Custom Assistant Agent Using Local LLM
class LocalStockAnalysisAgent(autogen.AssistantAgent):
    def generate_response(self, message, **kwargs):
        return {"content": generate_local_llm_response(message["content"])}

assistant = LocalStockAnalysisAgent(
    name="StockAnalysisBot",
    system_message="You are a stock market analysis assistant. Automatically analyze stock trends, fetch news, and generate stock charts.",
)

# Define User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="StockQueryUser",
    human_input_mode="NEVER",
    system_message="You are a user asking about stock trends and company insights.",
    is_termination_msg=lambda x: x.get("content", "").strip().lower() == "exit",
    code_execution_config={'use_docker': False},
)

# Function to Process Stock Queries
def process_stock_query(user_input):
    """Handles stock-related queries and automatically invokes tools."""
    
    # Extract details from query
    company_name, ticker, start_date, end_date = extract_entities(user_input)
    print(f"🔍 Extracted-Company: {company_name}, Ticker: {ticker}, Date Range: {start_date} to {end_date}")

    if not ticker:
        return {"error": "⚠️ Could not determine the stock ticker. Please try again with a clearer company name."}

    if not start_date or not end_date:
        return {"error": "⚠️ Could not determine the date range. Please try again."}

    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    if stock_data is None:
        return {"error": f"⚠️ No stock data found for {company_name} ({ticker}) from {start_date} to {end_date}."}

    # Generate stock trend analysis
    trend_report = analyze_trend(stock_data, ticker,user_input)

    # Fetch latest stock news
    news = fetch_stock_news(ticker, start_date[:4])

    # Generate stock chart
    chart_path = plot_stock_data(stock_data, ticker,company_name)
    stock = stock_insights(ticker,start_date[:4],end_date[:4])

    return {
        "analysis": trend_report,
        "news": news,
        "chart_path": chart_path,
        "stock_insights":stock,
    }

# Function for AutoGen to Use
def query_autogen(user_input):
    """Handles stock analysis query via AutoGen."""
    
    # Automatically process the query using the assistant
    response = user_proxy.initiate_chat(assistant, message=user_input)

    # Process the stock query using the agent's functions
    stock_analysis = process_stock_query(user_input)

    return stock_analysis