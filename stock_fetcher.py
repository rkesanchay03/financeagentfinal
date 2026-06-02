# stock_fetcher.py - Enhanced Version for Detailed Data Collection

import yfinance as yf
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st

# --- Initialization ---
load_dotenv()

# ─────────────────────────────────────────────────────────────
# SYMBOL RESOLVER  —  Handles user inputs like "TCS", "tcs", "tcs stock"
# ─────────────────────────────────────────────────────────────
# A curated map of popular Indian company names / short codes → NSE symbols
_INDIAN_SYMBOL_MAP = {
    # IT
    "TCS":         "TCS.NS",
    "INFOSYS":     "INFY.NS",
    "INFY":        "INFY.NS",
    "WIPRO":       "WIPRO.NS",
    "HCL":         "HCLTECH.NS",
    "HCLTECH":     "HCLTECH.NS",
    "TECHM":       "TECHM.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "MPHASIS":     "MPHASIS.NS",
    "LTI":         "LTI.NS",
    "LTIMINDTREE": "LTIM.NS",
    # Banking & Finance
    "HDFC BANK":   "HDFCBANK.NS",
    "HDFCBANK":    "HDFCBANK.NS",
    "ICICI BANK":  "ICICIBANK.NS",
    "ICICIBANK":   "ICICIBANK.NS",
    "SBI":         "SBIN.NS",
    "SBIN":        "SBIN.NS",
    "AXIS BANK":   "AXISBANK.NS",
    "AXISBANK":    "AXISBANK.NS",
    "KOTAK":       "KOTAKBANK.NS",
    "KOTAKBANK":   "KOTAKBANK.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "BAJFINANCE":  "BAJFINANCE.NS",
    "HDFC":        "HDFC.NS",
    # FMCG & Consumer
    "HUL":         "HINDUNILVR.NS",
    "HINDUNILVR":  "HINDUNILVR.NS",
    "ITC":         "ITC.NS",
    "NESTLE":      "NESTLEIND.NS",
    "NESTLEIND":   "NESTLEIND.NS",
    "DABUR":       "DABUR.NS",
    "BRITANNIA":   "BRITANNIA.NS",
    # Auto
    "TATA MOTORS": "TATAMOTORS.NS",
    "TATAMOTORS":  "TATAMOTORS.NS",
    "MARUTI":      "MARUTI.NS",
    "M&M":         "M&M.NS",
    "MAHINDRA":    "M&M.NS",
    "HERO":        "HEROMOTOCO.NS",
    "BAJAJ AUTO":  "BAJAJ-AUTO.NS",
    # Energy & Infra
    "RELIANCE":    "RELIANCE.NS",
    "RIL":         "RELIANCE.NS",
    "ONGC":        "ONGC.NS",
    "NTPC":        "NTPC.NS",
    "POWERGRID":   "POWERGRID.NS",
    "ADANI":       "ADANIENT.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANIENT":    "ADANIENT.NS",
    "L&T":         "LT.NS",
    "LT":          "LT.NS",
    # Pharma
    "SUNPHARMA":   "SUNPHARMA.NS",
    "SUN PHARMA":  "SUNPHARMA.NS",
    "DR REDDY":    "DRREDDY.NS",
    "DRREDDYS":    "DRREDDY.NS",
    "CIPLA":       "CIPLA.NS",
    "DIVIS":       "DIVISLAB.NS",
    # Telecom
    "AIRTEL":      "BHARTIARTL.NS",
    "BHARTIARTL":  "BHARTIARTL.NS",
    "JIO":         "RELIANCE.NS",
    # Metals
    "TATA STEEL":  "TATASTEEL.NS",
    "TATASTEEL":   "TATASTEEL.NS",
    "HINDALCO":    "HINDALCO.NS",
    "JSW STEEL":   "JSWSTEEL.NS",
    "JSWSTEEL":    "JSWSTEEL.NS",
    # Index ETFs
    "NIFTY 50":    "^NSEI",
    "NIFTY":       "^NSEI",
    "SENSEX":      "^BSESN",
}


def resolve_symbol(user_input: str) -> str:
    """
    Tries to convert free-form user input (e.g. 'TCS stock', 'tell me about Infosys')
    into a proper yfinance ticker symbol.
    Priority:
      1. Already a valid-looking ticker (ends in .NS / .BSE / .BO)
      2. Direct token/multi-word match in curated Indian map
      3. Greedy noise cleaning fallback
      4. Default .NS suffix for single word tickers
    """
    cleaned = user_input.strip().upper()
    
    # 1. If already has exchange suffix, trust it
    if cleaned.endswith(".NS") or cleaned.endswith(".BSE") or cleaned.endswith(".BO"):
        return cleaned

    # 2. Tokenize and search for mapped Indian stock names
    # This prevents complex natural sentences from blocking matching (e.g. "Tell me about Reliance")
    words = [w.strip(".,?!()\"'") for w in cleaned.split()]
    words = [w for w in words if w]
    
    # Try 2-word combinations first (e.g. "HDFC BANK" or "TATA MOTORS")
    for i in range(len(words) - 1):
        combo = f"{words[i]} {words[i+1]}"
        if combo in _INDIAN_SYMBOL_MAP:
            return _INDIAN_SYMBOL_MAP[combo]
            
    # Try single word matches (e.g. "RELIANCE", "TCS")
    for w in words:
        if w in _INDIAN_SYMBOL_MAP:
            return _INDIAN_SYMBOL_MAP[w]

    # 3. Greedy noise word removal (longest phrases first to avoid sub-phrase collisions)
    noise = [
        "TELL ME ABOUT", "WHAT IS THE", "WHAT IS", "TELL ME", 
        "STOCK PRICE", "SHARE PRICE", "STOCK", "SHARE", "SHARES", 
        "PRICE", "ABOUT", "ANALYZE", "ANALYSIS", "REPORT", "ON"
    ]
    for word in noise:
        cleaned = cleaned.replace(word, "").strip()

    # Direct map lookup post-cleaning
    if cleaned in _INDIAN_SYMBOL_MAP:
        return _INDIAN_SYMBOL_MAP[cleaned]

    # Remove spaces and lookup (e.g. "HDFC BANK" -> "HDFCBANK")
    no_space = cleaned.replace(" ", "")
    if no_space in _INDIAN_SYMBOL_MAP:
        return _INDIAN_SYMBOL_MAP[no_space]

    # 4. Default suffixing
    if " " not in cleaned and len(cleaned) <= 12 and cleaned:
        return f"{cleaned}.NS"

    return cleaned  # return as-is for the LLM to handle



# ─────────────────────────────────────────────────────────────
# NEWS via yfinance (replaces Finnhub — supports Indian stocks)
# ─────────────────────────────────────────────────────────────
def _fetch_yfinance_news(ticker_obj, max_items=5):
    """
    Pull recent news headlines from the yfinance Ticker object.
    yfinance returns a list of dicts with keys: title, publisher, link, etc.
    """
    try:
        raw_news = ticker_obj.news or []
        news_items = []
        for item in raw_news[:max_items]:
            content = item.get("content", {})
            title = content.get("title") or item.get("title") or "No headline"
            publisher = content.get("provider", {}).get("displayName") or item.get("publisher") or "Unknown"
            url = content.get("canonicalUrl", {}).get("url") or item.get("link") or ""
            news_items.append({
                "headline": title,
                "source":   publisher,
                "url":      url,
            })
        if not news_items:
            news_items = [{"headline": "No recent news found via yfinance.", "source": "", "url": ""}]
        return news_items
    except Exception as e:
        print(f"yfinance news fetch error: {e}")
        return [{"headline": "News temporarily unavailable.", "source": "", "url": ""}]


# ─────────────────────────────────────────────────────────────
# MAIN DATA FETCH  (cached 10 minutes)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def fetch_detailed_stock_data(symbol: str):
    """
    Fetches a comprehensive set of data for a given stock symbol
    from yfinance to provide rich context for the LLM.
    """
    print(f"Fetching detailed data for {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # 1. Core Profile
        profile_data = {
            "companyName":        info.get("longName"),
            "sector":             info.get("sector"),
            "industry":           info.get("industry"),
            "country":            info.get("country"),
            "website":            info.get("website"),
            "longBusinessSummary": info.get("longBusinessSummary"),
            "marketCap":          info.get("marketCap"),
            "sharesOutstanding":  info.get("sharesOutstanding"),
        }

        # 2. Quote
        quote_data = {
            "currentPrice":    info.get("currentPrice"),
            "previousClose":   info.get("previousClose"),
            "open":            info.get("open"),
            "dayHigh":         info.get("dayHigh"),
            "dayLow":          info.get("dayLow"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow":  info.get("fiftyTwoWeekLow"),
            "volume":          info.get("volume"),
            "averageVolume":   info.get("averageVolume"),
        }

        # 3. Key Financial Ratios
        financial_ratios = {
            "peRatio":        info.get("trailingPE"),
            "forwardPeRatio": info.get("forwardPE"),
            "priceToBook":    info.get("priceToBook"),
            "priceToSales":   info.get("priceToSalesTrailing12Months"),
            "dividendYield":  info.get("dividendYield"),
            "earningsGrowth": info.get("earningsQuarterlyGrowth"),
            "revenueGrowth":  info.get("revenueQuarterlyGrowth"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity":   info.get("debtToEquity"),
            "profitMargins":  info.get("profitMargins"),
        }

        # 4. Analyst Ratings
        analyst_ratings = {
            "recommendation":         info.get("recommendationKey", "N/A").upper(),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            "targetMeanPrice":        info.get("targetMeanPrice"),
            "targetHighPrice":        info.get("targetHighPrice"),
            "targetLowPrice":         info.get("targetLowPrice"),
        }

        # 5. News via yfinance (works for Indian stocks!)
        news = _fetch_yfinance_news(ticker)

        consolidated_data = {
            "symbol":         symbol,
            "fetchTimestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "profile":        profile_data,
            "quote":          quote_data,
            "financialRatios": financial_ratios,
            "analystRatings": analyst_ratings,
            "news":           news,
        }

        print("Data fetching complete.")
        return consolidated_data

    except Exception as e:
        print(f"An error occurred while fetching data for {symbol}: {e}")
        return {"error": f"Could not retrieve data for symbol '{symbol}'. Please try a valid NSE ticker (e.g., TCS.NS, RELIANCE.NS)."}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_history(symbol: str, period: str = "6mo"):
    """
    Fetches historical OHLCV data for Plotly charting.
    Supported periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max
    """
    print(f"Fetching {period} history for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            return None
        hist.reset_index(inplace=True)
        return hist
    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return None


if __name__ == "__main__":
    import json
    sym = resolve_symbol("tcs")
    print(f"Resolved: {sym}")
    data = fetch_detailed_stock_data(sym)
    print(json.dumps(data, indent=2, default=str))
