# llm_analyzer.py - Enhanced with Smart Intent Classification

import requests
import os
import json
from dotenv import load_dotenv
from stock_fetcher import fetch_detailed_stock_data, resolve_symbol
from datetime import datetime

# --- Initialization ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _call_gemini_api(prompt, system_message):
    """A centralized helper to make REST calls to the Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key or api_key == "your_gemini_api_key_here":
        return "⚠️ **Configuration Error**: `GEMINI_API_KEY` is missing or has a placeholder value in your `.env` file. Please update it and restart the application."

    if not (api_key.startswith("AIza") or api_key.startswith("AQ.")):
        return "⚠️ **API Key Format Error**: The key does not appear to be a valid Google Gemini API key (expected `AIza` or `AQ.` prefix)."

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    if system_message:
        payload["system_instruction"] = {"parts": [{"text": system_message}]}

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        resp_json = response.json()
        candidates = resp_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return "Error: Could not extract response text from Gemini API."

    except requests.exceptions.HTTPError as he:
        error_details = ""
        try:
            err_msg = response.json().get("error", {}).get("message", "No message")
            error_details = f"\n\n**Google API Error**: `{err_msg}`"
        except Exception:
            pass
        return f"⚠️ **Gemini API HTTP Error ({response.status_code})**: {he}{error_details}"

    except requests.exceptions.ConnectionError:
        return "⚠️ **Network Error**: Could not connect to Google's servers. Check your internet connection."

    except Exception as e:
        return f"Sorry, an unexpected error occurred: {e}"


# ─────────────────────────────────────────────────────────────
# SMART INTENT CLASSIFIER
# ─────────────────────────────────────────────────────────────
_STOCK_TRIGGER_PHRASES = [
    "stock", "share", "price", "report", "analyze", "analysis",
    "tell me about", "what is", "show me", "give me", "nse", "bse",
    "equity", "market cap", "pe ratio", "dividend", "earnings"
]

def _looks_like_stock_query(text: str) -> bool:
    """
    Heuristic check: does the text appear to be asking about a specific stock?
    Covers:
      - Explicit ticker suffixes:   TCS.NS, RELIANCE.BSE
      - Pure short-code:            TCS, INFY, WIPRO
      - Natural language:           "tell me about TCS", "TCS stock price"
    """
    t = text.strip().upper()

    # Has exchange suffix → definitely a stock
    if t.endswith(".NS") or t.endswith(".BSE") or t.endswith(".BO"):
        return True

    # Check curated map
    from stock_fetcher import _INDIAN_SYMBOL_MAP
    cleaned = t
    for word in ["STOCK", "SHARE", "SHARES", "PRICE", "ABOUT", "TELL ME ABOUT",
                 "WHAT IS", "ANALYZE", "ANALYSIS", "REPORT", "ON"]:
        cleaned = cleaned.replace(word, "").strip()
    if cleaned in _INDIAN_SYMBOL_MAP or cleaned.replace(" ", "") in _INDIAN_SYMBOL_MAP:
        return True

    # Contains a trigger phrase alongside a short capitalised word (possible ticker)
    lower = text.lower()
    for phrase in _STOCK_TRIGGER_PHRASES:
        if phrase in lower:
            words = text.strip().split()
            for w in words:
                w_clean = w.strip(".,?!").upper()
                if (2 <= len(w_clean) <= 12 and w_clean.isalpha()
                        and w_clean not in {"THE", "A", "AN", "IS", "IN", "OF",
                                            "FOR", "ME", "MY", "AT", "ON", "BY",
                                            "ITS", "NSE", "BSE", "STOCK", "SHARE",
                                            "TELL", "SHOW", "GIVE", "WHAT", "ABOUT"}):
                    return True
    return False


def classify_intent(question: str, stock_context) -> str:
    """
    Classifies intent into: new_report | follow_up | general_query
    Uses a fast heuristic first; falls back to LLM if ambiguous.
    """
    # Fast heuristic path (avoids an extra LLM call)
    if _looks_like_stock_query(question):
        return "new_report"

    context_exists = "Yes" if stock_context else "No"

    system_message = (
        "You are an expert at classifying user intent in a financial chatbot. "
        "Respond with ONLY ONE of these three words (no punctuation, no explanation):\n"
        "new_report — user is asking for a stock report on a company or ticker\n"
        "follow_up  — user is asking a question about the stock already discussed\n"
        "general_query — any other finance question or concept"
    )
    prompt = (
        f"Stock context currently loaded: {context_exists}\n"
        f'User message: "{question}"\n\n'
        "Classification:"
    )

    raw = _call_gemini_api(prompt, system_message).strip().lower()
    # Normalise — handle extra words / newlines in the response
    for label in ["new_report", "follow_up", "general_query"]:
        if label in raw:
            return label
    return "general_query"


# ─────────────────────────────────────────────────────────────
# GENERAL QUESTION ANSWERING
# ─────────────────────────────────────────────────────────────
def answer_general_question(question: str) -> str:
    prompt = (
        f'Answer the following finance question clearly and concisely, '
        f'focusing on the Indian market where relevant.\n\n'
        f'Question: "{question}"\n\n'
        f'Current date: {datetime.now().strftime("%A, %B %d, %Y")}'
    )
    system_message = (
        "You are a knowledgeable finance expert who specialises in Indian markets "
        "(NSE, BSE, SEBI regulations, mutual funds, etc.). "
        "Provide clear, accurate, and helpful explanations for all experience levels."
    )
    return _call_gemini_api(prompt, system_message)


# ─────────────────────────────────────────────────────────────
# RAG QUESTION ANSWERING
# ─────────────────────────────────────────────────────────────
def answer_rag_question(question: str, document_name: str, retrieved_context: list) -> str:
    context_str = "\n\n---\n\n".join(retrieved_context)
    prompt = f"""
You are a highly analytical AI finance assistant. You have been provided with relevant retrieved extracts from the uploaded document: **{document_name}**.

Use the retrieved extracts below to answer the user's question accurately. If the answer cannot be found or inferred from the provided context, state clearly that the document does not contain that information, but answer to the best of your ability if general knowledge allows.

**Retrieved Extracts from {document_name}:**
{context_str}

**User's Question:** "{question}"

Provide a structured, detailed, and professional answer. Highlight key numbers, dates, or insights from the extracts where relevant.
"""
    return _call_gemini_api(
        prompt,
        f"You are a professional financial analyst AI answering questions based on the uploaded document: {document_name}."
    )


# ─────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────
def generate_report(user_input: str):
    """
    Resolves the user's input to a proper ticker symbol,
    fetches data, and generates a comprehensive AI report.
    Returns (report_text, data_dict).
    """
    symbol = resolve_symbol(user_input)
    print(f"Resolved '{user_input}' → '{symbol}'")

    data = fetch_detailed_stock_data(symbol)
    if data.get("error"):
        return data["error"], None

    data_json_string = json.dumps(data, indent=2, default=str)

    prompt = f"""
You are a senior equity analyst. Generate a clean, structured, **table-first** financial report for **{symbol}**.

Use the following live data exclusively (do not invent numbers):
```json
{data_json_string}
```

Output the report in this EXACT format using Markdown. Use tables wherever data is present. Keep prose sections concise (2–3 sentences max each).

---

## 🏢 Company Snapshot

| Field | Details |
|---|---|
| Company | _(longName)_ |
| Sector | _(sector)_ |
| Industry | _(industry)_ |
| Country | _(country)_ |
| Website | _(website)_ |
| Market Cap | _(formatted marketCap)_ |

_(One concise sentence on the company's core business and market position.)_

---

## 📊 Price & Market Data

| Metric | Value |
|---|---|
| Current Price | _(currentPrice with ₹ symbol)_ |
| Previous Close | _(previousClose)_ |
| Day Open | _(open)_ |
| Day High / Low | _(dayHigh / dayLow)_ |
| 52-Week High | _(fiftyTwoWeekHigh)_ |
| 52-Week Low | _(fiftyTwoWeekLow)_ |
| Volume | _(volume formatted)_ |
| Avg Volume | _(averageVolume formatted)_ |

_(One sentence on current price momentum or proximity to 52-week extremes.)_

---

## 💰 Key Financial Ratios

| Ratio | Value | Interpretation |
|---|---|---|
| P/E Ratio (Trailing) | _(trailingPE)_ | _(brief interpretation: cheap/fair/expensive vs sector)_ |
| P/E Ratio (Forward) | _(forwardPE)_ | _(brief)_ |
| Price-to-Book | _(priceToBook)_ | _(brief)_ |
| Price-to-Sales | _(priceToSales)_ | _(brief)_ |
| Return on Equity | _(returnOnEquity as %)_ | _(brief)_ |
| Profit Margin | _(profitMargins as %)_ | _(brief)_ |
| Earnings Growth | _(earningsGrowth as %)_ | _(positive/negative)_ |
| Revenue Growth | _(revenueGrowth as %)_ | _(positive/negative)_ |
| Debt-to-Equity | _(debtToEquity)_ | _(low/moderate/high leverage)_ |
| Dividend Yield | _(dividendYield as %)_ | _(income potential)_ |

---

## 🎯 Analyst Consensus

| Metric | Value |
|---|---|
| Recommendation | _(recommendation)_ |
| Number of Analysts | _(numberOfAnalystOpinions)_ |
| Target Price (Mean) | _(targetMeanPrice with ₹)_ |
| Target Price (High) | _(targetHighPrice with ₹)_ |
| Target Price (Low) | _(targetLowPrice with ₹)_ |
| Upside / Downside | _(calculate % upside from current price to mean target)_ |

_(One sentence on overall market sentiment — bullish / neutral / bearish.)_

---

## 📰 Recent News & Catalysts

| # | Headline | Source | Sentiment |
|---|---|---|---|
_(List each news item as a table row. Sentiment = Positive / Neutral / Negative)_

---

## ⚖️ Strengths vs Risks

| Strengths ✅ | Risks ⚠️ |
|---|---|
| _(strength 1)_ | _(risk 1)_ |
| _(strength 2)_ | _(risk 2)_ |
| _(strength 3)_ | _(risk 3)_ |

---

## 🤖 AI Synthesis

_(2–3 sentences of AI-driven forward-looking observations — key trend, inflection point, or watch item for investors.)_

---
*AI-generated report · Data as of {datetime.now().strftime('%d %b %Y')} · Not financial advice.*
"""
    report_text = _call_gemini_api(
        prompt,
        "You are an expert financial analyst AI specialising in Indian equity markets. "
        "Always output valid Markdown tables. Never skip a table. Fill every cell with real data from the JSON. "
        "Use ₹ for Indian prices. Use N/A if data is missing."
    )
    return report_text, data


# ─────────────────────────────────────────────────────────────
# FOLLOW-UP ANSWERING
# ─────────────────────────────────────────────────────────────
def generate_follow_up_answer(question: str, context: dict) -> str:
    context_json_string = json.dumps(context, indent=2, default=str)
    prompt = f"""
You have already provided a report on **{context.get('symbol', 'this stock')}**.
The user now asks a follow-up question. Answer it directly and concisely using the reference data below.

**Reference Data:**
```json
{context_json_string}
```

**User's Follow-up Question:** "{question}"

Provide a focused, accurate answer. Cite specific numbers from the data where relevant.
"""
    return _call_gemini_api(prompt, "You are a helpful AI assistant answering follow-up financial questions.")