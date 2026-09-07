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
        # Check if it's a 503 (Overloaded) or 429 (Rate Limit)
        if response.status_code in [503, 429] and model != "gemini-1.5-flash":
            print(f"Model {model} is overloaded. Falling back to gemini-1.5-flash...")
            fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            try:
                fb_response = requests.post(fallback_url, headers={"Content-Type": "application/json"}, json=payload)
                fb_response.raise_for_status()
                fb_resp_json = fb_response.json()
                fb_candidates = fb_resp_json.get("candidates", [])
                if fb_candidates:
                    fb_parts = fb_candidates[0].get("content", {}).get("parts", [])
                    if fb_parts:
                        return fb_parts[0].get("text", "")
            except Exception as fallback_e:
                return f"⚠️ **Gemini API Error**: Both primary and fallback models are currently unavailable. Please try again later."
                
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
from rag_processor import get_embeddings_batch, cosine_similarity

class SemanticRouter:
    """
    A fast, API-based Semantic Router that uses embeddings to classify intent.
    It embeds example phrases once and caches them. When a user asks a question,
    it embeds the question and uses Cosine Similarity to find the closest intent cluster.
    This avoids the slow LLM text generation API call!
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SemanticRouter, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # 1. Define example queries for each intent
        self.examples = {
            "new_report": [
                "TCS price", "give me a report on reliance", "how is hdfc doing", 
                "tata motors stock", "what is the market cap of infosys", "TCS.NS",
                "show me sbi", "analyze wipro", "I want to see adani", "nifty 50",
                "stock price of hcl", "how is itc performing"
            ],
            "follow_up": [
                "what about its debt?", "does it pay a dividend?", "who is the ceo?", 
                "how does that compare to the industry?", "what is the pe ratio of this company?",
                "is it a buy?", "tell me more about their revenue", "what are the risks?"
            ],
            "general_query": [
                "what is a pe ratio?", "explain mutual funds", "how do interest rates affect stocks", 
                "what is an etf", "how to invest in index funds", "what does market cap mean",
                "explain the stock market", "what is inflation", "how do I save money"
            ]
        }
        
        self.flat_examples = []
        self.flat_intents = []
        for intent, phrases in self.examples.items():
            for phrase in phrases:
                self.flat_examples.append(phrase)
                self.flat_intents.append(intent)
        
        self.embeddings = []

    def route(self, query: str, stock_context) -> str:
        # Fast local checks for explicit tickers to save even the embedding call
        t = query.strip().upper()
        if t.endswith(".NS") or t.endswith(".BSE") or t.endswith(".BO"):
            return "new_report"
            
        # Cache embeddings for the examples on the very first run
        if not self.embeddings:
            print("Initializing Semantic Router Embeddings...")
            self.embeddings = get_embeddings_batch(self.flat_examples)
            
        # Embed the user's query
        try:
            query_emb = get_embeddings_batch([query])[0]
        except Exception:
            # Fallback if API fails
            return "general_query"
        
        best_intent = "general_query"
        best_score = -1
        
        # Compare the query against all examples using cosine similarity
        for intent, emb in zip(self.flat_intents, self.embeddings):
            # If the user asks a follow-up but hasn't searched a stock yet, ignore follow_up matches
            if intent == "follow_up" and not stock_context:
                continue
                
            score = cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_intent = intent
                
        print(f"Semantic Router classified '{query}' as '{best_intent}' (Confidence: {best_score:.2f})")
        
        # If the highest match is very low confidence, default to general_query
        if best_score < 0.4:
            return "general_query"
            
        return best_intent


# Instantiate the singleton router
_semantic_router = SemanticRouter()

def classify_intent(question: str, stock_context) -> str:
    """
    Classifies intent into: new_report | follow_up | general_query
    Uses the fast Semantic Router instead of slow LLM generation.
    """
    return _semantic_router.route(question, stock_context)


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

    try:
        data = fetch_detailed_stock_data(symbol)
    except ValueError as e:
        return f"Could not retrieve data for symbol '{symbol}'. Please try a valid NSE ticker (e.g., TCS.NS, RELIANCE.NS).", None

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