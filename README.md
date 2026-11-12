# FinSight AI — Autonomous Equity & Document Intelligence Agent

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://finsightaii.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/rkesanchay03/financeagentfinal)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Flash%20%26%20Embeddings-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)

> **FinSight AI** is a dual-pipeline financial research assistant that bridges structured equity telemetry (NSE/BSE quotes, valuation ratios, consensus ratings) and unstructured qualitative corporate disclosures (annual filings, earnings transcripts) into a unified, hallucination-resistant analytical workspace.

---

## 📌 Problem Statement

Financial analysts and retail investors constantly face **"Tab Fatigue" & Data Fragmentation**:
1. **Structured Data:** Stored in tabular silos (Yahoo Finance, Moneycontrol, broker terminals) for real-time price action, P/E multiples, and momentum metrics.
2. **Unstructured Data:** Buried in static 100-to-500 page corporate PDF filings for qualitative risk factors, EBITDA disclosures, and management commentary.

Traditional AI agents fail in finance because general LLMs hallucinate financial metrics and introduce high latency by passing trivial ticker queries through expensive generative pipelines. 

**FinSight AI solves this through architectural separation**:
- An **on-demand telemetry pipeline** that injects deterministic JSON data into grounded prompts.
- A **zero-dependency native RAG pipeline** to query corporate disclosures via local semantic vector search.
- An **embedding-based Semantic Router** that classifies intent locally, bypassing LLM text generation for raw ticker queries.

---

## 🏛 System Architecture & End-to-End Data Flow

```
                                  ┌────────────────────────┐
                                  │   User Interface (UI)  │
                                  │         app.py         │
                                  └───────────┬────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                   │
           [Document Uploaded]                                 [No Document Active]
                    │                                                   │
                    ▼                                                   ▼
       ┌─────────────────────────┐                         ┌─────────────────────────┐
       │   Document RAG Engine   │                         │  Semantic Intent Router │
       │     rag_processor.py    │                         │     llm_analyzer.py     │
       └────────────┬────────────┘                         └────────────┬────────────┘
                    │                                                   │
      ┌─────────────┴─────────────┐                 ┌───────────────────┼───────────────────┐
      ▼                           ▼                 ▼                   ▼                   ▼
┌───────────┐               ┌───────────┐    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ PyPDF     │               │ Cosine    │    │"new_report" │     │ "follow_up" │     │"general_..."│
│ Sliding   │               │ Nearest   │    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
│ Window    │               │ Neighbor  │           │                   │                   │
│ Chunker   │               │ (top-k=3) │           ▼                   ▼                   │
└─────┬─────┘               └─────┬─────┘    ┌─────────────┐     ┌─────────────┐            │
      │                           │          │yfinance     │     │Read context │            │
      ▼                           │          │Telemetry    │     │from RAM     │            │
┌───────────┐                     │          │Fetch & Cache│     │session_state│            │
│Gemini     │                     │          └──────┬──────┘     └──────┬──────┘            │
│Embedding  │                     │                 │                   │                   │
│Batch API  │                     │                 └─────────────┬─────┘                   │
└─────┬─────┘                     │                               │                         │
      │                           │                               ▼                         ▼
      ▼                           │                 ┌─────────────────────────────────────────────┐
┌───────────────────┐             │                 │        Google Gemini REST API Gateway       │
│ SimpleVectorStore │             │                 │   (with automatic HTTP 503/429 fallback)    │
│  (Session RAM)    │◄────────────┘                 └──────────────────────┬──────────────────────┘
└───────────────────┘                                                      │
                                                                           ▼
                                                            ┌─────────────────────────────┐
                                                            │ Markdown Report + Ratios    │
                                                            │ Interactive Plotly Chart    │
                                                            └─────────────────────────────┘
```

---

## ⚡ Key Engineering Features

### 1. Dual-Layer Semantic Intent Router
- **Ticker Suffix Fast-Path:** Instant heuristic match for exchange identifiers (`.NS`, `.BSE`, `.BO`) bypassing all API calls in $< 0.01\text{ ms}$.
- **Vector Space Classification:** Embeds free-form natural language queries and computes cosine similarity against a cached cluster of 36 intent anchors (`new_report`, `follow_up`, `general_query`).
- **Confidence Gating:** Enforces a similarity threshold of $0.55$. Queries with low confidence or empty context fall through safely to educational general finance QA.

### 2. Zero-Hallucination Stock Synthesis
- **Deterministic Ingestion:** Extracts comprehensive multi-point telemetry from `yfinance` across Company Profile, Real-Time Quotes, Financial Ratios (Trailing/Forward P/E, PEG, Price-to-Book, ROE, Debt-to-Equity), Analyst Ratings, and Live News.
- **Strict Prompt Grounding:** Serializes live metrics into structured JSON and passes them inside triple-backtick markdown blocks to Google Gemini with an explicit mandate: *"Use the following live data exclusively (do not invent numbers)"*.
- **Stateful Follow-ups:** Caches the retrieved stock JSON in `st.session_state.stock_context`. Follow-up queries (e.g., *"What is its debt-to-equity ratio?"*) re-use the in-memory context directly, eliminating redundant network calls.

### 3. Lightweight In-Memory RAG Engine (`SimpleVectorStore`)
- **Sentence-Aware Sliding-Window Chunker:** Slices raw PDF text into overlapping windows ($1,000\text{ characters}$ with $200\text{ characters}$ overlap) using reverse lookahead for sentence boundaries (`. `, `? `, `! `, `\n`) to preserve semantic continuity.
- **Native Vector Store:** Implemented in pure Python without external vector database dependencies (Pinecone/Milvus/ChromaDB).
- **Exact Dot-Product Cosine Similarity:**
  $$\text{Cosine Similarity}(u, v) = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \cdot \sqrt{\sum_{i=1}^{d} v_i^2}}$$
- **Top-$K$ Grounded Retrieval ($k=3$):** Retrieves the 3 highest-ranking chunks and injects them into a constrained system prompt with a strict negative constraint (*"If the answer cannot be found in the extracts, respond that the document lacks sufficient information"*).

### 4. Enterprise Fault Tolerance & Resilience
- **Automated Model Circuit-Breaking:** The centralized Gemini API client automatically catches HTTP `503 Service Unavailable` (demand spikes) and HTTP `429 Too Many Requests` (quota rate limits), seamlessly redirecting payloads to a fallback model (`gemini-1.5-flash`).
- **Cache Pollution Prevention:** In-memory caching (`@st.cache_data(ttl=600)`) raises structured `ValueError` on upstream network failures rather than returning error dictionaries, preventing failed states from polluting the cache.

---

## 📂 Repository Structure

```bash
financeagentfinal/
├── app.py              # Presentation layer: Streamlit UI, responsive styles, state orchestrator
├── stock_fetcher.py    # Data layer: Ticker normalization, yfinance telemetry fetcher & LRU cache
├── llm_analyzer.py     # AI & Intelligence layer: SemanticRouter, prompt synthesis, API fallback
├── rag_processor.py    # Vector search layer: PyPDF extractor, sliding chunker, SimpleVectorStore
├── benchmark.py        # Offline profiling suite: Router latency, chunking throughput, vector scan
├── requirements.txt    # Dependency definitions
├── .env.template       # Environment variable specifications
└── README.md           # Project documentation & technical specifications
```

---

## 🔬 Tech Stack & Architectural Decisions

| Layer | Technology | Decision Rationale |
| :--- | :--- | :--- |
| **Frontend / Orchestrator** | `Streamlit` | Full-stack reactive Python framework eliminating frontend/backend separation overhead for rapid prototyping; handles state via isolated session state dictionaries. |
| **Data Ingestion** | `yfinance` | Direct access to consolidated multi-metric NSE/BSE equity telemetry, historical OHLCV data, and current market catalysts without paid subscription barriers. |
| **LLM Inference** | `Google Gemini 2.5 Flash` | Sub-second generative token latency, large context window support, and high accuracy for complex markdown table synthesis. |
| **Embeddings** | `gemini-embedding-2` | 768-dimensional dense semantic vectors optimized for semantic search and classification. |
| **Document Processing** | `pypdf` | Zero-dependency, lightweight in-memory PDF stream parser executing without native C-binding requirements. |
| **Visualization** | `Plotly Graph Objects` | Hardware-accelerated client-side interactive financial charting (candlesticks, moving averages, volume). |

---

## ⚙️ Local Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Google Gemini API Key ([Get one at Google AI Studio](https://aistudio.google.com/))

### 1. Clone the Repository
```bash
git clone https://github.com/rkesanchay03/financeagentfinal.git
cd financeagentfinal
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🧪 Profiling & Benchmarking

The repository includes an offline micro-benchmarking suite to profile core sub-algorithms without running the UI:

```bash
python benchmark.py
```

This runs three tests:
1. **Intent Classification Throughput:** Profiles the heuristic ticker parsing rules against thousands of synthetic queries.
2. **Chunking Performance:** Measures sliding-window boundary lookahead throughput over large simulated text corpora ($1.5\text{M}+$ characters).
3. **In-Memory Vector Search Latency:** Times exhaustive dot-product cosine similarity scoring and top-$k$ sorting across $10,000$ 768-dimensional float embeddings in RAM.

---

## 🚀 Production Scaling Roadmap (V1 $\rightarrow$ V2)

- [ ] **Decoupled Client-Server:** Replace Streamlit monolith with **FastAPI** ASGI backend and **Next.js** frontend for asynchronous, non-blocking I/O.
- [ ] **Distributed Cache:** Migrate in-memory `@st.cache_data` to a distributed **Redis cluster** to share normalized stock contexts across auto-scaled worker nodes.
- [ ] **Vector Index Optimization:** Upgrade `SimpleVectorStore` to **PostgreSQL (`pgvector`)** with **HNSW** (Hierarchical Navigable Small World) indexing for $O(\log N)$ approximate nearest neighbor search.
- [ ] **Automated Offline Evals:** Integrate **Ragas** to track RAG Triad metrics (*Faithfulness*, *Answer Relevance*, *Context Precision*) across regression runs.

---

## 📜 License & Disclaimer

Distributed under the MIT License. See `LICENSE` for more information.

*Disclaimer: FinSight AI is an educational project and AI-driven market intelligence demonstration. All generated reports, valuations, and analyses are for informational purposes only and do not constitute certified financial or investment advice.*
