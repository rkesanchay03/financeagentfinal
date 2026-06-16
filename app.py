import streamlit as st
import plotly.graph_objects as go
from llm_analyzer import (
    classify_intent,
    generate_report,
    generate_follow_up_answer,
    answer_general_question,
    answer_rag_question,
)
from stock_fetcher import fetch_stock_history, resolve_symbol
from rag_processor import (
    extract_text_from_pdf,
    extract_text_from_txt,
    chunk_text,
    get_embeddings_batch,
    SimpleVectorStore,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI — Indian Market Intelligence",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME — Near-black + Teal
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:          #0b0e12;
    --surface:     #131720;
    --surface2:    #1a2030;
    --surface3:    #202a38;
    --border:      #232b38;
    --border2:     #2e3a4a;
    --text:        #e4e8ef;
    --text-muted:  #7a8694;
    --text-dim:    #3e4a58;
    --teal:        #2dd4bf;
    --teal-dim:    #0f9f8f;
    --teal-deep:   #0a7a6e;
    --teal-glow:   rgba(45,212,191,0.10);
    --teal-border: rgba(45,212,191,0.22);
    --green:       #34d399;
    --red:         #f87171;
    --blue:        #3b82f6;
    --purple:      #a855f7;
    --font-mono:   'DM Mono', 'Courier New', monospace;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
    font-size: 14px;
}

/* ── Main area ── */
[data-testid="stMain"], [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
}

/* .block-container is Streamlit's internal class for the centred content block.
   layout="centered" handles centering automatically — we just expand the max-width. */
.block-container {
    max-width: 1080px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
    width: 100% !important;
}

/* Responsive layout: adjust when sidebar is collapsed */
@media (max-width: 1400px) {
    .block-container {
        max-width: 95vw !important;
    }
}

/* ════════════════════════════════════
   SIDEBAR  — Modern & Organized
   ════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--surface) 0%, var(--surface) 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.4) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stSidebar"][aria-expanded="true"] {
    background: linear-gradient(180deg, var(--surface) 0%, var(--surface) 100%) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 240px !important;
    max-width: 280px !important;
    box-shadow: 2px 0 12px rgba(0,0,0,0.4) !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0px !important;
    max-width: 0px !important;
    border-right: none !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

/* Remove Streamlit's default top padding so content starts at the very top */
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Reset ALL text inside sidebar to one consistent size */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] label {
    font-size: 13px !important;
    line-height: 1.5 !important;
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar emoji — exempt from the 13px reset */
[data-testid="stSidebar"] .sb-icon {
    font-size: 20px !important;
    line-height: 1 !important;
    display: inline-block;
    flex-shrink: 0;
}

/* Sidebar heading (branding only) */
[data-testid="stSidebar"] .sb-brand-name {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] .sb-brand-sub {
    font-size: 10px !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
    font-weight: 500;
}
[data-testid="stSidebar"] .sb-section-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--teal) !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    margin-bottom: 10px !important;
    padding-bottom: 8px !important;
    border-bottom: 1.5px solid var(--border);
}
[data-testid="stSidebar"] .sb-section-icon {
    font-size: 14px !important;
    opacity: 0.8 !important;
}
[data-testid="stSidebar"] .sb-tip {
    font-size: 11px !important;
    color: var(--text-dim) !important;
    line-height: 1.8 !important;
}
[data-testid="stSidebar"] .sb-tip code {
    font-size: 11px !important;
    font-family: var(--font-mono) !important;
    color: var(--teal) !important;
    background: var(--surface2);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid var(--border);
}

/* Sidebar buttons — enhanced with icons and better hover */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 13px !important;
    text-align: left !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 6px !important;
    letter-spacing: 0.01em;
    line-height: 1.4 !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
}

[data-testid="stSidebar"] .stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: var(--teal-glow);
    transition: left 0.3s ease;
    z-index: -1;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--teal-glow) !important;
    border-color: var(--teal) !important;
    color: var(--teal) !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 12px rgba(45, 212, 191, 0.15) !important;
}

/* Sidebar dividers */
[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] [data-testid="stDivider"] {
    border-color: var(--border) !important;
    margin: 14px 0 !important;
    opacity: 0.7 !important;
}

/* ════════════════════════════════════
   CHAT — Enhanced with avatars & polish
   ════════════════════════════════════ */
[data-testid="stChatMessageContainer"] {
    background: transparent !important;
    gap: 12px !important;
}

[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
    transition: all 0.2s ease !important;
    position: relative !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: var(--border2) !important;
    box-shadow: 0 4px 12px rgba(45, 212, 191, 0.08) !important;
}

/* Assistant messages get a teal accent */
[data-testid="stChatMessage"][data-testid*="assistant"] {
    border-left: 3px solid var(--teal) !important;
    background: linear-gradient(135deg, var(--surface) 0%, rgba(45, 212, 191, 0.03) 100%) !important;
}

/* User messages get a subtle blue accent */
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid var(--blue) !important;
    background: linear-gradient(135deg, var(--surface) 0%, rgba(59, 130, 246, 0.03) 100%) !important;
    margin-left: 30px !important;
}

/* Chat input — rounded pill-shaped bar with enhanced styling */
[data-testid="stChatInputContainer"] {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 28px !important;
    padding-left: 18px !important;
    padding-right: 16px !important;
    max-width: 750px !important;
    margin: 0 auto !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

[data-testid="stChatInputContainer"]:focus-within {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-glow), 0 4px 12px rgba(45, 212, 191, 0.2) !important;
    background: var(--surface) !important;
}

[data-testid="stChatInputContainer"] textarea {
    border-radius: 28px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--text-dim) !important;
    font-style: italic !important;
}

/* ─────────────────────────────────────────────────────────────
   ATTACHMENT BUTTON — Premium floating design
   ───────────────────────────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    position: fixed !important;
    bottom: 50px !important;
    width: 40px !important;
    height: 40px !important;
    min-height: 40px !important;
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dim) 100%) !important;
    border: 2px solid var(--teal) !important;
    border-radius: 50% !important;
    z-index: 1000 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: left 0.3s cubic-bezier(0.4, 0, 0.2, 1), all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    overflow: hidden !important;
    box-shadow: 0 6px 20px rgba(45, 212, 191, 0.3) !important;
    
    left: calc(50% - 365px - 66px) !important;
}

/* Sidebar expanded: shift right so the button clears the sidebar */
[data-testid="stSidebar"][aria-expanded="true"] ~ .main div[data-testid="stFileUploader"] {
    left: calc(50% - 365px - 66px + 130px) !important;
}

@media (max-width: 767px) {
    div[data-testid="stFileUploader"] {
        left: 16px !important;
        bottom: 32px !important;
    }
}

/* Enhanced hover style with animation */
div[data-testid="stFileUploader"]:hover {
    background: linear-gradient(135deg, var(--teal) 0%, var(--blue) 100%) !important;
    transform: translateY(-4px) scale(1.15) !important;
    box-shadow: 0 10px 30px rgba(45, 212, 191, 0.4) !important;
}

/* Plus sign styling */
div[data-testid="stFileUploader"]::before {
    content: "+" !important;
    color: var(--bg) !important;
    font-size: 26px !important;
    font-weight: 600 !important;
    line-height: 1 !important;
    position: absolute !important;
    pointer-events: none !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stFileUploader"]:hover::before {
    color: var(--text) !important;
    transform: scale(1.2) !important;
}

/* Hide default file uploader text elements */
div[data-testid="stFileUploader"] * {
    opacity: 0 !important;
    cursor: pointer !important;
}








/* ════════════════════════════════════
   TYPOGRAPHY IN CHAT — Enhanced readability
   ════════════════════════════════════ */
[data-testid="stChatMessage"] p {
    color: #c2c8d4 !important;
    line-height: 1.9 !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatMessage"] h2 {
    color: var(--teal) !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    border-bottom: 2px solid var(--teal-border);
    padding-bottom: 8px;
    margin: 20px 0 12px !important;
    letter-spacing: -0.01em;
}

[data-testid="stChatMessage"] h3 {
    color: var(--teal) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin: 16px 0 8px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

[data-testid="stChatMessage"] strong { 
    color: var(--text) !important; 
    font-size: 1rem !important; 
    font-weight: 600 !important;
}

[data-testid="stChatMessage"] code {
    background: var(--surface2) !important;
    color: var(--teal) !important;
    border-radius: 5px;
    padding: 2px 7px;
    font-family: var(--font-mono) !important;
    font-size: 0.88em;
    border: 1px solid var(--border);
}

[data-testid="stChatMessage"] ul { 
    padding-left: 20px !important;
    margin: 10px 0 !important;
}

[data-testid="stChatMessage"] ul li { 
    color: #a4acb8 !important; 
    margin-bottom: 6px !important; 
    font-size: 0.95rem !important;
    line-height: 1.6;
}

[data-testid="stChatMessage"] table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 20px;
    font-size: 0.92rem;
}

[data-testid="stChatMessage"] table th {
    background: var(--surface2) !important;
    color: var(--teal) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 11px 15px !important;
    border: 1px solid var(--border) !important;
    text-align: left;
}

[data-testid="stChatMessage"] table td {
    padding: 10px 15px !important;
    border: 1px solid var(--border) !important;
    color: #c2c8d4 !important;
    font-size: 0.92rem !important;
    vertical-align: top;
    line-height: 1.6;
}

[data-testid="stChatMessage"] table tr:nth-child(even) td {
    background: rgba(255,255,255,0.025) !important;
}

[data-testid="stChatMessage"] table tr:hover td {
    background: var(--teal-glow) !important;
}

/* ════════════════════════════════════
   HERO — Modern & Striking
   ════════════════════════════════════ */
.hero {
    background: linear-gradient(135deg, var(--surface2) 0%, var(--surface) 100%);
    border: 1.5px solid var(--border2);
    border-left: 4px solid var(--teal);
    border-radius: 14px;
    padding: 32px 36px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(45, 212, 191, 0.08) !important;
    transition: all 0.3s ease;
}

.hero:hover {
    border-left-color: var(--blue);
    box-shadow: 0 12px 32px rgba(45, 212, 191, 0.12) !important;
}

.hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.hero-eyebrow::before {
    content: '✨';
}

.hero h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 12px 0;
    letter-spacing: -0.03em;
    line-height: 1.3;
}

.hero h1 span { 
    color: var(--teal);
    background: linear-gradient(135deg, var(--teal) 0%, var(--blue) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero p {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin: 0 0 18px 0;
    line-height: 1.7;
}

.hero-pills { 
    display: flex; 
    flex-wrap: wrap; 
    gap: 8px; 
}

.pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.75rem;
    color: var(--text-muted);
    font-weight: 500;
    letter-spacing: 0.04em;
    transition: all 0.2s ease;
    white-space: nowrap;
    text-transform: uppercase;
}

.pill:hover {
    background: var(--teal-glow);
    border-color: var(--teal);
    color: var(--teal);
}

/* ════════════════════════════════════
   METRIC CARDS — Premium Layout
   ════════════════════════════════════ */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin: 18px 0 12px;
}

.metric-card {
    background: linear-gradient(135deg, var(--surface2) 0%, rgba(45, 212, 191, 0.05) 100%);
    border: 1.5px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    transition: all 0.2s ease;
    cursor: default;
}

.metric-card:hover {
    border-color: var(--teal);
    background: linear-gradient(135deg, var(--surface2) 0%, rgba(45, 212, 191, 0.1) 100%);
    box-shadow: 0 4px 12px rgba(45, 212, 191, 0.12);
    transform: translateY(-2px);
}

.metric-card .mc-label {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

.metric-card .mc-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 7px;
    font-family: var(--font-mono);
    letter-spacing: -0.01em;
}

.mc-value.up   { color: var(--green) !important; }
.mc-value.down { color: var(--red)   !important; }
.mc-value.teal { color: var(--teal)  !important; }

/* ════════════════════════════════════
   SECTION BADGE
   ════════════════════════════════════ */
.badge {
    display: inline-block;
    background: var(--teal-glow);
    border: 1.5px solid var(--teal-border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 10px;
    color: var(--teal);
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.badge:hover {
    border-color: var(--teal);
    background: rgba(45, 212, 191, 0.15);
}

/* ════════════════════════════════════
   MISC — Polished Details
   ════════════════════════════════════ */
hr { 
    border-color: var(--border) !important; 
    margin: 14px 0 !important; 
}

.stSpinner > div { 
    border-top-color: var(--teal) !important; 
}

/* Loading animations */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

/* Scroll styling */
::-webkit-scrollbar { 
    width: 6px; 
    height: 6px;
}

::-webkit-scrollbar-track { 
    background: transparent; 
}

::-webkit-scrollbar-thumb { 
    background: var(--border2); 
    border-radius: 10px;
    transition: all 0.2s ease;
}

::-webkit-scrollbar-thumb:hover { 
    background: var(--teal); 
}

#MainMenu, footer, header { 
    visibility: hidden; 
}

/* Error message styling */
.stAlert {
    border-radius: 10px !important;
    border-left: 4px solid var(--red) !important;
}

/* Success message styling */
.stSuccess {
    border-radius: 10px !important;
    border-left: 4px solid var(--green) !important;
}

/* Info message styling */
.stInfo {
    border-radius: 10px !important;
    border-left: 4px solid var(--teal) !important;
}
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Welcome to FinSight AI!**\n\nType any stock name or symbol — `TCS`, `Reliance`, `HDFC Bank`, `Infosys` — and I'll generate a full structured report with charts.",
            "chart": None,
            "metrics": None,
        }
    ]
if "stock_context" not in st.session_state:
    st.session_state.stock_context = None
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = None
if "rag_store" not in st.session_state:
    st.session_state.rag_store = None
if "rag_filename" not in st.session_state:
    st.session_state.rag_filename = None
if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = 0


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def add_to_history(symbol: str):
    if symbol not in st.session_state.search_history:
        st.session_state.search_history.insert(0, symbol)
        st.session_state.search_history = st.session_state.search_history[:6]


def fmt_num(n):
    if n is None: return "N/A"
    if n >= 1e12: return f"₹{n/1e12:.2f}T"
    if n >= 1e9:  return f"₹{n/1e9:.2f}B"
    if n >= 1e7:  return f"₹{n/1e7:.2f}Cr"
    return f"₹{n:,.2f}"


def fmt_pct(n):
    if n is None: return "N/A"
    return f"{n*100:.2f}%"


def fmt_ratio(n, suffix="x"):
    if n is None: return "N/A"
    return f"{n:.2f}{suffix}"


def build_metrics_html(data: dict) -> str:
    q  = data.get("quote", {})
    fr = data.get("financialRatios", {})
    ar = data.get("analystRatings", {})
    pr = data.get("profile", {})

    cp = q.get("currentPrice")
    pc = q.get("previousClose")

    chg_cls, chg_str = "", "N/A"
    if cp and pc:
        chg = cp - pc
        pct = (chg / pc) * 100
        chg_cls = "up" if chg >= 0 else "down"
        arrow   = "▲" if chg >= 0 else "▼"
        chg_str = f"{arrow} {abs(chg):.2f} ({abs(pct):.2f}%)"

    rec = ar.get("recommendation", "N/A")
    rec_cls = "teal" if rec in ("BUY", "STRONG_BUY") else ("down" if rec in ("SELL", "STRONG_SELL") else "")

    items = [
        ("Price",      f"₹{cp:,.2f}" if cp else "N/A", ""),
        ("Change",     chg_str,                         chg_cls),
        ("52W High",   f"₹{q.get('fiftyTwoWeekHigh'):,.2f}" if q.get("fiftyTwoWeekHigh") else "N/A", "up"),
        ("52W Low",    f"₹{q.get('fiftyTwoWeekLow'):,.2f}"  if q.get("fiftyTwoWeekLow")  else "N/A", "down"),
        ("P/E",        fmt_ratio(fr.get("peRatio")),     ""),
        ("Fwd P/E",    fmt_ratio(fr.get("forwardPeRatio")), ""),
        ("ROE",        fmt_pct(fr.get("returnOnEquity")), ""),
        ("Mkt Cap",    fmt_num(pr.get("marketCap")),     ""),
        ("Analyst",    rec,                              rec_cls),
        ("Target",     fmt_num(ar.get("targetMeanPrice")), "teal"),
    ]
    cards = "".join(
        f'<div class="metric-card">'
        f'<div class="mc-label">{lbl}</div>'
        f'<div class="mc-value {cls}">{val}</div>'
        f'</div>'
        for lbl, val, cls in items
    )
    return f'<div class="metric-row">{cards}</div>'


def create_candlestick_chart(symbol: str, period: str = "6mo"):
    df = fetch_stock_history(symbol, period=period)
    if df is None or df.empty:
        return None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"], name=symbol,
        increasing_line_color="#34d399", decreasing_line_color="#f87171",
        increasing_fillcolor="#34d399",  decreasing_fillcolor="#f87171",
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA20"], name="SMA 20",
        line=dict(color="#2dd4bf", width=1.3, dash="dot"), opacity=0.9
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA50"], name="SMA 50",
        line=dict(color="#0f9f8f", width=1.3, dash="dash"), opacity=0.8
    ))
    vol_colors = ["#34d399" if c >= o else "#f87171"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.25, yaxis="y2"
    ))

    axis_style = dict(
        gridcolor="rgba(255,255,255,0.04)",
        showgrid=True,
        linecolor="#232b38",
        tickfont=dict(color="#3e4a58", size=10, family="DM Mono"),
        color="#3e4a58",
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b>  ·  Price History ({period})",
            font=dict(color="#7a8694", size=12, family="Inter")
        ),
        xaxis=dict(**axis_style, rangeslider=dict(visible=False)),
        yaxis=dict(**axis_style, title="Price (₹)",
                   title_font=dict(color="#3e4a58", size=10)),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color="#3e4a58", size=9),
                    title="Vol", title_font=dict(color="#3e4a58", size=10)),
        legend=dict(
            bgcolor="rgba(19,23,32,0.9)", bordercolor="#232b38",
            borderwidth=1, font=dict(color="#7a8694", size=11, family="Inter")
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(19,23,32,0.97)",
        margin=dict(l=8, r=8, t=38, b=8),
        height=410,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(19,23,32,0.97)", bordercolor="#2e3a4a",
            font=dict(color="#e4e8ef", size=11, family="Inter")
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR  — Modern & Organized
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 12px 16px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="sb-icon">📈</span>
            <div>
                <div class="sb-brand-name">FinSight <span style="color:#2dd4bf;">AI</span></div>
                <div class="sb-brand-sub">Market Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Recent Searches Section
    st.markdown('<div class="sb-section-label"><span class="sb-section-icon">🕐</span>Recent Searches</div>', unsafe_allow_html=True)
    if not st.session_state.search_history:
        st.markdown('<p class="sb-tip" style="padding:4px 0;opacity:0.7;">No searches yet. Start typing!</p>', unsafe_allow_html=True)
    else:
        for sym in st.session_state.search_history:
            if st.button(sym, key=f"hist_{sym}", use_container_width=True):
                st.session_state.trigger_search = sym
                st.rerun()

    st.divider()

    # Quick Look Section
    st.markdown('<div class="sb-section-label"><span class="sb-section-icon">⭐</span>Quick Look</div>', unsafe_allow_html=True)
    for q in ["TCS", "Reliance", "HDFC Bank", "Infosys", "Wipro", "SBI"]:
        if st.button(q, key=f"quick_{q}", use_container_width=True):
            st.session_state.trigger_search = q
            st.rerun()

    st.divider()

    # Help & Info Section
    st.markdown("""
    <div style="padding:0 2px;">
        <div class="sb-section-label"><span class="sb-section-icon">❓</span>How to Use</div>
        <div class="sb-tip" style="margin-bottom:12px;">
            <b style="color:#a4acb8;">Examples:</b><br>
            • <code>TCS</code> or <code>TCS.NS</code><br>
            • Tell me about Reliance<br>
            • What is a P/E ratio?<br>
            • Compare TCS and Infosys
        </div>
        <div style="padding:10px 12px;background:var(--surface2);border:1px solid var(--border);border-radius:8px;margin-bottom:12px;">
            <div style="font-size:10px;color:var(--teal);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;font-weight:600;">⚠️ Disclaimer</div>
            <div style="font-size:11px;color:var(--text-dim);line-height:1.6;">AI-generated insights. Not financial advice. Verify with professionals.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HERO BANNER — Enhanced Design
# ─────────────────────────────────────────────────────────────
if not st.session_state.rag_store:
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">NSE &nbsp;·&nbsp; BSE &nbsp;·&nbsp; AI-POWERED RESEARCH</div>
        <h1>Indian Equity <span>Intelligence</span>, Reimagined</h1>
        <p>Ask for any stock by name — no tickers required. Get structured reports, live charts, key ratios, and AI-driven insights powered by Gemini LLM.</p>
        <div class="hero-pills">
            <span class="pill">📊 Structured Reports</span>
            <span class="pill">📈 Live Charts + SMA</span>
            <span class="pill">📋 Key Ratios</span>
            <span class="pill">📰 News Digest</span>
            <span class="pill">🎯 Analyst Targets</span>
            <span class="pill">✨ AI Synthesis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">DOCUMENT RAG &nbsp;·&nbsp; NATIVE EMBEDDINGS &nbsp;·&nbsp; AI ANALYSIS</div>
        <h1>Financial <span>Document</span> Analyst</h1>
        <p>FinSight AI has processed and indexed your document using Google Gemini embeddings. Ask any question about your document below and get instant AI-powered answers.</p>
        <div class="hero-pills">
            <span class="pill">📄 PDF & TXT Parsing</span>
            <span class="pill">🔗 Smart Chunks</span>
            <span class="pill">🧠 Gemini Embeddings</span>
            <span class="pill">🔍 Similarity Search</span>
            <span class="pill">💬 Context-Aware Q&A</span>
            <span class="pill">⚡ Zero Framework Overhead</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RENDER CHAT HISTORY
# ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metrics"):
            st.markdown(msg["metrics"], unsafe_allow_html=True)
        if msg.get("chart"):
            st.plotly_chart(msg["chart"], use_container_width=True)


# ─────────────────────────────────────────────────────────────
# RAG ATTACHMENT WIDGET (Pinned to chat area)
# ─────────────────────────────────────────────────────────────
st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)

# Visual status bar if a document is loaded
if st.session_state.rag_store:
    col_status, col_clear = st.columns([8.5, 1.5], gap="small")
    with col_status:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 15px; background: linear-gradient(135deg, var(--teal-glow) 0%, rgba(45,212,191,0.08) 100%); border: 1.5px solid var(--teal-border); border-radius: 24px; box-shadow: 0 4px 12px rgba(45,212,191,0.1);">
            <div style="display: flex; align-items: center; gap: 10px; flex: 1;">
                <span style="font-size:16px;animation:pulse 2s infinite;">📎</span>
                <div>
                    <span style="font-size: 13px; color: var(--text); font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 350px; display: block;">{st.session_state.rag_filename}</span>
                    <span style="font-size: 11px; color: var(--text-muted);">{st.session_state.rag_chunks} chunks indexed</span>
                </div>
            </div>
            <span style="font-size: 10px; color: var(--teal); background: rgba(45,212,191,0.2); padding: 4px 10px; border-radius: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; margin-left: 10px;">RAG Enabled</span>
        </div>
        """, unsafe_allow_html=True)
    with col_clear:
        if st.button("🗑️ Clear", key="clear_chat_rag", use_container_width=True):
            st.session_state.rag_store = None
            st.session_state.rag_filename = None
            st.session_state.rag_chunks = 0
            st.rerun()

# Circular "+" button attachment box — absolutely positioned inside left chat bar margin
uploaded_file = st.file_uploader(
    "Attach Financial PDF or TXT...",
    type=["pdf", "txt"],
    key="chat_rag_attachment",
    label_visibility="collapsed"
)

# Process the attachment
if uploaded_file is not None:
    if st.session_state.rag_filename != uploaded_file.name:
        try:
            with st.spinner("Processing..."):
                text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_txt(uploaded_file)
                if not text.strip(): 
                    st.error("Empty file.")
                else:
                    chunks = chunk_text(text)
                    store = SimpleVectorStore()
                    store.add_documents(chunks, get_embeddings_batch(chunks))
                    st.session_state.rag_store = store
                    st.session_state.rag_filename = uploaded_file.name
                    st.session_state.rag_chunks = len(chunks)
                    st.rerun()
        except Exception as e: 
            st.error(f"Error: {e}")


input_placeholder = f"💡 Ask about {st.session_state.rag_filename}…" if st.session_state.rag_store else "💬 Ask about a stock, trend, or financial concept…"
prompt = st.chat_input(input_placeholder)

if st.session_state.trigger_search:
    prompt = st.session_state.trigger_search
    st.session_state.trigger_search = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "chart": None, "metrics": None})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                response, chart, metrics_html = "", None, None

                if st.session_state.rag_store:
                    # RAG Mode Active: Search in uploaded document
                    results = st.session_state.rag_store.similarity_search(prompt, top_k=3)
                    if not results:
                        response = "I couldn't retrieve any relevant information from the document to answer your question."
                    else:
                        chunks = [text for text, score in results]
                        response = answer_rag_question(prompt, st.session_state.rag_filename, chunks)
                else:
                    # Stock Intelligence Mode Active
                    intent = classify_intent(prompt, st.session_state.stock_context)

                    if intent == "new_report":
                        symbol = resolve_symbol(prompt)
                        add_to_history(symbol)
                        response, data = generate_report(prompt)
                        if data and "error" not in data:
                            st.session_state.stock_context = data
                            metrics_html = build_metrics_html(data)
                            chart = create_candlestick_chart(symbol)
                        else:
                            st.session_state.stock_context = None

                    elif intent == "follow_up" and st.session_state.stock_context:
                        response = generate_follow_up_answer(prompt, st.session_state.stock_context)

                    else:
                        response = answer_general_question(prompt)

                st.markdown(response)
                if metrics_html:
                    st.markdown(metrics_html, unsafe_allow_html=True)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "chart": chart,
                    "metrics": metrics_html,
                })

            except Exception as e:
                err = f"⚠️ Unexpected error: `{e}`"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant", "content": err, "chart": None, "metrics": None
                })





