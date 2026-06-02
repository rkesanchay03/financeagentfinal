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
}

/* ════════════════════════════════════
   SIDEBAR  — fully normalised
   ════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 230px !important;
    max-width: 260px !important;
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
    font-size: 15px !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}
[data-testid="stSidebar"] .sb-brand-sub {
    font-size: 10px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim) !important;
}
[data-testid="stSidebar"] .sb-section-label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--teal-dim) !important;
    display: block;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}
[data-testid="stSidebar"] .sb-tip {
    font-size: 11px !important;
    color: var(--text-dim) !important;
    line-height: 1.7 !important;
}
[data-testid="stSidebar"] .sb-tip code {
    font-size: 11px !important;
    font-family: var(--font-mono) !important;
    color: var(--teal) !important;
    background: var(--surface2);
    padding: 0 4px;
    border-radius: 3px;
}

/* Sidebar buttons — uniform size */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--text-muted) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 7px 11px !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    margin-bottom: 3px !important;
    letter-spacing: 0.01em;
    line-height: 1 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--teal-glow) !important;
    border-color: var(--teal-deep) !important;
    color: var(--teal) !important;
    transform: translateX(2px);
}

/* Sidebar dividers */
[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] [data-testid="stDivider"] {
    border-color: var(--border) !important;
    margin: 10px 0 !important;
}

/* ════════════════════════════════════
   CHAT
   ════════════════════════════════════ */
[data-testid="stChatMessageContainer"] {
    background: transparent !important;
}
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.35) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stChatMessage"]:hover {
    border-color: var(--border2) !important;
}

/* Chat input — rounded pill-shaped bar matching screenshot */
[data-testid="stChatInputContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 26px !important;
    padding-left: 16px !important; /* normal, clean padding since "+" is outside */
    max-width: 730px !important; /* Locks width to match Streamlit centered layout */
    margin: 0 auto !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: var(--teal-dim) !important;
    box-shadow: 0 0 0 2px var(--teal-glow) !important;
}
[data-testid="stChatInputContainer"] textarea {
    border-radius: 26px !important;
}

/* ─────────────────────────────────────────────────────────────
   NATIVE "+" ATTACHMENT DESIGN & TRANSITIONS (PREMIUM OUTSIDE BUTTON)
   ───────────────────────────────────────────────────────────── */
div[data-testid="stFileUploader"] {
    position: fixed !important;
    bottom: 50px !important; /* perfectly aligned vertically with the chat input container */
    width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    background-color: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50% !important; /* circular button UI */
    z-index: 1000 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    overflow: hidden !important;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.35) !important;
    
    /* Default position (sidebar collapsed or absent): exactly 12px to the left of the centered 730px chat bar */
    left: calc(50% - 365px + 100px) !important; /* 365px (half of 730px) + 32px (button width) + 12px (gap) = 44px offset */
}

/* Sidebar expanded: shift right by exactly 122px (half of 244px sidebar width) */
[data-testid="stSidebar"][data-collapsed="false"] ~ .main div[data-testid="stFileUploader"],
[data-testid="stSidebar"]:not([data-collapsed="true"]) ~ .main div[data-testid="stFileUploader"] {
    left: calc(50% - 365px - 44px + 122px) !important;
}

/* Sidebar collapsed: position at collapsed coordinate */
[data-testid="stSidebar"][data-collapsed="true"] ~ .main div[data-testid="stFileUploader"] {
    left: calc(50% - 365px - 44px) !important;
}

@media (max-width: 767px) {
    div[data-testid="stFileUploader"] {
        left: 12px !important;
        bottom: 32px !important;
    }
}

/* Hover style: elegant teal border glow, scale, and background change */
div[data-testid="stFileUploader"]:hover {
    background-color: var(--teal-glow) !important;
    border-color: var(--teal) !important;
    transform: translateY(-1px) scale(1.08) !important;
    box-shadow: 0 0 10px rgba(45, 212, 191, 0.25) !important;
}

/* Plus sign styling */
div[data-testid="stFileUploader"]::before {
    content: "+" !important;
    color: var(--text-muted) !important;
    font-size: 22px !important;
    font-weight: 400 !important;
    line-height: 1 !important;
    position: absolute !important;
    pointer-events: none !important;
    margin-top: -2px; /* vertical tuning */
    transition: color 0.2s ease !important;
}

div[data-testid="stFileUploader"]:hover::before {
    color: var(--teal) !important; /* glowing teal on hover */
}

/* Hide default file uploader text elements */
div[data-testid="stFileUploader"] * {
    opacity: 0 !important;
    cursor: pointer !important;
}








/* ════════════════════════════════════
   TYPOGRAPHY IN CHAT
   ════════════════════════════════════ */
[data-testid="stChatMessage"] p {
    color: #c2c8d4 !important;
    line-height: 1.8 !important;
    font-size: 1rem !important;
}
[data-testid="stChatMessage"] h2 {
    color: var(--text) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 18px 0 10px !important;
}
[data-testid="stChatMessage"] h3 {
    color: var(--teal) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    margin: 14px 0 6px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stChatMessage"] strong { color: var(--text) !important; font-size: 1rem !important; }
[data-testid="stChatMessage"] code {
    background: var(--surface2) !important;
    color: var(--teal) !important;
    border-radius: 4px;
    padding: 1px 5px;
    font-family: var(--font-mono) !important;
    font-size: 0.9em;
    border: 1px solid var(--border);
}
[data-testid="stChatMessage"] ul { padding-left: 18px !important; }
[data-testid="stChatMessage"] ul li { color: #8d97a5 !important; margin-bottom: 4px !important; font-size: 0.97rem !important; }

[data-testid="stChatMessage"] table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 18px;
    font-size: 0.95rem;
}
[data-testid="stChatMessage"] table th {
    background: var(--surface2) !important;
    color: var(--teal) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 9px 14px !important;
    border: 1px solid var(--border) !important;
    text-align: left;
}
[data-testid="stChatMessage"] table td {
    padding: 8px 14px !important;
    border: 1px solid var(--border) !important;
    color: #c2c8d4 !important;
    font-size: 0.93rem !important;
    vertical-align: top;
    line-height: 1.55;
}
[data-testid="stChatMessage"] table tr:nth-child(even) td {
    background: rgba(255,255,255,0.02) !important;
}
[data-testid="stChatMessage"] table tr:hover td {
    background: var(--teal-glow) !important;
}

/* ════════════════════════════════════
   HERO
   ════════════════════════════════════ */
.hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.hero-eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--teal-dim);
    margin-bottom: 8px;
}
.hero h1 {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 7px 0;
    letter-spacing: -0.025em;
    line-height: 1.25;
}
.hero h1 span { color: var(--teal); }
.hero p {
    color: var(--text-muted);
    font-size: 0.88rem;
    margin: 0 0 16px 0;
    line-height: 1.6;
}
.hero-pills { display: flex; flex-wrap: wrap; gap: 7px; }
.pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* ════════════════════════════════════
   METRIC CARDS
   ════════════════════════════════════ */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 8px;
    margin: 14px 0 6px;
}
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px 13px;
}
.metric-card .mc-label {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 600;
}
.metric-card .mc-value {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 5px;
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
    border: 1px solid var(--teal-border);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    color: var(--teal-dim);
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ════════════════════════════════════
   MISC
   ════════════════════════════════════ */
hr { border-color: var(--border) !important; margin: 12px 0 !important; }
.stSpinner > div { border-top-color: var(--teal) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal-deep); }
#MainMenu, footer, header { visibility: hidden; }
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
# SIDEBAR  — clean, uniform
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:18px 12px 12px;">
        <div style="display:flex;align-items:center;gap:9px;">
            <span class="sb-icon">📈</span>
            <div>
                <div class="sb-brand-name">FinSight <span style="color:#2dd4bf;">AI</span></div>
                <div class="sb-brand-sub">Market Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<span class="sb-section-label">Recent Searches</span>', unsafe_allow_html=True)
    if not st.session_state.search_history:
        st.markdown('<p class="sb-tip" style="padding:2px 0;">No searches yet.</p>', unsafe_allow_html=True)
    else:
        for sym in st.session_state.search_history:
            if st.button(sym, key=f"hist_{sym}", use_container_width=True):
                st.session_state.trigger_search = sym
                st.rerun()

    st.divider()

    st.markdown('<span class="sb-section-label">Quick Look</span>', unsafe_allow_html=True)
    for q in ["TCS", "Reliance", "HDFC Bank", "Infosys", "Wipro", "SBI"]:
        if st.button(q, key=f"quick_{q}", use_container_width=True):
            st.session_state.trigger_search = q
            st.rerun()

    st.divider()

    st.markdown("""
    <div class="sb-tip">
        <b style="color:#7a8694;">How to use</b><br>
        → <code>TCS</code> or <code>TCS.NS</code><br>
        → Tell me about Reliance<br>
        → What is a P/E ratio?<br>
        → Compare TCS and Infosys
    </div>
    <div style="margin-top:14px;padding:9px 10px;background:#131720;border:1px solid #232b38;border-radius:7px;">
        <div style="font-size:10px;color:#3e4a58;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:3px;">Disclaimer</div>
        <div style="font-size:11px;color:#3e4a58;">AI-generated. Not financial advice.</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────
if not st.session_state.rag_store:
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">NSE &nbsp;·&nbsp; BSE &nbsp;·&nbsp; AI-Powered Research</div>
        <h1>Indian Equity <span>Intelligence</span>, Reimagined</h1>
        <p>Ask for any stock by name — no tickers required. Get structured reports, live charts, key ratios, and AI-driven insights.</p>
        <div class="hero-pills">
            <span class="pill">Structured Reports</span>
            <span class="pill">Live Charts + SMA</span>
            <span class="pill">Key Ratios Table</span>
            <span class="pill">Latest News</span>
            <span class="pill">Analyst Targets</span>
            <span class="pill">AI Synthesis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">DOCUMENT RAG &nbsp;·&nbsp; NATIVE EMBEDDINGS &nbsp;·&nbsp; AI ANALYSIS</div>
        <h1>Financial <span>Document</span> RAG Analyser</h1>
        <p>FinSight AI has chunked and indexed your attached file using Google Gemini embeddings. Ask questions directly in the chat input below.</p>
        <div class="hero-pills">
            <span class="pill">PDF & TXT Parsing</span>
            <span class="pill">Smart Overlapping Chunks</span>
            <span class="pill">Gemini Embedding Models</span>
            <span class="pill">Cosine Similarity Index</span>
            <span class="pill">Context-Augmented Generation</span>
            <span class="pill">Zero Framework Overhead</span>
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
    col_status, col_clear = st.columns([8.2, 1.8], gap="small")
    with col_status:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 12px; background: var(--teal-glow); border: 1px solid var(--teal-border); border-radius: 20px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size:14px;">📎</span>
                <span style="font-size: 13px; color: var(--text); font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 400px;">{st.session_state.rag_filename}</span>
                <span style="font-size: 10px; color: var(--teal); background: rgba(45,212,191,0.15); padding: 2px 7px; border-radius: 12px; font-family: var(--font-mono);">{st.session_state.rag_chunks} chunks</span>
            </div>
            <span style="font-size: 10px; color: var(--teal); font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; padding-right: 5px;">RAG Enabled</span>
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


input_placeholder = f"Ask about {st.session_state.rag_filename}…" if st.session_state.rag_store else "Ask about a stock or market trend…"
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





