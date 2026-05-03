"""
vOiD.py — Voices of the Void | Alpen Signal Observatorium
Master Frontend — All Systems Wired
Streamlit Cloud Entry Point

"Something has been trying to make contact for a very long time.
 You are the only one listening."

Station: ASO-DUNKELTALER-01
Operator: Dr. Kel
Location: Dunkeltaler Forest, Swiss Alps

Navigation routes to all 6 backend modules:
  1. signal_engine.py       — Signal Acquisition & Processing
  2. spectral_analyzer.py   — Deep Spectral Analysis
  3. anomaly_detector.py    — Anomaly Detection & Threat Assessment
  4. ml_predictor.py        — ML Prediction Engine
  5. environment_monitor.py — Observatory Environment & Systems
  6. hq_reporter.py         — HQ Reporting & Data Submission
"""

import time
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Voices of the Void — ASO-Dunkeltaler",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES — VotV terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
/* ── Google Font Import ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Root variables ── */
:root {
    --void-bg:      #030d07;
    --void-bg2:     #050f08;
    --void-fg:      #00ff88;
    --void-fg-dim:  #004422;
    --void-accent:  #00ffcc;
    --void-warn:    #ff8800;
    --void-danger:  #ff2222;
    --void-grid:    #001a0e;
    --void-axis:    #335544;
    --void-gold:    #ffcc00;
    --void-blue:    #0088ff;
    --void-purple:  #8844ff;
    --void-sidebar: #020c05;
    --void-border:  #00ff4430;
    --font-mono:    'Share Tech Mono', 'Courier New', monospace;
}

/* ── Global body ── */
html, body, .stApp {
    background-color: var(--void-bg) !important;
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--void-bg); }
::-webkit-scrollbar-thumb { background: var(--void-fg-dim); border-radius: 1px; }
::-webkit-scrollbar-thumb:hover { background: var(--void-fg); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--void-sidebar) !important;
    border-right: 1px solid var(--void-border) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--font-mono) !important;
    color: var(--void-fg) !important;
}

/* ── Sidebar selectbox ── */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--void-bg2) !important;
    border: 1px solid var(--void-border) !important;
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
    border-radius: 2px !important;
}

/* ── Main area headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-mono) !important;
    color: var(--void-fg) !important;
    letter-spacing: 0.08em;
}

/* ── Metric boxes ── */
[data-testid="stMetricValue"] {
    color: var(--void-accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 1.05rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em;
}
[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--void-bg2) !important;
    color: var(--void-fg) !important;
    border: 1px solid var(--void-fg-dim) !important;
    border-radius: 2px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    transition: border-color 0.15s, color 0.15s, background 0.15s;
    padding: 0.35rem 0.6rem !important;
}
.stButton > button:hover {
    background: var(--void-fg-dim) !important;
    border-color: var(--void-fg) !important;
    color: var(--void-accent) !important;
}
.stButton > button:active {
    background: var(--void-fg) !important;
    color: var(--void-bg) !important;
}

/* ── Primary button (large acquire) ── */
.stButton > button[kind="primary"] {
    border-color: var(--void-fg) !important;
    background: #001a0e !important;
}

/* ── Sliders ── */
.stSlider > div > div > div > div {
    background: var(--void-fg) !important;
}
.stSlider [data-baseweb="slider"] {
    background: var(--void-bg2) !important;
}
.stSlider [aria-label] {
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
}

/* ── Number inputs ── */
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: var(--void-bg2) !important;
    color: var(--void-fg) !important;
    border: 1px solid var(--void-border) !important;
    border-radius: 2px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
}
.stNumberInput input:focus, .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--void-fg) !important;
    box-shadow: 0 0 0 1px var(--void-fg-dim) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--void-bg2) !important;
    border: 1px solid var(--void-border) !important;
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    border-radius: 2px !important;
}
.stSelectbox [data-baseweb="select"] span {
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
}

/* ── Checkboxes ── */
.stCheckbox > label {
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.70rem !important;
}
.stCheckbox span[aria-checked="true"] {
    background: var(--void-fg) !important;
    border-color: var(--void-fg) !important;
}

/* ── Dataframes / tables ── */
.stDataFrame {
    border: 1px solid var(--void-border) !important;
    border-radius: 2px !important;
}
.stDataFrame thead tr th {
    background: var(--void-bg2) !important;
    color: var(--void-accent) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--void-border) !important;
}
.stDataFrame tbody tr td {
    background: var(--void-bg) !important;
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.62rem !important;
    border-bottom: 1px solid var(--void-grid) !important;
}
.stDataFrame tbody tr:hover td {
    background: var(--void-bg2) !important;
}

/* ── Tab strip ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--void-border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--void-fg-dim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.25rem 0.6rem !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--void-fg) !important;
    border-bottom: 2px solid var(--void-fg) !important;
    background: var(--void-bg2) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--void-accent) !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--void-bg2) !important;
    color: var(--void-fg) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.70rem !important;
    border: 1px solid var(--void-border) !important;
    border-radius: 2px !important;
    letter-spacing: 0.06em;
}
.streamlit-expanderContent {
    background: var(--void-bg) !important;
    border: 1px solid var(--void-border) !important;
    border-top: none !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: var(--void-fg) !important;
}
.stProgress > div > div {
    background: var(--void-bg2) !important;
    border: 1px solid var(--void-border) !important;
    border-radius: 1px !important;
}

/* ── Spinner ── */
.stSpinner > div > div {
    border-top-color: var(--void-fg) !important;
}

/* ── Pyplot figures ── */
.stImage img, .stPlotlyChart, .stPyplot {
    border: 1px solid var(--void-border) !important;
    border-radius: 2px !important;
    background: var(--void-bg) !important;
}

/* ── Info / Warning / Error / Success boxes ── */
.stAlert {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    border-radius: 2px !important;
}
div[data-baseweb="notification"] {
    font-family: var(--font-mono) !important;
}

/* ── Code blocks ── */
.stCodeBlock, code, pre {
    background: var(--void-bg2) !important;
    color: var(--void-accent) !important;
    font-family: var(--font-mono) !important;
    border: 1px solid var(--void-border) !important;
    border-radius: 2px !important;
}

/* ── Sidebar navigation items ── */
.nav-item {
    display: block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--void-fg-dim);
    letter-spacing: 0.06em;
    padding: 0.25rem 0.5rem;
    border-left: 2px solid transparent;
    margin-bottom: 0.1rem;
    cursor: pointer;
    transition: all 0.12s;
}
.nav-item.active {
    color: var(--void-fg);
    border-left: 2px solid var(--void-fg);
    background: var(--void-bg2);
}
.nav-item:hover {
    color: var(--void-accent);
    border-left: 2px solid var(--void-accent);
}

/* ── Top status bar ── */
.status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--void-bg2);
    border: 1px solid var(--void-border);
    border-radius: 2px;
    padding: 0.3rem 0.8rem;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    color: var(--void-accent);
    margin-bottom: 0.8rem;
}
.status-bar span.ok  { color: var(--void-fg); }
.status-bar span.err { color: var(--void-danger); }
.status-bar span.wrn { color: var(--void-warn); }

/* ── Splash screen terminal ── */
.splash-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem;
}
.splash-title {
    font-family: var(--font-mono);
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--void-fg);
    letter-spacing: 0.18em;
    text-shadow: 0 0 40px #00ff8880, 0 0 80px #00ff8840;
    margin-bottom: 0.2rem;
    text-align: center;
}
.splash-subtitle {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--void-accent);
    letter-spacing: 0.22em;
    text-align: center;
    margin-bottom: 0.6rem;
}
.splash-divider {
    width: 100%;
    max-width: 600px;
    border: none;
    border-top: 1px solid var(--void-border);
    margin: 0.6rem 0;
}
.splash-log-line {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--void-fg-dim);
    letter-spacing: 0.06em;
    text-align: left;
    width: 100%;
    max-width: 600px;
    margin: 0.06rem 0;
}
.splash-log-line.hi { color: var(--void-fg); }
.splash-log-line.ac { color: var(--void-accent); }
.splash-log-line.wr { color: var(--void-warn); }
.splash-log-line.er { color: var(--void-danger); }

/* ── Sidebar logo area ── */
.sidebar-logo {
    font-family: var(--font-mono);
    font-size: 1.05rem;
    font-weight: 900;
    color: var(--void-fg);
    letter-spacing: 0.14em;
    text-shadow: 0 0 12px #00ff8860;
    text-align: center;
    padding: 0.5rem 0 0.2rem;
}
.sidebar-station {
    font-family: var(--font-mono);
    font-size: 0.60rem;
    color: var(--void-accent);
    letter-spacing: 0.10em;
    text-align: center;
    margin-bottom: 0.4rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid var(--void-border);
    margin: 0.4rem 0;
}

/* ── Section sub-labels ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--void-fg-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0.5rem 0 0.2rem;
}

/* ── Scanline overlay (purely decorative) ── */
.scanline-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 9999;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
}

/* ── Void pulse animation ── */
@keyframes void-pulse {
    0%, 100% { text-shadow: 0 0 10px #00ff8840; }
    50%       { text-shadow: 0 0 30px #00ff8880, 0 0 60px #00ff8830; }
}
.void-pulse { animation: void-pulse 3s ease-in-out infinite; }

/* ── Cursor blink ── */
@keyframes cursor-blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.cursor { animation: cursor-blink 1s step-end infinite; }

/* ── Threat level badge ── */
.threat-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.60rem;
    letter-spacing: 0.08em;
    padding: 0.1rem 0.4rem;
    border-radius: 1px;
    border: 1px solid;
}
.threat-nominal    { color: var(--void-fg);     border-color: var(--void-fg);     background: #00180a; }
.threat-elevated   { color: var(--void-warn);   border-color: var(--void-warn);   background: #1a0e00; }
.threat-critical   { color: #ff4400;             border-color: #ff4400;             background: #1a0500; }
.threat-containment{ color: var(--void-danger); border-color: var(--void-danger); background: #1a0000;
                     animation: cursor-blink 0.7s step-end infinite; }

/* ── Day counter ── */
.day-counter {
    font-family: var(--font-mono);
    font-size: 0.60rem;
    color: var(--void-accent);
    letter-spacing: 0.10em;
    text-align: center;
    background: var(--void-bg2);
    border: 1px solid var(--void-border);
    padding: 0.2rem 0.5rem;
    border-radius: 1px;
    margin-bottom: 0.4rem;
}

/* ── Sidebar module list ── */
.mod-list-item {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    color: var(--void-fg-dim);
    padding: 0.2rem 0.3rem;
    border-left: 2px solid transparent;
    margin-bottom: 0.08rem;
}
.mod-list-item.selected {
    color: var(--void-fg);
    border-left: 2px solid var(--void-fg);
    background: rgba(0,255,136,0.04);
}

/* ── Matplotlib figure ── */
.stPyplot > div { background: transparent !important; }

/* ── Hide default streamlit branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
"""

# Scanline overlay (purely aesthetic — no interaction impact)
SCANLINE_HTML = '<div class="scanline-overlay"></div>'


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND MODULE IMPORTS (with graceful error handling)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import(module_name: str):
    """Import a backend module, returning it or an error placeholder."""
    try:
        import importlib
        return importlib.import_module(module_name), None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_backends():
    """Load all six backend modules once, cache across reruns."""
    results = {}
    for name in ["signal_engine", "spectral_analyzer", "anomaly_detector",
                  "ml_predictor", "environment_monitor", "hq_reporter"]:
        mod, err = _try_import(name)
        results[name] = (mod, err)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SPLASH / BOOT SCREEN
# ─────────────────────────────────────────────────────────────────────────────

BOOT_LINES = [
    ("hi", "[ BIOS v3.7.2 — ALPEN SIGNAL OBSERVATORIUM ]"),
    ("",   "CPU: Intel Xeon W-2295 @ 3.0GHz  |  RAM: 128 GB ECC"),
    ("",   "STORAGE: 4× 2TB NVMe RAID-6  |  GPU: RTX A5000 24GB"),
    ("",   ""),
    ("ac", "[ LOADING KERNEL MODULES ]"),
    ("",   "  signal_engine.py       ..... OK"),
    ("",   "  spectral_analyzer.py   ..... OK"),
    ("",   "  anomaly_detector.py    ..... OK"),
    ("",   "  ml_predictor.py        ..... OK"),
    ("",   "  environment_monitor.py ..... OK"),
    ("",   "  hq_reporter.py         ..... OK"),
    ("",   ""),
    ("ac", "[ HARDWARE INITIALISATION ]"),
    ("",   "  Dish AZ/EL motors       .... NOMINAL"),
    ("",   "  LNA cryo-cooler         .... 25.0 K"),
    ("",   "  Backend ADC             .... 2.0 MSPS locked"),
    ("",   "  Satellite uplink        .... 128 kbps"),
    ("",   "  Generator               .... RUNNING — 38.4 L"),
    ("",   "  Battery bank            .... 80% — 16.0 kWh"),
    ("",   ""),
    ("wr", "[ WARNING ] Last anomaly detected: 03:33 local time."),
    ("wr", "[ WARNING ] Drive DRV-A1F contains restricted class signal."),
    ("",   ""),
    ("ac", "[ ARIRAL REPUTATION ] NEUTRAL  |  Rep: 0"),
    ("",   ""),
    ("hi", "[ SYSTEM READY — BEGIN OBSERVATIONS ]"),
    ("",   "Type: listen."),
]


def render_boot_screen():
    """Full-screen terminal boot animation."""
    html_lines = ""
    for cls, text in BOOT_LINES:
        css = f"splash-log-line {cls}" if cls else "splash-log-line"
        html_lines += f'<div class="{css}">{text}</div>\n'

    st.markdown(f"""
    <div class="splash-container">
        <div class="splash-title void-pulse">vOiD</div>
        <div class="splash-subtitle">VOICES OF THE VOID — SIGNAL OBSERVATORY SYSTEM</div>
        <hr class="splash-divider"/>
        <div style="width:100%;max-width:600px;">
            {html_lines}
            <div class="splash-log-line hi">▋<span class="cursor"> </span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────────────────────────────────────────

def render_status_bar(backends: dict):
    day    = st.session_state.get("day", 1)
    pts    = st.session_state.get("total_points", 0)
    threat = st.session_state.get("threat_level", None)
    threat_name = threat.name if threat is not None else "NOMINAL"
    threat_col = ("err" if threat_name == "CONTAINMENT" else
                   "wrn" if threat_name in ("CRITICAL", "ELEVATED") else "ok")

    rep_mgr = st.session_state.get("rep_manager", None)
    rep_val = int(getattr(getattr(rep_mgr, "rep", None), "current", 0)) if rep_mgr else 0
    rep_state = ("ALLIED" if rep_val >= 50 else "FRIENDLY" if rep_val >= 10 else
                 "NEUTRAL" if rep_val >= -10 else "WARY" if rep_val >= -50 else "HOSTILE")

    dm = st.session_state.get("drive_manager", None)
    drive_info = "NO DRIVE"
    if dm and dm.active_drive:
        d = dm.active_drive
        drive_info = f"DRV:{d.drive_id[-6:]} {d.fill_fraction*100:.0f}%"

    # Module load status
    mod_status = ""
    for name, (mod, err) in backends.items():
        short = name.replace("_", "")[:6]
        if mod:
            mod_status += f'<span class="ok">[{short}✓]</span> '
        else:
            mod_status += f'<span class="err">[{short}✗]</span> '

    st.markdown(f"""
    <div class="status-bar">
        <span>ASO-DUNKELTALER-01</span>
        <span>DAY <span class="ok">{day:03d}</span></span>
        <span>PTS <span class="ok">{pts:,}</span></span>
        <span>THREAT <span class="{threat_col}">{threat_name}</span></span>
        <span>ARIRAL <span class="ok">{rep_state}</span> ({rep_val:+d})</span>
        <span>{drive_info}</span>
        <span style="font-size:0.55rem">{mod_status}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

MODULE_REGISTRY = [
    ("📡", "SIGNAL ACQUISITION",  "signal_engine",       "signal_engine_page"),
    ("🌊", "SPECTRAL ANALYSIS",   "spectral_analyzer",   "spectral_analyzer_page"),
    ("👁", "ANOMALY DETECTION",   "anomaly_detector",    "anomaly_detector_page"),
    ("🧠", "ML PREDICTION",       "ml_predictor",        "ml_predictor_page"),
    ("🏔", "ENVIRONMENT MONITOR", "environment_monitor", "environment_monitor_page"),
    ("📤", "HQ REPORTING",        "hq_reporter",         "hq_reporter_page"),
]


def render_sidebar(backends: dict):
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo void-pulse">📡 vOiD</div>
        <div class="sidebar-station">ALPEN SIGNAL OBSERVATORIUM<br>DUNKELTALER FOREST — 1840m ASL</div>
        <hr class="sidebar-divider"/>
        """, unsafe_allow_html=True)

        # Day counter
        day = st.session_state.get("day", 1)
        pts = st.session_state.get("total_points", 0)
        st.markdown(f"""
        <div class="day-counter">
            DAY {day:03d} — {pts:,} PTS
        </div>
        """, unsafe_allow_html=True)

        # Module selection
        st.markdown('<div class="section-label">[ SYSTEM MODULES ]</div>', unsafe_allow_html=True)
        module_labels = [f"{icon}  {name}" for icon, name, _, _ in MODULE_REGISTRY]

        selected_label = st.selectbox(
            "Navigation",
            ["⊙  BOOT SEQUENCE"] + module_labels,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)

        # Quick system stats
        st.markdown('<div class="section-label">[ SYSTEM STATUS ]</div>', unsafe_allow_html=True)

        # Drive status
        dm = st.session_state.get("drive_manager", None)
        if dm and dm.active_drive:
            d = dm.active_drive
            frac = d.fill_fraction
            col = "#ff2222" if frac > 0.9 else "#ff8800" if frac > 0.7 else "#00ff88"
            st.progress(frac, text=f"DRIVE: {d.drive_id[-6:]} {frac*100:.0f}%")

        # Threat
        threat = st.session_state.get("threat_level", None)
        t_name = threat.name if threat else "NOMINAL"
        t_col  = ("threat-containment" if t_name == "CONTAINMENT" else
                   "threat-critical"   if t_name == "CRITICAL" else
                   "threat-elevated"   if t_name == "ELEVATED" else
                   "threat-nominal")
        st.markdown(f'<div style="margin:0.2rem 0"><span class="threat-badge {t_col}">THREAT: {t_name}</span></div>',
                    unsafe_allow_html=True)

        # Signal log summary
        sig_log = st.session_state.get("signal_log", [])
        anomaly_recs = st.session_state.get("anomaly_records", [])
        n_void = sum(1 for r in sig_log if "VOID" in str(r.get("Class", "")))

        st.markdown(f"""
        <div style="font-family:var(--font-mono);font-size:0.60rem;color:var(--void-accent);
                    line-height:1.7;margin-top:0.3rem;">
          SIGNALS: {len(sig_log)}<br>
          ANOMALIES: {len(anomaly_recs)}<br>
          VOID EVENTS: <span style="color:{'#ff2222' if n_void else 'var(--void-fg-dim)'}">
                       {n_void}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)

        # Module load status table
        st.markdown('<div class="section-label">[ MODULE STATUS ]</div>', unsafe_allow_html=True)
        for icon, name, mod_key, _ in MODULE_REGISTRY:
            mod, err = backends.get(mod_key, (None, "not loaded"))
            status = "OK" if mod else "ERR"
            col    = "var(--void-fg)" if mod else "var(--void-danger)"
            st.markdown(
                f'<div style="font-family:var(--font-mono);font-size:0.58rem;'
                f'color:{col};margin:0.06rem 0;">'
                f'{icon} {name[:18]:<18} [{status}]</div>',
                unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:var(--font-mono);font-size:0.52rem;
                    color:var(--void-fg-dim);text-align:center;
                    line-height:1.6;">
            Dr. Kel Station<br>
            Dunkeltaler Forest, CH<br>
            46.80°N 8.10°E 1840m<br>
            H-I: 1420.405751768 MHz<br>
            © 2024 ASO — All Rights Reserved<br>
            <br>
            "Something is listening."
        </div>
        """, unsafe_allow_html=True)

    return selected_label


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def render_module_error(name: str, error: str):
    st.markdown(f"""
    <div style="
        background:#1a0000;
        border:2px solid #ff2222;
        border-radius:2px;
        padding:1rem 1.2rem;
        font-family:var(--font-mono);
        font-size:0.72rem;
        color:#ff4444;
        margin:1rem 0;">
        [ MODULE LOAD FAILURE ]<br><br>
        Module  : {name}<br>
        Error   : {error[:200]}<br><br>
        Ensure all backend files are in the same directory as vOiD.py.<br>
        Check requirements.txt is satisfied.
    </div>
    """, unsafe_allow_html=True)


def route_page(selected_label: str, backends: dict):
    """Dispatch to the selected backend page function."""

    if selected_label.startswith("⊙"):
        render_boot_screen()
        return

    # Match label to registry
    for icon, name, mod_key, fn_name in MODULE_REGISTRY:
        if name in selected_label:
            mod, err = backends.get(mod_key, (None, "module not found"))
            if mod is None:
                render_module_error(mod_key, err or "unknown error")
                return
            page_fn = getattr(mod, fn_name, None)
            if page_fn is None:
                render_module_error(mod_key, f"function '{fn_name}' not found in module")
                return
            # Guard: ensure session state is bootstrapped
            if hasattr(mod, "init_session_state"):
                mod.init_session_state()
            # Call the backend page
            page_fn()
            return

    # Fallback
    render_boot_screen()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_shared_state(backends: dict):
    """
    Ensure all cross-module shared state variables are initialised.
    Called once per session before any page renders.
    """
    sig_mod, _ = backends.get("signal_engine", (None, None))
    if sig_mod and hasattr(sig_mod, "init_session_state"):
        sig_mod.init_session_state()

    if "day" not in st.session_state:
        st.session_state.day = 1
    if "total_points" not in st.session_state:
        st.session_state.total_points = 0
    if "threat_level" not in st.session_state:
        # Import ThreatLevel from signal_engine if available
        if sig_mod:
            try:
                st.session_state.threat_level = sig_mod.ThreatLevel.NOMINAL
            except Exception:
                st.session_state.threat_level = None
    if "signal_log" not in st.session_state:
        st.session_state.signal_log = []
    if "anomaly_records" not in st.session_state:
        st.session_state.anomaly_records = []
    if "anomaly_features" not in st.session_state:
        st.session_state.anomaly_features = []

    # ARIRAL reputation manager (from anomaly_detector)
    if "rep_manager" not in st.session_state:
        anom_mod, _ = backends.get("anomaly_detector", (None, None))
        if anom_mod:
            try:
                st.session_state.rep_manager = anom_mod.AriralReputationManager()
            except Exception:
                pass

    # Drive manager (from signal_engine)
    if "drive_manager" not in st.session_state:
        if sig_mod:
            try:
                dm = sig_mod.DriveManager()
                dm.insert_drive(512.0)
                st.session_state.drive_manager = dm
            except Exception:
                pass

    # Environment subsystems
    env_mod, _ = backends.get("environment_monitor", (None, None))
    if env_mod:
        for attr, factory in [
            ("dish",           env_mod.DishHealth),
            ("power_system",   env_mod.PowerSystem),
            ("sleep_rec",      env_mod.SleepRecord),
            ("inventory",      env_mod.Inventory),
            ("station_log",    env_mod.StationLog),
        ]:
            if attr not in st.session_state:
                try:
                    st.session_state[attr] = factory()
                except Exception:
                    pass
        for attr, factory, args in [
            ("weather_sim",    env_mod.WeatherSimulator,   {"seed": 42, "season": "winter"}),
            ("event_engine",   env_mod.RandomEventEngine,  {}),
            ("sq_calc",        env_mod.SignalQualityCalculator, {}),
            ("astro_calc",     env_mod.AstronomicalCalc,   {}),
        ]:
            if attr not in st.session_state:
                try:
                    st.session_state[attr] = factory(**args)
                except Exception:
                    pass
        if "current_weather" not in st.session_state:
            try:
                st.session_state.current_weather = st.session_state.weather_sim.step(12.0)
            except Exception:
                pass

    # HQ reporter state
    hq_mod, _ = backends.get("hq_reporter", (None, None))
    if hq_mod:
        for attr, factory in [
            ("archive",           hq_mod.ArchiveManager),
            ("session_stats",     hq_mod.SessionStats),
            ("achievements",      hq_mod.AchievementSystem),
            ("hq_protocol",       hq_mod.HQProtocol),
            ("report_gen",        hq_mod.ReportGenerator),
            ("leaderboard",       hq_mod.Leaderboard),
        ]:
            if attr not in st.session_state:
                try:
                    st.session_state[attr] = factory()
                except Exception:
                    pass
        if "submitted_packages" not in st.session_state:
            st.session_state.submitted_packages = []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Inject global CSS and scanline
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(SCANLINE_HTML, unsafe_allow_html=True)

    # Load all backends (cached)
    backends = load_backends()

    # Bootstrap shared cross-module state
    bootstrap_shared_state(backends)

    # Render sidebar and get selection
    selected = render_sidebar(backends)

    # Top status bar
    render_status_bar(backends)

    # Route to selected page
    route_page(selected, backends)


if __name__ == "__main__":
    main()
