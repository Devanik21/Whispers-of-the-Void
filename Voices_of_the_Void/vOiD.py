"""
vOiD.py ─ Voices of the Void: Alpen Signal Observatorium
Definitive Frontend ─ All 7 Backends Wired
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Station  : ASO-DUNKELTALER-01
Operator : Dr. Kel  (recent hire, unknown research company)
Coord    : 46.80°N  8.10°E  1840 m ASL  ─  Dunkeltaler Forest
H-I line : 1420.405 751 768 MHz

Job description:
    Locate and process signals. Filter them.
    Send drives to supervisors with hash codes.
    Do not open the door at 03:33.
    Do not look outside.

"Something has been trying to make contact
 for a very long time.  You are the only one listening."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Backends:
  1. signal_engine.py        ─ signal_engine_page()
  2. spectral_analyzer.py    ─ spectral_analyzer_page()
  3. anomaly_detector.py     ─ anomaly_detector_page()
  4. ml_predictor.py         ─ ml_predictor_page()
  5. environment_monitor.py  ─ environment_monitor_page()
  6. hq_reporter.py          ─ hq_reporter_page()
  7. crypto_decoder.py       ─ crypto_decoder_page()
"""

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────
import base64
import importlib
import math
import os
import random
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  ─  must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="vOiD ─ ASO-Dunkeltaler",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND IMAGE LOADER
# Drop any .png named "bg.png" next to this file; it will auto-load.
# If absent the page still works with the dark gradient fallback.
# ─────────────────────────────────────────────────────────────────────────────
def _load_bg_b64() -> Optional[str]:
    """Locate bg.png adjacent to vOiD.py and return base64 data URI."""
    candidates = [
        Path(__file__).parent / "bg.png",
        Path(os.getcwd()) / "bg.png",
        Path("/mnt/user-data/uploads/bg.png"),
    ]
    for p in candidates:
        if p.exists():
            try:
                data = p.read_bytes()
                b64  = base64.b64encode(data).decode()
                return f"data:image/png;base64,{b64}"
            except Exception:
                pass
    return None


_BG_URI: Optional[str] = _load_bg_b64()

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# Dark Streamlit theme kept as base; we ADD a texture layer on top.
# All our custom widgets are glass/transparent so bg image shows through.
# ─────────────────────────────────────────────────────────────────────────────
_PAL = dict(
    fg      = "#96a868",    # earthy lichen green (was #00ff88)
    fg2     = "#7a8a52",    # dimmer lichen (was #00cc6a)
    dim     = "#1a1e14",    # very dim earthy surface (was #0d2e1a)
    acc     = "#c2b878",    # anomalous bone/sand accent (was #00ffcc)
    gold    = "#ffd700",    # ARIRAL gold
    warn    = "#ff8c00",    # warning orange
    dng     = "#ff2020",    # danger red
    purp    = "#9955ff",    # void purple
    blue    = "#0099ff",    # data blue
    glass   = "rgba(22,25,18,0.68)",     # earthy glass panel bg
    glass2  = "rgba(14,16,12,0.82)",     # darker earthy glass
    border  = "rgba(0,255,136,0.18)",  # ghost border 
    border2 = "rgba(0,255,136,0.08)",  # even lighter border
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  ─  3-layer system:
#   1. Background image / gradient (z-index 0)
#   2. Scanline + vignette overlay (z-index 9997/9998, pointer-events:none)
#   3. Glass morphism panels (semi-transparent, show bg)
# ─────────────────────────────────────────────────────────────────────────────
def _build_css(bg_uri: Optional[str]) -> str:
    if bg_uri:
        bg_rule = f"""
        background-image: url('{bg_uri}') !important;
        background-size: cover !important;
        background-position: center top !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
        """
    else:
        bg_rule = """
        background: radial-gradient(ellipse at 20% 40%,
            rgba(0,40,20,0.95) 0%,
            rgba(2,10,6,0.98) 55%,
            rgba(0,0,3,1) 100%) !important;
        """

    return f"""
<style>
/* ─── FONT ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ─── ROOT ──────────────────────────────────────────────────────────────── */
:root {{
  --fg:      {_PAL['fg']};
  --fg2:     {_PAL['fg2']};
  --dim:     {_PAL['dim']};
  --acc:     {_PAL['acc']};
  --gold:    {_PAL['gold']};
  --warn:    {_PAL['warn']};
  --dng:     {_PAL['dng']};
  --purp:    {_PAL['purp']};
  --blue:    {_PAL['blue']};
  --glass:   {_PAL['glass']};
  --glass2:  {_PAL['glass2']};
  --bdr:     {_PAL['border']};
  --bdr2:    {_PAL['border2']};
  --mono:    'Share Tech Mono','Courier New',monospace;
  --body:    'Rajdhani','Share Tech Mono',sans-serif;
}}

/* ─── LAYER 0 ─ background ──────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
.stApp {{
  {bg_rule}
}}

/* Darken layer above bg for readability */
[data-testid="stAppViewContainer"]::before {{
  content: "";
  position: fixed; top:0; left:0; width:100%; height:100%;
  pointer-events: none; z-index: 0;
  background: rgba(0,0,0,0.52);
}}

/* ─── LAYER 1 ─ scanlines ───────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]::after {{
  content: "";
  position: fixed; top:0; left:0; width:100%; height:100%;
  pointer-events: none; z-index: 9998;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 3px,
    rgba(0,0,0,0.04) 3px, rgba(0,0,0,0.04) 4px);
}}

/* ─── LAYER 2 ─ vignette ────────────────────────────────────────────────── */
body::after {{
  content: "";
  position: fixed; top:0; left:0; width:100%; height:100%;
  pointer-events: none; z-index: 9997;
  background: radial-gradient(ellipse at center,
    transparent 55%, rgba(0,0,0,0.65) 100%);
}}

/* ─── APP CHROME ─────────────────────────────────────────────────────────── */
* {{ box-sizing: border-box; }}
.stApp {{ color: var(--fg) !important; font-family: var(--mono) !important; }}
h1,h2,h3,h4,h5,h6 {{
  font-family: var(--mono) !important;
  color: var(--fg) !important;
  letter-spacing: .07em;
}}

/* ─── SCROLLBAR ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width:3px; height:3px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(0,255,136,.25); border-radius:0; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--fg); }}

/* ─── SIDEBAR ─────────────────────────────────────────────────────────────  */
[data-testid="stSidebar"] {{
  background: {_PAL['glass2']} !important;
  border-right: 1px solid var(--bdr) !important;
  backdrop-filter: blur(18px) saturate(160%) !important;
  -webkit-backdrop-filter: blur(18px) saturate(160%) !important;
}}
[data-testid="stSidebar"] * {{
  font-family: var(--mono) !important;
  color: var(--fg) !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}

/* ─── MAIN AREA ──────────────────────────────────────────────────────────── */
[data-testid="block-container"] {{
  background: transparent !important;
  padding-top: .6rem !important;
}}

/* ─── GLASS PANELS  (applied via st.markdown containers) ─────────────────── */
.glass-panel {{
  background: var(--glass);
  border: 1px solid var(--bdr);
  border-radius: 4px;
  backdrop-filter: blur(14px) saturate(130%);
  -webkit-backdrop-filter: blur(14px) saturate(130%);
  padding: .60rem .75rem;
  margin: .25rem 0;
}}
.glass-dark {{
  background: var(--glass2);
  border: 1px solid var(--bdr2);
  border-radius: 3px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  padding: .50rem .65rem;
  margin: .20rem 0;
}}

/* ─── METRICS ────────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{
  font-family: var(--mono) !important;
  color: var(--acc) !important;
  font-size: 1.10rem !important;
  letter-spacing: .04em;
  text-shadow: 0 0 8px rgba(0,255,204,.35);
}}
[data-testid="stMetricLabel"] {{
  font-family: var(--mono) !important;
  color: var(--fg2) !important;
  font-size: .58rem !important;
  letter-spacing: .09em;
  text-transform: uppercase;
}}
[data-testid="metric-container"] {{
  background: var(--glass) !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 3px !important;
  backdrop-filter: blur(12px) !important;
  padding: .4rem .55rem !important;
}}

/* ─── BUTTONS ─────────────────────────────────────────────────────────────── */
.stButton > button {{
  background: rgba(0,255,136,0.07) !important;
  color: var(--fg) !important;
  border: 1px solid rgba(0,255,136,0.30) !important;
  border-radius: 2px !important;
  font-family: var(--mono) !important;
  font-size: .69rem !important;
  letter-spacing: .07em;
  padding: .28rem .60rem !important;
  backdrop-filter: blur(8px) !important;
  transition: all .14s ease;
  text-transform: uppercase;
}}
.stButton > button:hover {{
  background: rgba(0,255,136,0.18) !important;
  border-color: var(--fg) !important;
  color: var(--acc) !important;
  box-shadow: 0 0 10px rgba(0,255,136,.22), inset 0 0 6px rgba(0,255,136,.06);
}}
.stButton > button:active {{
  background: rgba(0,255,136,0.35) !important;
  color: #000 !important;
}}

/* ─── INPUTS ──────────────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {{
  background: rgba(5,20,10,0.70) !important;
  color: var(--fg) !important;
  border: 1px solid rgba(0,255,136,0.22) !important;
  border-radius: 2px !important;
  font-family: var(--mono) !important;
  font-size: .70rem !important;
  backdrop-filter: blur(8px) !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
  border-color: var(--fg) !important;
  box-shadow: 0 0 0 1px rgba(0,255,136,.18) !important;
  outline: none !important;
}}
.stTextInput label, .stNumberInput label, .stTextArea label,
.stSelectbox label, .stSlider label, .stCheckbox label {{
  font-family: var(--mono) !important;
  font-size: .62rem !important;
  color: var(--fg2) !important;
  letter-spacing: .07em;
  text-transform: uppercase;
}}

/* ─── SELECTBOX ───────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
[data-baseweb="select"] > div {{
  background: rgba(5,20,10,0.72) !important;
  border: 1px solid rgba(0,255,136,0.22) !important;
  border-radius: 2px !important;
  color: var(--fg) !important;
  font-family: var(--mono) !important;
  font-size: .70rem !important;
  backdrop-filter: blur(8px) !important;
}}
[data-baseweb="select"] span {{ color: var(--fg) !important; font-family: var(--mono) !important; }}
[data-baseweb="menu"] {{
  background: rgba(5,18,10,0.94) !important;
  border: 1px solid var(--bdr) !important;
  backdrop-filter: blur(20px) !important;
}}
[role="option"] {{
  color: var(--fg) !important;
  font-family: var(--mono) !important;
  font-size: .67rem !important;
  background: transparent !important;
}}
[role="option"]:hover, [aria-selected="true"] {{
  background: rgba(0,255,136,.10) !important;
  color: var(--acc) !important;
}}

/* ─── SLIDERS ─────────────────────────────────────────────────────────────── */
[data-baseweb="slider"] [role="slider"] {{
  background: var(--fg) !important;
  border: none !important;
  box-shadow: 0 0 8px rgba(0,255,136,.45) !important;
}}
[data-baseweb="slider"] div[class*="Track"],
[data-baseweb="slider"] div[class*="track"] {{
  background: rgba(0,255,136,.15) !important;
}}
[data-testid="stSlider"] {{
  background: transparent !important;
}}

/* ─── CHECKBOXES ──────────────────────────────────────────────────────────── */
.stCheckbox label {{ color: var(--fg) !important; }}
.stCheckbox [role="checkbox"] {{
  border: 1px solid rgba(0,255,136,.35) !important;
  background: rgba(5,20,10,.60) !important;
  border-radius: 1px !important;
  backdrop-filter: blur(6px) !important;
}}
.stCheckbox [aria-checked="true"] {{
  background: var(--fg) !important;
  border-color: var(--fg) !important;
}}

/* ─── TABS ────────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {{
  background: rgba(5,18,10,0.55) !important;
  border-bottom: 1px solid var(--bdr) !important;
  backdrop-filter: blur(12px) !important;
  gap: 0 !important;
}}
[data-baseweb="tab"] {{
  background: transparent !important;
  color: rgba(0,255,136,.35) !important;
  font-family: var(--mono) !important;
  font-size: .61rem !important;
  letter-spacing: .08em;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  padding: .22rem .52rem !important;
  border-radius: 0 !important;
  text-transform: uppercase;
}}
[aria-selected="true"][data-baseweb="tab"] {{
  color: var(--fg) !important;
  border-bottom-color: var(--fg) !important;
  background: rgba(0,255,136,.05) !important;
  text-shadow: 0 0 8px rgba(0,255,136,.4);
}}
[data-baseweb="tab"]:hover {{ color: var(--acc) !important; }}

/* ─── DATAFRAMES ──────────────────────────────────────────────────────────── */
.stDataFrame {{
  background: rgba(5,18,10,0.55) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 3px !important;
  backdrop-filter: blur(10px) !important;
}}
.stDataFrame thead tr th {{
  background: rgba(0,255,136,0.07) !important;
  color: var(--acc) !important;
  font-family: var(--mono) !important;
  font-size: .58rem !important;
  letter-spacing: .05em;
  border-bottom: 1px solid var(--bdr) !important;
  text-transform: uppercase;
}}
.stDataFrame tbody tr td {{
  background: transparent !important;
  color: var(--fg2) !important;
  font-family: var(--mono) !important;
  font-size: .61rem !important;
  border-bottom: 1px solid rgba(0,255,136,.05) !important;
}}
.stDataFrame tbody tr:hover td {{
  background: rgba(0,255,136,.05) !important;
  color: var(--fg) !important;
}}

/* ─── EXPANDERS ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {{
  background: rgba(5,20,12,0.60) !important;
  color: var(--fg) !important;
  font-family: var(--mono) !important;
  font-size: .66rem !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 2px !important;
  letter-spacing: .06em;
  backdrop-filter: blur(10px) !important;
  text-transform: uppercase;
}}
.streamlit-expanderHeader:hover {{
  background: rgba(0,255,136,.08) !important;
  border-color: var(--bdr) !important;
}}
.streamlit-expanderContent {{
  background: rgba(3,12,6,0.70) !important;
  border: 1px solid var(--bdr2) !important;
  border-top: none !important;
  backdrop-filter: blur(12px) !important;
}}
details > summary {{ color: var(--fg) !important; font-family: var(--mono) !important; }}

/* ─── PROGRESS BARS ───────────────────────────────────────────────────────── */
.stProgress > div > div {{
  background: rgba(0,255,136,.12) !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 1px !important;
}}
.stProgress > div > div > div {{ background: var(--fg) !important; border-radius: 1px !important; }}

/* ─── ALERTS ──────────────────────────────────────────────────────────────── */
.stAlert {{
  font-family: var(--mono) !important;
  font-size: .64rem !important;
  border-radius: 2px !important;
  background: rgba(5,20,10,0.75) !important;
  backdrop-filter: blur(10px) !important;
  border-left: 3px solid var(--fg) !important;
}}
div[data-baseweb="notification"] {{
  font-family: var(--mono) !important;
  background: rgba(5,20,10,0.85) !important;
}}
/* success/warning/error alert border colours */
.stAlert[data-baseweb="notification"] {{ border-left-color: var(--fg) !important; }}

/* ─── CODE / PRE ──────────────────────────────────────────────────────────── */
code, pre, .stCodeBlock {{
  background: rgba(0,0,0,0.55) !important;
  color: var(--acc) !important;
  font-family: var(--mono) !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 2px !important;
}}

/* ─── PYPLOT ──────────────────────────────────────────────────────────────── */
.stPyplot > div {{ background: transparent !important; }}
[data-testid="stImage"] img {{
  border: 1px solid var(--bdr2) !important;
  border-radius: 2px;
}}

/* ─── SPINNER ─────────────────────────────────────────────────────────────── */
.stSpinner > div {{ border-top-color: var(--fg) !important; }}

/* ─── HIDE DEFAULT CHROME ─────────────────────────────────────────────────── */
#MainMenu, header, [data-testid="stToolbar"] {{ visibility: hidden !important; }}

/* ════════════════════════════════════════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
   ════════════════════════════════════════════════════════════════════════════ */

/* ── Terminal header strip ─────────────────────────────────────────────────── */
.term-hdr {{
  font-family: var(--mono);
  font-size: .72rem;
  color: var(--fg);
  letter-spacing: .14em;
  border-bottom: 1px solid var(--bdr);
  padding-bottom: .28rem;
  margin-bottom: .70rem;
  text-transform: uppercase;
  text-shadow: 0 0 8px rgba(0,255,136,.30);
}}
.term-hdr-gold {{
  font-family: var(--mono);
  font-size: .72rem;
  color: var(--gold);
  letter-spacing: .14em;
  border-bottom: 1px solid rgba(255,215,0,.25);
  padding-bottom: .28rem;
  margin-bottom: .65rem;
  text-transform: uppercase;
  text-shadow: 0 0 10px rgba(255,215,0,.35);
}}

/* ── Dim label ─────────────────────────────────────────────────────────────── */
.t-lbl {{
  font-family: var(--mono);
  font-size: .60rem;
  color: rgba(0,255,136,.45);
  letter-spacing: .11em;
  margin: .40rem 0 .15rem;
  text-transform: uppercase;
}}

/* ── Data block (monospace pre-formatted) ──────────────────────────────────── */
.data-block {{
  font-family: var(--mono);
  font-size: .63rem;
  color: var(--acc);
  background: rgba(0,0,0,0.50);
  border: 1px solid var(--bdr2);
  border-radius: 2px;
  padding: .38rem .55rem;
  margin: .22rem 0;
  white-space: pre-wrap;
  overflow-x: auto;
  backdrop-filter: blur(6px);
}}

/* ── STATUS BAR ────────────────────────────────────────────────────────────── */
.sbar {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0;
  background: rgba(2,10,5,0.82);
  border: 1px solid var(--bdr2);
  border-radius: 2px;
  padding: .22rem .55rem;
  font-family: var(--mono);
  font-size: .58rem;
  letter-spacing: .06em;
  margin-bottom: .55rem;
  backdrop-filter: blur(16px);
}}
.sbar-seg {{
  padding: 0 .60rem;
  border-right: 1px solid rgba(0,255,136,.10);
  white-space: nowrap;
  line-height: 1.6;
}}
.sbar-seg:last-child {{ border-right: none; }}
.c-ok   {{ color: var(--fg); }}
.c-acc  {{ color: var(--acc); }}
.c-gold {{ color: var(--gold); text-shadow: 0 0 6px rgba(255,215,0,.4); }}
.c-warn {{ color: var(--warn); }}
.c-dng  {{ color: var(--dng); animation: blink-dng .7s step-end infinite; }}
.c-dim  {{ color: rgba(0,255,136,.30); }}
.c-purp {{ color: var(--purp); }}
@keyframes blink-dng {{ 0%,100%{{opacity:1}} 50%{{opacity:.20}} }}

/* ── SIDEBAR LOGO ──────────────────────────────────────────────────────────── */
.sb-logo {{
  font-family: var(--mono);
  font-size: 2.10rem;
  font-weight: 900;
  color: var(--fg);
  letter-spacing: .28em;
  text-align: center;
  padding: .60rem 0 .04rem;
  text-shadow: 0 0 30px rgba(0,255,136,.55), 0 0 70px rgba(0,255,136,.20);
  animation: logo-pulse 4s ease-in-out infinite;
}}
@keyframes logo-pulse {{
  0%,100% {{ text-shadow: 0 0 22px rgba(0,255,136,.45), 0 0 60px rgba(0,255,136,.15); }}
  50%      {{ text-shadow: 0 0 55px rgba(0,255,136,.80), 0 0 120px rgba(0,255,136,.35); }}
}}
.sb-sub {{
  font-family: var(--mono);
  font-size: .52rem;
  color: var(--acc);
  letter-spacing: .20em;
  text-align: center;
  margin-bottom: .25rem;
  opacity: .75;
}}
.sb-hr {{ border: none; border-top: 1px solid var(--bdr2); margin: .32rem 0; }}
.sb-section {{
  font-family: var(--mono);
  font-size: .52rem;
  color: rgba(0,255,136,.35);
  letter-spacing: .14em;
  text-transform: uppercase;
  margin: .42rem 0 .16rem .10rem;
}}

/* ── Navigation item ────────────────────────────────────────────────────────── */
.nav-item {{
  display: flex;
  align-items: center;
  gap: .40rem;
  font-family: var(--mono);
  font-size: .66rem;
  color: rgba(0,255,136,.50);
  letter-spacing: .05em;
  padding: .16rem .42rem;
  border-left: 2px solid transparent;
  cursor: pointer;
  text-decoration: none;
  transition: all .12s;
  text-transform: uppercase;
}}
.nav-item:hover {{ color: var(--acc); border-left-color: var(--acc); background: rgba(0,255,136,.04); }}
.nav-item.active {{ color: var(--fg); border-left-color: var(--fg); background: rgba(0,255,136,.07); text-shadow: 0 0 6px rgba(0,255,136,.30); }}

/* ── BOOT SCREEN ─────────────────────────────────────────────────────────────── */
.boot-wrap {{
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.60rem .80rem;
}}
.boot-title {{
  font-family: var(--mono);
  font-size: 5.0rem;
  font-weight: 900;
  color: var(--fg);
  letter-spacing: .38em;
  text-shadow: 0 0 80px rgba(0,255,136,.65), 0 0 160px rgba(0,255,136,.25);
  text-align: center;
  margin-bottom: .10rem;
  animation: logo-pulse 4s ease-in-out infinite;
}}
.boot-sub1 {{
  font-family: var(--mono);
  font-size: .80rem;
  color: var(--acc);
  letter-spacing: .35em;
  text-align: center;
  margin-bottom: .20rem;
}}
.boot-sub2 {{
  font-family: var(--mono);
  font-size: .56rem;
  color: rgba(0,255,136,.45);
  letter-spacing: .16em;
  text-align: center;
  margin-bottom: .55rem;
}}
.boot-divider {{
  width: 100%;
  max-width: 700px;
  border: none;
  border-top: 1px solid var(--bdr2);
  margin: .48rem 0;
}}
.boot-log {{ width: 100%; max-width: 700px; }}
.bl {{ font-family: var(--mono); font-size: .64rem; margin: .07rem 0; letter-spacing: .04em; line-height:1.5; }}
.bl-hi  {{ color: var(--fg); }}
.bl-dim {{ color: rgba(0,255,136,.28); }}
.bl-acc {{ color: var(--acc); }}
.bl-wr  {{ color: var(--warn); }}
.bl-er  {{ color: var(--dng); }}
.bl-gd  {{ color: var(--gold); }}
.bl-pu  {{ color: var(--purp); }}
.cursor {{ animation: cursor-blink 1.2s step-end infinite; }}
@keyframes cursor-blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0}} }}

/* ── THREAT BADGES ────────────────────────────────────────────────────────────── */
.threat {{
  display: inline-block;
  font-family: var(--mono);
  font-size: .57rem;
  letter-spacing: .07em;
  padding: .06rem .38rem;
  border-radius: 1px;
  border: 1px solid;
  text-transform: uppercase;
}}
.t-nom {{ color: var(--fg);   border-color: rgba(0,255,136,.30); background: rgba(0,255,136,.06); }}
.t-el  {{ color: var(--warn); border-color: rgba(255,140,0,.45); background: rgba(255,140,0,.07); }}
.t-cr  {{ color: #ff4400;     border-color: rgba(255,68,0,.50);   background: rgba(255,68,0,.06);  }}
.t-co  {{ color: var(--dng);  border-color: rgba(255,32,32,.60);  background: rgba(255,0,0,.08);
           animation: blink-dng .55s step-end infinite; }}

/* ── GLASS CARD ────────────────────────────────────────────────────────────────  */
.g-card {{
  background: rgba(5,20,12,0.64);
  border: 1px solid var(--bdr2);
  border-radius: 3px;
  backdrop-filter: blur(16px) saturate(130%);
  -webkit-backdrop-filter: blur(16px) saturate(130%);
  padding: .48rem .62rem;
  margin: .20rem 0;
}}
.g-card-title {{
  font-family: var(--mono);
  font-size: .58rem;
  color: rgba(0,255,136,.50);
  letter-spacing: .10em;
  text-transform: uppercase;
  margin-bottom: .28rem;
}}
.g-card-val {{
  font-family: var(--mono);
  font-size: 1.05rem;
  color: var(--acc);
  letter-spacing: .05em;
  text-shadow: 0 0 8px rgba(0,255,204,.30);
}}
.g-card-sub {{
  font-family: var(--mono);
  font-size: .54rem;
  color: rgba(0,255,136,.38);
  letter-spacing: .06em;
  margin-top: .14rem;
}}

/* ── ARIRAL COMMS BOX ──────────────────────────────────────────────────────── */
.ariral-box {{
  background: rgba(10,14,0,0.72);
  border: 1px solid rgba(255,215,0,.40);
  border-radius: 2px;
  padding: .48rem .72rem;
  font-family: var(--mono);
  font-size: .72rem;
  color: var(--gold);
  letter-spacing: .10em;
  margin: .32rem 0;
  text-shadow: 0 0 10px rgba(255,215,0,.35);
  backdrop-filter: blur(10px);
}}
.ariral-box-dim {{
  background: rgba(10,14,0,0.55);
  border: 1px solid rgba(255,215,0,.15);
  border-radius: 2px;
  padding: .36rem .60rem;
  font-family: var(--mono);
  font-size: .65rem;
  color: rgba(255,215,0,.45);
  letter-spacing: .08em;
  margin: .22rem 0;
}}

/* ── VOID / CONTAINMENT ALERT ──────────────────────────────────────────────── */
.void-alert {{
  background: rgba(24,0,0,0.82);
  border: 2px solid var(--dng);
  border-radius: 2px;
  padding: .55rem .80rem;
  font-family: var(--mono);
  font-size: .72rem;
  color: var(--dng);
  letter-spacing: .11em;
  animation: blink-dng .65s step-end infinite;
  margin: .32rem 0;
  text-shadow: 0 0 12px rgba(255,32,32,.55);
  backdrop-filter: blur(10px);
}}

/* ── 03:33 WINDOW ALERT ────────────────────────────────────────────────────── */
.alert-0333 {{
  background: rgba(12,0,22,0.82);
  border: 1px solid var(--purp);
  border-radius: 2px;
  padding: .48rem .72rem;
  font-family: var(--mono);
  font-size: .70rem;
  color: var(--purp);
  letter-spacing: .10em;
  margin: .32rem 0;
  text-shadow: 0 0 10px rgba(153,85,255,.45);
  backdrop-filter: blur(12px);
}}

/* ── EMAIL CARD ─────────────────────────────────────────────────────────────── */
.email-card {{
  background: rgba(5,18,10,0.72);
  border: 1px solid var(--bdr);
  border-left: 3px solid var(--fg);
  border-radius: 2px;
  padding: .55rem .70rem;
  margin: .30rem 0;
  font-family: var(--mono);
  backdrop-filter: blur(12px);
}}
.email-card.unread {{ border-left-color: var(--acc); }}
.email-card.urgent {{ border-left-color: var(--dng); }}
.email-from {{
  font-size: .60rem;
  color: var(--acc);
  letter-spacing: .07em;
  text-transform: uppercase;
  margin-bottom: .18rem;
}}
.email-subject {{
  font-size: .72rem;
  color: var(--fg);
  letter-spacing: .05em;
  margin-bottom: .20rem;
}}
.email-body {{
  font-size: .63rem;
  color: rgba(0,255,136,.55);
  letter-spacing: .04em;
  line-height: 1.5;
}}

/* ── SIGNAL ROW ─────────────────────────────────────────────────────────────── */
.sig-row {{
  display: flex;
  align-items: center;
  gap: .50rem;
  font-family: var(--mono);
  font-size: .60rem;
  color: rgba(0,255,136,.55);
  border-bottom: 1px solid rgba(0,255,136,.05);
  padding: .10rem .10rem;
  transition: background .10s;
}}
.sig-row:hover {{ background: rgba(0,255,136,.04); color: var(--fg); }}
.sig-class {{ color: var(--acc); min-width: 100px; }}
.sig-wow-hi {{ color: var(--dng); text-shadow: 0 0 6px rgba(255,32,32,.40); }}
.sig-wow-md {{ color: var(--warn); }}
.sig-wow-lo {{ color: var(--fg2); }}

/* ── TELEMETRY MINI PANEL ───────────────────────────────────────────────────── */
.telem-row {{
  display: flex;
  justify-content: space-between;
  font-family: var(--mono);
  font-size: .60rem;
  color: rgba(0,255,136,.45);
  padding: .07rem .10rem;
  line-height: 1.55;
}}
.telem-val {{ color: var(--acc); }}
.telem-warn {{ color: var(--warn); }}
.telem-dng  {{ color: var(--dng); }}

/* ── PIPELINE NODES ─────────────────────────────────────────────────────────── */
.pipe-node {{
  display: inline-block;
  font-family: var(--mono);
  font-size: .58rem;
  color: rgba(0,0,0,0.9);
  background: var(--fg);
  padding: .10rem .35rem;
  border-radius: 1px;
  margin: .05rem;
  letter-spacing: .05em;
  text-transform: uppercase;
}}
.pipe-node-err {{
  display: inline-block;
  font-family: var(--mono);
  font-size: .58rem;
  color: var(--dng);
  background: rgba(255,32,32,.12);
  border: 1px solid rgba(255,32,32,.40);
  padding: .10rem .35rem;
  border-radius: 1px;
  margin: .05rem;
  letter-spacing: .05em;
  text-transform: uppercase;
}}
.pipe-arrow {{
  font-family: var(--mono);
  color: rgba(0,255,136,.30);
  font-size: .68rem;
  margin: 0 .08rem;
}}

/* ── LOG LINE ────────────────────────────────────────────────────────────────── */
.log-ln {{
  display: flex;
  gap: .50rem;
  font-family: var(--mono);
  font-size: .60rem;
  color: rgba(0,255,136,.45);
  border-bottom: 1px solid rgba(0,255,136,.04);
  padding: .10rem .15rem;
}}
.log-ln.C {{ color: var(--dng); }}
.log-ln.H {{ color: var(--dng); animation: blink-dng 1s step-end infinite; }}
.log-ln.W {{ color: var(--warn); }}

/* ── MODULE STATUS PILL ──────────────────────────────────────────────────────── */
.mod-pill {{
  display: inline-flex;
  align-items: center;
  gap: .32rem;
  font-family: var(--mono);
  font-size: .59rem;
  padding: .09rem .38rem;
  border-radius: 1px;
  border: 1px solid;
  margin: .07rem;
  text-transform: uppercase;
  letter-spacing: .04em;
}}
.mod-ok  {{ color: var(--fg);   border-color: rgba(0,255,136,.25); background: rgba(0,255,136,.06); }}
.mod-err {{ color: var(--dng);  border-color: rgba(255,32,32,.35);  background: rgba(255,0,0,.06);  }}

/* ── ACHIEVEMENT BADGE ────────────────────────────────────────────────────────── */
.ach-badge {{
  background: rgba(255,215,0,.08);
  border: 1px solid rgba(255,215,0,.35);
  border-radius: 2px;
  padding: .35rem .55rem;
  font-family: var(--mono);
  font-size: .63rem;
  color: var(--gold);
  margin: .15rem 0;
  letter-spacing: .05em;
  backdrop-filter: blur(8px);
}}
.ach-locked {{
  background: rgba(5,15,8,0.55);
  border: 1px solid rgba(0,255,136,.08);
  border-radius: 2px;
  padding: .35rem .55rem;
  font-family: var(--mono);
  font-size: .63rem;
  color: rgba(0,255,136,.22);
  margin: .15rem 0;
  letter-spacing: .05em;
}}

/* ── OVERVIEW GRID CELL ──────────────────────────────────────────────────────── */
.ov-cell {{
  background: rgba(5,18,10,0.62);
  border: 1px solid var(--bdr2);
  border-radius: 3px;
  padding: .42rem .55rem;
  backdrop-filter: blur(14px);
}}
.ov-cell-title {{
  font-family: var(--mono);
  font-size: .56rem;
  color: rgba(0,255,136,.38);
  letter-spacing: .10em;
  text-transform: uppercase;
  margin-bottom: .22rem;
}}
.ov-cell-big {{
  font-family: var(--mono);
  font-size: 1.20rem;
  color: var(--acc);
  text-shadow: 0 0 8px rgba(0,255,204,.28);
}}
.ov-cell-sub {{
  font-family: var(--mono);
  font-size: .55rem;
  color: rgba(0,255,136,.35);
  margin-top: .12rem;
  letter-spacing: .05em;
}}

/* ── MINI GAUGE ──────────────────────────────────────────────────────────────── */
.mgauge-ok   {{ color: var(--fg); }}
.mgauge-warn {{ color: var(--warn); }}
.mgauge-dng  {{ color: var(--dng); }}

/* ─────────────────────────────────────────────────────────────────────────── */
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND MANIFEST  ─  maps module names to page functions
# ─────────────────────────────────────────────────────────────────────────────
BACKENDS: List[Tuple[str, str, str, str]] = [
    # (module_name, page_fn, icon, label)
    ("signal_engine",       "signal_engine_page",       "📡", "SIGNAL ACQUISITION"),
    ("spectral_analyzer",   "spectral_analyzer_page",   "🌊", "SPECTRAL ANALYSIS"),
    ("anomaly_detector",    "anomaly_detector_page",    "👁", "ANOMALY DETECTION"),
    ("ml_predictor",        "ml_predictor_page",        "🧠", "ML PREDICTION"),
    ("environment_monitor", "environment_monitor_page", "🏔", "ENVIRONMENT"),
    ("hq_reporter",         "hq_reporter_page",         "📤", "HQ REPORTING"),
    ("crypto_decoder",      "crypto_decoder_page",      "🔐", "CRYPTO / DECODE"),
]

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND LOADER (cached per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_backends() -> Dict[str, Tuple[Any, Optional[str]]]:
    result: Dict[str, Tuple[Any, Optional[str]]] = {}
    for mod_name, _, _, _ in BACKENDS:
        try:
            result[mod_name] = (importlib.import_module(mod_name), None)
        except Exception:
            result[mod_name] = (None, traceback.format_exc()[-500:])
    return result

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)

def _set(key: str, val: Any) -> None:
    st.session_state[key] = val

def _mod(name: str, backends: Dict) -> Optional[Any]:
    return backends.get(name, (None, None))[0]

def _threat_name(t) -> str:
    if t is None: return "NOMINAL"
    return t.name if hasattr(t, "name") else str(t)

def _threat_css(t) -> str:
    n = _threat_name(t)
    return {"NOMINAL":"t-nom","ELEVATED":"t-el","CRITICAL":"t-cr","CONTAINMENT":"t-co"}.get(n, "t-nom")

def _rep_state(v: int) -> str:
    if v >= 50: return "LOYAL"
    if v >= 10: return "GOOD"
    if v >= -10: return "NEUTRAL"
    if v >= -50: return "INCONVENIENT"
    return "MEAN"

def _rep_col(v: int) -> str:
    if v >= 50: return "c-gold"
    if v >= 10: return "c-ok"
    if v >= -10: return "c-dim"
    if v >= -50: return "c-warn"
    return "c-dng"

def _mini_gauge(frac: float, width: int = 12) -> str:
    frac = float(np.clip(frac, 0, 1))
    n = int(frac * width)
    bar = "█" * n + "░" * (width - n)
    css = "mgauge-ok" if frac > .55 else "mgauge-warn" if frac > .25 else "mgauge-dng"
    return f'<span class="{css}" style="font-family:var(--mono);font-size:.70rem">{bar}</span>'

def _spark(vals: List[float], w: int = 18) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not vals: return "─" * w
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    sample = (vals[-w:] if len(vals) >= w else vals)
    return "".join(blocks[int((v - lo) / rng * 8)] for v in sample)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# Wires ALL 7 backend shared objects into st.session_state exactly once.
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap(B: Dict) -> None:
    # ── scalar defaults ───────────────────────────────────────────────────
    for k, v in {
        "day": 1, "ingame_hour": 12.0, "total_points": 0,
        "signal_log": [], "anomaly_records": [], "anomaly_features": [],
        "submitted_packages": [], "decoder_archive": [], "current_messages": [],
        "_ingested_n": 0, "_active": "BOOT",
        "_sig_history": [],          # WoW values over time for sparkline
        "_pts_history": [],          # cumulative points history
        "_event_log": [],            # internal cross-module event log
    }.items():
        if k not in st.session_state:
            _set(k, v)

    # ── signal_engine ──────────────────────────────────────────────────────
    se = _mod("signal_engine", B)
    if se:
        if hasattr(se, "init_session_state"):
            se.init_session_state()
        if "threat_level" not in st.session_state:
            try: _set("threat_level", se.ThreatLevel.NOMINAL)
            except Exception: _set("threat_level", None)

    # ── anomaly_detector ───────────────────────────────────────────────────
    ad = _mod("anomaly_detector", B)
    if ad:
        if "rep_manager" not in st.session_state:
            try: _set("rep_manager", ad.AriralReputationManager())
            except Exception: pass
        if "pipeline" not in st.session_state:
            try:
                p = ad.AnomalyPipeline(sample_rate=2e6)
                p._ensure_trained()
                _set("pipeline", p)
            except Exception: pass

    # ── ml_predictor ───────────────────────────────────────────────────────
    ml = _mod("ml_predictor", B)
    if ml and "master_predictor" not in st.session_state:
        try: _set("master_predictor", ml.MasterPredictor())
        except Exception: pass

    # ── environment_monitor ────────────────────────────────────────────────
    em = _mod("environment_monitor", B)
    if em:
        _env = [
            ("dish",          em.DishHealth,          {}),
            ("power_system",  em.PowerSystem,         {}),
            ("sleep_rec",     em.SleepRecord,         {}),
            ("inventory",     em.Inventory,           {}),
            ("station_log",   em.StationLog,          {}),
            ("weather_sim",   em.WeatherSimulator,    {"seed": 42, "season": "winter"}),
            ("event_engine",  em.RandomEventEngine,   {}),
            ("sq_calc",       em.SignalQualityCalculator, {}),
            ("astro_calc",    em.AstronomicalCalc,    {}),
            ("env_viz",       em.EnvironmentVisualizer, {}),
        ]
        for key, cls, kw in _env:
            if key not in st.session_state:
                try: _set(key, cls(**kw))
                except Exception: pass
        if "current_weather" not in st.session_state:
            try:
                wx = _get("weather_sim")
                _set("current_weather", wx.step(12.0))
            except Exception: pass

    # ── hq_reporter ────────────────────────────────────────────────────────
    hq = _mod("hq_reporter", B)
    if hq:
        _hq = [
            ("archive",       hq.ArchiveManager),
            ("session_stats", hq.SessionStats),
            ("achievements",  hq.AchievementSystem),
            ("hq_protocol",   hq.HQProtocol),
            ("report_gen",    hq.ReportGenerator),
            ("leaderboard",   hq.Leaderboard),
        ]
        for key, cls in _hq:
            if key not in st.session_state:
                try: _set(key, cls())
                except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-MODULE AUTO-HARVEST
# Runs every rerender. Propagates new signals → archive → stats → achievements.
# ─────────────────────────────────────────────────────────────────────────────

def _harvest(B: Dict) -> None:
    sig_log  = _get("signal_log", [])
    archive  = _get("archive")
    ss       = _get("session_stats")
    ach      = _get("achievements")
    rep_mgr  = _get("rep_manager")
    rep_val  = int(getattr(getattr(rep_mgr, "rep", None), "current", 0)) if rep_mgr else 0
    ingested = _get("_ingested_n", 0)
    new_rows = sig_log[ingested:]
    if not new_rows or not archive or not ss:
        return

    for row in new_rows:
        try:
            rec = archive.ingest(
                row, rep_value=rep_val,
                day=_get("day", 1),
                ingame_time=_get("ingame_hour", 12.0))
            if rec is not None:
                ss.total_signals   += 1
                ss.total_points    += rec.final_points
                ss.highest_wow      = max(ss.highest_wow, rec.wow_factor)
                ss.highest_snr      = max(ss.highest_snr, rec.snr_db)
                if rec.signal_class in ("ANOMALOUS","ARIRAL","ARIRAL_COMMS"):
                    ss.anomaly_count += 1
                if rec.signal_class == "ARIRAL":
                    ss.ariral_count += 1
                    if rep_mgr:
                        rep_mgr.apply_action("ariral_signal_detected")
                if rec.signal_class == "VOID_CARRIER":
                    ss.void_carrier_count += 1
                # update global pts
                cur = _get("total_points", 0)
                _set("total_points", cur + rec.final_points)
                # history for sparkline
                hist = _get("_sig_history", [])
                hist.append(rec.wow_factor)
                _set("_sig_history", hist[-80:])
                ph = _get("_pts_history", [])
                ph.append(_get("total_points", 0))
                _set("_pts_history", ph[-80:])
        except Exception:
            pass

    _set("_ingested_n", len(sig_log))

    # achievements
    if ach and hasattr(ach, "check_and_unlock"):
        try: ach.check_and_unlock(ss, archive, rep_val)
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-ACQUIRE  ─  one-click signal capture wired to signal_engine
# ─────────────────────────────────────────────────────────────────────────────

def _quick_acquire(B: Dict, force_class: Optional[str] = None) -> Optional[str]:
    """
    Generate and classify one signal using signal_engine.
    Appends to signal_log. Returns classification string or None on failure.
    """
    se = _mod("signal_engine", B)
    if not se:
        return None
    try:
        rng = np.random.default_rng(int(time.time_ns()) % (2**31))
        all_cls = list(se.SignalClass)
        hour = float(_get("ingame_hour", 12.0))

        if force_class:
            chosen = se.SignalClass(force_class)
        else:
            weights = np.ones(len(all_cls))
            # 03:33 window boosts anomalous/ARIRAL
            if abs(hour - 3.55) < 0.22:
                for i, c in enumerate(all_cls):
                    if c.value in ("ANOMALOUS","ARIRAL"):
                        weights[i] = 7.0
            # Day events
            day = _get("day", 1)
            if day >= 8:          # Ariral picnic from day 8
                for i, c in enumerate(all_cls):
                    if c.value == "ARIRAL":
                        weights[i] = max(weights[i], 3.0)
            weights /= weights.sum()
            chosen = rng.choice(all_cls, p=weights)

        snr   = float(rng.uniform(4, 32))
        dm    = float(rng.exponential(25))
        gen   = _get("generator", se.SignalGenerator(sample_rate=2e6))
        if "generator" not in st.session_state:
            _set("generator", gen)

        t, sig = se._generate_for_class(gen, chosen, 5.0, snr, dm)

        dsp = _get("dsp", se.DSPEngine(sample_rate=2e6))
        if "dsp" not in st.session_state:
            _set("dsp", dsp)

        feats = dsp.extract_features(sig, freq_mhz=1420.405751768)
        drift, _ = dsp.detect_drift_rate(sig)
        feats["drift_rate_hz_s"] = drift

        clf = _get("classifier")
        if clf is None:
            clf = se.SignalClassifier(); clf.train()
            _set("classifier", clf)
        pred_cls, conf, probs = clf.predict(feats)

        snr_m = dsp.estimate_snr(sig, (-5e4, 5e4), [(-5e5, -1e5), (1e5, 5e5)])

        from signal_engine import (SignalRecord, ProcessingStage,
                                    compute_wow_factor, _assess_threat, _infer_origin)
        rec = SignalRecord(
            signal_class=pred_cls,
            origin=_infer_origin(dm, 1420.405751768),
            center_freq_mhz=1420.405751768 + float(rng.normal(0, 0.5)),
            bandwidth_hz=float(feats.get("bw_10db_hz", 1000)),
            snr_db=float(snr_m),
            dispersion_measure=dm,
            ra_hours=float(rng.uniform(0, 24)),
            dec_degrees=float(rng.uniform(-90, 90)),
            classifier_confidence=conf,
            stage=ProcessingStage.CLASSIFIED)
        rec.wow_factor   = compute_wow_factor(rec, feats)
        rec.threat_level = _assess_threat(rec)
        rec.drift_rate_hz_s = drift

        # Write to drive
        dm_mgr = _get("drive_manager")
        if dm_mgr:
            ok, msg = dm_mgr.write(rec)
            rec.stage = ProcessingStage.ARCHIVED if ok else ProcessingStage.CLASSIFIED

        # Update session threat
        cur_t = _get("threat_level", se.ThreatLevel.NOMINAL)
        if hasattr(rec.threat_level, "value") and hasattr(cur_t, "value"):
            if rec.threat_level.value > cur_t.value:
                _set("threat_level", rec.threat_level)

        # Log
        row = rec.to_dataframe_row()
        log = _get("signal_log", [])
        log.append(row)
        _set("signal_log", log)

        pts_gain = int(rec.wow_factor * 85 + rec.snr_db * 2.5)
        _set("total_points", _get("total_points", 0) + pts_gain)

        # Station log
        slog = _get("station_log")
        if slog:
            from environment_monitor import EventSeverity
            sev = (EventSeverity.CRITICAL if "VOID" in rec.signal_class.value
                   else EventSeverity.WARNING if "ANOMALOUS" in rec.signal_class.value
                   else EventSeverity.INFO)
            slog.log(sev, "ACQ",
                     f"{rec.signal_class.value} | WoW={rec.wow_factor:.2f} | SNR={rec.snr_db:.1f}dB",
                     _get("day",1), _get("ingame_hour",12.0))

        return f"{pred_cls.value} | WoW={rec.wow_factor:.2f} | +{pts_gain}pts"
    except Exception as e:
        return f"ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# HQ EMAIL SYSTEM  ─  game-accurate daily email simulation
# ─────────────────────────────────────────────────────────────────────────────

class EmailSystem:
    """
    Simulates HQ / ARIRAL email inbox matching VotV email mechanics.
    Emails arrive on specific days or conditions.
    """

    INITIAL_EMAILS: List[Dict[str, Any]] = [
        {"id":"E001","from":"HQ <research@hq.aso>","subject":"Welcome to ASO-DUNKELTALER",
         "body":"Dr. Kel,\n\nYour assignment begins today.\nLocate and process signals. Filter them.\nSubmit drives with hash codes weekly.\nDo not leave equipment unattended.\n\n— Director",
         "day":1,"read":False,"urgent":False},
        {"id":"E002","from":"SYSTEM <auto@hq.aso>","subject":"[AUTO] Signal quota: Day 1",
         "body":"Minimum quota: 5 signals per day.\nPriority signals: NARROWBAND_CW, PULSAR.\nBonus: ANOMALOUS class signals +400% value.\n\nDrive submissions accepted via uplink 22:00–23:59.",
         "day":1,"read":False,"urgent":False},
        {"id":"E003","from":"HQ <research@hq.aso>","subject":"RE: Array calibration",
         "body":"Calibrate dish at 03:00 each night.\nDo not disturb neighbours.\nDo not look toward the valley after dark.\n\n— Director",
         "day":2,"read":False,"urgent":False},
        {"id":"E004","from":"user","subject":"blank",
         "body":"####OUTSIDE####OUTSIDE####OUTSIDE####",
         "day":13,"read":False,"urgent":True},
        {"id":"E005","from":"ARIRAL <◈@void>","subject":"◈ ⊕ ◉ △",
         "body":"◈ signal detected\n◉ observer confirmed\n⊕ contact initiated\n△ approach authorised\n◎ acknowledge",
         "day":8,"read":False,"urgent":False},
        {"id":"E006","from":"HQ <research@hq.aso>","subject":"PRIORITY: Drive hash required",
         "body":"Submit drive hash for satellite SIERRA.\nHash code must match our records.\nDeadline: end of current day.\n\nFailure to comply: contract review.",
         "day":3,"read":False,"urgent":True},
        {"id":"E007","from":"HQ <research@hq.aso>","subject":"Signal 'evil' — PROTOCOL 7-OMEGA",
         "body":"If you detect a VOID_CARRIER class signal:\n1. Do NOT process to level 3.\n2. Delete ALL copies immediately.\n3. Erase the drive.\n4. Submit erasure confirmation.\n\nNon-compliance is a termination offence.",
         "day":5,"read":False,"urgent":True},
    ]

    def __init__(self):
        if "emails" not in st.session_state:
            _set("emails", list(self.INITIAL_EMAILS))
        self._emails: List[Dict] = _get("emails", [])

    def visible_emails(self) -> List[Dict]:
        day = _get("day", 1)
        return [e for e in self._emails if e["day"] <= day]

    def unread_count(self) -> int:
        return sum(1 for e in self.visible_emails() if not e["read"])

    def mark_read(self, eid: str) -> None:
        for e in self._emails:
            if e["id"] == eid:
                e["read"] = True
        _set("emails", self._emails)

    def inject(self, from_: str, subject: str, body: str, urgent: bool = False) -> None:
        new_id = f"E{len(self._emails)+1:03d}"
        self._emails.append({
            "id":new_id, "from":from_, "subject":subject, "body":body,
            "day":_get("day",1), "read":False, "urgent":urgent})
        _set("emails", self._emails)

    def render(self) -> None:
        visible = self.visible_emails()
        if not visible:
            st.markdown('<div class="t-lbl">No emails.</div>', unsafe_allow_html=True)
            return

        for em in reversed(visible):
            badge = "🔴 " if em["urgent"] else ("📭 " if em["read"] else "📬 ")
            css   = "email-card urgent" if em["urgent"] else (
                    "email-card" if not em["read"] else "email-card")
            unread_str = "" if em["read"] else " [UNREAD]"
            with st.expander(f'{badge}{em["subject"]}{unread_str} — {em["from"][:30]}'):
                st.markdown(
                    f'<div class="{css}">'
                    f'<div class="email-from">FROM: {em["from"]}</div>'
                    f'<div class="email-subject">{em["subject"]}</div>'
                    f'<div class="email-body">{em["body"].replace(chr(10),"<br>")}</div>'
                    f'</div>', unsafe_allow_html=True)
                self.mark_read(em["id"])


# ─────────────────────────────────────────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────────────────────────────────────────

def render_status_bar(B: Dict) -> None:
    day   = _get("day", 1)
    pts   = _get("total_points", 0)
    t     = _get("threat_level", None)
    t_name= _threat_name(t)
    t_css = {"NOMINAL":"c-dim","ELEVATED":"c-warn","CRITICAL":"c-warn","CONTAINMENT":"c-dng"}.get(t_name,"c-dim")
    hour  = float(_get("ingame_hour", 12.0))
    hh, mm = int(hour), int((hour%1)*60)
    ts    = f"{hh:02d}:{mm:02d}"
    w333  = abs(hour - 3.55) < 0.22

    rep_mgr = _get("rep_manager")
    rep_val = int(getattr(getattr(rep_mgr,"rep",None),"current",0)) if rep_mgr else 0
    rep_st  = _rep_state(rep_val)
    rep_css = _rep_col(rep_val)

    dm = _get("drive_manager")
    drv_txt, drv_css = "NO DRIVE", "c-warn"
    if dm and dm.active_drive:
        d = dm.active_drive
        pct = d.fill_fraction * 100
        drv_txt = f"DRV:{d.drive_id[-6:]} {pct:.0f}%"
        drv_css = "c-dng" if pct > 90 else "c-warn" if pct > 70 else "c-ok"

    n_sig = len(_get("signal_log", []))
    ok_n  = sum(1 for mn,_,_,_ in BACKENDS if B.get(mn,(None,None))[0])
    mod_css = "c-ok" if ok_n == len(BACKENDS) else "c-warn" if ok_n >= 5 else "c-dng"

    pwr = _get("power_system")
    pwr_txt, pwr_css = "PWR:─", "c-dim"
    if pwr:
        src  = pwr.active_source.value if hasattr(pwr,"active_source") else "?"
        fuel = getattr(pwr,"generator_fuel_l",0)
        pwr_txt = f"PWR:{src}"
        pwr_css = "c-dng" if src=="OFFLINE" else "c-warn" if fuel < 8 else "c-ok"

    wx   = _get("current_weather")
    wx_txt, wx_css = "WX:─", "c-dim"
    if wx:
        wx_txt = f"WX:{wx.state.value[:3]} {wx.temperature_c:.0f}°C"
        wx_css = "c-warn" if wx.state.value in ("STORM","FOG") else "c-ok"

    em_sys = EmailSystem()
    uc = em_sys.unread_count()
    mail_css = "c-warn" if uc > 0 else "c-dim"

    ts_css = "c-dng" if w333 else "c-dim"
    segs = [
        ("c-dim",   "ASO-DKL-01"),
        ("c-dim",   f"DAY:{day:03d}"),
        (ts_css,    ts),
        (t_css,     f"THREAT:{t_name}"),
        (rep_css,   f"ARIRAL:{rep_st}({rep_val:+d})"),
        (drv_css,   drv_txt),
        (pwr_css,   pwr_txt),
        (wx_css,    wx_txt),
        ("c-dim",   f"SIGS:{n_sig}"),
        ("c-acc",   f"PTS:{pts:,}"),
        (mail_css,  f"MAIL:{uc}"),
        (mod_css,   f"MOD:{ok_n}/{len(BACKENDS)}"),
    ]
    inner = "".join(
        f'<span class="sbar-seg"><span class="{css}">{txt}</span></span>'
        for css, txt in segs)
    st.markdown(f'<div class="sbar">{inner}</div>', unsafe_allow_html=True)

    # Critical alerts
    if t_name == "CONTAINMENT":
        st.markdown(
            '<div class="void-alert">⚠  CONTAINMENT — VOID_CARRIER DETECTED — '
            'ERASE DRIVE IMMEDIATELY — DO NOT TRANSMIT — DO NOT LOOK OUTSIDE</div>',
            unsafe_allow_html=True)
    if w333:
        st.markdown(
            '<div class="alert-0333">⚠  03:33 TEMPORAL WINDOW ACTIVE — '
            'ANOMALY PROBABILITY ELEVATED — LOG ALL ACQUISITIONS</div>',
            unsafe_allow_html=True)
    if t_name == "ELEVATED" or t_name == "CRITICAL":
        ariral_msgs = [m for m in _get("decoder_archive",[])
                       if hasattr(m,"ariral_phrase") and m.ariral_phrase]
        if ariral_msgs:
            ph = ariral_msgs[-1].ariral_phrase
            st.markdown(f'<div class="ariral-box">◈ LATEST ARIRAL: {ph[:80]}</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(B: Dict) -> str:
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sb-logo">vOiD</div>
        <div class="sb-sub">VOICES&nbsp;OF&nbsp;THE&nbsp;VOID</div>
        <div class="sb-sub" style="font-size:.48rem;opacity:.55;">
          ALPEN&nbsp;SIGNAL&nbsp;OBSERVATORIUM&nbsp;·&nbsp;1840 m&nbsp;ASL
        </div>
        <hr class="sb-hr"/>
        """, unsafe_allow_html=True)

        # Day/pts strip
        day = _get("day", 1)
        pts = _get("total_points", 0)
        em_sys = EmailSystem()
        uc = em_sys.unread_count()
        st.markdown(f"""
        <div style="background:rgba(0,0,0,0.45);border:1px solid rgba(0,255,136,.12);
                    border-radius:2px;padding:.22rem .45rem;text-align:center;
                    font-family:var(--mono);font-size:.60rem;
                    backdrop-filter:blur(10px);margin-bottom:.32rem;">
          DAY&nbsp;<span style="color:var(--fg)">{day:03d}</span>
          &nbsp;&middot;&nbsp;
          <span style="color:var(--gold)">{pts:,}</span>&nbsp;PTS
          &nbsp;&middot;&nbsp;
          <span style="color:{'var(--warn)' if uc>0 else 'rgba(0,255,136,.30)'}">
          MAIL:{uc}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">[ Navigation ]</div>', unsafe_allow_html=True)
        all_pages = [
            ("BOOT",     "⊙", "BOOT SEQUENCE"),
            ("OVERVIEW", "◈", "STATION OVERVIEW"),
            ("EMAILS",   "📬", "EMAILS / HQ COMMS"),
            ("EVENTS",   "📖", "EVENT JOURNAL"),
        ] + [(mn.upper(), icon, label) for mn, _, icon, label in BACKENDS]

        labels = [f"{icon}  {lbl}" for _, icon, lbl in all_pages]
        sel = st.selectbox("Module", labels, index=0, label_visibility="collapsed",
                           key="nav_select")
        sel_key = all_pages[labels.index(sel)][0]

        st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)

        # Drive progress
        dm = _get("drive_manager")
        if dm and dm.active_drive:
            d = dm.active_drive
            st.progress(d.fill_fraction, text=f"DRIVE: {d.drive_id[-8:]} {d.fill_fraction*100:.0f}%")

        # Threat badge
        t = _get("threat_level")
        t_name = _threat_name(t)
        t_css  = _threat_css(t)
        st.markdown(f'<div style="margin:.18rem 0"><span class="threat {t_css}">THREAT: {t_name}</span></div>',
                    unsafe_allow_html=True)

        # ARIRAL rep bar
        rep_mgr = _get("rep_manager")
        rep_val = int(getattr(getattr(rep_mgr,"rep",None),"current",0)) if rep_mgr else 0
        rep_frac = (rep_val + 100) / 200
        st.progress(float(np.clip(rep_frac,0,1)),
                    text=f"ARIRAL: {_rep_state(rep_val)} ({rep_val:+d})")

        st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)

        # ── Quick actions ─────────────────────────────────────────────────
        st.markdown('<div class="sb-section">[ Quick Actions ]</div>', unsafe_allow_html=True)

        if st.button("⬡ ACQUIRE SIGNAL", use_container_width=True, key="qa_acq"):
            result = _quick_acquire(B)
            if result:
                st.toast(f"ACQ: {result}", icon="📡")

        c1sb, c2sb = st.columns(2)
        with c1sb:
            if st.button("⬡ +1 HOUR", use_container_width=True, key="qa_1h"):
                _advance_time(B, 1.0)
                st.toast("Hour advanced")
        with c2sb:
            if st.button("⬡ SLEEP 8H", use_container_width=True, key="qa_sleep"):
                slp = _get("sleep_rec")
                if slp:
                    q = slp.sleep(8.0)
                    inv = _get("inventory")
                    if inv: inv.consume("food_rations", 1)
                    st.toast(f"Slept 8h | Quality: {q:.2f}", icon="💤")

        if st.button("⬡ SEAL + SUBMIT DRIVE", use_container_width=True, key="qa_seal"):
            _seal_and_submit(B)
            st.toast("Drive sealed and submitted to HQ", icon="📤")

        if st.button("⬡ ADVANCE DAY", use_container_width=True, key="qa_day"):
            _advance_day(B)
            st.toast(f"Day {_get('day',1)} begins", icon="🌑")

        st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)

        # ── Telemetry mini panel ──────────────────────────────────────────
        st.markdown('<div class="sb-section">[ Live Telemetry ]</div>', unsafe_allow_html=True)

        wx  = _get("current_weather")
        pwr = _get("power_system")
        slp = _get("sleep_rec")
        dsh = _get("dish")
        inv = _get("inventory")

        def _trow(lbl: str, val: str, css: str = "telem-val") -> str:
            return (f'<div class="telem-row">'
                    f'<span>{lbl}</span><span class="{css}">{val}</span></div>')

        telem_html = ""
        telem_html += _trow("SIGNALS", str(len(_get("signal_log",[]))))
        telem_html += _trow("ANOMALIES", str(len(_get("anomaly_records",[]))))
        if dsh:
            eff = dsh.effective_area_fraction
            css = "telem-ok" if eff > .80 else "telem-warn" if eff > .50 else "telem-dng"
            telem_html += _trow("DISH EFF", f"{eff*100:.0f}%",
                                 "telem-val" if eff > .80 else "telem-warn" if eff > .50 else "telem-dng")
        if pwr:
            bc = pwr.battery_fraction
            telem_html += _trow("BATTERY", f"{bc*100:.0f}%",
                                 "telem-val" if bc > .55 else "telem-warn" if bc > .20 else "telem-dng")
            telem_html += _trow("FUEL", f"{pwr.generator_fuel_l:.0f}L",
                                 "telem-val" if pwr.generator_fuel_l > 15 else "telem-warn")
        if wx:
            telem_html += _trow("WEATHER", f"{wx.state.value[:5]} {wx.temperature_c:.0f}°C")
            telem_html += _trow("RADIO-Q", f"{wx.radio_seeing_factor:.2f}",
                                 "telem-val" if wx.radio_seeing_factor > .75 else "telem-warn")
        if slp:
            telem_html += _trow("FATIGUE", slp.fatigue_level.value[:8],
                                 "telem-val" if slp.hours_awake < 20 else
                                 "telem-warn" if slp.hours_awake < 36 else "telem-dng")
        if inv:
            telem_html += _trow("FOOD", f"{inv.food_rations}d",
                                 "telem-val" if inv.food_rations > 3 else "telem-warn")
            telem_html += _trow("DRIVES", str(inv.blank_drives),
                                 "telem-val" if inv.blank_drives > 2 else "telem-warn")

        st.markdown(f'<div style="background:rgba(0,0,0,0.35);border:1px solid rgba(0,255,136,.08);'
                    f'border-radius:2px;padding:.28rem .40rem;backdrop-filter:blur(8px);">'
                    f'{telem_html}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)

        # ── Module health ─────────────────────────────────────────────────
        st.markdown('<div class="sb-section">[ Module Health ]</div>', unsafe_allow_html=True)
        for mn, _, icon, lbl in BACKENDS:
            mod, err = B.get(mn, (None, ""))
            css = "mod-ok" if mod else "mod-err"
            sym = "✓" if mod else "✗"
            st.markdown(
                f'<span class="mod-pill {css}">{icon}&nbsp;{lbl[:12]}&nbsp;{sym}</span>',
                unsafe_allow_html=True)

        st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:var(--mono);font-size:.49rem;
                    color:rgba(0,255,136,.28);text-align:center;line-height:1.9;
                    padding:.10rem 0;">
          46.80°N · 8.10°E · 1840m ASL<br>
          H-I: 1420.405&thinsp;751&thinsp;768 MHz<br>
          Dr. Kel · ASO-DKL-01<br>
          <br>
          <span style="color:rgba(0,255,136,.18)">
          "Something is listening."</span>
        </div>
        """, unsafe_allow_html=True)

    return sel_key


# ─────────────────────────────────────────────────────────────────────────────
# TIME / DAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _advance_time(B: Dict, hours: float) -> None:
    """Advance in-game time and update all environment subsystems."""
    cur = float(_get("ingame_hour", 12.0))
    new_h = (cur + hours) % 24.0
    _set("ingame_hour", new_h)

    em = _mod("environment_monitor", B)
    if not em:
        return
    try:
        wx_sim = _get("weather_sim")
        if wx_sim:
            new_wx = wx_sim.step(new_h)
            _set("current_weather", new_wx)
        dsh = _get("dish")
        wx  = _get("current_weather")
        slp = _get("sleep_rec")
        pwr = _get("power_system")
        if dsh and wx:
            dsh.degrade(hours, wx, np.random.default_rng(int(time.time())))
        if slp:
            slp.update(hours)
        if pwr and wx:
            ac = em.AstronomicalCalc
            doy = int(_get("env_doy", 355))
            el  = ac.solar_elevation(46.8, 8.1, new_h, doy)
            dp  = ac.day_phase(el)
            pe  = pwr.update(hours, wx, dp)
            # Power events
            slog = _get("station_log")
            if slog and pe:
                from environment_monitor import EventSeverity
                for ev_key in pe:
                    slog.log(EventSeverity.WARNING, "POWER",
                             ev_key.replace("_"," ").upper(),
                             _get("day",1), new_h)
        # Random events
        ev_eng = _get("event_engine")
        if ev_eng:
            game_state = {
                "dish_health_fraction": dsh.effective_area_fraction if dsh else 1.0,
                "generator_fuel_l":     pwr.generator_fuel_l if pwr else 50.0,
                "hours_awake":          slp.hours_awake if slp else 8.0,
                "weather_state":        wx.state.value if wx else "CLEAR",
                "ingame_hour":          new_h,
            }
            new_evs = ev_eng.sample(hours, game_state)
            slog = _get("station_log")
            evlog = _get("_event_log", [])
            for ev in new_evs:
                evlog.append(ev)
                if slog:
                    from environment_monitor import EventSeverity
                    slog.log(ev.severity, ev.category,
                             f"{ev.title}: {ev.description[:80]}",
                             _get("day",1), new_h)
                # Apply effects
                _apply_event_effects(ev, B)
            _set("_event_log", evlog[-100:])
    except Exception:
        pass


def _apply_event_effects(ev: Any, B: Dict) -> None:
    """Apply RandomEvent effects to session state objects."""
    eff = getattr(ev, "effects", {})
    dsh = _get("dish")
    inv = _get("inventory")
    slp = _get("sleep_rec")
    rep_mgr = _get("rep_manager")

    try:
        if "surface_contamination" in eff and dsh:
            dsh.surface_contamination = float(np.clip(
                dsh.surface_contamination + eff["surface_contamination"], 0, 1))
        if "lna_noise_figure_db" in eff and dsh:
            dsh.lna_noise_figure_db = float(np.clip(
                dsh.lna_noise_figure_db + eff["lna_noise_figure_db"], 0.3, 5))
        if "az_motor_health" in eff and dsh:
            dsh.az_motor_health = float(np.clip(
                dsh.az_motor_health + eff["az_motor_health"], 0, 1))
        if "connector_vswr" in eff and dsh:
            dsh.connector_vswr = float(np.clip(
                dsh.connector_vswr + eff["connector_vswr"], 1.0, 5.0))
        if "feed_alignment_deg" in eff and dsh:
            dsh.feed_alignment_deg = float(
                dsh.feed_alignment_deg + eff["feed_alignment_deg"])
        if "shrimp_packs" in eff and inv:
            inv.add("shrimp_packs", int(eff["shrimp_packs"]))
            if rep_mgr:
                rep_mgr.apply_action("gift_received_from_ariral")
        if "food_rations" in eff and inv:
            inv.add("food_rations", int(eff["food_rations"]))
        if "blank_drives" in eff and inv:
            inv.add("blank_drives", int(eff["blank_drives"]))
        if "cleaning_kits" in eff and inv:
            inv.add("cleaning_kits", int(eff["cleaning_kits"]))
        if "hours_awake_penalty" in eff and slp:
            slp.hours_awake += float(eff["hours_awake_penalty"])
        if "void_fragments" in eff and inv:
            inv.add("void_fragments", int(eff["void_fragments"]))
        if "rep_delta" in eff and rep_mgr:
            rep_mgr.apply_action(None, custom_delta=int(eff["rep_delta"]))
        if "stress_delta" in eff and slp:
            slp.stress_level = float(np.clip(
                slp.stress_level + eff["stress_delta"], 0, 1))
    except Exception:
        pass


def _advance_day(B: Dict) -> None:
    cur = _get("day", 1)
    _set("day", cur + 1)
    _set("ingame_hour", 8.0)
    inv = _get("inventory")
    if inv:
        inv.consume("food_rations", 1)
    slp = _get("sleep_rec")
    if slp:
        slp.stress_level = max(0.0, slp.stress_level - 0.1)

    # Leaderboard record
    lb = _get("leaderboard")
    ss = _get("session_stats")
    rep_mgr = _get("rep_manager")
    rep_val = int(getattr(getattr(rep_mgr,"rep",None),"current",0)) if rep_mgr else 0
    sig_log = _get("signal_log", [])
    if lb and ss and hasattr(lb, "record_day"):
        try:
            wow_max = max((float(r.get("WoW",0)) for r in sig_log), default=0.0)
            lb.record_day(cur, _get("total_points",0),
                          len(sig_log), getattr(ss,"anomaly_count",0),
                          rep_val, wow_max)
            if hasattr(ss,"day_number"): ss.day_number = cur + 1
        except Exception:
            pass

    # Day-8 picnic email
    if cur + 1 == 8:
        em_sys = EmailSystem()
        em_sys.inject(
            "SYSTEM <auto@hq.aso>",
            "[ALERT] Anomalous readings in valley — Day 8",
            "Two warp signatures detected at X:-177 / Y:-510.\n"
            "Servers SIERRA and ROMEO offline 01:45.\n"
            "Do not approach the valley.\nDo not interact.\n"
            "Photograph if possible (reputation +5).",
            urgent=True)

    # Week survival
    if cur + 1 == 7:
        ach = _get("achievements")
        ss2 = _get("session_stats")
        arch = _get("archive")
        if ach and ss2 and arch:
            try: ach.check_and_unlock(ss2, arch, rep_val,
                                       recent_event="WEEK_SURVIVAL")
            except Exception: pass


def _seal_and_submit(B: Dict) -> None:
    dm_mgr  = _get("drive_manager")
    archive = _get("archive")
    hq_prot = _get("hq_protocol")
    ss      = _get("session_stats")
    rep_mgr = _get("rep_manager")
    ach     = _get("achievements")

    if not dm_mgr or not dm_mgr.active_drive:
        return
    did = dm_mgr.active_drive_id
    hq_hash = dm_mgr.seal_and_submit(did)
    dm_mgr.insert_drive(512.0)      # insert fresh blank

    if archive and hq_prot:
        try:
            pkg = archive.build_drive_package(did, max_records=200)
            resp = hq_prot.submit(pkg)
            pkgs = _get("submitted_packages", [])
            pkgs.append(pkg)
            _set("submitted_packages", pkgs)
            bonus = resp.bonus_points
            _set("total_points", _get("total_points",0) + bonus)
            if ss and hasattr(ss,"drives_submitted"):
                ss.drives_submitted += 1
            if rep_mgr:
                rep_mgr.apply_action("drive_submitted_clean")
            if resp.rep_delta != 0 and rep_mgr:
                rep_mgr.apply_action(None, custom_delta=resp.rep_delta)
            if ach and ss and archive:
                rep_val = int(getattr(getattr(rep_mgr,"rep",None),"current",0)) if rep_mgr else 0
                try: ach.check_and_unlock(ss, archive, rep_val)
                except Exception: pass
            # Email from HQ
            em_sys = EmailSystem()
            em_sys.inject(
                "HQ <research@hq.aso>",
                f"RE: Drive {did[-8:]} received",
                f"Drive received.\nStatus: {resp.response_code.value}\n"
                f"Verified: {resp.verified_count} records\n"
                f"Bonus: +{bonus} pts\n\n{resp.bulletin}",
                urgent=(resp.response_code.value in ("PRIORITY_ESCALATE","REJECT_VOID_CONT")))
        except Exception as e:
            st.error(f"Submission error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BOOT SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def render_boot(B: Dict) -> None:
    def bl(cls, txt):
        return f'<div class="bl {cls}">{txt}</div>'

    mod_lines = ""
    for mn, _, _, lbl in BACKENDS:
        mod, err = B.get(mn,(None,""))
        ok = mod is not None
        mod_lines += bl(
            "bl-hi" if ok else "bl-er",
            f"  {lbl:<26} {'·····  OK' if ok else '·····  FAILED'}")

    st.markdown(f"""
<div class="boot-wrap">
  <div class="boot-title">vOiD</div>
  <div class="boot-sub1">VOICES&nbsp;&nbsp;OF&nbsp;&nbsp;THE&nbsp;&nbsp;VOID</div>
  <div class="boot-sub2">ALPEN&nbsp;SIGNAL&nbsp;OBSERVATORIUM&nbsp;·&nbsp;DUNKELTALER&nbsp;FOREST&nbsp;·&nbsp;SWITZERLAND&nbsp;·&nbsp;1840 m&nbsp;ASL</div>
  <hr class="boot-divider"/>
  <div class="boot-log">
    {bl("bl-acc","[ SYSTEM INIT v3.7.2 ]")}
    {bl("bl-dim","  Station  : ASO-DUNKELTALER-01")}
    {bl("bl-dim","  Operator : Dr. Kel")}
    {bl("bl-dim","  Coord    : 46.80°N  8.10°E  1840 m")}
    {bl("bl-dim","  H-I line : 1420.405 751 768 MHz")}
    {bl("bl-dim","")}
    {bl("bl-acc","[ HARDWARE ]")}
    {bl("bl-dim","  CPU  Intel Xeon W-2295 @ 3.0 GHz   ·····  OK")}
    {bl("bl-dim","  RAM  128 GB ECC DDR4                ·····  OK")}
    {bl("bl-dim","  GPU  RTX A5000 24 GB                ·····  OK")}
    {bl("bl-dim","  ADC  Ettus USRP X310 @ 2.0 MSPS    ·····  LOCKED")}
    {bl("bl-dim","  LNA  Cryo-cooler    T = 25.0 K      ·····  OK")}
    {bl("bl-dim","  DISH Az/El motors   enc drift 0.00° ·····  NOMINAL")}
    {bl("bl-dim","  UPLINK Satellite 128 kbps           ·····  CONNECTED")}
    {bl("bl-dim","  GEN   Generator     38.4 L          ·····  RUNNING")}
    {bl("bl-dim","  BAT   Battery bank  80%  16.0 kWh   ·····  OK")}
    {bl("bl-dim","")}
    {bl("bl-acc","[ BACKEND MODULES ]")}
    {mod_lines}
    {bl("bl-dim","")}
    {bl("bl-wr", "[ WARNING ]  Last anomaly detected at 03:33 local time.")}
    {bl("bl-wr", "[ WARNING ]  Drive DRV-A1F contains restricted-class signal.")}
    {bl("bl-wr", "[ WARNING ]  ARIRAL contact confirmed — valley sector — Day 8.")}
    {bl("bl-pu", "[ CLASSIFIED ]  VOID_CARRIER class protocol active.")}
    {bl("bl-dim","")}
    {bl("bl-gd", "[ ARIRAL REPUTATION ]  NEUTRAL  ·  Rep: 0")}
    {bl("bl-dim", "[ HQ COMMS ]  Uplink window: 22:00 – 23:59")}
    {bl("bl-dim","")}
    {bl("bl-hi", "[ SYSTEM READY ]")}
    {bl("bl-dim", "  Locate and process signals.  Filter them.")}
    {bl("bl-dim", "  Submit drives to supervisors with hash codes.")}
    {bl("bl-dim", "  Do not leave the base after 03:00.")}
    {bl("bl-dim","")}
    <div class="bl bl-hi">▋<span class="cursor">&nbsp;</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STATION OVERVIEW PAGE  ─  most sophisticated page: wires ALL 7 backends
# ─────────────────────────────────────────────────────────────────────────────

def render_overview(B: Dict) -> None:
    st.markdown('<div class="term-hdr">◈ STATION OVERVIEW — ALL SYSTEMS LIVE</div>',
                unsafe_allow_html=True)

    # ─── Row 0: Pipeline health ────────────────────────────────────────────
    pipe_html = ""
    for mn, _, icon, lbl in BACKENDS:
        ok = B.get(mn,(None,None))[0] is not None
        css = "pipe-node" if ok else "pipe-node-err"
        pipe_html += f'<span class="{css}">{icon}&nbsp;{lbl[:8]}</span>'
        if (mn, _, icon, lbl) != BACKENDS[-1]:
            pipe_html += '<span class="pipe-arrow">→</span>'
    st.markdown(f'<div style="margin:.15rem 0 .55rem;">{pipe_html}</div>',
                unsafe_allow_html=True)

    # ─── Row 1: Core metrics ───────────────────────────────────────────────
    sig_log  = _get("signal_log", [])
    an_recs  = _get("anomaly_records", [])
    dec_arch = _get("decoder_archive", [])
    pts      = _get("total_points", 0)
    day      = _get("day", 1)

    n_sig   = len(sig_log)
    n_anom  = len(an_recs)
    n_ariral= sum(1 for r in sig_log if "ARIRAL" in str(r.get("Class","")))
    n_void  = sum(1 for r in sig_log if "VOID" in str(r.get("Class","")))
    wow_max = max((float(r.get("WoW",0)) for r in sig_log), default=0.0)
    pkgs    = _get("submitted_packages",[])

    c = st.columns(8)
    c[0].metric("SIGNALS",    n_sig)
    c[1].metric("ANOMALIES",  n_anom)
    c[2].metric("ARIRAL",     n_ariral,
                delta="CONTACT" if n_ariral > 0 else None)
    c[3].metric("VOID EVTS",  n_void,
                delta="⚠" if n_void > 0 else None)
    c[4].metric("MAX WoW",    f"{wow_max:.2f}")
    c[5].metric("TOTAL PTS",  f"{pts:,}")
    c[6].metric("DRIVES SUB", len(pkgs))
    c[7].metric("DAY",        day)

    st.markdown("---")

    # ─── Row 2: Three columns ─────────────────────────────────────────────
    col_env, col_sigs, col_intel = st.columns([1.15, 1.25, 1.0])

    # ── Column A: Environment ──────────────────────────────────────────────
    with col_env:
        st.markdown('<div class="t-lbl">[ Environment & Station ]</div>',
                    unsafe_allow_html=True)

        dsh = _get("dish");  pwr = _get("power_system")
        slp = _get("sleep_rec"); wx = _get("current_weather")
        inv = _get("inventory"); sq = _get("sq_calc")
        em  = _mod("environment_monitor", B)

        if dsh:
            eff = dsh.effective_area_fraction
            st.markdown(
                f'<div class="ov-cell">'
                f'<div class="ov-cell-title">Dish Efficiency</div>'
                f'<div class="ov-cell-big">{eff*100:.0f}%</div>'
                f'{_mini_gauge(eff)}'
                f'<div class="ov-cell-sub">Status: {dsh.overall_status.value}'
                f' · AZ motor: {dsh.az_motor_health*100:.0f}%'
                f' · LNA NF: {dsh.lna_noise_figure_db:.1f} dB</div>'
                f'</div>', unsafe_allow_html=True)

        if pwr:
            bc = pwr.battery_fraction
            fl = pwr.generator_fuel_l / 50.0
            st.markdown(
                f'<div class="ov-cell">'
                f'<div class="ov-cell-title">Power System</div>'
                f'BATT {_mini_gauge(bc)} {pwr.battery_fraction*100:.0f}%<br>'
                f'FUEL {_mini_gauge(fl)} {pwr.generator_fuel_l:.0f} L<br>'
                f'<div class="ov-cell-sub">Source: {pwr.active_source.value}'
                f' · Solar: {pwr.solar_power_kw:.2f} kW</div>'
                f'</div>', unsafe_allow_html=True)

        if slp:
            aw = slp.hours_awake
            fatigue_col = (
                "var(--dng)" if aw > 36 else
                "var(--warn)" if aw > 20 else "var(--fg)")
            st.markdown(
                f'<div class="ov-cell">'
                f'<div class="ov-cell-title">Operator Fatigue</div>'
                f'{_mini_gauge(max(0, 1 - aw/48))}'
                f'<div class="ov-cell-sub" style="color:{fatigue_col}">'
                f'{slp.fatigue_level.value} — {aw:.1f}h awake'
                f' · debt: {slp.sleep_debt_h:.1f}h'
                f' · hall: {slp.hallucination_probability*100:.0f}%</div>'
                f'</div>', unsafe_allow_html=True)

        if wx:
            rq = wx.radio_seeing_factor
            atm = wx.atmospheric_noise_k
            
            # Extract conditional HTML to avoid backslashes inside the f-string expression
            lightning_html = '&nbsp;<span style="color:var(--dng)">⚡ LIGHTNING</span>' if wx.lightning_active else ''
            
            st.markdown(
                f'<div class="ov-cell">'
                f'<div class="ov-cell-title">Weather & Atmosphere</div>'
                f'<span style="color:var(--acc)">{wx.state.value}</span>'
                f'  {wx.temperature_c:.1f}°C · wind {wx.wind_speed_ms:.1f} m/s<br>'
                f'RADIO-Q {_mini_gauge(rq)} {rq:.2f}'
                f'{lightning_html}'
                f'<div class="ov-cell-sub">Atm noise: {atm:.1f} K'
                f' · Precip: {wx.precipitation_mm_h:.1f} mm/h'
                f' · Vis: {wx.visibility_km:.1f} km</div>'
                f'</div>', unsafe_allow_html=True)

        # Signal quality composite
        if sq and dsh and wx and pwr and slp and em:
            try:
                from environment_monitor import AstronomicalCalc
                doy = int(_get("env_doy", 355))
                el  = AstronomicalCalc.solar_elevation(46.8, 8.1,
                        _get("ingame_hour",12.0), doy)
                q   = sq.compute(dsh, wx, pwr, slp, 45.0, 1420.4, doy,
                                  _get("ingame_hour",12.0))
                comp = q.get("composite", 0)
                pen  = q.get("snr_penalty_db", 0)
                st.markdown(
                    f'<div class="ov-cell">'
                    f'<div class="ov-cell-title">Signal Quality Composite</div>'
                    f'{_mini_gauge(comp)}'
                    f'<div class="ov-cell-sub">'
                    f'Q={comp:.3f} · SNR penalty {pen:.1f} dB'
                    f' · T_sys {q.get("t_sys_effective_k",0):.0f} K'
                    f' · Iono {q.get("ionospheric_quality",1):.2f}</div>'
                    f'</div>', unsafe_allow_html=True)
            except Exception:
                pass

        # Inventory shortages
        if inv:
            shortages = inv.critical_shortages()
            if shortages:
                for s in shortages:
                    st.markdown(
                        f'<div style="font-family:var(--mono);font-size:.60rem;'
                        f'color:var(--dng);border:1px solid rgba(255,32,32,.35);'
                        f'padding:.15rem .42rem;margin:.10rem 0;'
                        f'background:rgba(24,0,0,.55);backdrop-filter:blur(8px);">⚠ {s}</div>',
                        unsafe_allow_html=True)

    # ── Column B: Signal intelligence ──────────────────────────────────────
    with col_sigs:
        st.markdown('<div class="t-lbl">[ Signal Intelligence ]</div>',
                    unsafe_allow_html=True)

        # WoW sparkline
        sig_hist = _get("_sig_history", [])
        if sig_hist:
            spark_str = _spark(sig_hist, 30)
            pts_hist  = _get("_pts_history", [])
            pts_spark = _spark(pts_hist, 30) if pts_hist else "─"*30
            st.markdown(
                f'<div class="data-block">'
                f'WoW hist: <span style="color:var(--gold)">{spark_str}</span>  max={wow_max:.2f}<br>'
                f'Pts hist: <span style="color:var(--acc)">{pts_spark}</span>  total={pts:,}'
                f'</div>', unsafe_allow_html=True)

        # Recent 10 signals
        if sig_log:
            rows_html = ""
            for row in reversed(sig_log[-10:]):
                cls = str(row.get("Class","?"))
                wow = float(row.get("WoW",0))
                snr = float(row.get("SNR(dB)", row.get("snr_db",0)))
                thr = str(row.get("Threat","?"))
                uid = str(row.get("UID","?"))[-6:]
                wow_css = "sig-wow-hi" if wow>=8 else "sig-wow-md" if wow>=5 else "sig-wow-lo"
                rows_html += (
                    f'<div class="sig-row">'
                    f'<span style="color:rgba(0,255,136,.30);min-width:50px">{uid}</span>'
                    f'<span class="sig-class">{cls[:14]}</span>'
                    f'<span>SNR:<span style="color:var(--fg)">{snr:.0f}dB</span></span>'
                    f'<span>WoW:<span class="{wow_css}">{wow:.1f}</span></span>'
                    f'<span style="color:rgba(0,255,136,.30)">{thr[:4]}</span>'
                    f'</div>')
            st.markdown(
                f'<div class="t-lbl">Recent Signals (last 10)</div>'
                f'<div style="background:rgba(0,0,0,.38);border:1px solid rgba(0,255,136,.08);'
                f'border-radius:2px;padding:.12rem .25rem;max-height:220px;overflow-y:auto;">'
                f'{rows_html}</div>',
                unsafe_allow_html=True)

        # Class breakdown
        if sig_log:
            cls_counts = Counter(str(r.get("Class","?")) for r in sig_log)
            st.markdown('<div class="t-lbl">Signal Class Breakdown</div>',
                        unsafe_allow_html=True)
            breakdown_html = ""
            total = len(sig_log)
            for cls_name, cnt in cls_counts.most_common(8):
                frac = cnt / total
                col_b = ("var(--gold)" if "ARIRAL" in cls_name else
                          "var(--dng)" if "VOID" in cls_name else
                          "var(--warn)" if "ANOMALOUS" in cls_name else
                          "var(--acc)")
                breakdown_html += (
                    f'<div style="display:flex;align-items:center;gap:.40rem;'
                    f'font-family:var(--mono);font-size:.60rem;margin:.07rem 0;">'
                    f'<span style="color:{col_b};min-width:120px">{cls_name[:15]}</span>'
                    f'<span style="flex:1;background:rgba(0,0,0,.35);height:8px;border-radius:1px;">'
                    f'<span style="display:block;width:{frac*100:.0f}%;height:100%;'
                    f'background:{col_b};border-radius:1px;"></span></span>'
                    f'<span style="color:rgba(0,255,136,.45);min-width:30px">{cnt}</span>'
                    f'</div>')
            st.markdown(
                f'<div style="background:rgba(0,0,0,.35);border:1px solid rgba(0,255,136,.07);'
                f'padding:.30rem .42rem;border-radius:2px;">{breakdown_html}</div>',
                unsafe_allow_html=True)

        # ARIRAL rep tiers
        st.markdown('<div class="t-lbl">ARIRAL Reputation</div>', unsafe_allow_html=True)
        rep_mgr = _get("rep_manager")
        rep_val = int(getattr(getattr(rep_mgr,"rep",None),"current",0)) if rep_mgr else 0
        tiers = [(-100,-50,"MEAN","var(--dng)"),
                 (-50, -10,"INCONVENIENT","var(--warn)"),
                 (-10,  10,"NEUTRAL","rgba(0,255,136,.35)"),
                 ( 10,  50,"GOOD","var(--fg2)"),
                 ( 50, 100,"LOYAL","var(--gold)")]
        tier_html = ""
        for lo,hi,name,col in tiers:
            active = lo <= rep_val < hi
            bg = col if active else "rgba(0,0,0,.25)"
            fc = "rgba(0,0,0,.8)" if active else "rgba(0,255,136,.20)"
            tier_html += (f'<div style="flex:1;text-align:center;background:{bg};'
                          f'color:{fc};font-family:var(--mono);font-size:.50rem;'
                          f'padding:.14rem .05rem;border:1px solid rgba(0,255,136,.08);">{name}</div>')
        st.markdown(
            f'<div style="display:flex;gap:1px;margin:.18rem 0 .10rem;">{tier_html}</div>'
            f'<div style="font-family:var(--mono);font-size:.65rem;color:var(--acc);'
            f'text-align:center;text-shadow:0 0 8px rgba(0,255,204,.35);">{rep_val:+d}</div>',
            unsafe_allow_html=True)

    # ── Column C: Crypto + HQ intel ────────────────────────────────────────
    with col_intel:
        st.markdown('<div class="t-lbl">[ Crypto / ARIRAL / HQ ]</div>',
                    unsafe_allow_html=True)

        # Latest ARIRAL message
        ariral_msgs = [m for m in dec_arch
                       if hasattr(m,"ariral_phrase") and m.ariral_phrase]
        if ariral_msgs:
            ph = ariral_msgs[-1].ariral_phrase
            syms = " ".join(getattr(ariral_msgs[-1],"ariral_symbols",["◈"])[:6])
            st.markdown(f'<div class="ariral-box">◈ {ph[:70]}<br>'
                        f'<span style="font-size:.60rem;opacity:.65">{syms[:50]}</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="ariral-box-dim">◈ NO ARIRAL COMMS DECODED</div>',
                        unsafe_allow_html=True)

        # Crypto stats
        n_dec = len(dec_arch)
        n_ariral_d = sum(1 for m in dec_arch if "ARIRAL" in str(getattr(m,"message_class","")))
        n_nat = sum(1 for m in dec_arch if "NATURAL" in str(getattr(m,"message_class","")))
        best_c = max((getattr(m,"confidence_score",0) for m in dec_arch), default=0.0)
        st.markdown(
            f'<div class="ov-cell">'
            f'<div class="ov-cell-title">Decoded Messages</div>'
            f'<div class="ov-cell-big">{n_dec}</div>'
            f'<div class="ov-cell-sub">ARIRAL:{n_ariral_d} · Natural:{n_nat}'
            f' · Best conf:{best_c*100:.0f}%</div>'
            f'</div>', unsafe_allow_html=True)

        # Drive status
        dm = _get("drive_manager")
        if dm:
            n_drives = len(dm.drives)
            active_d = dm.active_drive
            st.markdown(
                f'<div class="ov-cell">'
                f'<div class="ov-cell-title">Drive Manager</div>'
                f'<div class="ov-cell-big">{n_drives}</div>'
                f'<div class="ov-cell-sub">drives loaded · '
                f'Active: {active_d.drive_id[-8:] if active_d else "─"}'
                f' {active_d.fill_fraction*100:.0f}% full' if active_d else "─"
                f'</div></div>', unsafe_allow_html=True)

        # HQ submission history
        pkgs = _get("submitted_packages", [])
        if pkgs:
            last_pkg = pkgs[-1]
            last_status = getattr(last_pkg,"status",None)
            last_pts    = getattr(last_pkg,"total_points",0)
            st.markdown(
                f'<div class="ov-cell">'
                f'<div class="ov-cell-title">Last HQ Submission</div>'
                f'<span style="color:var(--fg);font-family:var(--mono);font-size:.68rem">'
                f'{getattr(last_pkg,"drive_id","?")[-8:]}</span><br>'
                f'<div class="ov-cell-sub">'
                f'{getattr(last_status,"value","?")}'
                f' · {getattr(last_pkg,"total_signals",0)} sigs'
                f' · {last_pts:,} pts</div>'
                f'</div>', unsafe_allow_html=True)

        # ML model status
        pred = _get("master_predictor")
        ml_t = getattr(pred,"_trained",False) if pred else False
        n_ver = len(getattr(pred,"versions",[])) if pred else 0
        if n_ver > 0 and pred:
            last_ver = pred.versions[-1]
            f1  = getattr(last_ver,"cv_f1_weighted",0)
            auc = getattr(last_ver,"cv_roc_auc",0)
        else:
            f1 = auc = 0.0
        st.markdown(
            f'<div class="ov-cell">'
            f'<div class="ov-cell-title">ML Prediction Engine</div>'
            f'<span style="color:{"var(--fg)" if ml_t else "var(--warn)"};">'
            f'{"TRAINED" if ml_t else "UNTRAINED"}</span>'
            f'<div class="ov-cell-sub">Versions:{n_ver}'
            f'{"  F1:"+str(round(f1,3)) if f1 else ""}{"  AUC:"+str(round(auc,3)) if auc else ""}'
            f'</div></div>', unsafe_allow_html=True)

        # Achievements
        ach = _get("achievements")
        if ach:
            unlocked = sum(1 for a in ach._achievements.values() if a.unlocked)
            total    = len(ach._achievements)
            ach_pts  = ach.total_achievement_points()
            st.markdown(
                f'<div class="ov-cell">'
                f'<div class="ov-cell-title">Achievements</div>'
                f'<div class="ov-cell-big">{unlocked}/{total}</div>'
                f'<div class="ov-cell-sub">Achievement pts: {ach_pts:,}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ─── Row 3: Quick-action bar ───────────────────────────────────────────
    st.markdown('<div class="t-lbl">[ Quick Actions ]</div>', unsafe_allow_html=True)
    qa_cols = st.columns(7)
    qa_actions = [
        ("📡 ACQUIRE",      lambda: _quick_acquire(B),         "qa_ov_acq"),
        ("🌊 SPECTRAL",     lambda: _quick_acquire(B, "NARROWBAND_CW"), "qa_ov_spec"),
        ("👁 ANOMALOUS",    lambda: _quick_acquire(B, "ANOMALOUS"), "qa_ov_an"),
        ("⬡ +1 HOUR",      lambda: _advance_time(B, 1.0),     "qa_ov_1h"),
        ("💤 SLEEP 8H",     lambda: _sleep_action(),           "qa_ov_sleep"),
        ("📤 SEAL DRIVE",   lambda: _seal_and_submit(B),       "qa_ov_seal"),
        ("🌑 NEXT DAY",     lambda: _advance_day(B),           "qa_ov_day"),
    ]
    for col, (lbl, fn, key) in zip(qa_cols, qa_actions):
        with col:
            if st.button(lbl, use_container_width=True, key=key):
                result = fn()
                if result:
                    st.toast(str(result)[:60])
                else:
                    st.toast("Done")

    # ─── Row 4: Station log ────────────────────────────────────────────────
    st.markdown("---")
    col_log, col_evts = st.columns([1.4, 1])

    with col_log:
        st.markdown('<div class="t-lbl">[ Station Log — Recent Entries ]</div>',
                    unsafe_allow_html=True)
        slog = _get("station_log")
        if slog and hasattr(slog, "recent"):
            df = slog.recent(12)
            if not df.empty:
                log_html = ""
                for _, row in df.iterrows():
                    sev = str(row.get("severity","INFO"))
                    msg = str(row.get("message",""))[:85]
                    cat = str(row.get("category",""))[:6]
                    ts  = str(row.get("game_time",""))
                    sev_map = {"HORROR":"c-dng","CRITICAL":"c-dng","WARNING":"c-warn","INFO":"c-dim"}
                    sc = sev_map.get(sev,"c-dim")
                    log_html += (
                        f'<div class="log-ln {sev[:1]}">'
                        f'<span class="c-dim" style="min-width:38px">{ts}</span>'
                        f'<span class="{sc}" style="min-width:26px">[{sev[:1]}]</span>'
                        f'<span class="c-dim" style="min-width:36px">{cat}</span>'
                        f'<span style="color:rgba(0,255,136,.55)">{msg}</span>'
                        f'</div>')
                st.markdown(
                    f'<div style="background:rgba(0,0,0,.38);border:1px solid rgba(0,255,136,.07);'
                    f'border-radius:2px;padding:.12rem .25rem;max-height:220px;overflow-y:auto;">'
                    f'{log_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="t-lbl">No log entries yet.</div>', unsafe_allow_html=True)

    with col_evts:
        st.markdown('<div class="t-lbl">[ Recent Events ]</div>', unsafe_allow_html=True)
        evlog = _get("_event_log", [])
        if evlog:
            ev_html = ""
            for ev in reversed(evlog[-8:]):
                sev  = getattr(ev,"severity",None)
                title= getattr(ev,"title","?")[:45]
                cat  = getattr(ev,"category","?")[:8]
                sev_name = sev.name if sev and hasattr(sev,"name") else "INFO"
                sev_col  = ("var(--dng)" if sev_name in ("HORROR","CRITICAL")
                             else "var(--warn)" if sev_name == "WARNING"
                             else "rgba(0,255,136,.40)")
                ev_html += (
                    f'<div style="font-family:var(--mono);font-size:.60rem;'
                    f'border-bottom:1px solid rgba(0,255,136,.04);'
                    f'padding:.10rem .14rem;display:flex;gap:.42rem;">'
                    f'<span style="color:{sev_col};min-width:44px">[{sev_name[:4]}]</span>'
                    f'<span style="color:rgba(0,255,136,.32);min-width:52px">{cat}</span>'
                    f'<span style="color:rgba(0,255,136,.52)">{title}</span>'
                    f'</div>')
            st.markdown(
                f'<div style="background:rgba(0,0,0,.38);border:1px solid rgba(0,255,136,.07);'
                f'border-radius:2px;padding:.10rem .20rem;max-height:220px;overflow-y:auto;">'
                f'{ev_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="t-lbl">No events yet. Advance time to generate.</div>',
                        unsafe_allow_html=True)


def _sleep_action() -> str:
    slp = _get("sleep_rec")
    if slp:
        q = slp.sleep(8.0)
        inv = _get("inventory")
        if inv: inv.consume("food_rations", 1)
        return f"Slept 8h | Quality {q:.2f} | {slp.fatigue_level.value}"
    return "No sleep module"


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_emails() -> None:
    st.markdown('<div class="term-hdr">📬 EMAILS / HQ COMMUNICATIONS</div>',
                unsafe_allow_html=True)
    em_sys = EmailSystem()
    uc = em_sys.unread_count()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown(f'<div class="t-lbl">{uc} unread · '
                    f'{len(em_sys.visible_emails())} total visible</div>',
                    unsafe_allow_html=True)
        em_sys.render()

    with col_right:
        st.markdown('<div class="t-lbl">[ Compose Message to HQ ]</div>',
                    unsafe_allow_html=True)
        subject = st.text_input("Subject", "Status report")
        body    = st.text_area("Body", height=100,
                                placeholder="Enter your message to HQ…")
        if st.button("⬡ SEND TO HQ", use_container_width=True):
            if subject and body:
                em_sys.inject("Dr. Kel <kel@aso-dkl.ch>", subject, body)
                # Mock HQ auto-reply
                em_sys.inject("HQ <research@hq.aso>",
                               f"RE: {subject}",
                               f"Message received.\nWe will review your report.\n\n— Director",
                               urgent=False)
                st.success("Message sent. HQ auto-reply received.")

        st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)
        st.markdown('<div class="t-lbl">[ HQ Drive Requests ]</div>', unsafe_allow_html=True)

        dm = _get("drive_manager")
        if dm:
            pkgs = _get("submitted_packages",[])
            if pkgs:
                last = pkgs[-1]
                st.markdown(
                    f'<div class="data-block">'
                    f'Last drive : {getattr(last,"drive_id","?")}\n'
                    f'Hash       : {getattr(last,"drive_hash","─")[:24]}\n'
                    f'Status     : {getattr(getattr(last,"status",None),"value","?")}\n'
                    f'Signals    : {getattr(last,"total_signals",0)}\n'
                    f'Points     : {getattr(last,"total_points",0):,}\n'
                    f'HQ code    : {getattr(last,"hq_response","─") or "─"}'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="t-lbl">No drives submitted yet.</div>',
                            unsafe_allow_html=True)

        # Satellite hash codes (mock)
        st.markdown('<div class="t-lbl">[ Satellite Hash Codes ]</div>', unsafe_allow_html=True)
        import hashlib
        day = _get("day", 1)
        sats = [("SIERRA","SRA"), ("ROMEO","ROM"), ("ALPHA","ALP")]
        for sat_name, prefix in sats:
            h = hashlib.sha256(f"{sat_name}{day}".encode()).hexdigest()[:12].upper()
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:.62rem;'
                f'display:flex;justify-content:space-between;'
                f'border-bottom:1px solid rgba(0,255,136,.06);padding:.08rem .10rem;">'
                f'<span style="color:rgba(0,255,136,.45)">{sat_name}</span>'
                f'<span style="color:var(--acc)">{h}</span>'
                f'</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EVENT JOURNAL PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_event_journal() -> None:
    st.markdown('<div class="term-hdr">📖 EVENT JOURNAL — Dr. Kel\'s Log</div>',
                unsafe_allow_html=True)

    evlog   = _get("_event_log", [])
    sig_log = _get("signal_log", [])
    day     = _get("day", 1)

    col_j, col_ref = st.columns([1.6, 1])

    with col_j:
        # Journal narrative
        st.markdown('<div class="t-lbl">[ Observation Journal ]</div>', unsafe_allow_html=True)
        n_sig   = len(sig_log)
        n_anom  = sum(1 for r in sig_log if "ANOMALOUS" in str(r.get("Class","")))
        n_ariral= sum(1 for r in sig_log if "ARIRAL" in str(r.get("Class","")))
        wow_max = max((float(r.get("WoW",0)) for r in sig_log), default=0.0)
        rep_mgr = _get("rep_manager")
        rep_val = int(getattr(getattr(rep_mgr,"rep",None),"current",0)) if rep_mgr else 0

        journal_text = f"""Day {day:03d} — Dunkeltaler Forest Observatory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total signals acquired this session : {n_sig}
Anomalous signals detected          : {n_anom}
ARIRAL communications recorded      : {n_ariral}
Highest WoW score                   : {wow_max:.3f}
ARIRAL reputation                   : {rep_val:+d} ({_rep_state(rep_val)})
Total points                        : {_get("total_points",0):,}

"""
        if day >= 8:
            journal_text += "Day 8 note: Two warp signatures in the valley.\n"
            journal_text += "The picnic is still there. I have not approached it.\n\n"
        if n_ariral > 0:
            journal_text += f"The signals are structured. Prime-encoded. {n_ariral} logged.\n"
            journal_text += "They know we're listening.\n\n"
        if wow_max >= 7.0:
            journal_text += f"WoW {wow_max:.2f} — this is extraordinary.\n"
            journal_text += "I've submitted the drive hash. HQ will want to see this.\n\n"

        slp = _get("sleep_rec")
        if slp and slp.hours_awake > 28:
            journal_text += "I haven't slept in a long time.\n"
            journal_text += "The trees outside are moving differently.\n"
            journal_text += "I should sleep.\n\n"

        journal_text += "The array is running. The sky is quiet.\n"
        journal_text += "Something will come through tonight.\n"
        journal_text += "It always does."

        st.markdown(f'<div class="data-block">{journal_text}</div>', unsafe_allow_html=True)

        # Event log full table
        st.markdown('<div class="t-lbl">[ All Events This Session ]</div>', unsafe_allow_html=True)
        if evlog:
            ev_rows = []
            for ev in evlog:
                ev_rows.append({
                    "Category": getattr(ev,"category","?"),
                    "Title":    getattr(ev,"title","?")[:40],
                    "Severity": getattr(getattr(ev,"severity",None),"name","?"),
                    "Duration": f"{getattr(ev,'duration_h',0):.1f}h",
                    "Resolved": "✓" if getattr(ev,"resolved",False) else "—",
                })
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="t-lbl">No events yet. Advance time to generate random events.</div>',
                        unsafe_allow_html=True)

    with col_ref:
        # VotV story reference panel
        st.markdown('<div class="t-lbl">[ Story Reference ]</div>', unsafe_allow_html=True)

        story_events = [
            (1,  "Begin signal monitoring at ASO-DKL-01"),
            (3,  "HQ requests drive hash for SIERRA"),
            (5,  "PROTOCOL 7-OMEGA briefing received"),
            (8,  "ARIRAL picnic detected — valley sector"),
            (8,  "Servers SIERRA + ROMEO offline 01:45"),
            (10, "ARIRAL ships depart valley"),
            (13, "Emails from 'user' — subject: blank"),
            (14, "Look away from radar — something appears"),
            (22, "Two water guns found in treehouse"),
        ]
        for d, desc in story_events:
            active = day >= d
            css = "c-ok" if active else "c-dim"
            sym = "✓" if active else "○"
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:.60rem;'
                f'display:flex;gap:.45rem;padding:.08rem .10rem;'
                f'border-bottom:1px solid rgba(0,255,136,.04);">'
                f'<span class="{css}">{sym}</span>'
                f'<span class="c-dim" style="min-width:36px">D{d:02d}</span>'
                f'<span style="color:{_PAL["fg"] if active else "rgba(0,255,136,.30)"}">{desc}</span>'
                f'</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)
        st.markdown('<div class="t-lbl">[ Entity Reference ]</div>', unsafe_allow_html=True)

        entities = [
            ("ARIRAL",    "var(--gold)", "Benign ET · shrimp · tetrahedron ships"),
            ("INSOMNIAC", "var(--warn)", "Sleep entity · dream events"),
            ("RUFUS",     "var(--dng)",  "Hostile · broadband chaos signals"),
            ("LOOKER",    "var(--purp)", "Sequential prime signals · do not miss"),
            ("BAD SUN",   "var(--dng)",  "Day 24 · evacuate · sun expands"),
            ("THE END",   "var(--dng)",  "VOID_CARRIER class · game over"),
        ]
        for name, col, desc in entities:
            st.markdown(
                f'<div style="font-family:var(--mono);font-size:.59rem;'
                f'padding:.10rem .12rem;border-bottom:1px solid rgba(0,255,136,.04);">'
                f'<span style="color:{col};font-weight:bold">{name}</span>'
                f'<span style="color:rgba(0,255,136,.38)"> — {desc}</span>'
                f'</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)
        st.markdown('<div class="t-lbl">[ Inventory Status ]</div>', unsafe_allow_html=True)
        inv = _get("inventory")
        if inv:
            inv_d = inv.to_dict()
            for k, v in inv_d.items():
                if isinstance(v, (int, float)) and v > 0:
                    css = ("c-warn" if (isinstance(v,int) and v < 3 and
                            k in ("food_rations","blank_drives")) else "c-ok")
                    st.markdown(
                        f'<div class="telem-row">'
                        f'<span>{k.replace("_"," ").title()[:20]}</span>'
                        f'<span class="{css}">{v}</span>'
                        f'</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ERROR PAGE  ─  shown when a backend fails to import
# ─────────────────────────────────────────────────────────────────────────────

def render_module_error(mod_name: str, err: str) -> None:
    st.markdown(f"""
    <div style="background:rgba(20,0,0,0.80);
                border:2px solid var(--dng);
                border-radius:2px;
                padding:.80rem 1.0rem;
                font-family:var(--mono);
                font-size:.68rem;
                color:var(--dng);
                margin:.80rem 0;
                backdrop-filter:blur(12px);">
      ⚠  MODULE LOAD FAILURE<br><br>
      Module : <span style="color:var(--acc)">{mod_name}</span><br><br>
      <pre style="background:rgba(0,0,0,0.55);color:#ff6666;
                  padding:.40rem;font-size:.59rem;
                  white-space:pre-wrap;border-radius:2px;">{err[:800]}</pre>
      <br>Verify all .py backend files are in the same directory as vOiD.py.<br>
      Verify requirements.txt is installed on Streamlit Cloud.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def route(page_key: str, B: Dict) -> None:
    if page_key == "BOOT":
        render_boot(B); return
    if page_key == "OVERVIEW":
        render_overview(B); return
    if page_key == "EMAILS":
        render_emails(); return
    if page_key == "EVENTS":
        render_event_journal(); return

    # Map to backend
    lookup = {mn.upper(): (mn, fn) for mn, fn, _, _ in BACKENDS}
    if page_key not in lookup:
        render_boot(B); return

    mod_name, fn_name = lookup[page_key]
    mod, err = B.get(mod_name, (None, "not loaded"))
    if mod is None:
        render_module_error(mod_name, err or "unknown error"); return

    page_fn = getattr(mod, fn_name, None)
    if page_fn is None:
        render_module_error(mod_name, f"function '{fn_name}' not found"); return

    # Pre-call state init
    if hasattr(mod, "init_session_state"):
        mod.init_session_state()

    page_fn()


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB GLOBAL STYLE  ─  all backend plots use this
# ─────────────────────────────────────────────────────────────────────────────

def _apply_mpl_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   "#030d07",
        "axes.facecolor":     "#040f08",
        "axes.edgecolor":     "#0d2e1a",
        "axes.labelcolor":    "#00ff88",
        "axes.grid":          True,
        "grid.color":         "#071408",
        "grid.linestyle":     ":",
        "grid.alpha":         0.50,
        "xtick.color":        "#335544",
        "ytick.color":        "#335544",
        "xtick.labelsize":    6,
        "ytick.labelsize":    6,
        "axes.labelsize":     7,
        "axes.titlesize":     8,
        "axes.titlecolor":    "#00ff88",
        "text.color":         "#00ff88",
        "font.family":        "monospace",
        "legend.facecolor":   "#040f08",
        "legend.edgecolor":   "#0d2e1a",
        "legend.labelcolor":  "#00ff88",
        "legend.fontsize":    6,
        "figure.dpi":         110,
        "savefig.facecolor":  "#030d07",
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Inject CSS (background + glass morphism + all custom components)
    st.markdown(_build_css(_BG_URI), unsafe_allow_html=True)

    # 2. Apply matplotlib dark style for all backend plots
    _apply_mpl_style()

    # 3. Load all 7 backends (cached per session)
    B = _load_backends()

    # 4. Bootstrap all shared state
    bootstrap(B)

    # 5. Cross-module signal harvest (lightweight, every rerender)
    _harvest(B)

    # 6. Sidebar → returns selected page key
    page_key = render_sidebar(B)

    # 7. Status bar (persistent across all pages)
    render_status_bar(B)

    # 8. Dispatch to selected page
    route(page_key, B)


if __name__ == "__main__":
    main()
