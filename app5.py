import requests
import datetime as _dt
import base64
import os
import html
import re
import time
from model import MOILManganeseAI
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MOIL Predictive Intelligence",
    layout="wide",
    page_icon="image.png",
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION FOR OFFLINE SYNC
# -----------------------------------------------------------------------------
if 'offline_queue' not in st.session_state:
    st.session_state['offline_queue'] = []
if 'last_synced_data' not in st.session_state:
    st.session_state['last_synced_data'] = {
        'rain': 15.0, 'downtime': 3.5, 'blast': 1.0, 'wagon': 0.95}

# -----------------------------------------------------------------------------
# THEME — dark SaaS / terminal aesthetic
# -----------------------------------------------------------------------------
ACCENT = "#10b981"
ACCENT_SOFT = "rgba(16, 185, 129, 0.12)"
BG = "#08090a"
BG_CARD = "#111214"
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#e6e6e6"
TEXT_MUTED = "#8b8f97"
PICKAXE_CURSOR_B64 = "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"  # user-supplied pickaxe artwork, bg removed

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {BG};
    color: {TEXT};
}}

[data-testid="stHeader"] {{
    background-color: transparent;
}}
.block-container {{
    padding-top: 2.5rem;
    max-width: 1200px;
}}

.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.35rem;
}}
.eyebrow::before {{
    content: "// ";
    opacity: 0.6;
}}

.section-title {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: {TEXT};
    margin: 0 0 0.5rem 0;
    line-height: 1.15;
}}
.section-sub {{
    color: {TEXT_MUTED};
    font-size: 0.98rem;
    max-width: 720px;
    margin-bottom: 1.25rem;
}}

.hero-title {{
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}}
.hero-title span {{
    color: {ACCENT};
}}
.hero-sub {{
    color: {TEXT_MUTED};
    font-size: 1.05rem;
    max-width: 620px;
    margin-bottom: 1.5rem;
}}

/* Standalone single-call cards (e.g. the insight box) */
.moil-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem;
    transition: border-color 0.2s ease;
}}
.moil-card:hover {{
    border-color: {ACCENT};
}}

.stat-label {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 2.5rem 0;
}}

/* ---------- Dial-style sliders ---------- */
[data-testid="stSlider"] {{
    padding-top: 0.5rem;
    padding-bottom: 0.9rem;
}}
[data-testid="stSlider"] [data-baseweb="slider"] {{
    position: relative;
    padding: 34px 0 26px;
    --tick-top: 20px;      /* nudge these if ticks don't line up with your track */
    --tick-bottom: 14px;
}}

/* tick rows: above (::before) and below (::after) the track.
   major tick every 25%, minor every 5%, plus an end cap at 100% */
[data-testid="stSlider"] [data-baseweb="slider"]::before,
[data-testid="stSlider"] [data-baseweb="slider"]::after {{
    content: "";
    position: absolute;
    left: 0; right: 0;
    height: 10px;
    pointer-events: none;
}}
[data-testid="stSlider"] [data-baseweb="slider"]::before {{
    top: var(--tick-top);
    background:
        linear-gradient(270deg, rgba(255,255,255,0.55) 0 2px, transparent 2px) 100% 0 / 2px 100% no-repeat,
        linear-gradient(90deg, rgba(255,255,255,0.55) 0 2px, transparent 2px) 0 100% / 25% 100% repeat-x,
        linear-gradient(90deg, rgba(255,255,255,0.22) 0 1px, transparent 1px) 0 100% / 5% 55% repeat-x;
}}
[data-testid="stSlider"] [data-baseweb="slider"]::after {{
    bottom: var(--tick-bottom);
    background:
        linear-gradient(270deg, rgba(255,255,255,0.55) 0 2px, transparent 2px) 100% 0 / 2px 100% no-repeat,
        linear-gradient(90deg, rgba(255,255,255,0.55) 0 2px, transparent 2px) 0 0 / 25% 100% repeat-x,
        linear-gradient(90deg, rgba(255,255,255,0.22) 0 1px, transparent 1px) 0 0 / 5% 55% repeat-x;
}}

/* track: thicker and rounded. The fill colour comes from primaryColor (see step 2),
   so it is not overridden here */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {{
    height: 14px !important;
    border-radius: 999px !important;
    box-shadow:
        inset 0 2px 4px rgba(0,0,0,0.6),
        0 0 0 1px rgba(255,255,255,0.08) !important;
}}

/* knob: metallic ring with an orange centre */
[data-testid="stSlider"] [role="slider"] {{
    width: 34px !important;
    height: 34px !important;
    border-radius: 50% !important;
    background:
        radial-gradient(circle at 50% 50%, #ff7a00 0 24%, transparent 26%),
        radial-gradient(circle at 50% 50%, #f3f4f6 0 38%, #b3b9c2 64%, #eceef1 100%) !important;
    border: 2px solid #8d949e !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.55), 0 0 0 0 rgba(249,115,22,0) !important;
    transition: box-shadow 0.2s ease;
}}
/* oversized invisible grab area, so the knob is easy to catch */
[data-testid="stSlider"] [role="slider"]::after {{
    content: "";
    position: absolute;
    inset: -10px;
    border-radius: 50%;
}}
[data-testid="stSlider"] [role="slider"]:hover,
[data-testid="stSlider"] [role="slider"]:focus,
[data-testid="stSlider"] [role="slider"]:active {{
    box-shadow: 0 4px 10px rgba(0,0,0,0.55), 0 0 0 6px rgba(249,115,22,0.28) !important;
    outline: none !important;
}}

/* value bubble above the knob */
[data-testid="stThumbValue"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: #fff !important;
    background: #f97316;
    padding: 2px 8px;
    border-radius: 6px;
}}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: {TEXT_MUTED} !important;
}}

/* ---------- Shortfall predictor panel ---------- */
.status-pill {{
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.3rem 0.8rem; border-radius: 999px; border: 1px solid;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.06em; text-transform: uppercase;
}}
.status-pill.online  {{ color: #34d399; background: rgba(16,185,129,0.10); border-color: rgba(16,185,129,0.35); }}
.status-pill.offline {{ color: #f87171; background: rgba(239,68,68,0.10);  border-color: rgba(239,68,68,0.35); }}
.pulse-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: currentColor; animation: pulse 1.8s ease-in-out infinite;
}}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} }}

.tile-row {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem; margin: 0.9rem 0 0.4rem;
}}
.tile {{
    background: rgba(255,255,255,0.03); border: 1px solid {BORDER};
    border-radius: 10px; padding: 0.6rem 0.75rem;
}}
.tile .k {{ color: {TEXT_MUTED}; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.tile .v {{ font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600; color: {TEXT}; }}
.tile .u {{ font-size: 0.7rem; color: {TEXT_MUTED}; margin-left: 0.2rem; }}

.verdict {{
    display: flex; align-items: center; gap: 0.9rem;
    padding: 0.9rem 1.1rem; border-radius: 12px; border: 1px solid;
    margin: 0.4rem 0 1.2rem;
}}
.verdict.ok  {{ background: linear-gradient(90deg, rgba(16,185,129,0.18), rgba(16,185,129,0.03)); border-color: rgba(16,185,129,0.4); }}
.verdict.bad {{ background: linear-gradient(90deg, rgba(239,68,68,0.20),  rgba(239,68,68,0.03));  border-color: rgba(239,68,68,0.45); }}
.verdict .ico {{ font-size: 1.5rem; }}
.verdict .t   {{ font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; font-size: 0.95rem; }}
.verdict.ok .t  {{ color: #34d399; }}
.verdict.bad .t {{ color: #f87171; }}
.verdict .d   {{ color: {TEXT_MUTED}; font-size: 0.88rem; }}

.reco {{
    display: flex; gap: 0.75rem; align-items: flex-start;
    padding: 0.7rem 0.85rem; margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.02); border: 1px solid {BORDER};
    border-left: 3px solid; border-radius: 8px;
}}
.reco.ok  {{ border-left-color: {ACCENT}; }}
.reco.bad {{ border-left-color: #ef4444; }}
.reco .tag {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 0.18rem 0.5rem; border-radius: 6px; white-space: nowrap;
}}
.reco.ok .tag  {{ color: #34d399; background: rgba(16,185,129,0.12); }}
.reco.bad .tag {{ color: #f87171; background: rgba(239,68,68,0.12); }}
.reco .msg {{ color: {TEXT}; font-size: 0.92rem; }}

.queue-note {{
    margin-top: 0.9rem; padding: 0.6rem 0.9rem; border-radius: 8px;
    border: 1px dashed rgba(245,158,11,0.5); background: rgba(245,158,11,0.07);
    color: #fbbf24; font-size: 0.85rem;
}}

[data-testid="stAlert"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 8px;
}}
div[data-testid="stAlert"][data-baseweb] p {{
    color: {TEXT};
}}

:root {{
    --pickaxe-cursor: url("data:image/png;base64,{PICKAXE_CURSOR_B64}") 6 8, auto;
}}
html, body, .stApp {{
    cursor: var(--pickaxe-cursor) !important;
}}
button, [role="button"], .stButton button, a, input, select, textarea,
[data-baseweb="slider"], [data-baseweb="select"], [data-testid="stSlider"] * {{
    cursor: var(--pickaxe-cursor) !important;
}}

/* Chart cards: hover = slight bulge, click = readable zoom */
[class*="st-key-chartcard"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem;
    position: relative;
    transition: transform 0.3s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}}
/* zoom outward toward the free side so cards at the edges don't leave the screen */
.st-key-chartcard_trend {{ transform-origin: left center; }}
.st-key-chartcard_grade {{ transform-origin: center center; }}
.st-key-chartcard_pareto,
.st-key-chartcard_reserves {{ transform-origin: right center; }}

[class*="st-key-chartcard"]:hover {{
    transform: scale(1.04);
    border-color: {ACCENT};
    box-shadow: 0 0 0 1px {ACCENT}, 0 12px 32px rgba(16, 185, 129, 0.18);
    z-index: 10;
}}
[class*="st-key-chartcard"].zoomed {{
    transform: scale(1.7);
    border-color: {ACCENT};
    box-shadow: 0 0 0 1px {ACCENT}, 0 24px 64px rgba(0, 0, 0, 0.7);
    z-index: 999;
}}

/* remove Streamlit's built-in fullscreen button on images */
[class*="st-key-chartcard"] [data-testid="stElementToolbar"],
[class*="st-key-chartcard"] [data-testid="stImage"] button {{
    display: none !important;
}}

/* Panel cards: static border, no green glow */
[class*="st-key-card_"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem;
}}
/* only the map card keeps the hover glow (delete this rule to remove it too) */
.st-key-card_map:hover {{
    border-color: {ACCENT};
    box-shadow: 0 0 0 1px {ACCENT};
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}}
/* ---------- Big connectivity switch ---------- */
.st-key-net_switch {{
    width: 300px;
    max-width: 100%;
    gap: 0 !important;
    margin-top: 0.6rem;
}}

/* track: real height, so it can never collapse */
.netsw {{
    position: relative;
    width: 300px;
    max-width: 100%;
    height: 72px;
    border-radius: 999px;
    overflow: hidden;
    background: linear-gradient(135deg, #f87171 0%, #dc2626 55%, #9f1239 100%);
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.35), 0 0 28px rgba(239,68,68,0.28);
    transition: background 0.6s cubic-bezier(.65,0,.35,1), box-shadow 0.6s cubic-bezier(.65,0,.35,1);
}}
/* a second gradient layer that fades in for the online state (gradients can't transition directly) */
.netsw::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, #6ee7b7 0%, #10b981 50%, #0d9488 100%);
    opacity: 0;
    transition: opacity 0.6s cubic-bezier(.65,0,.35,1);
}}

/* white knob */
.netsw-thumb {{
    position: absolute;
    z-index: 2;
    top: 8px;
    left: 8px;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #fff;
    box-shadow: 0 6px 16px rgba(0,0,0,0.40), 0 2px 4px rgba(0,0,0,0.25);
    transition: left 0.6s cubic-bezier(.68,-0.25,.27,1.25);
}}
.netsw-thumb svg {{
    position: absolute;
    width: 28px;
    height: 28px;
    transition: opacity 0.35s ease, transform 0.6s cubic-bezier(.68,-0.25,.27,1.25);
}}
.netsw-thumb .ic-on  {{ color: #059669; opacity: 0; transform: scale(0.4) rotate(-90deg); }}
.netsw-thumb .ic-off {{ color: #dc2626; opacity: 1; transform: scale(1) rotate(0deg); }}

/* labels */
.netsw-label {{
    position: absolute;
    z-index: 1;
    top: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.14em;
    color: #fff;
    text-shadow: 0 1px 3px rgba(0,0,0,0.35);
    transition: opacity 0.4s ease, transform 0.6s cubic-bezier(.65,0,.35,1);
}}
.netsw-off {{ right: 26px; opacity: 1; transform: translateX(0); }}
.netsw-on  {{ left: 26px;  opacity: 0; transform: translateX(18px); }}

/* the real toggle: invisible, pulled up over the track */
.st-key-net_switch [data-testid="stElementContainer"]:has([data-testid="stCheckbox"]) {{
    position: relative !important;
    z-index: 5;
    width: 300px;
    max-width: 100%;
    height: 72px;
    margin-top: -72px;
    opacity: 0;
}}
.st-key-net_switch [data-testid="stCheckbox"],
.st-key-net_switch [data-testid="stCheckbox"] label {{
    width: 100%;
    height: 100%;
    cursor: pointer;
}}

/* ---- checked = online ---- */
.st-key-net_switch:has(input:checked) .netsw {{
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.35), 0 0 28px rgba(16,185,129,0.35);
}}
.st-key-net_switch:has(input:checked) .netsw::before {{ opacity: 1; }}
.st-key-net_switch:has(input:checked) .netsw-thumb {{ left: calc(100% - 64px); }}
.st-key-net_switch:has(input:checked) .netsw-thumb .ic-on  {{ opacity: 1; transform: scale(1) rotate(0deg); }}
.st-key-net_switch:has(input:checked) .netsw-thumb .ic-off {{ opacity: 0; transform: scale(0.4) rotate(90deg); }}
.st-key-net_switch:has(input:checked) .netsw-off {{ opacity: 0; transform: translateX(-18px); }}
.st-key-net_switch:has(input:checked) .netsw-on  {{ opacity: 1; transform: translateX(0); }}

/* pulse ring on the knob while online */
.st-key-net_switch:has(input:checked) .netsw-thumb::after {{
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.8);
    animation: netring 2s ease-out infinite;
}}
@keyframes netring {{
    0%   {{ transform: scale(1);   opacity: 0.8; }}
    100% {{ transform: scale(1.6); opacity: 0; }}
}}

/* keyboard focus */
.st-key-net_switch:has(input:focus-visible) .netsw {{
    outline: 2px solid rgba(255,255,255,0.6);
    outline-offset: 3px;
}}
/* ---------- Problem / flaw cards ---------- */
.flaw-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem; margin: 0.5rem 0 2rem;
}}
.flaw {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-top: 2px solid #ef4444; border-radius: 12px; padding: 1rem 1.1rem;
}}
.flaw .n {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: #f87171; letter-spacing: 0.08em; margin-bottom: 0.35rem;
}}
.flaw .h {{ font-weight: 700; font-size: 1rem; color: {TEXT}; margin-bottom: 0.35rem; }}
.flaw .b {{ color: {TEXT_MUTED}; font-size: 0.88rem; line-height: 1.5; }}
.problem-banner {{
    padding: 1rem 1.2rem; border-radius: 12px; margin-bottom: 1.2rem;
    border: 1px solid rgba(239,68,68,0.35);
    background: linear-gradient(90deg, rgba(239,68,68,0.12), rgba(239,68,68,0.02));
    color: {TEXT}; font-size: 1rem; line-height: 1.55;
}}
.problem-banner b {{ color: #f87171; }}
@media (max-width: 900px) {{ .flaw-grid {{ grid-template-columns: 1fr; }} }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_PAPER_BG = BG_CARD
PLOTLY_FONT = dict(family="Inter, sans-serif", color=TEXT)


def section_header(label, title, sub=None):
    st.markdown(f'<div class="eyebrow">{label}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(
            f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def render_reco(action, bad):
    """Turn '[Label]: message' into a styled recommendation card."""
    m = re.match(r"^\s*\[(.+?)\]\s*:?\s*(.*)$", action, re.S)
    tag, msg = (m.group(1), m.group(2)) if m else ("Action", action)
    tag = html.escape(tag)
    msg = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(msg))
    cls = "bad" if bad else "ok"
    return (f'<div class="reco {cls}"><span class="tag">{tag}</span>'
            f'<span class="msg">{msg}</span></div>')


@st.cache_resource
def init_system():
    ai = MOILManganeseAI()
    res_df, ops_df = ai.train_models()
    return ai, res_df, ops_df


ai_engine, res_df, ops_df = init_system()

# Hero Section
st.markdown(
    '<div class="eyebrow">MOIL AI &amp; Space Tech</div>'
    '<div class="hero-title">Predictive intelligence for<br><span>manganese reserve mapping.</span></div>'
    '<div class="hero-sub">A command center for reserve discovery and operational '
    'shortfall mitigation — fusing satellite telemetry, geological data, and '
    'machine learning into a single live view.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


def render_chart(filepath, column, caption, key):
    with column:
        with st.container(key=f"chartcard_{key}"):
            if os.path.exists(filepath):
                st.image(filepath, use_container_width=True, caption=caption)
            else:
                st.warning(
                    f"Chart missing: Ensure '{filepath}' exists in the repository.")


# Section 1: Strategic Overview
section_header(
    "Platform",
    "Strategic overview.",
    "Based on data extracted from the IBM Indian Minerals Yearbook 2024 and Annual Report 2019-20.",
)

# ---- The problem -----------------------------------------------------------
st.markdown(
    '<div class="problem-banner"><b>The problem —</b> manganese output is '
    'concentrated in a handful of mines, most known resources are still unproven, '
    'and day-to-day production is exposed to disruptions that are noticed only after '
    'targets have already been missed.</div>',
    unsafe_allow_html=True,
)

FLAWS = [
    ("01 · Concentrated output",
     "Production is top-heavy",
     "A small fraction of mines delivers most of the ore (see the chart below). "
     "One disrupted large mine can swing the national number, and the long tail of "
     "small mines adds little output."),
    ("02 · Unproven resources",
     "Resources are not converted to reserves",
     "States such as Odisha, Karnataka and Goa hold large 'Remaining Resources' "
     "that are not yet 'Proved'. Conversion relies on slow, costly manual drilling."),
    ("03 · Grade variability",
     "Much of the ore is low-grade",
     "District output is split across high, medium and low grades. Without "
     "sub-surface grade prediction, planning around high-yield (>46% Mn) ore is guesswork."),
    ("04 · Reactive operations",
     "Shortfalls are spotted too late",
     "Rain, equipment downtime, blasting delays and rail-wagon availability all "
     "cut output. Today they are handled after the fact, not predicted."),
    ("05 · Connectivity gaps",
     "Deep shafts are offline",
     "Underground crews often have no signal, so shift telemetry is logged late "
     "or on paper and never reaches decision-makers in time."),
    ("06 · Uneven state trajectories",
     "Growth is not uniform",
     "Some states are ramping up while others stay flat or are declining, so a single "
     "national strategy misses where the real gaps are."),
]
st.markdown(
    '<div class="flaw-grid">' + "".join(
        f'<div class="flaw"><div class="n">{n}</div><div class="h">{h}</div>'
        f'<div class="b">{b}</div></div>' for n, h, b in FLAWS
    ) + '</div>',
    unsafe_allow_html=True,
)

# ---- Interactive Pareto chart ---------------------------------------------
# NOTE: values below were read off your static PNG. Replace with the real
# numbers from the script/CSV that generates charts/06_mine_size_pareto.png.
PARETO_BUCKETS = ["Up to 1000", "1001-5000", "5001-10000", "10001-20000",
                  "20001-30000", "30001-40000", "40001-50000", "50001 and above"]
PARETO_MINES = [63, 24, 10, 11, 5, 7, 2, 14]
PARETO_PROD_T = [10_000, 70_000, 90_000, 190_000,
                 150_000, 230_000, 100_000, 2_650_000]  # tonnes, 2023-24(P)

pareto_df = pd.DataFrame({"bucket": PARETO_BUCKETS,
                          "mines": PARETO_MINES,
                          "prod": PARETO_PROD_T})
pareto_df["mine_pct"] = pareto_df["mines"] / pareto_df["mines"].sum() * 100
pareto_df["prod_pct"] = pareto_df["prod"] / pareto_df["prod"].sum() * 100

top = pareto_df.iloc[-1]

section_header(
    "The core imbalance",
    "Pareto law: a few mines carry the output.",
    f"The largest bucket is only {top['mine_pct']:.0f}% of mines but produces "
    f"{top['prod_pct']:.0f}% of output. Hover for details, click legend items to "
    "toggle series, drag to zoom, double-click to reset.",
)

with st.container(key="card_pareto"):
    view = st.radio(
        "View", ["By size bucket", "Cumulative (Pareto curve)"],
        horizontal=True, label_visibility="collapsed", key="pareto_view",
    )

    fig_p = go.Figure()
    if view == "By size bucket":
        fig_p.add_bar(
            x=pareto_df["bucket"], y=pareto_df["mines"], name="No. of mines",
            marker_color="#1f4e8c", yaxis="y",
            customdata=pareto_df["mine_pct"],
            hovertemplate="<b>%{x}</b><br>Mines: %{y} (%{customdata:.1f}%)<extra></extra>",
        )
        fig_p.add_bar(
            x=pareto_df["bucket"], y=pareto_df["prod"], name="Production 2023-24 (t)",
            marker_color="#f59e0b", yaxis="y2",
            customdata=pareto_df["prod_pct"],
            hovertemplate="<b>%{x}</b><br>Production: %{y:,.0f} t (%{customdata:.1f}%)<extra></extra>",
        )
        fig_p.update_layout(
            barmode="group",
            yaxis=dict(title="No. of mines",
                       gridcolor="rgba(255,255,255,0.06)"),
            yaxis2=dict(title="Production (tonnes)", overlaying="y", side="right",
                        showgrid=False, tickformat=","),
        )
    else:
        # largest mines first, so the curve shows how fast output accumulates
        rev = pareto_df.iloc[::-1].copy()
        rev["cum_mines"] = rev["mine_pct"].cumsum()
        rev["cum_prod"] = rev["prod_pct"].cumsum()
        fig_p.add_bar(
            x=rev["bucket"], y=rev["cum_mines"], name="Cumulative % of mines",
            marker_color="#1f4e8c",
            hovertemplate="<b>%{x}</b><br>Cumulative mines: %{y:.1f}%<extra></extra>",
        )
        fig_p.add_scatter(
            x=rev["bucket"], y=rev["cum_prod"], name="Cumulative % of production",
            mode="lines+markers", line=dict(color="#f59e0b", width=3),
            marker=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Cumulative production: %{y:.1f}%<extra></extra>",
        )
        fig_p.add_hline(y=80, line_dash="dot", line_color="rgba(255,255,255,0.35)",
                        annotation_text="80%", annotation_font_color=TEXT_MUTED)
        fig_p.update_layout(
            yaxis=dict(title="Cumulative %", range=[0, 105],
                       gridcolor="rgba(255,255,255,0.06)"),
        )

    fig_p.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT, height=460, hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(title="Mine size bucket (tonnes per year)", tickangle=-25),
    )
    st.plotly_chart(fig_p, use_container_width=True)

st.markdown("---")
# Section 2: Sub-surface Reserve Mapping
section_header(
    "Technology",
    "Sub-surface reserve mapping.",
    "Simulated Sentinel-2 / Bhuvan satellite telemetry (NDVI, LST) fused with geological "
    "drill hole depth. A Random Forest regressor maps spatial continuity to predict "
    "high-yield (>46% Mn) ore bodies.",
)

map_col, chart_col = st.columns([2, 1])

with map_col:
    with st.container(key="card_map"):
        selected_belt = st.selectbox(
            "Manganese belt", list(ai_engine.BELTS.keys()), index=0, label_visibility="collapsed"
        )
        belt_cfg = ai_engine.BELTS[selected_belt]

        sample_df = ai_engine.generate_belt_data(selected_belt, n_samples=250)
        sample_df["Predicted_Mn_Grade"] = ai_engine.predict_reserve_grid(
            sample_df[["latitude", "longitude", "depth_m", "ndvi",
                       "lst_celsius", "soil_moisture_pct", "mag_susceptibility"]]
        )

        fig_map = px.scatter_map(
            sample_df,
            lat="latitude",
            lon="longitude",
            color="Predicted_Mn_Grade",
            size="depth_m",
            color_continuous_scale="Turbo",
            zoom=belt_cfg["zoom"],
            center=belt_cfg["center"],
            title=f"Predictive Mineralization Heatmap ({selected_belt})",
            hover_data=["depth_m", "ndvi", "lst_celsius"],
        )
        fig_map.update_layout(
            map_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            paper_bgcolor=PLOTLY_PAPER_BG,
            font=PLOTLY_FONT,
        )
        st.plotly_chart(fig_map, use_container_width=True)

with chart_col:
    render_chart("charts/03_reserves_by_state.png", chart_col,
                 "Current Proved vs Remaining Resources", "reserves")
    st.markdown(
        '<div class="moil-card" style="margin-top:0.75rem;">'
        f'<span style="color:{ACCENT};font-weight:600;">Insight —</span> '
        'Notice the massive "Remaining Resources" volume in Odisha, Karnataka, and Goa. '
        'AI-driven mapping allows MOIL to efficiently convert these into "Proved Reserves" without excessive manual drilling.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# Section 3: Operational Shortfall Predictor with Offline Sync
section_header(
    "Live Simulation",
    "Operational shortfall predictor (Offline-First Architecture).",
    "Deep underground mine shafts often lack internet connectivity. This interface demonstrates how shift managers can log telemetry locally on their device, which auto-syncs to the central AI model once surface connectivity is restored.",
)
WIFI_ON = ('<svg class="ic-on" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
           'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
           '<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/>'
           '<path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>')
WIFI_OFF = ('<svg class="ic-off" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
            '<line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/>'
            '<path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"/><path d="M10.71 5.05A16 16 0 0 1 22.58 9"/>'
            '<path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/>'
            '<line x1="12" y1="20" x2="12.01" y2="20"/></svg>')

NET_SWITCH_HTML = (
    '<div class="netsw">'
    '<span class="netsw-label netsw-on">ONLINE</span>'
    '<span class="netsw-label netsw-off">OFFLINE</span>'
    f'<span class="netsw-thumb">{WIFI_ON}{WIFI_OFF}</span>'
    '</div>'
)
# Network switch
col_net1, col_net2 = st.columns([1, 2])
with col_net1:
    st.markdown('<div class="stat-label">Device Connectivity Status</div>',
                unsafe_allow_html=True)
    with st.container(key="net_switch"):
        st.markdown(NET_SWITCH_HTML, unsafe_allow_html=True)
        is_online = st.toggle("Network", value=True, key="net_toggle",
                              label_visibility="collapsed")

network_mode = ("🟢 Online (Surface / Wi-Fi)" if is_online
                else "🔴 Offline (Deep Shaft / No Signal)")

st.write("")
sc1, sc2 = st.columns([1, 2])

with sc1:
    with st.container(key="card_telemetry"):
        st.markdown('<div class="stat-label">Log Telemetry</div>',
                    unsafe_allow_html=True)
        st.write("")

        in_rain = st.slider("Rainfall Forecast (mm)", 0.0,
                            100.0, 15.0, format="%.1f")
        in_downtime = st.slider(
            "Total Equipment Downtime (hrs)", 0.0, 24.0, 3.5, format="%.1f")
        in_blast = st.slider("Blasting Delay (hrs)", 0.0,
                             12.0, 1.0, format="%.1f")
        in_wagon = st.slider("Rail Wagon Availability Ratio",
                             0.0, 1.0, 0.95, format="%.2f")

        st.write("")

        # Offline Logic: Button behavior changes based on network state
        if "Offline" in network_mode:
            if st.button("💾 Save Locally (Offline)", use_container_width=True):
                st.session_state['offline_queue'].append({
                    'rain': in_rain,
                    'downtime': in_downtime,
                    'blast': in_blast,
                    'wagon': in_wagon,
                    'timestamp': time.strftime("%H:%M:%S")
                })
                st.success("Saved to local device storage.")
        else:
            # Online Logic: Sync automatically or process instantly
            if len(st.session_state['offline_queue']) > 0:
                st.warning(
                    f"You have {len(st.session_state['offline_queue'])} pending offline logs.")
                if st.button("☁️ Sync Pending Logs to AI Engine", use_container_width=True):
                    # Simulate pushing the latest log to the backend model
                    latest_log = st.session_state['offline_queue'][-1]
                    st.session_state['last_synced_data'] = latest_log
                    st.session_state['offline_queue'].clear()
                    st.success(
                        "Successfully synced all local telemetry to the MOIL central server.")
            else:
                if st.button("🚀 Process Telemetry Live", use_container_width=True):
                    st.session_state['last_synced_data'] = {
                        'rain': in_rain, 'downtime': in_downtime, 'blast': in_blast, 'wagon': in_wagon
                    }

with sc2:
    with st.container(key="card_results"):
        is_offline = "Offline" in network_mode
        eval_data = st.session_state['last_synced_data']

        if is_offline:
            st.markdown(
                '<span class="status-pill offline"><span class="pulse-dot"></span>'
                'Offline · viewing last cached result</span>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<span class="status-pill online"><span class="pulse-dot"></span>'
                'Online · live AI evaluation</span>',
                unsafe_allow_html=True)

        prob_shortfall, pred_label = ai_engine.predict_shortfall_risk(
            eval_data['rain'], eval_data['downtime'], eval_data['blast'], eval_data['wagon']
        )
        actions = ai_engine.get_prescriptive_actions(
            eval_data['rain'], eval_data['downtime'], eval_data['blast'], eval_data['wagon']
        )
        bad = pred_label == 1
        risk_pct = prob_shortfall * 100
        bar_color = "#10b981" if risk_pct < 40 else "#f59e0b" if risk_pct < 70 else "#ef4444"

        # Inputs currently being evaluated
        st.markdown(
            '<div class="tile-row">'
            f'<div class="tile"><div class="k">Rainfall</div><div class="v">{eval_data["rain"]:.1f}<span class="u">mm</span></div></div>'
            f'<div class="tile"><div class="k">Downtime</div><div class="v">{eval_data["downtime"]:.1f}<span class="u">hrs</span></div></div>'
            f'<div class="tile"><div class="k">Blast delay</div><div class="v">{eval_data["blast"]:.1f}<span class="u">hrs</span></div></div>'
            f'<div class="tile"><div class="k">Wagons</div><div class="v">{eval_data["wagon"]:.2f}<span class="u">ratio</span></div></div>'
            '</div>',
            unsafe_allow_html=True)

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={"suffix": "%", "valueformat": ".1f",
                        "font": {"color": bar_color, "size": 54,
                                 "family": "JetBrains Mono, monospace"}},
                title={"text": "PROBABILITY OF PRODUCTION SHORTFALL",
                       "font": {"color": TEXT_MUTED, "size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickvals": [0, 20, 40, 60, 80, 100],
                             "tickcolor": TEXT_MUTED,
                             "tickfont": {"size": 11, "color": TEXT_MUTED}},
                    "bar": {"color": bar_color, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16,185,129,0.14)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.14)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.14)"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            height=280,
            margin={"r": 30, "t": 50, "l": 30, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            font=PLOTLY_FONT,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Verdict banner
        if bad:
            st.markdown(
                '<div class="verdict bad"><div class="ico">⚠️</div><div>'
                '<div class="t">Shortfall imminent</div>'
                '<div class="d">AI model flags a high probability of missing daily production targets.</div>'
                '</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="verdict ok"><div class="ico">✅</div><div>'
                '<div class="t">Targets on track</div>'
                '<div class="d">Current operating parameters are within safety margins.</div>'
                '</div></div>', unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="stat-label" style="margin-bottom:0.5rem;">Prescriptive recommendations</div>',
                    unsafe_allow_html=True)
        st.markdown("".join(render_reco(a, bad) for a in actions),
                    unsafe_allow_html=True)

        # Offline queue note
        queued = len(st.session_state['offline_queue'])
        if is_offline and queued > 0:
            st.markdown(
                f'<div class="queue-note">⏳ <b>{queued}</b> record(s) saved locally, awaiting sync</div>',
                unsafe_allow_html=True)
# =============================================================================
# SECTION 5: ALERTS & PUSH NOTIFICATIONS
# Paste AFTER Section 3 (it needs `is_online`, `ai_engine`, `selected_belt`,
# `belt_cfg`) and before the final components.html(...) call.
# Uses names already in your app: st, pd, np, go, os, re, html, time,
# section_header, ACCENT, ACCENT_SOFT, BG_CARD, BORDER, TEXT, TEXT_MUTED,
# PLOTLY_TEMPLATE, PLOTLY_FONT.
# =============================================================================

st.markdown("---")
section_header(
    "Alerts & Notifications",
    "Forecast-driven alerts for shift managers.",
    "Managers on the ground are not watching a dashboard. The 3-7 day forecast is scored by the "
    "shortfall model, and anyone whose threshold is crossed gets a short, actionable message on "
    "WhatsApp, SMS or a phone push notification. Offline devices queue alerts and auto-send on reconnect.",
)

st.markdown(f"""
<style>
.chan-row {{ display:flex; gap:0.5rem; flex-wrap:wrap; margin:0.2rem 0 0.9rem; }}
.chan {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.05em;
         text-transform:uppercase; padding:0.25rem 0.65rem; border-radius:999px; border:1px solid; }}
.chan.on  {{ color:#34d399; background:rgba(16,185,129,0.10); border-color:rgba(16,185,129,0.35); }}
.chan.off {{ color:{TEXT_MUTED}; background:rgba(255,255,255,0.03); border-color:{BORDER}; }}

.phone {{ max-width:330px; margin:0.4rem auto 0; background:#0b0c0e;
          border:1px solid rgba(255,255,255,0.14); border-radius:30px; padding:14px 12px 20px;
          box-shadow:0 18px 40px rgba(0,0,0,0.5); }}
.phone .notch {{ width:84px; height:5px; border-radius:99px; background:rgba(255,255,255,0.12);
                 margin:0 auto 10px; }}
.phone .hdr {{ display:flex; align-items:center; gap:0.55rem; padding:0.3rem 0.4rem 0.65rem;
               border-bottom:1px solid {BORDER}; margin-bottom:0.8rem; }}
.phone .av {{ width:30px; height:30px; border-radius:50%; background:{ACCENT}; color:#04120c;
              display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.8rem; }}
.phone .nm {{ font-weight:600; font-size:0.86rem; color:{TEXT}; line-height:1.1; }}
.phone .st {{ font-size:0.68rem; color:{TEXT_MUTED}; }}
.bubble {{ border-radius:12px; border-top-left-radius:3px; padding:0.65rem 0.8rem;
           font-size:0.83rem; line-height:1.5; color:#e9edef; }}
.bubble.wa  {{ background:#202c33; }}
.bubble.sms {{ background:#26272b; }}
.bubble .meta {{ text-align:right; font-size:0.66rem; color:rgba(255,255,255,0.5); margin-top:0.35rem; }}
.pushcard {{ background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.10);
             border-radius:16px; padding:0.7rem 0.85rem; }}
.pushcard .app {{ display:flex; justify-content:space-between; font-size:0.66rem; color:{TEXT_MUTED};
                  text-transform:uppercase; letter-spacing:0.06em; }}
.pushcard .t {{ font-weight:700; font-size:0.92rem; color:{TEXT}; margin:0.3rem 0 0.2rem; }}
.pushcard .b {{ font-size:0.83rem; line-height:1.5; color:{TEXT}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def _cfg(key):
    """Read a credential from env vars first, then .streamlit/secrets.toml."""
    v = os.environ.get(key)
    if v:
        return v
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def _latlon(cfg):
    c = cfg.get("center") if isinstance(cfg, dict) else None
    if isinstance(c, dict):
        return float(c.get("lat", 21.8)), float(c.get("lon", 80.2))
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return float(c[0]), float(c[1])
    return 21.8, 80.2


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_rain_forecast(lat, lon, days):
    """Daily rainfall (mm) from Open-Meteo. Free, no API key."""
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "daily": "precipitation_sum",
                "forecast_days": days, "timezone": "auto"},
        timeout=6,
    )
    r.raise_for_status()
    d = r.json()["daily"]
    return d["time"], [float(x or 0.0) for x in d["precipitation_sum"]]


def _level(p):
    return "Critical" if p >= 70 else "Watch" if p >= 40 else "Normal"


LEVEL_MIN = {"Watch": 40, "Critical": 70}
CHANNELS = ["WhatsApp", "SMS", "Push"]
# Planned allowances: losses only count what exceeds these (assumed values, tune to MOIL data)
ALLOW = {"downtime": 4.0, "blast": 1.5, "wagon": 0.90}


def estimate_shortfall(rain, downtime, blast, wagon):
    """Estimated % of daily production target lost, with the contribution of each driver.
    Placeholder heuristic: replace with a trained regressor when available."""
    rain_l = 0.30 * (1 - np.exp(-max(rain - 10.0, 0.0) / 40.0)
                     )    # saturates near 30%
    down_l = min(max(downtime - ALLOW["downtime"], 0.0) / 24.0 * 0.9, 0.6)
    blast_l = min(max(blast - ALLOW["blast"], 0.0) / 12.0 * 0.7, 0.5)
    wag_l = min(max(ALLOW["wagon"] - wagon, 0.0) * 0.8, 0.6)
    total = 1 - (1 - rain_l) * (1 - down_l) * (1 - blast_l) * (1 - wag_l)
    parts = {"Rainfall": rain_l * 100, "Downtime": down_l * 100,
             "Blast delay": blast_l * 100, "Wagons": wag_l * 100}
    return total * 100, parts


# ---------------------------------------------------------------- state
if "alert_roster_default" not in st.session_state:
    st.session_state["alert_roster_default"] = pd.DataFrame([
        ["Paras Medhi",  "Shift Manager (A)",   "+910000000001",
         "WhatsApp", "moil-demo-shift-a", "Watch",    True],
        ["Anita Das", "Shift Manager (B)",   "+910000000002",
         "SMS",      "moil-demo-shift-b", "Critical", True],
        ["Vinod Rao",  "Mine Superintendent", "+910000000003",
            "Push",     "moil-demo-super",   "Critical", True],
    ], columns=["Name", "Role", "Phone", "Channel", "Push topic", "Notify from", "Active"])
if "alert_outbox" not in st.session_state:
    st.session_state["alert_outbox"] = []
if "alert_sent_keys" not in st.session_state:
    st.session_state["alert_sent_keys"] = []

site = selected_belt
lat, lon = _latlon(belt_cfg)
base = st.session_state["last_synced_data"]

# ---------------------------------------------------------------- forecast + risk
fc_col, set_col = st.columns([2, 1], gap="medium")

with set_col:
    with st.container(key="card_alert_settings"):
        st.markdown('<div class="stat-label">Forecast settings</div>',
                    unsafe_allow_html=True)
        st.write("")
        source = st.radio("Forecast source",
                          ["Demo: monsoon spike",
                              "Live forecast (Open-Meteo)"],
                          key="alert_source")
        horizon_lbl = st.radio("Horizon", ["3 days", "5 days", "7 days"], index=2,
                               horizontal=True, key="alert_horizon")
        horizon = int(horizon_lbl.split()[0])
        target_t = st.number_input("Daily production target (t/day)", 100, 50000, 3000,
                                   step=100, key="alert_target",
                                   help="Placeholder. Used to convert shortfall % into tonnes.")

        st.caption(f"Site: **{site}**. Downtime {base['downtime']:.1f} h, blast delay "
                   f"{base['blast']:.1f} h and wagon ratio {base['wagon']:.2f} are held at the "
                   "latest logged telemetry; rainfall varies by day.")

today = _dt.date.today()
dates, rains, src_note = None, None, ""
if source.startswith("Live"):
    try:
        iso, rains = fetch_rain_forecast(lat, lon, horizon)
        dates = [_dt.date.fromisoformat(x) for x in iso]
        src_note = "Live forecast from Open-Meteo"
    except Exception:
        src_note = "Live forecast unavailable, showing the demo scenario instead"
if dates is None:
    demo = [8, 12, 35, 68, 84, 40, 15]
    rains = demo[:horizon]
    dates = [today + _dt.timedelta(days=i) for i in range(horizon)]
    src_note = src_note or "Demo scenario: a monsoon spike builds mid-week"

rows = []
for d, rain in zip(dates, rains):
    rain = float(np.clip(rain, 0, 100))
    prob, _ = ai_engine.predict_shortfall_risk(
        rain, base["downtime"], base["blast"], base["wagon"])
    risk = float(prob) * 100
    short_pct, parts = estimate_shortfall(
        rain, base["downtime"], base["blast"], base["wagon"])
    rows.append({"date": d, "label": f"{d:%a %d %b}", "short": f"{d:%a %d}",
                 "lead": (d - today).days, "rain": rain, "risk": risk,
                 "level": _level(risk), "short_pct": short_pct, "parts": parts,
                 "t_lost": target_t * short_pct / 100})

with fc_col:
    with st.container(key="card_alert_forecast"):
        st.markdown(
            '<div class="chart-title">Production shortfall forecast</div>'
            f'<div class="chart-sub">{html.escape(src_note)}. Bars = expected output lost vs target; '
            'colour = alert level; diamonds = model risk probability (right axis). '
            '<b>Click a bar</b> for the breakdown.</div>', unsafe_allow_html=True)

        colors = ["#ef4444" if r["level"] == "Critical" else "#f59e0b" if r["level"] == "Watch"
                  else "#10b981" for r in rows]
        xs = [f"{r['short']}<br>{r['rain']:.0f} mm" for r in rows]
        custom = [[r["risk"], r["level"], r["t_lost"], r["rain"]]
                  for r in rows]

        fig_fc = go.Figure()
        fig_fc.add_bar(
            x=xs, y=[r["short_pct"] for r in rows], name="Production shortfall (%)",
            marker_color=colors, customdata=custom, yaxis="y",
            text=[f"{r['short_pct']:.1f}%" for r in rows],
            textposition="outside", cliponaxis=False,
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.45)),
            hovertemplate=("<b>%{x}</b><br>Shortfall: %{y:.1f}% of target"
                           "<br>Output lost: %{customdata[2]:,.0f} t"
                           "<br>Risk: %{customdata[0]:.0f}% (%{customdata[1]})"
                           "<extra></extra>"))
        fig_fc.add_scatter(
            x=xs, y=[r["risk"] for r in rows], name="Risk probability (%)",
            mode="lines+markers", yaxis="y2",
            line=dict(color="rgba(255,255,255,0.55)", width=2, dash="dot"),
            marker=dict(symbol="diamond", size=10, color="#e6e6e6"),
            hovertemplate="%{x}<br>Risk probability: %{y:.0f}%<extra></extra>")

        # Watch / Critical thresholds belong to the risk axis (y2)
        for lvl, col in ((40, "rgba(245,158,11,0.6)"), (70, "rgba(239,68,68,0.6)")):
            fig_fc.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=lvl, y1=lvl,
                             line=dict(color=col, width=1, dash="dot"))

        y_top = max(30.0, max(r["short_pct"] for r in rows) * 1.35)
        fig_fc.update_layout(
            template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=PLOTLY_FONT, height=380, bargap=0.35, clickmode="event+select",
            hovermode="closest",
            legend=dict(orientation="h", y=1.1, x=0),
            yaxis=dict(title="Production shortfall (%)", range=[0, y_top],
                       gridcolor="rgba(255,255,255,0.06)"),
            yaxis2=dict(title="Risk probability (%)", overlaying="y", side="right",
                        range=[0, 100], showgrid=False),
            xaxis=dict(title="Day / forecast rainfall"),
            margin=dict(l=5, r=5, t=30, b=50))

        ev = st.plotly_chart(fig_fc, use_container_width=True, key="alert_fc_chart",
                             on_select="rerun", selection_mode="points")

        # selected day (defaults to the worst day)
        pts = (ev.selection.points if ev and ev.selection else []) or []
        sel = pts[0]["point_index"] if pts else max(
            range(len(rows)), key=lambda i: rows[i]["short_pct"])
        sel = min(sel, len(rows) - 1)
        r = rows[sel]
        top_driver = max(r["parts"], key=r["parts"].get)

        st.markdown(
            f'<div class="stat-label" style="margin:0.3rem 0 0.2rem;">'
            f'{r["label"]} · {r["level"]} · est. loss {r["t_lost"]:,.0f} t '
            f'({r["short_pct"]:.1f}% of target)</div>'
            '<div class="tile-row">' + "".join(
                f'<div class="tile"><div class="k">{k}{" ★" if k == top_driver and r["parts"][k] > 0 else ""}</div>'
                f'<div class="v">{v:.1f}<span class="u">% loss</span></div></div>'
                for k, v in r["parts"].items()) + '</div>',
            unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- roster + preview
ros_col, prev_col = st.columns([3, 2], gap="medium")

with ros_col:
    with st.container(key="card_alert_roster"):
        st.markdown('<div class="chart-title">Alert recipients</div>'
                    '<div class="chart-sub">Who gets notified, on which channel, and from which risk '
                    'level. Add rows for more staff.</div>', unsafe_allow_html=True)
        roster = st.data_editor(
            st.session_state["alert_roster_default"],
            num_rows="dynamic", use_container_width=True, hide_index=True, key="alert_roster_editor",
            column_config={
                "Name": st.column_config.TextColumn(required=True),
                "Phone": st.column_config.TextColumn(help="E.164 format, e.g. +919876543210"),
                "Channel": st.column_config.SelectboxColumn(options=CHANNELS, required=True),
                "Push topic": st.column_config.TextColumn(
                    help="ntfy.sh topic. Use a long, unguessable name: anyone who knows it can read it."),
                "Notify from": st.column_config.SelectboxColumn(options=list(LEVEL_MIN), required=True),
                "Active": st.column_config.CheckboxColumn(),
            },
        )


def _build_message(mgr, peak, flagged):
    when = "today" if peak["lead"] == 0 else "tomorrow" if peak[
        "lead"] == 1 else f"in {peak['lead']} days"
    actions = ai_engine.get_prescriptive_actions(
        peak["rain"], base["downtime"], base["blast"], base["wagon"])
    action = ""
    if actions:
        m = re.match(r"^\s*\[(.+?)\]\s*:?\s*(.*)$", actions[0], re.S)
        action = (f"{m.group(1)}: {m.group(2)}" if m else actions[0])
        action = re.sub(r"\*\*", "", action).strip()
        action = action if len(action) <= 130 else action[:127] + "..."
        lines = [f"Hi {str(mgr['Name']).split()[0]}, {peak['level']} shortfall risk {peak['risk']:.0f}% "
                 f"on {peak['label']} ({when}). Forecast rain {peak['rain']:.0f} mm, "
                 f"est. output loss {peak['short_pct']:.0f}% (~{peak['t_lost']:,.0f} t)."]
    if action:
        lines.append(f"Do: {action}")
        others = [
            f"{r['short']} {r['short_pct']:.0f}% loss" for r in flagged if r is not peak]
    if others:
        lines.append("Also flagged: " + ", ".join(others[:3]))
    lines.append("Reply ACK to confirm.")
    ascii_site = site.encode("ascii", "ignore").decode()
    return f"MOIL Alert | {ascii_site}", "\n".join(lines)


plan = []
for _, m in roster.iterrows():
    active = m.get("Active")
    if active is None or pd.isna(active) or not bool(active):
        continue
    if pd.isna(m.get("Name")) or not str(m["Name"]).strip():
        continue
    minp = LEVEL_MIN.get(m.get("Notify from"), 40)
    flagged = [r for r in rows if r["risk"] >= minp]
    if not flagged:
        continue
    peak = max(flagged, key=lambda r: r["risk"])
    title, body = _build_message(m, peak, flagged)
    plan.append({
        "name": str(m["Name"]).strip(), "role": str(m.get("Role") or ""),
        "phone": str(m.get("Phone") or "").replace(" ", ""),
        "channel": m.get("Channel") or "WhatsApp",
        "topic": str(m.get("Push topic") or "").strip(),
        "level": peak["level"], "title": title, "body": body,
        "key": f"{str(m['Name']).strip()}|{peak['date']}|{peak['level']}",
    })

with prev_col:
    with st.container(key="card_alert_preview"):
        st.markdown('<div class="chart-title">Message preview</div>',
                    unsafe_allow_html=True)
        names = [str(n) for n in roster["Name"].dropna() if str(n).strip()]
        if not names:
            st.info("Add a recipient to see a preview.")
            who = None
        else:
            who = st.selectbox(
                "Preview for", names, label_visibility="collapsed", key="alert_preview_who")
            item = next((p for p in plan if p["name"] == who), None)
            row = roster[roster["Name"] == who].iloc[0]
            chan = row.get("Channel") or "WhatsApp"
            if item is None:
                st.info(f"No alert for {who}: the forecast stays below their "
                        f"{row.get('Notify from')} level.")
            else:
                stamp = time.strftime("%H:%M")
                full = html.escape(
                    f"{item['title']}\n{item['body']}").replace("\n", "<br>")
                if chan == "Push":
                    inner = (f'<div class="pushcard"><div class="app"><span>MOIL Alerts</span>'
                             f'<span>now</span></div><div class="t">{html.escape(item["title"])}</div>'
                             f'<div class="b">{html.escape(item["body"]).replace(chr(10), "<br>")}</div></div>')
                    sender, sub = "Notification", "Lock screen"
                else:
                    cls = "wa" if chan == "WhatsApp" else "sms"
                    inner = f'<div class="bubble {cls}">{full}<div class="meta">{stamp}</div></div>'
                    sender, sub = ("MOIL Alerts", "business account") if chan == "WhatsApp" else (
                        "MOIL-ALERT", "SMS")
                st.markdown(
                    f'<div class="phone"><div class="notch"></div><div class="hdr">'
                    f'<div class="av">M</div><div><div class="nm">{sender}</div>'
                    f'<div class="st">{sub}</div></div></div>{inner}</div>',
                    unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- delivery


def _twilio(phone, text, whatsapp):
    sid, tok = _cfg("TWILIO_ACCOUNT_SID"), _cfg("TWILIO_AUTH_TOKEN")
    frm = _cfg("TWILIO_WHATSAPP_FROM" if whatsapp else "TWILIO_SMS_FROM")
    if not (sid and tok and frm):
        return "Failed: Twilio credentials missing"
    if whatsapp:
        frm = frm if frm.startswith("whatsapp:") else f"whatsapp:{frm}"
        to = f"whatsapp:{phone}"
    else:
        to = phone
    try:
        r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                          data={"From": frm, "To": to, "Body": text}, auth=(sid, tok), timeout=8)
        return "Delivered (live)" if r.status_code in (200, 201) else f"Failed: HTTP {r.status_code}"
    except Exception as e:
        return f"Failed: {type(e).__name__}"


def _ntfy(topic, title, body, level):
    if not topic:
        return "Failed: no push topic"
    try:
        r = requests.post(
            f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), timeout=8,
            headers={"Title": title, "Priority": "urgent" if level == "Critical" else "high",
                     "Tags": "rotating_light" if level == "Critical" else "warning"})
        return "Delivered (live)" if r.status_code == 200 else f"Failed: HTTP {r.status_code}"
    except Exception as e:
        return f"Failed: {type(e).__name__}"


def _deliver(item, live):
    if not live:
        return "Delivered (simulated)"
    if item["channel"] == "Push":
        return _ntfy(item["topic"], item["title"], item["body"], item["level"])
    if not item["phone"]:
        return "Failed: no phone number"
    return _twilio(item["phone"], f"{item['title']}\n{item['body']}", item["channel"] == "WhatsApp")


def _record(item, status):
    st.session_state["alert_outbox"].append({
        **item, "time": time.strftime("%H:%M:%S"), "status": status})


wa_ok = all(_cfg(k) for k in ("TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"))
sms_ok = all(_cfg(k) for k in ("TWILIO_ACCOUNT_SID",
             "TWILIO_AUTH_TOKEN", "TWILIO_SMS_FROM"))

with st.container(key="card_alert_dispatch"):
    st.markdown('<div class="chart-title">Dispatch</div>'
                '<div class="chart-sub">Send the alerts above. Demo mode simulates delivery; live mode '
                'uses your Twilio credentials and ntfy.sh push.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chan-row">'
        f'<span class="chan {"on" if wa_ok else "off"}">WhatsApp {"ready" if wa_ok else "no credentials"}</span>'
        f'<span class="chan {"on" if sms_ok else "off"}">SMS {"ready" if sms_ok else "no credentials"}</span>'
        '<span class="chan on">Push ready (no key needed)</span></div>', unsafe_allow_html=True)

    d1, d2 = st.columns([2, 3])
    with d1:
        mode = st.radio("Delivery mode", [
                        "Demo (simulated)", "Live"], horizontal=True, key="alert_mode")
    live = mode == "Live"
    with d2:
        st.write("")
        b1, b2, b3 = st.columns(3)
        go_send = b1.button("Dispatch alerts", type="primary",
                            use_container_width=True)
        go_test = b2.button("Send test", use_container_width=True)
        go_clear = b3.button("Clear log", use_container_width=True)

    # offline-first: queued alerts flush automatically once the device is back online
    if is_online:
        flushed = 0
        for row in st.session_state["alert_outbox"]:
            if row["status"].startswith("Queued"):
                row["status"] = _deliver(row, live)
                flushed += 1
        if flushed:
            st.success(
                f"Connectivity restored: {flushed} queued alert(s) sent.")

    if go_send:
        if not plan:
            st.info(
                "Nothing to send: no recipient's threshold is crossed in this forecast.")
        else:
            sent = skipped = queued = 0
            for item in plan:
                if item["key"] in st.session_state["alert_sent_keys"]:
                    skipped += 1
                    continue
                if is_online:
                    _record(item, _deliver(item, live))
                    sent += 1
                else:
                    _record(item, "Queued (offline)")
                    queued += 1
                st.session_state["alert_sent_keys"].append(item["key"])
            parts = []
            if sent:
                parts.append(f"{sent} sent")
            if queued:
                parts.append(f"{queued} queued until connectivity returns")
            if skipped:
                parts.append(
                    f"{skipped} skipped (already alerted for that day)")
            st.info("; ".join(parts) + ".")

    if go_test and who:
        row = roster[roster["Name"] == who].iloc[0]
        test = {"name": who, "role": str(row.get("Role") or ""),
                "phone": str(row.get("Phone") or "").replace(" ", ""),
                "channel": row.get("Channel") or "WhatsApp",
                "topic": str(row.get("Push topic") or "").strip(), "level": "Watch",
                "title": "MOIL Alert | TEST",
                "body": "This is a test notification. If you can read this, alerts are working.",
                "key": f"test|{who}|{time.time()}"}
        _record(test, _deliver(test, live)
                if is_online else "Queued (offline)")

    if go_clear:
        st.session_state["alert_outbox"] = []
        st.session_state["alert_sent_keys"] = []

    outbox = st.session_state["alert_outbox"]
    if outbox:
        log_df = pd.DataFrame([{
            "Time": r["time"], "To": r["name"], "Channel": r["channel"], "Level": r["level"],
            "Status": r["status"], "Message": r["body"].split("\n")[0][:70]} for r in reversed(outbox[-15:])])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No alerts sent yet.")

    st.caption("Live setup: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_SMS_FROM and "
               "TWILIO_WHATSAPP_FROM as environment variables or in .streamlit/secrets.toml. For push, "
               "install the free ntfy app on the manager's phone and subscribe to their topic. "
               "Phone numbers and topics here are placeholders.")
# ============================================================================
# SECTION 4: FINANCIAL ROI - TRADITIONAL vs AI-ASSISTED  (v2 redesign)
# Paste after the `with sc2:` block, before the final components.html(...) call.
# Uses: pd, np, go, st, os, section_header, ACCENT, BG_CARD, BORDER, TEXT,
#       TEXT_MUTED, PLOTLY_TEMPLATE, PLOTLY_FONT (all already in your app).
# =============================================================================
st.markdown("---")
section_header(
    "Financial Impact",
    "ROI breakdown: traditional vs AI-assisted.",
    "Enter the real cost heads of a manganese mining operation and the expected AI-driven "
    "reduction for each. The AI platform runs on free/open satellite data and a near-zero-cost "
    "API, so its own cost is entered separately.",
)

st.markdown(f"""
<style>
/* ---------- ROI section ---------- */
.st-key-card_roi_admin {{ border-top: 2px solid {ACCENT} !important; }}

.roi-head {{ display:flex; justify-content:space-between; align-items:flex-start;
             gap:1rem; margin-bottom:1rem; }}
.roi-head .ttl {{ font-weight:700; font-size:1.15rem; color:{TEXT}; letter-spacing:-0.01em; }}
.roi-head .sub {{ color:{TEXT_MUTED}; font-size:0.86rem; margin-top:0.15rem; }}
.chip {{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; letter-spacing:0.06em;
         text-transform:uppercase; padding:0.28rem 0.7rem; border-radius:999px; white-space:nowrap;
         color:{ACCENT}; background:{ACCENT_SOFT}; border:1px solid rgba(16,185,129,0.35); }}

.mini-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0.6rem; margin-bottom:1.1rem; }}
.mini {{ background:rgba(255,255,255,0.03); border:1px solid {BORDER};
         border-left:3px solid {ACCENT}; border-radius:10px; padding:0.6rem 0.85rem; }}
.mini .k {{ color:{TEXT_MUTED}; font-size:0.66rem; text-transform:uppercase; letter-spacing:0.06em; }}
.mini .v {{ font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:600; color:{TEXT}; }}
.mini.amber {{ border-left-color:#f59e0b; }}
.mini.blue  {{ border-left-color:#3b82f6; }}

.kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin:0.6rem 0 1.1rem; }}
.kpi {{ background:{BG_CARD}; border:1px solid {BORDER}; border-top:2px solid {BORDER};
        border-radius:12px; padding:1rem 1.1rem; }}
.kpi.red   {{ border-top-color:#ef4444; }}
.kpi.green {{ border-top-color:{ACCENT}; }}
.kpi.blue  {{ border-top-color:#3b82f6; }}
.kpi .k {{ color:{TEXT_MUTED}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; }}
.kpi .v {{ font-family:'JetBrains Mono',monospace; font-size:1.55rem; font-weight:700;
           color:{TEXT}; margin:0.3rem 0 0.15rem; line-height:1.2; }}
.kpi .v.good {{ color:{ACCENT}; }}
.kpi .s {{ color:{TEXT_MUTED}; font-size:0.78rem; }}

.chart-title {{ font-weight:700; font-size:1.05rem; color:{TEXT}; }}
.chart-sub {{ color:{TEXT_MUTED}; font-size:0.82rem; margin:0.15rem 0 0.9rem; }}
.legend-strip {{ display:flex; gap:1.2rem; margin-bottom:0.6rem; font-size:0.82rem; color:{TEXT_MUTED}; }}
.legend-strip i {{ display:inline-block; width:10px; height:10px; border-radius:3px;
                   margin-right:0.4rem; vertical-align:middle; }}

@media (max-width: 900px) {{
  .mini-row, .kpi-grid {{ grid-template-columns:repeat(2,1fr); }}
}}
</style>
""", unsafe_allow_html=True)

COST_COLS = ["Cost head", "Cost (Rs lakh)",
             "Recurs yearly", "AI reduction (%)"]
AI_COLS = ["AI platform item", "Cost (Rs lakh)", "Recurs yearly"]
SCENARIOS = {"Conservative \u00b7 60%": 0.6,
             "Expected \u00b7 100%": 1.0, "Optimistic \u00b7 125%": 1.25}


def _default_cost_df():
    # PLACEHOLDER numbers - replace with real figures
    return pd.DataFrame([
        ["Exploratory drilling",        400.0, False, 35.0],
        ["Equipment (purchase/lease)",  800.0, False,  5.0],
        ["Labour & manpower",           300.0, True,   5.0],
        ["Geophysical & survey",        120.0, True,  40.0],
        ["Blasting & explosives",       150.0, True,   8.0],
        ["Fuel & power",                200.0, True,   6.0],
        ["Rail / road logistics",       180.0, True,  10.0],
        ["Downtime & shortfall losses", 250.0, True,  30.0],
        ["Environmental & statutory",   100.0, True,   0.0],
        ["Other / contingency",          80.0, True,   0.0],
    ], columns=COST_COLS)


def _default_ai_df():
    return pd.DataFrame([
        ["Satellite data & API (Sentinel-2 / Bhuvan)", 0.0, True],
        ["Model hosting / compute",                    0.5, True],
        ["Integration & deployment",                   2.0, False],
        ["Training & change management",               1.0, False],
        ["Maintenance & support",                      0.5, True],
    ], columns=AI_COLS)


def _clean(df, name_col):
    d = df.copy()
    d = d[d[name_col].notna() & (d[name_col].astype(str).str.strip() != "")]
    d["Cost (Rs lakh)"] = pd.to_numeric(
        d["Cost (Rs lakh)"], errors="coerce").fillna(0.0)
    d["Recurs yearly"] = d["Recurs yearly"].fillna(False).astype(bool)
    if "AI reduction (%)" in d.columns:
        d["AI reduction (%)"] = pd.to_numeric(
            d["AI reduction (%)"], errors="coerce").fillna(0.0).clip(0, 100)
    return d.reset_index(drop=True)


def fmt(x):
    """Rs lakh -> readable string (switches to crore above 100 lakh)."""
    return f"Rs {x / 100:,.2f} Cr" if abs(x) >= 100 else f"Rs {x:,.1f} L"


if "roi_cost_df" not in st.session_state:
    st.session_state["roi_cost_df"] = _default_cost_df()
    st.session_state["roi_ai_df"] = _default_ai_df()
    st.session_state["roi_years"] = 5
    st.session_state["roi_scenario"] = "Expected \u00b7 100%"

# ---------------- Admin console ----------------
saved_cost = _clean(st.session_state["roi_cost_df"], "Cost head")
saved_ai = _clean(st.session_state["roi_ai_df"], "AI platform item")

with st.container(key="card_roi_admin"):
    st.markdown(
        '<div class="roi-head"><div>'
        '<div class="ttl">Admin console</div>'
        '<div class="sub">Edit the cost model below, then press <b>Apply &amp; recompute</b> '
        'to refresh the comparison.</div></div>'
        f'<span class="chip">Scenario \u00b7 {html.escape(st.session_state["roi_scenario"])}</span></div>',
        unsafe_allow_html=True,
    )

    # live summary of what is currently saved
    one_time = saved_cost.loc[~saved_cost["Recurs yearly"],
                              "Cost (Rs lakh)"].sum()
    recurring = saved_cost.loc[saved_cost["Recurs yearly"],
                               "Cost (Rs lakh)"].sum()
    ai_yearly = saved_ai.loc[saved_ai["Recurs yearly"], "Cost (Rs lakh)"].sum()
    st.markdown(
        '<div class="mini-row">'
        f'<div class="mini"><div class="k">Cost heads</div><div class="v">{len(saved_cost)}</div></div>'
        f'<div class="mini blue"><div class="k">One-time (capex)</div><div class="v">{fmt(one_time)}</div></div>'
        f'<div class="mini amber"><div class="k">Recurring / year</div><div class="v">{fmt(recurring)}</div></div>'
        f'<div class="mini"><div class="k">AI platform / year</div><div class="v">{fmt(ai_yearly)}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("roi_form", border=False):
        tab_cost, tab_ai, tab_set = st.tabs(
            ["\u26cf  Cost heads", "\U0001f916  AI platform", "\u2699\ufe0f  Assumptions"])

        with tab_cost:
            st.caption("Add rows for any other cost head. Untick *Recurs yearly* for one-time "
                       "capex. *AI reduction* is the expected saving on that head.")
            cost_in = st.data_editor(
                st.session_state["roi_cost_df"],
                num_rows="dynamic", use_container_width=True, hide_index=True,
                key="roi_cost_editor", height=420,
                column_config={
                    "Cost head": st.column_config.TextColumn(width="large", required=True),
                    "Cost (Rs lakh)": st.column_config.NumberColumn(
                        min_value=0.0, format="%.1f", width="medium"),
                    "Recurs yearly": st.column_config.CheckboxColumn(width="small"),
                    "AI reduction (%)": st.column_config.NumberColumn(
                        min_value=0.0, max_value=100.0, format="%.0f", width="medium",
                        help="Expected cost reduction on this head with the AI system."),
                },
            )

        with tab_ai:
            st.caption("Everything it costs to run the AI system. Free satellite data and a "
                       "near-zero-cost API keep the recurring part small.")
            ai_in = st.data_editor(
                st.session_state["roi_ai_df"],
                num_rows="dynamic", use_container_width=True, hide_index=True,
                key="roi_ai_editor",
                column_config={
                    "AI platform item": st.column_config.TextColumn(width="large", required=True),
                    "Cost (Rs lakh)": st.column_config.NumberColumn(
                        min_value=0.0, format="%.2f", width="medium"),
                    "Recurs yearly": st.column_config.CheckboxColumn(width="small"),
                },
            )

        with tab_set:
            a1, a2 = st.columns(2)
            with a1:
                years_in = st.number_input(
                    "Analysis horizon (years)", 1, 15, int(st.session_state["roi_years"]))
            with a2:
                scenario_in = st.radio(
                    "AI effectiveness scenario", list(SCENARIOS.keys()),
                    index=list(SCENARIOS.keys()).index(
                        st.session_state["roi_scenario"]),
                    help="Scales every AI reduction %. 100% = exactly as entered.",
                )

        b1, b2 = st.columns([3, 1])
        applied = b1.form_submit_button("Apply & recompute", type="primary",
                                        use_container_width=True)
        reset = b2.form_submit_button("Reset", use_container_width=True)

    if applied:
        st.session_state["roi_cost_df"] = cost_in
        st.session_state["roi_ai_df"] = ai_in
        st.session_state["roi_years"] = int(years_in)
        st.session_state["roi_scenario"] = scenario_in
        st.rerun()
    if reset:
        st.session_state["roi_cost_df"] = _default_cost_df()
        st.session_state["roi_ai_df"] = _default_ai_df()
        st.session_state["roi_years"] = 5
        st.session_state["roi_scenario"] = "Expected \u00b7 100%"
        st.rerun()

# ---------------- Calculation ----------------
cost_df = saved_cost.copy()
ai_df = saved_ai.copy()
years = int(st.session_state["roi_years"])
scale = SCENARIOS[st.session_state["roi_scenario"]]

if cost_df.empty:
    st.info("Enter at least one cost head to see the comparison.")
else:
    cost_df["eff_red"] = (cost_df["AI reduction (%)"] * scale).clip(0, 100)
    cost_df["AI-assisted cost"] = cost_df["Cost (Rs lakh)"] * (
        1 - cost_df["eff_red"] / 100)

    def horizon_total(cost, recurs):
        return cost * np.where(recurs, years, 1)

    cost_df["trad_total"] = horizon_total(
        cost_df["Cost (Rs lakh)"], cost_df["Recurs yearly"])
    cost_df["ai_total"] = horizon_total(
        cost_df["AI-assisted cost"], cost_df["Recurs yearly"])
    ai_platform_total = float(horizon_total(
        ai_df["Cost (Rs lakh)"], ai_df["Recurs yearly"]).sum())

    trad_total = float(cost_df["trad_total"].sum())
    ai_total = float(cost_df["ai_total"].sum()) + ai_platform_total
    net_savings = trad_total - ai_total
    saved_pct = net_savings / trad_total * 100 if trad_total else 0.0

    def one_and_rec(costs, recurs):
        return float(costs[~recurs].sum()), float(costs[recurs].sum())

    t_one, t_rec = one_and_rec(
        cost_df["Cost (Rs lakh)"], cost_df["Recurs yearly"])
    a1_, a2_ = one_and_rec(
        cost_df["AI-assisted cost"], cost_df["Recurs yearly"])
    p1_, p2_ = one_and_rec(ai_df["Cost (Rs lakh)"], ai_df["Recurs yearly"])
    a_one, a_rec = a1_ + p1_, a2_ + p2_

    yrs = list(range(0, years + 1))
    cum_trad = [t_one + t_rec * y for y in yrs]
    cum_ai = [a_one + a_rec * y for y in yrs]

    payback = next((y for y, (t, a) in enumerate(
        zip(cum_trad, cum_ai)) if a <= t), None)
    payback_txt = ("Immediate" if payback ==
                   0 else f"Year {payback}" if payback else "Beyond horizon")
    roi_txt = f"{net_savings / ai_platform_total * 100:,.0f}%" if ai_platform_total > 0 else "\u221e"

    # ---------------- KPI tiles ----------------
    st.markdown(
        '<div class="kpi-grid">'
        f'<div class="kpi red"><div class="k">Traditional \u00b7 {years} yr</div>'
        f'<div class="v">{fmt(trad_total)}</div><div class="s">Total operating outlay</div></div>'
        f'<div class="kpi green"><div class="k">AI-assisted \u00b7 {years} yr</div>'
        f'<div class="v">{fmt(ai_total)}</div><div class="s">Incl. AI platform cost</div></div>'
        f'<div class="kpi green"><div class="k">Net savings</div>'
        f'<div class="v good">{fmt(net_savings)}</div><div class="s">{saved_pct:.1f}% lower than traditional</div></div>'
        f'<div class="kpi blue"><div class="k">ROI on AI spend</div>'
        f'<div class="v">{roi_txt}</div><div class="s">Payback: {payback_txt}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    legend_html = ('<div class="legend-strip">'
                   '<span><i style="background:#ef4444"></i>Traditional</span>'
                   f'<span><i style="background:{ACCENT}"></i>AI-assisted</span></div>')

    # ---------------- Side-by-side charts ----------------
    import textwrap

    n_rows = len(cost_df) + 1
    # both charts share one height
    CHART_H = max(460, 80 + 48 * n_rows)
    # one shared legend for both cards
    st.markdown(legend_html, unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="medium")

    # ---- Chart 1: cost by head ----
    with ch1:
        with st.container(key="card_roi_bar"):
            st.markdown(
                '<div class="chart-title">Cost by head</div>'
                f'<div class="chart-sub">Total spend over {years} year(s), Rs lakh. '
                'Hover a bar for the % reduction.</div>',
                unsafe_allow_html=True)

            bar_df = pd.DataFrame({
                "head": list(cost_df["Cost head"]) + ["AI platform (API + hosting)"],
                "trad": list(cost_df["trad_total"]) + [0.0],
                "ai": list(cost_df["ai_total"]) + [ai_platform_total],
                # biggest head ends up on top
            }).sort_values("trad", ascending=True)
            bar_df["label"] = ["<br>".join(
                textwrap.wrap(str(h), 20)) for h in bar_df["head"]]
            bar_df["red"] = [(1 - a / t) * 100 if t > 0 else 0.0
                             for t, a in zip(bar_df["trad"], bar_df["ai"])]

            trad_txt = [f"{t:,.0f}" if t > 0 else "" for t in bar_df["trad"]]
            ai_txt = [(f"{a:,.0f}" if a >= 10 else f"{a:,.1f}") if a > 0 else ""
                      for a in bar_df["ai"]]
            x_max = max(float(bar_df["trad"].max()),
                        float(bar_df["ai"].max()), 1.0) * 1.25

            fig_bar = go.Figure()
            # AI first, Traditional second -> Traditional sits on top of each pair
            fig_bar.add_bar(
                y=bar_df["label"], x=bar_df["ai"], name="AI-assisted", orientation="h",
                marker_color=ACCENT, text=ai_txt, textposition="outside", cliponaxis=False,
                textfont=dict(size=11, color=ACCENT), customdata=bar_df["red"],
                hovertemplate="<b>%{y}</b><br>AI-assisted: Rs %{x:,.1f} L"
                              "<br>Reduction: %{customdata:.0f}%<extra></extra>")
            fig_bar.add_bar(
                y=bar_df["label"], x=bar_df["trad"], name="Traditional", orientation="h",
                marker_color="#ef4444", text=trad_txt, textposition="outside", cliponaxis=False,
                textfont=dict(size=11, color=TEXT_MUTED),
                hovertemplate="<b>%{y}</b><br>Traditional: Rs %{x:,.1f} L<extra></extra>")
            fig_bar.update_layout(
                template=PLOTLY_TEMPLATE, barmode="group", bargap=0.30, bargroupgap=0.05,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=PLOTLY_FONT, showlegend=False, height=CHART_H,
                xaxis=dict(title=dict(text="Rs lakh", standoff=12), range=[0, x_max],
                           gridcolor="rgba(255,255,255,0.06)", zeroline=False, nticks=5),
                yaxis=dict(automargin=True, tickfont=dict(size=11)),
                margin=dict(l=5, r=10, t=5, b=55),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Chart 2: cumulative spend ----
    with ch2:
        with st.container(key="card_roi_line"):
            st.markdown(
                '<div class="chart-title">Cumulative spend</div>'
                f'<div class="chart-sub">Shaded gap = money saved: '
                f'<b style="color:{ACCENT};">{fmt(net_savings)}</b> over {years} year(s)</div>',
                unsafe_allow_html=True)

            fig_cum = go.Figure()
            fig_cum.add_scatter(x=yrs, y=cum_trad, name="Traditional", mode="lines+markers",
                                line=dict(color="#ef4444", width=3), marker=dict(size=8))
            fig_cum.add_scatter(x=yrs, y=cum_ai, name="AI-assisted", mode="lines+markers",
                                line=dict(color=ACCENT, width=3), marker=dict(size=8),
                                fill="tonexty", fillcolor="rgba(16,185,129,0.14)")
            fig_cum.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=PLOTLY_FONT, showlegend=False, hovermode="x unified",
                height=CHART_H,
                xaxis=dict(title=dict(text="Year (0 = upfront)", standoff=12),
                           tickvals=yrs, range=[-0.2, years + 0.2],
                           gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title=dict(text="Rs lakh", standoff=8), rangemode="tozero",
                           tickformat=",", nticks=6, gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(l=5, r=15, t=5, b=55),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

    st.caption("All figures come from the admin inputs above; defaults are placeholders, not MOIL "
               "data. One-time heads count once, recurring heads count every year. Undiscounted.")
components.html(
    """
    <script>
    const doc = window.parent.document;
    if (!doc.__chartZoomBound) {
      doc.__chartZoomBound = true;
      const SEL = '[class*="st-key-chartcard"]';
      const closeAll = (except) =>
        doc.querySelectorAll(SEL + '.zoomed').forEach(c => { if (c !== except) c.classList.remove('zoomed'); });

      // capture phase: runs before Streamlit's own click handlers, so no fullscreen
      doc.addEventListener('click', (e) => {
        const card = e.target.closest(SEL);
        closeAll(card);
        if (card) {
          e.preventDefault();
          e.stopPropagation();
          card.classList.toggle('zoomed');
        }
      }, true);

      doc.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeAll(null); });
    }
    </script>
    """,
    height=0,
)
