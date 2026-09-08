import base64
import os
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
# THEME — dark SaaS / terminal aesthetic (inspired by v0 "Nexus AI" style)
# Change ACCENT to re-skin the whole app to a different single accent color.
# -----------------------------------------------------------------------------
# emerald — swap to "#3b82f6" (blue) / "#f59e0b" (amber) etc.
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

/* Kill Streamlit's default top padding/header bar look */
[data-testid="stHeader"] {{
    background-color: transparent;
}}
.block-container {{
    padding-top: 2.5rem;
    max-width: 1200px;
}}

/* Section eyebrow label — "// SECTION" style */
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

/* Big headline under the eyebrow */
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

/* Hero */
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

/* Card container used for image / chart blocks */
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

/* Streamlit image captions */
[data-testid="stImage"] > div {{
    border-radius: 8px;
    overflow: hidden;
}}
[data-testid="caption"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* Metric-style number chips */
.stat-chip {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: {ACCENT};
}}
.stat-label {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* Divider */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 2.5rem 0;
}}

/* Sliders */
[data-testid="stSlider"] [role="slider"] {{
    background-color: {ACCENT} !important;
}}
.stSlider [data-baseweb="slider"] > div > div {{
    background: {ACCENT} !important;
}}

/* Alert boxes recolored to fit dark theme */
[data-testid="stAlert"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 8px;
}}
div[data-testid="stAlert"][data-baseweb] p {{
    color: {TEXT};
}}

/* Subheader monospace tag look for st.subheader calls we keep as-is */
h2, h3 {{
    color: {TEXT};
    font-weight: 700;
}}

/* Remove default translucent panel Streamlit adds on vertical blocks */
div[data-testid="stVerticalBlock"] > div {{
    background-color: transparent;
    padding: 0;
}}
/* Custom cursor — user-supplied pickaxe artwork, background removed */
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

/* Top loading bar — filled in by JS below, shown while Streamlit reruns */
#top-loading-bar {{
    position: fixed;
    top: 0; left: 0;
    height: 3px;
    width: 100%;
    background: linear-gradient(90deg, transparent, {ACCENT}, transparent);
    background-size: 50% 100%;
    z-index: 999999;
    opacity: 0;
    transition: opacity 0.2s ease;
    pointer-events: none;
}}
#top-loading-bar.active {{
    opacity: 1;
    animation: top-bar-scan 1s linear infinite;
}}
@keyframes top-bar-scan {{
    0% {{ background-position: 150% 0; }}
    100% {{ background-position: -50% 0; }}
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Inject the top loading bar element into the *parent* document (the CSS
# above only styles it) and toggle it on/off based on Streamlit's own
# "running" status widget. This only needs to run once — it guards against
# re-injecting itself on every rerun.
_LOADING_BAR_JS = """
<script>
(function() {
    const doc = window.parent.document;
    if (doc.getElementById('top-loading-bar')) return;

    const bar = doc.createElement('div');
    bar.id = 'top-loading-bar';
    doc.body.appendChild(bar);

    const observer = new MutationObserver(function() {
        const running = doc.querySelector('[data-testid="stStatusWidget"]');
        bar.classList.toggle('active', !!running);
    });
    observer.observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""
components.html(_LOADING_BAR_JS, height=0)

# Ore-mining animation that plays when the user scrolls to the bottom of the
# page: a pickaxe swings at a rock, it shatters into ore chunks, and a
# "vein detected" pill fades in. Re-triggers (with a cooldown) each time the
# user re-enters the bottom of the page, not just once per session.
_MINING_ANIMATION_JS = f"""
<script>
(function() {{
    const doc = window.parent.document;
    if (doc.getElementById('mining-overlay')) return;

    const style = doc.createElement('style');
    style.innerHTML = `
        #mining-overlay {{
            position: fixed;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999999;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.4s ease;
        }}
        #mining-overlay.active {{ opacity: 1; }}
        #mining-overlay .mining-scene {{
            position: relative;
            width: 120px;
            height: 70px;
        }}
        #mining-overlay .rock {{
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            font-size: 40px;
            transition: opacity 0.15s ease, transform 0.15s ease;
        }}
        #mining-overlay .rock.shatter {{
            opacity: 0;
            transform: translate(-50%, -50%) scale(1.4);
        }}
        #mining-overlay .rock.shake {{
            animation: rock-shake 0.3s ease-in-out;
        }}
        @keyframes rock-shake {{
            0%, 100% {{ transform: translate(-50%, -50%) rotate(0deg); }}
            25%      {{ transform: translate(-56%, -50%) rotate(-6deg); }}
            75%      {{ transform: translate(-44%, -50%) rotate(6deg); }}
        }}
        #mining-overlay .pick {{
            position: absolute;
            left: 6px;
            bottom: 2px;
            font-size: 32px;
            transform-origin: 85% 85%;
            opacity: 0;
        }}
        #mining-overlay .pick.swing {{
            opacity: 1;
            animation: pick-swing 0.4s ease-in-out 3;
        }}
        @keyframes pick-swing {{
            0%   {{ transform: rotate(-25deg); }}
            50%  {{ transform: rotate(20deg); }}
            100% {{ transform: rotate(-25deg); }}
        }}
        #mining-overlay .ore-bit {{
            position: absolute;
            left: 50%;
            top: 50%;
            font-size: 18px;
            opacity: 0;
        }}
        #mining-overlay .ore-bit.pop {{
            animation: ore-pop 0.9s ease-out forwards;
        }}
        @keyframes ore-pop {{
            0%   {{ opacity: 1; transform: translate(-50%, -50%) scale(0.6); }}
            100% {{ opacity: 0; transform: translate(calc(-50% + var(--dx)), calc(-50% + var(--dy))) scale(1.1); }}
        }}
        #mining-overlay .mining-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            color: {ACCENT};
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 999px;
            padding: 4px 14px;
            opacity: 0;
            transition: opacity 0.3s ease;
            white-space: nowrap;
        }}
        #mining-overlay .mining-label.show {{ opacity: 1; }}
    `;
    doc.head.appendChild(style);

    const overlay = doc.createElement('div');
    overlay.id = 'mining-overlay';
    overlay.innerHTML = `
        <div class="mining-scene">
            <div class="rock">🪨</div>
            <div class="pick">⛏️</div>
        </div>
        <div class="mining-label">Manganese ore vein detected</div>
    `;
    doc.body.appendChild(overlay);

    const rock = overlay.querySelector('.rock');
    const pick = overlay.querySelector('.pick');
    const label = overlay.querySelector('.mining-label');
    const scene = overlay.querySelector('.mining-scene');

    let cooldown = false;

    function playMiningAnimation() {{
        if (cooldown) return;
        cooldown = true;

        overlay.classList.add('active');
        pick.classList.remove('swing');
        rock.classList.remove('shatter', 'shake');
        label.classList.remove('show');
        void overlay.offsetWidth; // force reflow so animations restart cleanly

        pick.classList.add('swing');

        setTimeout(function() {{ rock.classList.add('shake'); }}, 900);

        setTimeout(function() {{
            rock.classList.add('shatter');
            pick.classList.remove('swing');

            for (let i = 0; i < 6; i++) {{
                const bit = doc.createElement('div');
                bit.className = 'ore-bit pop';
                bit.textContent = '💎';
                const angle = (Math.PI * 2 * i) / 6;
                const dist = 40 + Math.random() * 20;
                bit.style.setProperty('--dx', (Math.cos(angle) * dist) + 'px');
                bit.style.setProperty('--dy', (Math.sin(angle) * dist) + 'px');
                scene.appendChild(bit);
                setTimeout(function() {{ bit.remove(); }}, 1000);
            }}

            label.classList.add('show');
        }}, 1250);

        setTimeout(function() {{ overlay.classList.remove('active'); }}, 4200);
        setTimeout(function() {{ cooldown = false; }}, 6000);
    }}

    function isNearBottom() {{
        const scrollY = doc.documentElement.scrollTop || doc.body.scrollTop;
        const viewport = window.parent.innerHeight;
        const full = doc.documentElement.scrollHeight;
        return scrollY + viewport >= full - 40;
    }}

    // Debug hook — run window.__playMiningAnimation() in the browser
    // console to fire it manually regardless of scroll position.
    window.parent.__playMiningAnimation = playMiningAnimation;

    let wasAtBottom = false;
    window.parent.addEventListener('scroll', function() {{
        const atBottom = isNearBottom();
        if (atBottom && !wasAtBottom) {{
            playMiningAnimation();
        }}
        wasAtBottom = atBottom;
    }}, {{ passive: true }});

    // Also check shortly after load — handles pages short enough that
    // they're already "at the bottom" with no scroll event ever firing,
    // and restored scroll positions after a refresh.
    setTimeout(function() {{
        wasAtBottom = isNearBottom();
        if (wasAtBottom) {{
            playMiningAnimation();
        }}
    }}, 1200);
}})();
</script>
"""
components.html(_MINING_ANIMATION_JS, height=0)

# Dark plotly template so charts match the theme
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


@st.cache_resource
def init_system():
    ai = MOILManganeseAI()
    res_df, ops_df = ai.train_models()
    return ai, res_df, ops_df


ai_engine, res_df, ops_df = init_system()

# 4. Hero Section
st.markdown(
    '<div class="eyebrow">MOIL AI &amp; Space Tech</div>'
    '<div class="hero-title">Predictive intelligence for<br><span>manganese reserve mapping.</span></div>'
    '<div class="hero-sub">A command center for reserve discovery and operational '
    'shortfall mitigation — fusing satellite telemetry, geological data, and '
    'machine learning into a single live view.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# 5. Section 1: Strategic Overview
section_header(
    "Platform",
    "Strategic overview.",
    "Based on data extracted from the IBM Indian Minerals Yearbook 2024 and Annual Report 2019-20.",
)

col_c1, col_c2, col_c3 = st.columns(3)


def render_chart(filepath, column, caption):
    # Works whether 'column' is a real st.columns() object or the st module
    # itself — both expose .markdown/.image/.warning, but only the former
    # supports "with column:".
    column.markdown('<div class="moil-card">', unsafe_allow_html=True)
    if os.path.exists(filepath):
        column.image(filepath, use_container_width=True, caption=caption)
    else:
        column.warning(
            f"Chart missing: Ensure '{filepath}' exists in the repository.")
    column.markdown("</div>", unsafe_allow_html=True)


render_chart(
    "charts/01_state_production_trend.png",
    col_c1,
    "State-wise Production Trajectory",
)
render_chart(
    "charts/05_district_grade_stacked.png",
    col_c2,
    "District Output by Grade (Balaghat Dominance)",
)
render_chart(
    "charts/06_mine_size_pareto.png",
    col_c3,
    "Pareto Law: 10% Mines = 76% Output",
)

st.markdown("---")

# 6. Section 2: Sub-surface Reserve Mapping
section_header(
    "Technology",
    "Sub-surface reserve mapping.",
    "Simulated Sentinel-2 / Bhuvan satellite telemetry (NDVI, LST) fused with geological "
    "drill hole depth. A Random Forest regressor maps spatial continuity to predict "
    "high-yield (>46% Mn) ore bodies.",
)

map_col, chart_col = st.columns([2, 1])

with map_col:
    st.markdown('<div class="moil-card">', unsafe_allow_html=True)

    selected_belt = st.selectbox(
        "Manganese belt", list(ai_engine.BELTS.keys()), index=0,
        label_visibility="collapsed",
    )
    belt_cfg = ai_engine.BELTS[selected_belt]

    sample_df = ai_engine.generate_belt_data(selected_belt, n_samples=250)
    sample_df["Predicted_Mn_Grade"] = ai_engine.predict_reserve_grid(
        sample_df[[
            "latitude",
            "longitude",
            "depth_m",
            "ndvi",
            "lst_celsius",
            "soil_moisture_pct",
            "mag_susceptibility",
        ]]
    )

    # scatter_map (MapLibre) replaces the deprecated scatter_mapbox.
    # "open-street-map" is a light, always-free basemap (no API key) —
    # unlike CARTO's darkmatter/positron tiles, which now require one.
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
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col:
    render_chart(
        "charts/03_reserves_by_state.png",
        st,
        "Current Proved vs Remaining Resources",
    )
    st.markdown(
        '<div class="moil-card" style="margin-top:0.75rem;">'
        '<span style="color:{0};font-weight:600;">Insight —</span> '
        'Notice the massive "Remaining Resources" volume in Odisha, Karnataka, '
        'and Goa. AI-driven mapping allows MOIL to efficiently convert these into '
        '"Proved Reserves" without excessive manual drilling.'
        '</div>'.format(ACCENT),
        unsafe_allow_html=True,
    )

st.markdown("---")

# 7. Section 3: Operational Shortfall Predictor
section_header(
    "Live Simulation",
    "Operational shortfall predictor.",
    "Adjust the daily telemetry sliders to simulate mine-site conditions and observe "
    "the AI engine's prescriptive actions in real time.",
)

sc1, sc2 = st.columns([1, 2])

with sc1:
    st.markdown('<div class="moil-card">', unsafe_allow_html=True)
    st.markdown('<div class="stat-label">Input Parameters</div>',
                unsafe_allow_html=True)
    st.write("")
    in_rain = st.slider("Rainfall Forecast (mm)", 0.0,
                        100.0, 15.0, format="%.1f")
    in_downtime = st.slider(
        "Total Equipment Downtime (hrs)", 0.0, 24.0, 3.5, format="%.1f"
    )
    in_blast = st.slider("Blasting Delay (hrs)", 0.0, 12.0, 1.0, format="%.1f")
    in_wagon = st.slider(
        "Rail Wagon Availability Ratio", 0.0, 1.0, 0.95, format="%.2f"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with sc2:
    prob_shortfall, pred_label = ai_engine.predict_shortfall_risk(
        in_rain, in_downtime, in_blast, in_wagon
    )

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_shortfall * 100,
            title={"text": "Probability of Production Shortfall (%)", "font": {
                "color": TEXT}},
            number={"font": {"color": ACCENT}},
            gauge={
                "axis": {"range": [None, 100], "tickcolor": TEXT_MUTED},
                "bar": {"color": ACCENT},
                "bgcolor": BG_CARD,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [0, 40], "color": "rgba(16,185,129,0.25)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.25)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.25)"},
                ],
            },
        )
    )
    fig_gauge.update_layout(
        height=250,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        paper_bgcolor=PLOTLY_PAPER_BG,
        font=PLOTLY_FONT,
    )
    st.markdown('<div class="moil-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True)

    actions = ai_engine.get_prescriptive_actions(
        in_rain, in_downtime, in_blast, in_wagon
    )

    if pred_label == 1:
        st.error(
            "**SHORTFALL IMMINENT** — AI model flags a high probability of "
            "missing daily production targets."
        )
    else:
        st.success(
            "**TARGETS ON TRACK** — Current operating parameters are within "
            "safety margins."
        )

    st.markdown('<div class="stat-label" style="margin-top:0.75rem;">Prescriptive Recommendations</div>',
                unsafe_allow_html=True)
    for action in actions:
        st.markdown(f"- {action}")
    st.markdown("</div>", unsafe_allow_html=True)
