import base64
import os
from model import MOILManganeseAI
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="MOIL Predictive Intelligence",
    layout="wide",
    page_icon="image.png",
)

def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()

        ext = image_path.split(".")[-1].lower()
        mime = "png" if ext == "png" else "jpeg"

        style = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:background/{mime};base64,{b64_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0,0,0,0);
        }}
        /* Adds a slight dark overlay to card containers for readability */
        .st-bw, div[data-testid="stVerticalBlock"] > div {{
            background-color: rgba(17, 24, 39, 0.7);
            border-radius: 8px;
            padding: 8px;
        }}
        </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    else:
        st.warning(f"Background image '{image_path}' not found in folder.")

set_background("background.png")

@st.cache_resource
def init_system():
    ai = MOILManganeseAI()
    res_df, ops_df = ai.train_models()
    return ai, res_df, ops_df


ai_engine, res_df, ops_df = init_system()

col_logo, col_title = st.columns([1, 8])
with col_title:
    st.title("MOIL AI & Space Tech Command Center")
    st.markdown(
        "**Predictive Intelligence for Manganese Reserve Mapping & Operational"
        " Shortfall Mitigation**"
    )

st.markdown("---")

st.subheader("1. Strategic Overview (Historical & Market Context)")
st.markdown(
    "Based on data extracted from the IBM Indian Minerals Yearbook 2024 and"
    " Annual Report 2019-20."
)

col_c1, col_c2, col_c3 = st.columns(3)


def render_chart(filepath, column, caption):
    if os.path.exists(filepath):
        column.image(filepath, use_container_width=True, caption=caption)
    else:
        column.warning(
            f"Chart missing: Ensure '{filepath}' exists in the repository."
        )


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

st.subheader("2. Sub-surface Reserve Mapping (AI + Earth Observation)")
st.markdown("""
Using simulated Sentinel-2/Bhuvan satellite telemetry (NDVI, LST) fused with geological drill hole depth. 
The Random Forest regressor maps spatial continuity to predict high-yield (>46% Mn) ore bodies.
""")

map_col, chart_col = st.columns([2, 1])

with map_col:
    sample_df = res_df.sample(250).copy()
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

    fig_map = px.scatter_mapbox(
        sample_df,
        lat="latitude",
        lon="longitude",
        color="Predicted_Mn_Grade",
        size="depth_m",
        color_continuous_scale="Turbo",
        zoom=7,
        center={"lat": 21.65, "lon": 80.0},
        title="Predictive Mineralization Heatmap (Balaghat-Nagpur Belt)",
        hover_data=["depth_m", "ndvi", "lst_celsius"],
    )
    fig_map.update_layout(
        mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

with chart_col:
    render_chart(
        "charts/03_reserves_by_state.png",
        st,
        "Current Proved vs Remaining Resources",
    )
    st.info(
        "**Insight:** Notice the massive 'Remaining Resources' volume in"
        " Odisha, Karnataka, and Goa. AI-driven mapping allows MOIL to efficiently"
        " convert these into 'Proved Reserves' without excessive manual drilling."
    )

st.markdown("---")

st.subheader("3. Real-Time Operational Shortfall Predictor")
st.markdown(
    "Adjust the daily telemetry sliders below to simulate mine-site conditions"
    " and observe the AI engine's prescriptive actions."
)

sc1, sc2 = st.columns([1, 2])

with sc1:
    st.markdown("#### Input Operational Parameters")
    in_rain = st.slider("Rainfall Forecast (mm)", 0.0,
                        100.0, 15.0, format="%.1f")
    in_downtime = st.slider(
        "Total Equipment Downtime (hrs)", 0.0, 24.0, 3.5, format="%.1f"
    )
    in_blast = st.slider("Blasting Delay (hrs)", 0.0, 12.0, 1.0, format="%.1f")
    in_wagon = st.slider(
        "Rail Wagon Availability Ratio", 0.0, 1.0, 0.95, format="%.2f"
    )

with sc2:
    st.markdown("#### AI Risk Assessment & Prescriptive Actions")
    prob_shortfall, pred_label = ai_engine.predict_shortfall_risk(
        in_rain, in_downtime, in_blast, in_wagon
    )

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_shortfall * 100,
            title={"text": "Probability of Production Shortfall (%)"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "gold"},
                    {"range": [70, 100], "color": "tomato"},
                ],
            },
        )
    )
    fig_gauge.update_layout(
        height=250, margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    actions = ai_engine.get_prescriptive_actions(
        in_rain, in_downtime, in_blast, in_wagon
    )

    if pred_label == 1:
        st.error(
            "**SHORTFALL IMMINENT:** AI Model flags a high probability of"
            " missing daily production targets."
        )
    else:
        st.success(
            "**TARGETS ON TRACK:** Current operating parameters are within"
            " safety margins."
        )

    st.markdown("### Prescriptive Recommendations:")
    for action in actions:
        st.markdown(f"- {action}")
