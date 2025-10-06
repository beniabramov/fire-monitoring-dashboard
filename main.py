import pandas as pd
import streamlit as st
from io import StringIO
import requests
import pydeck as pdk
import numpy as np
from enum import Enum
from typing import Optional
from datetime import datetime
import plotly.express as px


# ---------------------------
# Constants
# ---------------------------
class Config:
    """Configuration constants for the dashboard"""

    MAP_KEY = "5ed184d05a60adad4d942f0b52d8c2ea"
    PRODUCT = "VIIRS_NOAA20_NRT"
    DAYS = 1
    CACHE_TTL_SECONDS = 900
    REQUEST_TIMEOUT = 60


class MapConfig:
    """Map visualization constants"""

    DEFAULT_LATITUDE = 0
    DEFAULT_LONGITUDE = 0
    DEFAULT_ZOOM = 1.5
    MIN_RADIUS = 800
    MAX_RADIUS = 6000
    OPACITY = 0.65
    FIRE_COLOR = [255, 50, 0, 220]  # Bright red-orange for excellent visibility


class DataColumns(Enum):
    """Data column names"""

    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    FRP = "frp"
    BRIGHTNESS = "brightness"
    BRIGHT_TI4 = "bright_ti4"
    BRIGHT_TI5 = "bright_ti5"
    CONFIDENCE = "confidence"
    ACQ_DATE = "acq_date"
    ACQ_TIME = "acq_time"
    INTENSITY = "intensity"


# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Global Fire Monitoring Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for stunning UI/UX
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main, .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%) !important;
        color: #1a1a1a !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stText {
        color: #1a1a1a !important;
    }
    
    /* Stunning Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff4500 0%, #ff8c00 50%, #ffa500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 2px 2px 20px rgba(255, 69, 0, 0.3);
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Beautiful Info Box */
    .info-box {
        background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 100%);
        padding: 2rem;
        border-radius: 16px;
        border: none;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(33, 150, 243, 0.15);
        position: relative;
        overflow: hidden;
        animation: fadeIn 1s ease-out;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(180deg, #2196f3 0%, #64b5f6 100%);
    }
    
    .info-box strong {
        color: #1565c0 !important;
        font-size: 1.1rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Modern Tab-Style Navigation */
    div.stButton > button {
        width: 100%;
        border-radius: 16px;
        height: 4rem;
        font-weight: 700;
        font-size: 1.15rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        background: white;
        color: #666 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        position: relative;
        overflow: hidden;
    }
    
    div.stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    div.stButton > button:hover::before {
        left: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        color: #333 !important;
    }
    
    /* Active Tab - Stunning Gradient */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ff4500 0%, #ff6b35 50%, #ff8c00 100%) !important;
        color: white !important;
        box-shadow: 0 8px 24px rgba(255, 69, 0, 0.4) !important;
        border: none !important;
        transform: translateY(-2px);
    }
    
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 12px 32px rgba(255, 69, 0, 0.5) !important;
        transform: translateY(-6px) scale(1.02);
    }
    
    /* Hide Sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    button[kind="header"] {
        display: none !important;
    }
    
    /* Remove layout wrapper padding */
    .stLayoutWrapper {
        padding: 0 !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Stunning Metrics Cards */
    div[data-testid="stMetric"] {
        background: white;
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: none;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ff4500, #ff8c00);
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    
    div[data-testid="stMetric"] label {
        color: #666 !important;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        line-height: 1.2;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Tooltip Icon - Subtle and Clean */
    [data-testid="stTooltipHoverTarget"] {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: #666 !important;
        opacity: 0.85 !important;
        visibility: visible !important;
        margin-left: 0.4rem !important;
        vertical-align: middle !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stTooltipHoverTarget"] svg {
        width: 0.9rem !important;
        height: 0.9rem !important;
        fill: currentColor !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
    }
    
    [data-testid="stTooltipHoverTarget"]:hover {
        color: #ff6347 !important;
        opacity: 1 !important;
        transform: scale(1.15) !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stTooltipHoverTarget"] {
        display: inline-flex !important;
        opacity: 0.8 !important;
        visibility: visible !important;
    }
    
    div[data-testid="stMetric"]:hover [data-testid="stTooltipHoverTarget"] {
        opacity: 1 !important;
    }
    
    /* Force show tooltips in metrics */
    div[data-testid="stMetric"] label [data-testid="stTooltipHoverTarget"] {
        display: inline-flex !important;
    }
    
    /* Beautiful Headers */
    h1, h2, h3 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        font-size: 2rem !important;
        margin-top: 2rem !important;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #ff4500, #ff8c00);
        border-radius: 2px;
    }
    
    h3 {
        font-size: 1.5rem !important;
        color: #2c3e50 !important;
    }
    
    /* Premium Dataframe */
    .dataframe {
        background-color: white !important;
        border: none !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        color: #2c3e50 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        padding: 1rem !important;
        border: none !important;
    }
    
    .dataframe tbody tr td {
        color: #1a1a1a !important;
        padding: 0.9rem 1rem !important;
        border-bottom: 1px solid #f0f0f0 !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f8f9fa !important;
    }
    
    /* Chart Containers */
    .stPlotlyChart, .stPydeckChart {
        background-color: white !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
        border: none !important;
    }
    
    /* Elegant Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
        margin: 2.5rem 0;
    }
    
    /* Vibrant Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff4500, #ff8c00) !important;
        border-radius: 10px;
    }
    
    .stProgress > div {
        background-color: #f0f0f0;
        border-radius: 10px;
        height: 12px;
    }
    
    /* Alert Boxes */
    .stAlert {
        background-color: white !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
        padding: 1.5rem !important;
    }
    
    /* Slider Styling */
    .stSlider {
        padding: 1.5rem 0;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #ff4500, #ff8c00);
    }
    
    /* Captions */
    .stCaption {
        color: #666 !important;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Smooth Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff4500, #ff8c00);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #ff3500, #ff7c00);
    }
    
    /* Info message styling */
    div[data-testid="stMarkdownContainer"] p {
        line-height: 1.6;
    }
    
    /* Expander Styling - Light Theme */
    div[data-testid="stExpander"] {
        background: white !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    
    div[data-testid="stExpander"] details {
        background: white !important;
    }
    
    div[data-testid="stExpander"] summary {
        background: white !important;
        color: #333 !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    div[data-testid="stExpander"] summary:hover {
        background: #f8f9fa !important;
    }
    
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background: white !important;
        color: #333 !important;
        padding: 1rem !important;
    }
    
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] li,
    div[data-testid="stExpander"] strong {
        color: #333 !important;
        background: transparent !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%) !important;
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
    }
    
    /* Selectbox styling */
    div[data-testid="stSelectbox"] {
        color: #1a1a1a !important;
    }
    
    div[data-testid="stSelectbox"] label {
        color: #1a1a1a !important;
        font-weight: 600;
    }
    
    /* Multiselect styling */
    div[data-testid="stMultiSelect"] label {
        color: #1a1a1a !important;
        font-weight: 600;
    }
    
    /* Enhanced map container */
    .stPydeckChart > div {
        border-radius: 16px !important;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for routing
if "current_view" not in st.session_state:
    st.session_state.current_view = "overview"


# ---------------------------
# Data Fetching
# ---------------------------
@st.cache_data(ttl=Config.CACHE_TTL_SECONDS, show_spinner=False)
def fetch_firms_data(map_key: str, product: str, days: int) -> pd.DataFrame:
    """
    Fetch fire detection data from NASA FIRMS API

    Args:
        map_key: API key for FIRMS
        product: Satellite product identifier
        days: Number of days of data to fetch

    Returns:
        DataFrame with fire detection data
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{product}/world/{days}"
    response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def get_intensity_metric(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Determine and apply the best intensity metric from available columns

    Args:
        df: DataFrame with fire data

    Returns:
        Tuple of (processed DataFrame, metric column name)
    """
    intensity_candidates = [
        DataColumns.FRP.value,
        DataColumns.BRIGHTNESS.value,
        DataColumns.BRIGHT_TI4.value,
        DataColumns.BRIGHT_TI5.value,
    ]

    for candidate in intensity_candidates:
        if candidate in df.columns:
            df_copy = df.copy()
            df_copy.rename(
                columns={candidate: DataColumns.INTENSITY.value}, inplace=True
            )
            return df_copy, DataColumns.INTENSITY.value

    # Fallback: create dummy intensity
    df_copy = df.copy()
    df_copy[DataColumns.INTENSITY.value] = 1.0
    return df_copy, DataColumns.INTENSITY.value


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate key statistics from fire data"""
    # Calculate high confidence - handle both numeric and categorical confidence values
    high_confidence = 0
    if "confidence" in df.columns:
        if pd.api.types.is_numeric_dtype(df["confidence"]):
            # Numeric confidence (e.g., percentage)
            high_confidence = len(df[df["confidence"] > 80])
        else:
            # Categorical confidence (h/n/l)
            high_confidence = len(df[df["confidence"] == "h"])

    return {
        "total_fires": len(df),
        "countries": (
            df["latitude"].apply(lambda x: "Multiple").nunique() if len(df) > 0 else 0
        ),
        "avg_brightness": df["brightness"].mean() if "brightness" in df.columns else 0,
        "high_confidence": high_confidence,
    }


# ---------------------------
# Dashboard Header
# ---------------------------
st.markdown(
    '<h1 class="main-header">🌍 Global Fire Monitoring Dashboard</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; font-size: 1.2rem; color: #666; margin-top: -1rem; margin-bottom: 1rem;"><strong>Today\'s Data</strong> | Real-time Satellite Data</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    <strong>Real-time Fire Detection System - Today's Data Only</strong><br>
    Monitor thermal anomalies and fire detections worldwide using NASA's FIRMS (Fire Information for Resource Management System) data.<br>
    <span style="color: #ff6347; font-weight: 600;">⏰ All data shown represents detections from today (UTC timezone)</span> - fires may have been extinguished since detection.<br>
    Data from VIIRS NOAA-20 satellite, updated every 15 minutes.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Fetch Data
# ---------------------------
with st.spinner("🔄 Loading fire detection data..."):
    try:
        df_raw = fetch_firms_data(Config.MAP_KEY, Config.PRODUCT, Config.DAYS)
        data_loaded = True
        last_update = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        data_loaded = False

# ---------------------------
# Navigation Buttons
# ---------------------------
if data_loaded:
    # Calculate statistics
    stats = calculate_statistics(df_raw)

    # Create navigation buttons
    nav_col1, nav_col2, nav_col3 = st.columns(3)

    with nav_col1:
        if st.button(
            "📊 Overview",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.current_view == "overview"
                else "secondary"
            ),
        ):
            st.session_state.current_view = "overview"
            st.rerun()

    with nav_col3:
        if st.button(
            "🗺️ Interactive Map",
            use_container_width=True,
            type="primary" if st.session_state.current_view == "map" else "secondary",
        ):
            st.session_state.current_view = "map"
            st.rerun()

    with nav_col2:
        if st.button(
            "⏰ Hourly Analysis",
            use_container_width=True,
            type=(
                "primary" if st.session_state.current_view == "hourly" else "secondary"
            ),
        ):
            st.session_state.current_view = "hourly"
            st.rerun()

    # ---------------------------
    # Main Content Area - Routing
    # ---------------------------

    if st.session_state.current_view == "overview":
        st.markdown("## 📊 Fire Detection Overview - Today")
        st.markdown(
            "Real-time global fire and thermal anomaly monitoring powered by NASA satellite data"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Calculate additional metrics
        countries_affected = df_raw[DataColumns.LATITUDE.value].count()

        # Recent fires (last 6 hours)
        if DataColumns.ACQ_TIME.value in df_raw.columns:
            df_with_time = df_raw.copy()
            df_with_time["hour"] = (
                df_with_time[DataColumns.ACQ_TIME.value]
                .astype(str)
                .str.zfill(4)
                .str[:2]
                .astype(int)
            )
            current_hour = datetime.now().hour
            recent_hours = [(current_hour - i) % 24 for i in range(6)]
            recent_fires = df_with_time[df_with_time["hour"].isin(recent_hours)].shape[
                0
            ]
        else:
            recent_fires = 0

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="🔥 Total Fire Detections",
                value=f"{stats['total_fires']:,}",
                help="Total number of thermal anomalies and fire detections identified by satellites today",
            )

        with col2:
            st.metric(
                label="⏰ Recent Detections",
                value=f"{recent_fires:,}",
                help="Fires detected in the last 6 hours - indicates current fire activity trends",
            )

        with col3:
            # Calculate fire intensity distribution
            intense_fires = stats["high_confidence"]
            st.metric(
                label="🔥 High Confidence Detections",
                value=f"{intense_fires:,}",
                help="Fire detections with high confidence level ('h' rating) - strong thermal signatures indicating likely active fires",
            )

        with col4:
            update_time = last_update.split()[1][:5] if " " in last_update else "N/A"
            st.metric(
                label="🕐 Last Updated",
                value=update_time,
                help=f"Latest data update from satellite passes - {last_update}",
            )

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Data Analysis Section
        st.markdown("### 📊 Fire Analysis Insights")

        # Key insights explanation
        with st.expander("ℹ️ What do these metrics mean?", expanded=False):
            st.markdown(
                """
            **What are "Fire Detections"?**  
            NASA FIRMS provides thermal anomaly detections from satellites. When a satellite passes over an area 
            and detects unusual heat signatures, it's recorded as a fire detection. This dashboard shows all 
            detections from **today**. Note: A detection indicates a fire was present when the 
            satellite passed overhead, but it may have been extinguished since then.
            
            **Fire Radiative Power (FRP):**  
            Measured in Megawatts (MW), FRP indicates the intensity of heat released by a fire at the time of detection. 
            Higher values suggest more intense fires that may be spreading rapidly or consuming large amounts of fuel.
            
            **Day vs Night Detection:**  
            Fires detected during day (D) vs night (N). Some fires are more visible at night due to 
            their heat signature, while daytime detection can be affected by cloud cover and sun glare.
            
            **Confidence Levels:**  
            - **High (h):** Strong indication of fire - requires immediate attention
            - **Nominal (n):** Standard detection - likely fire
            - **Low (l):** Possible fire but may need verification - could be other heat sources
            
            **Satellite Coverage:**  
            Data from VIIRS instrument on NOAA-20 satellite, providing 375m resolution fire detection.
            Satellites pass over the same location approximately every 12 hours.
            """
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Calculate additional statistics
        day_fires = (
            df_raw[df_raw["daynight"] == "D"].shape[0]
            if "daynight" in df_raw.columns
            else 0
        )
        night_fires = (
            df_raw[df_raw["daynight"] == "N"].shape[0]
            if "daynight" in df_raw.columns
            else 0
        )

        # FRP (Fire Radiative Power) statistics
        avg_frp = df_raw["frp"].mean() if "frp" in df_raw.columns else 0
        max_frp = df_raw["frp"].max() if "frp" in df_raw.columns else 0

        # Confidence distribution
        if "confidence" in df_raw.columns:
            confidence_counts = df_raw["confidence"].value_counts()
            high_conf = confidence_counts.get("h", 0)
            nominal_conf = confidence_counts.get("n", 0)
            low_conf = confidence_counts.get("l", 0)
        else:
            high_conf = nominal_conf = low_conf = 0

        # Display insights in cards
        insight_col1, insight_col2, insight_col3 = st.columns(3)

        with insight_col1:
            st.markdown("#### 🌡️ Fire Intensity")
            st.metric(
                "Avg Fire Power",
                f"{avg_frp:.2f} MW" if avg_frp > 0 else "N/A",
                help="Fire Radiative Power (FRP) - measures energy released by fires",
            )
            st.metric(
                "Peak Fire Power",
                f"{max_frp:.2f} MW" if max_frp > 0 else "N/A",
                help="Highest FRP detected - indicates most intense fire",
            )

        with insight_col2:
            st.markdown("#### 🕐 Detection Time")
            total_detections = day_fires + night_fires
            if total_detections > 0:
                day_percentage = (day_fires / total_detections) * 100
                night_percentage = (night_fires / total_detections) * 100
                st.metric(
                    "☀️ Day Fires",
                    f"{day_fires:,} ({day_percentage:.1f}%)",
                    help="Number of fires detected during daytime hours and their percentage of total detections",
                )
                st.metric(
                    "🌙 Night Fires",
                    f"{night_fires:,} ({night_percentage:.1f}%)",
                    help="Number of fires detected during nighttime hours and their percentage of total detections",
                )
            else:
                st.metric("☀️ Day Fires", "N/A")
                st.metric("🌙 Night Fires", "N/A")

        with insight_col3:
            st.markdown("#### 🎯 Detection Confidence")
            st.metric(
                "High (h)", f"{high_conf:,}", help="Fires detected with high certainty"
            )
            st.metric(
                "Nominal (n)",
                f"{nominal_conf:,}",
                help="Standard confidence detections",
            )
            if low_conf > 0:
                st.metric(
                    "Low (l)",
                    f"{low_conf:,}",
                    help="Lower confidence - may require verification",
                )

    elif st.session_state.current_view == "hourly":
        st.markdown("## ⏰ Fire Detections by Hour - Today")
        st.markdown(
            "Analyze today's fire patterns with detailed hourly breakdowns (UTC timezone)"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if DataColumns.ACQ_TIME.value in df_raw.columns:
            # Process hourly data
            time_str = df_raw[DataColumns.ACQ_TIME.value].astype(str).str.zfill(4)
            df_hourly = df_raw.copy()
            df_hourly["hour"] = time_str.str[:2].astype(int)
            fires_by_hour_all = (
                df_hourly.groupby("hour").size().reindex(range(24), fill_value=0)
            )

            # Get current hour in UTC
            current_hour_utc = datetime.utcnow().hour

            # Filter to show only hours that have passed (including current hour)
            hours_passed = range(current_hour_utc + 1)
            fires_by_hour = fires_by_hour_all[list(hours_passed)]

            # Create two columns for chart and insights
            col1, col2 = st.columns([2, 1])

            with col1:
                st.caption(
                    "Distribution of fire detections throughout the day (UTC timezone)"
                )

                # Create a clean DataFrame for Plotly - only hours that passed
                plot_df = pd.DataFrame(
                    {
                        "Hour": [f"{h:02d}:00" for h in hours_passed],
                        "Fires": fires_by_hour.values,
                    }
                )

                # Create Plotly Express bar chart
                fig = px.bar(
                    plot_df,
                    x="Hour",
                    y="Fires",
                    title="Fire Detections Throughout the Day (UTC)",
                    labels={"Fires": "Number of Fire Detections", "Hour": "Hour (UTC)"},
                    color_discrete_sequence=["#d32f2f"],
                )

                # Update layout for better appearance
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Inter, sans-serif", color="#000000"),
                    title=dict(
                        font=dict(size=18, color="#000000", family="Inter, sans-serif"),
                        x=0,
                    ),
                    height=450,
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="#f0f0f0",
                        tickfont=dict(size=11, color="#000000"),
                        title_font=dict(color="#000000"),
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="#f0f0f0",
                        zeroline=True,
                        tickformat=",d",
                        tickfont=dict(color="#000000"),
                        title_font=dict(color="#000000"),
                    ),
                    margin=dict(l=60, r=40, t=80, b=60),
                    hovermode="x unified",
                )

                # Update hover template
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>Fires: %{y}<extra></extra>"
                )

                # Configure to show only fullscreen button
                config = {
                    "modeBarButtonsToRemove": [
                        "zoom2d",
                        "pan2d",
                        "select2d",
                        "lasso2d",
                        "zoomIn2d",
                        "zoomOut2d",
                        "autoScale2d",
                        "resetScale2d",
                        "toImage",
                    ],
                    "displaylogo": False,
                }

                st.plotly_chart(fig, use_container_width=True, config=config)

            with col2:
                st.markdown("### 📋 Hourly Insights")

                peak_hour = fires_by_hour.idxmax()
                peak_count = fires_by_hour.max()
                quiet_hour = fires_by_hour.idxmin()
                quiet_count = fires_by_hour.min()
                total_fires = fires_by_hour.sum()

                st.metric(
                    "🔥 Total Fires",
                    f"{total_fires:,}",
                    help="Total number of fires detected across all hours today",
                )
                st.metric(
                    f"🔝 Peak Hour ({peak_count:,} fires)",
                    f"{peak_hour:02d}:00 UTC",
                )
                st.metric(
                    f"🌙 Quietest Hour ({quiet_count:,} fires)",
                    f"{quiet_hour:02d}:00 UTC",
                )

        else:
            st.warning("⚠️ Time data not available in current dataset")

    elif st.session_state.current_view == "map":
        st.markdown("## 🗺️ Interactive Fire Map - Today")
        st.markdown(
            "Explore today's fire detections and thermal anomalies across the globe with advanced filtering and visualization tools"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Prepare data
        required_columns = [
            DataColumns.LATITUDE.value,
            DataColumns.LONGITUDE.value,
            DataColumns.FRP.value,
            DataColumns.BRIGHTNESS.value,
            DataColumns.BRIGHT_TI4.value,
            DataColumns.BRIGHT_TI5.value,
            DataColumns.CONFIDENCE.value,
            DataColumns.ACQ_DATE.value,
            DataColumns.ACQ_TIME.value,
        ]

        cols_needed = [c for c in required_columns if c in df_raw.columns]
        dfm = (
            df_raw[cols_needed]
            .dropna(subset=[DataColumns.LATITUDE.value, DataColumns.LONGITUDE.value])
            .copy()
        )

        dfm, metric_col = get_intensity_metric(dfm)

        # Add hour column if time data exists
        if DataColumns.ACQ_TIME.value in dfm.columns:
            time_str = dfm[DataColumns.ACQ_TIME.value].astype(str).str.zfill(4)
            dfm["hour"] = time_str.str[:2].astype(int)

        # Add day/night column if exists
        has_daynight = "daynight" in dfm.columns

        # Initialize session state for filter presets
        if "filter_preset" not in st.session_state:
            st.session_state.filter_preset = "All Fires"

        # Quick Filter Presets Section
        st.markdown("### 🎯 Quick Filter Presets")
        preset_cols = st.columns(3)

        with preset_cols[0]:
            if st.button("🔥 All Fires", use_container_width=True):
                st.session_state.filter_preset = "All Fires"
                st.rerun()

        with preset_cols[1]:
            if st.button("🎯 High Confidence", use_container_width=True):
                st.session_state.filter_preset = "High Confidence"
                st.rerun()

        with preset_cols[2]:
            if st.button("⏰ Last 6 Hours", use_container_width=True):
                st.session_state.filter_preset = "Last 6 Hours"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Advanced Filters in Expander
        with st.expander("🎚️ Advanced Filters & Settings", expanded=False):
            filter_row1_col1, filter_row1_col2 = st.columns(2)

            with filter_row1_col1:
                # Intensity filter
                min_val = float(np.nanmin(dfm[metric_col])) if len(dfm) else 0.0
                max_val = float(np.nanmax(dfm[metric_col])) if len(dfm) else 1.0

                if not np.isfinite(min_val):
                    min_val = 0.0
                if not np.isfinite(max_val) or max_val <= min_val:
                    max_val = min_val + 1.0

                intensity_range = st.slider(
                    "🔥 Fire Intensity Range",
                    min_value=float(round(min_val, 2)),
                    max_value=float(round(max_val, 2)),
                    value=(float(round(min_val, 2)), float(round(max_val, 2))),
                    step=float(round((max_val - min_val) / 100 or 0.01, 2)),
                    help=f"Filter fires by {metric_col} level. Higher values indicate more intense fires.",
                )

            with filter_row1_col2:
                # Confidence filter
                conf_mask = pd.Series(True, index=dfm.index)
                if DataColumns.CONFIDENCE.value in dfm.columns:
                    if pd.api.types.is_numeric_dtype(dfm[DataColumns.CONFIDENCE.value]):
                        cmin = (
                            int(np.nanmin(dfm[DataColumns.CONFIDENCE.value]))
                            if dfm[DataColumns.CONFIDENCE.value].notna().any()
                            else 0
                        )
                        cmax = (
                            int(np.nanmax(dfm[DataColumns.CONFIDENCE.value]))
                            if dfm[DataColumns.CONFIDENCE.value].notna().any()
                            else 100
                        )
                        cmin = max(0, min(cmin, 100))
                        cmax = min(100, max(cmax, 0))
                        conf_range = st.slider(
                            "✅ Confidence Level (%)",
                            0,
                            100,
                            (cmin, cmax),
                            step=1,
                            help="Filter by detection confidence percentage",
                        )
                        conf_mask = dfm[DataColumns.CONFIDENCE.value].between(
                            conf_range[0], conf_range[1], inclusive="both"
                        )
                    else:
                        # Map confidence codes to readable names
                        confidence_map = {
                            "h": "🔥 High",
                            "n": "⚡ Nominal",
                            "l": "⚠️ Low",
                        }

                        available_codes = sorted(
                            [
                                str(x)
                                for x in dfm[DataColumns.CONFIDENCE.value]
                                .dropna()
                                .unique()
                            ]
                        )

                        # Create display options
                        display_options = [
                            confidence_map.get(code, code) for code in available_codes
                        ]

                        # Apply preset logic for confidence
                        if st.session_state.filter_preset == "High Confidence":
                            default_display = (
                                [confidence_map["h"]]
                                if "h" in available_codes
                                else display_options
                            )
                        else:
                            default_display = display_options

                        chosen_display = st.multiselect(
                            "✅ Confidence Categories",
                            options=display_options,
                            default=default_display,
                            help="Filter by detection confidence level",
                        )

                        # Map back to original codes
                        reverse_map = {v: k for k, v in confidence_map.items()}
                        chosen_codes = [
                            reverse_map.get(display, display)
                            for display in chosen_display
                        ]

                        conf_mask = (
                            dfm[DataColumns.CONFIDENCE.value]
                            .astype(str)
                            .isin(chosen_codes)
                        )

            # Second row of filters
            filter_row2_col1, filter_row2_col2 = st.columns(2)

            with filter_row2_col1:
                # Time-based filter
                if "hour" in dfm.columns:
                    current_hour = datetime.utcnow().hour

                    # Apply preset logic for time
                    if st.session_state.filter_preset == "Last 6 Hours":
                        # Last 6 hours
                        recent_hours = [(current_hour - i) % 24 for i in range(6)]
                        default_hours = recent_hours
                    else:
                        default_hours = list(range(24))

                    hour_options = [f"{h:02d}:00" for h in range(24)]
                    selected_hours = st.multiselect(
                        "⏰ Detection Time (UTC)",
                        options=hour_options,
                        default=[hour_options[h] for h in default_hours],
                        help="Filter by hour of detection (UTC timezone)",
                    )
                    selected_hour_nums = [int(h.split(":")[0]) for h in selected_hours]
                else:
                    selected_hour_nums = None

            with filter_row2_col2:
                # Day/Night filter
                if has_daynight:
                    daynight_filter = st.multiselect(
                        "☀️🌙 Day/Night Detection",
                        options=["D", "N"],
                        default=["D", "N"],
                        help="D=Day, N=Night detection",
                    )
                else:
                    daynight_filter = None

        # Set map style to Light
        map_style_url = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

        # Apply all filters
        mask = (
            dfm[metric_col].between(
                intensity_range[0], intensity_range[1], inclusive="both"
            )
            & conf_mask
        )

        # Apply time filter
        if selected_hour_nums is not None and "hour" in dfm.columns:
            mask &= dfm["hour"].isin(selected_hour_nums)

        # Apply day/night filter
        if daynight_filter is not None and has_daynight:
            mask &= dfm["daynight"].isin(daynight_filter)

        plot_columns = (
            [
                DataColumns.LATITUDE.value,
                DataColumns.LONGITUDE.value,
                metric_col,
                DataColumns.CONFIDENCE.value,
            ]
            if DataColumns.CONFIDENCE.value in dfm.columns
            else [DataColumns.LATITUDE.value, DataColumns.LONGITUDE.value, metric_col]
        )

        # Add additional columns if they exist
        if "hour" in dfm.columns and "hour" not in plot_columns:
            plot_columns.append("hour")
        if has_daynight and "daynight" not in plot_columns:
            plot_columns.append("daynight")

        df_plot = dfm.loc[mask, plot_columns].copy()

        # Calculate statistics for filtered data
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Filtered Data Statistics")

        col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)

        with col_stat1:
            st.metric(
                "🎯 Filtered Fires",
                f"{len(df_plot):,}",
                help="Number of fires matching your current filters",
            )

        with col_stat2:
            percentage = (len(df_plot) / len(dfm) * 100) if len(dfm) > 0 else 0
            st.metric(
                "📈 Showing",
                f"{percentage:.1f}%",
                help="Percentage of total fires being displayed",
            )

        with col_stat3:
            if len(df_plot) > 0:
                avg_intensity = df_plot[metric_col].mean()
                st.metric(
                    "🔥 Avg Intensity",
                    f"{avg_intensity:.2f}",
                    help=f"Average {metric_col} of filtered fires",
                )
            else:
                st.metric("🔥 Avg Intensity", "N/A")

        with col_stat4:
            if len(df_plot) > 0 and DataColumns.CONFIDENCE.value in df_plot.columns:
                if not pd.api.types.is_numeric_dtype(
                    df_plot[DataColumns.CONFIDENCE.value]
                ):
                    high_conf_count = (
                        df_plot[DataColumns.CONFIDENCE.value] == "h"
                    ).sum()
                    st.metric(
                        "✅ High Confidence",
                        f"{high_conf_count:,}",
                        help="Number of high confidence detections in filtered data",
                    )
                else:
                    st.metric("✅ High Confidence", "N/A")
            else:
                st.metric("✅ High Confidence", "N/A")

        with col_stat5:
            if len(df_plot) > 0:
                max_intensity = df_plot[metric_col].max()
                st.metric(
                    "🔝 Peak Intensity",
                    f"{max_intensity:.2f}",
                    help=f"Maximum {metric_col} in filtered fires",
                )
            else:
                st.metric("🔝 Peak Intensity", "N/A")

        st.markdown("<br>", unsafe_allow_html=True)

        # Compute visualization radius
        if len(df_plot) > 0:
            vmin, vmax = df_plot[metric_col].min(), df_plot[metric_col].max()
            if vmax == vmin:
                norm = np.ones(len(df_plot))
            else:
                norm = (df_plot[metric_col] - vmin) / (vmax - vmin)
            df_plot["radius"] = (
                MapConfig.MIN_RADIUS
                + (MapConfig.MAX_RADIUS - MapConfig.MIN_RADIUS) * norm
            ).astype(float)

            # Calculate center point for better initial view
            center_lat = df_plot[DataColumns.LATITUDE.value].median()
            center_lon = df_plot[DataColumns.LONGITUDE.value].median()

            # Adjust zoom based on data spread
            lat_range = (
                df_plot[DataColumns.LATITUDE.value].max()
                - df_plot[DataColumns.LATITUDE.value].min()
            )
            lon_range = (
                df_plot[DataColumns.LONGITUDE.value].max()
                - df_plot[DataColumns.LONGITUDE.value].min()
            )
            max_range = max(lat_range, lon_range)

            # Calculate appropriate zoom level
            if max_range < 10:
                zoom_level = 6
            elif max_range < 30:
                zoom_level = 4
            elif max_range < 60:
                zoom_level = 3
            elif max_range < 120:
                zoom_level = 2
            else:
                zoom_level = 1.5

            # Rename columns for map and add formatted columns
            df_map = df_plot.rename(
                columns={
                    DataColumns.LATITUDE.value: "lat",
                    DataColumns.LONGITUDE.value: "lon",
                }
            ).copy()

            # Add formatted columns for tooltip
            df_map["lat_formatted"] = df_map["lat"].apply(lambda x: f"{x:.4f}")
            df_map["lon_formatted"] = df_map["lon"].apply(lambda x: f"{x:.4f}")
            df_map["intensity_formatted"] = df_map[metric_col].apply(
                lambda x: f"{x:.2f}"
            )

            if "hour" in df_map.columns:
                df_map["time_formatted"] = df_map["hour"].apply(
                    lambda x: f"{x:02d}:00 UTC"
                )

            # Prepare enhanced tooltip
            tooltip_html = (
                "<b style='color: #ff4500; font-size: 14px;'>🔥 Fire Detection</b><br/>"
                "<b>📍 Location:</b> {lat_formatted}, {lon_formatted}<br/>"
                "<b>🔥 Intensity:</b> {intensity_formatted}<br/>"
            )

            if DataColumns.CONFIDENCE.value in df_map.columns:
                tooltip_html += (
                    f"<b>✅ Confidence:</b> {{{DataColumns.CONFIDENCE.value}}}<br/>"
                )

            if "hour" in df_map.columns:
                tooltip_html += "<b>⏰ Time:</b> {time_formatted}<br/>"

            if has_daynight and "daynight" in df_map.columns:
                tooltip_html += "<b>☀️ Day/Night:</b> {daynight}<br/>"

            tooltip = {
                "html": tooltip_html,
                "style": {
                    "backgroundColor": "rgba(255, 255, 255, 0.95)",
                    "color": "#1a1a1a",
                    "fontSize": "13px",
                    "fontFamily": "Inter, sans-serif",
                    "padding": "12px 16px",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
                },
            }

            # Create map with better initial view
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom_level,
                pitch=0,
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position="[lon, lat]",
                get_radius="radius",
                pickable=True,
                stroked=False,
                filled=True,
                opacity=MapConfig.OPACITY,
                get_fill_color=MapConfig.FIRE_COLOR,
            )

            # Add legend
            st.markdown(
                """
                <div style='background: white; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                    <b style='color: #1a1a1a; font-size: 14px;'>🗺️ Map Legend:</b>
                    <div style='margin-top: 0.5rem; color: #666;'>
                        <span style='color: #ff4500; font-weight: bold;'>⬤</span> Fire Detection Point • 
                        <span style='font-size: 20px;'>⬤</span> Larger = Higher Intensity • 
                        <span>Hover over points for details • Scroll to zoom • Drag to pan</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style=map_style_url,
                ),
                use_container_width=True,
                height=650,
            )

            # Add download filtered data option
            st.markdown("<br>", unsafe_allow_html=True)
            col_download, col_reset = st.columns([3, 1])

            with col_download:
                # Prepare CSV download
                csv_data = df_plot.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name=f"filtered_fires_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the currently filtered fire data as a CSV file",
                )

            with col_reset:
                if st.button("🔄 Reset All Filters", use_container_width=True):
                    st.session_state.filter_preset = "All Fires"
                    st.rerun()

        else:
            st.warning(
                "⚠️ No fires match the current filter criteria. Try adjusting the filters or click 'Reset All Filters'."
            )
            if st.button("🔄 Reset All Filters", type="primary"):
                st.session_state.filter_preset = "All Fires"
                st.rerun()

else:
    st.error(
        "Unable to load fire detection data. Please check your internet connection and try again."
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p style='margin-bottom: 0.5rem;'><strong>Data Source:</strong> NASA FIRMS • <strong>Satellite:</strong> VIIRS NOAA-20</p>
        <p style='font-size: 1rem; color: #ff6347; font-weight: 600; margin-bottom: 0.5rem;'>📅 Showing detections from TODAY only (UTC timezone)</p>
        <p style='font-size: 0.85rem; color: #999;'>Data refreshes every 15 minutes • Satellites pass over each location approximately every 12 hours</p>
    </div>
    """,
    unsafe_allow_html=True,
)
