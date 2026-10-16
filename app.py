import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path
from datetime import datetime, date
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import statistics_function as sf
import re
from pathlib import Path
import pickle
import base64
from pathlib import Path


# Current Directory
try:
    # For Everything except Jupyter Notebook
    current_dir = Path(__file__).resolve().parent
except NameError:
    # For Jupyter Notebooks
    current_dir = Path(os.path.abspath("")).resolve()
    

# DIRECTORIES

ICME_CATALOGUE = current_dir / "ICME_master_catalogue.csv"
PLOT_DATABASE = current_dir / "ICME_plots" / "plots1"
EVENT_DATABASE = current_dir / "EVENT_MASTER_PRE_POST.parquet"
WAVELET_DATABASE = current_dir / "WAVELET_MASTER.parquet"
SILSO_CSV = current_dir / "SN_m_tot_V2.0.csv"
STATS_BUNDLE = current_dir / "Statistical_plot_components.pkl"
BANNER_IMAGE_PATH = current_dir / "Chimera Banner.png"
BG_IMAGE_PATH = current_dir / "Chimera image.png"

# For Plots

ME_SHEATH_DIR = current_dir / "Plots" / "ME_vs_sheath_pdf"
ICME_SW_DIR = current_dir / "Plots" / "ICME_vs_non-ICME_pdf"



# PAGE CONFIG
   
# 1. THE ONLY PAGE CONFIG (Must be line 1 of Streamlit execution)
st.set_page_config(
    page_title="ICME Data Repository",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. INSTANTIATE THE SIDEBAR IMMEDIATELY
st.sidebar.title("Control Panel")
# st.sidebar.markdown("Use these controls to filter Epoch data.")

# Manage navigation and parameter expansion via session state
if "page_view" not in st.session_state:
    st.session_state.page_view = "main"
if "statistics_view" not in st.session_state:
    st.session_state.statistics_view = None
if "selected_icme_event" not in st.session_state:
    st.session_state.selected_icme_event = None
if "active_query" not in st.session_state:
    st.session_state.active_query = None

if "final_results" not in st.session_state:
    st.session_state.final_results = None

if "final_columns" not in st.session_state:
    st.session_state.final_columns = []

if "selected_parameters" not in st.session_state:
    st.session_state.selected_parameters = []
if "phase1_parameters" not in st.session_state:
    st.session_state.phase1_parameters = []

if "phase2_parameters" not in st.session_state:
    st.session_state.phase2_parameters = []
if "phase1_columns" not in st.session_state:
    st.session_state.phase1_columns = []

if "phase2_columns" not in st.session_state:
    st.session_state.phase2_columns = []

if "phase1_results" not in st.session_state:
    st.session_state.phase1_results = pd.DataFrame()
if "phase2_results" not in st.session_state:
    st.session_state.phase2_results = pd.DataFrame()

# 2. Function to convert your local image to Base64
@st.cache_data
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image_base64 = get_img_as_base64(BG_IMAGE_PATH)
banner_image_base64 = get_img_as_base64(BANNER_IMAGE_PATH)


# 3. Inject into your styling block
st.markdown(f"""
<style>
/* 
   FIX: Make header invisible instead of 'display: none' 
   so the sidebar expand button survives! 
*/
header[data-testid="stHeader"] {{
    background: transparent !important;
    box-shadow: none !important;
}}

/* Force both legacy and modern sidebar expansion buttons to stay visible */
[data-testid="collapsedControl"], 
[data-testid="stSidebarCollapseButton"], 
button[kind="header"] {{
    opacity: 1 !important;
    display: block !important;
    visibility: visible !important;
    z-index: 999999 !important;
    background-color: rgba(5,20,45,0.88) !important; /* Matches your sidebar background */
    border-radius: 5px;
}}

/* Remove extra top spacing created by header */
div[data-testid="stAppViewContainer"] {{
    padding-top: 0rem !important;
}}

/* Remove decoration strip (sometimes appears as black line) */
div[data-testid="stDecoration"] {{
    display: none !important;
}}

section[data-testid="stSidebar"] {{
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}}


/* RESPONSIVE FOREGROUND BANNER */
.main-banner {{
    width: 100%;
    height: auto;
    max-height: 200px; /* Constrains height while keeping aspect ratio */
    object-fit: cover;
    border-radius: 18px;
    border: 1px solid rgba(255,180,60,0.45);
    box-shadow: 0 8px 25px rgba(0,0,0,0.45);
    margin-bottom: 25px; /* Creates spacing before your actual app content starts */
    display: block;
}}

/* GLOBAL BACKGROUND (Loaded locally & anchored to top) */

.stApp {{
    background-image:
    linear-gradient(
        rgba(0,0,20,0.25),
        rgba(0,0,20,0.35)
    ),
    url("data:image/jpeg;base64,{bg_image_base64}");
    background-size: cover;
    background-position: center center; /* Shifted to center since banner is removed from BG */
    background-attachment: fixed;
}}

/* SIDEBAR BASE */

section[data-testid="stSidebar"] {{
    background: rgba(5,20,45,0.88);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-right: 2px solid rgba(255,180,60,0.45);
    box-shadow: 5px 0 25px rgba(0,0,0,0.35);
}}

/* Sidebar text */
section[data-testid="stSidebar"] * {{
    color:#F8EAD8;
}}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color:#FFB020;
    font-weight:700;
}}

/* SIDEBAR IMAGE / HEADER BLOCKS */
.sidebar-image {{
    width:100%;
    height:150px;
    object-fit:cover;
    border-radius:12px;
    margin-bottom:15px;
    border:1px solid rgba(255,180,60,0.45);
    box-shadow:0 5px 20px rgba(0,0,0,0.45);
}}
.sidebar-title {{
    text-align:center;
    color:#FFB020;
    font-size:22px;
    font-weight:700;
    margin-bottom:5px;
}}
.sidebar-subtitle {{
    text-align:center;
    color:#F8EAD8;
    font-size:14px;
    margin-bottom:20px;
}}

/*                                                                                                    
   MAIN CONTAINER                                                                                                
                                                                                                     */
.block-container {{
    background: rgba(10,25,50,0.18);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border:1px solid rgba(255,220,150,0.25);
    box-shadow:0 8px 30px rgba(0,0,0,0.25);
    padding:2rem;
    border-radius:22px;
}}

/*                                                                                                    
   TEXT STYLING                                                                                                
                                                                                                     */
h1,h2,h3 {{
    color:#FFE7B3;
}}
p,label,span {{
    color:#F8EAD8;
}}

/*                                                                                                    
   SIDEBAR INPUTS (FIXED CONTRAST SYSTEM)                                                                                                
                                                                                                     */
/* Selectbox container */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 176, 32, 0.55) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px);
}}
/* Selected text */
section[data-testid="stSidebar"] div[data-baseweb="select"] div {{
    color: #FFE7B3 !important;
    font-weight: 600;
}}
/* Placeholder */
section[data-testid="stSidebar"] div[data-baseweb="select"] [data-testid="stMarkdownContainer"] {{
    color: rgba(255, 231, 179, 0.65) !important;
}}
/* Dropdown panel */
section[data-testid="stSidebar"] ul[role="listbox"] {{
    background: rgba(15, 35, 60, 0.98) !important;
    border: 1px solid rgba(255, 176, 32, 0.35);
    border-radius: 10px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}}
/* Options */
section[data-testid="stSidebar"] li[role="option"] {{
    color: #F8EAD8 !important;
}}
/* Hover */
section[data-testid="stSidebar"] li[role="option"]:hover {{
    background: rgba(255, 176, 32, 0.15) !important;
    color: #FFFFFF !important;
}}
/* Selected option */
section[data-testid="stSidebar"] li[aria-selected="true"] {{
    background: rgba(255, 176, 32, 0.25) !important;
    color: #FFFFFF !important;
}}
/* Input fields */
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stDateInput input {{
    background: rgba(255, 255, 255, 0.07) !important;
    color: #FFE7B3 !important;
    border: 1px solid rgba(255, 176, 32, 0.55) !important;
    border-radius: 10px;
}}
/* Focus state */
section[data-testid="stSidebar"] input:focus {{
    border: 1px solid #F59E0B !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.25) !important;
    outline: none !important;
}}
/* Checkbox */
section[data-testid="stSidebar"] .stCheckbox label {{
    color:#FFE7B3 !important;
}}
/* Grouping feel */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stDateInput {{
    padding: 6px 8px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.03);
}}

/*                                                                                                    
   METRICS                                                                                                
                                                                                                     */
[data-testid="stMetricValue"] {{
    color:#FFB020;
    font-weight:700;
}}
[data-testid="stMetricLabel"] {{
    color:#FFE7B3;
}}

/*                                                                                                    
   BUTTONS                                                                                                
                                                                                                     */
.stButton > button,
.stDownloadButton > button {{
    background: linear-gradient(135deg, #F59E0B, #EA580C);
    color:white;
    border-radius:10px;
    border:none;
    font-weight:600;
}}
.stButton > button:hover,
.stDownloadButton > button:hover {{
    background: linear-gradient(135deg, #FDBA74, #F97316);
}}

/*                                                                                                    
   HIDE SIDEBAR NAV                                                                                                
                                                                                                     */
[data-testid="stSidebarNav"] {{
    display:none;
}}

/*                                                                                                    
   SOLAR IMAGE BANNER                                                                                                
                                                                                                     */
.solar-banner {{
    display:flex;
    width:100%;
    height:150px;
    gap:0px;
    overflow:hidden;
    border-radius:15px;
    border:1px solid rgba(255,180,60,0.45);
    box-shadow:0 8px 25px rgba(0,0,0,0.45);
}}
.solar-banner img {{
    width:33.33%;
    height:150px;
    object-fit:cover;
    transition:transform 0.4s ease;
}}
.solar-banner img:hover {{
    transform:scale(1.05);
}}
/* Fix R theta phi number input size */
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] {{
    width: 145px !important;
}}
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] > div {{
    width: 145px !important;
}}
/* Actual text box */
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {{
    width: 75px !important;
    min-width: 75px !important;
    height: 42px !important;
    font-size: 16px !important;
    padding-left: 8px !important;
}}
/* DISABLE STREAMLIT'S AUTOMATIC RE-RUN SCREEN BLUR */
div[data-testid="stAppViewContainer"] > div:first-child {{
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
}}
/* Hide the default "Running..." man in top right */
[data-testid="stStatusWidget"] {{
    display: none !important;
}}
/* HIDE THE REDUNDANT +/- BUTTONS IN NUMBER INPUTS */
button[data-testid="stNumberInputStepDown"], 
button[data-testid="stNumberInputStepUp"] {{
    display: none !important;
}}

/* Center the numbers inside the box for a cleaner look */
input[data-testid="stNumberInputInput"] {{
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# RENDER DYNAMIC BANNER (Respects sidebar width and stays crisp)
st.markdown(f'<img src="data:image/jpeg;base64,{banner_image_base64}" class="main-banner">', unsafe_allow_html=True)



# LOAD DATA
   
@st.cache_data
def load_data(event_parquet):

    df = pd.read_parquet(event_parquet)
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    # Recover r theta phi from Location only if needed
    if "Location" in df.columns:

        # Check if columns exist
        if not all(col in df.columns for col in ["r","theta","phi"]):

            coords = (
                df["Location"]
                .astype(str)
                .str.extract(
                    r'\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\)'
                )
            )

            coords.columns = [
                "r",
                "phi",
                "theta"
            ]

            df["r"] = coords["r"].astype(float)
            df["theta"] = coords["theta"].astype(float)
            df["phi"] = coords["phi"].astype(float)


        # If columns exist but contain NaN
        elif df[["r","theta","phi"]].isna().any().any():

            coords = (
                df["Location"]
                .astype(str)
                .str.extract(
                    r'\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\)'
                )
            )

            coords.columns = [
                "r",
                "phi",
                "theta"
            ]

            df["r"] = coords["r"].astype(float)
            df["theta"] = coords["theta"].astype(float)
            df["phi"] = coords["phi"].astype(float)


    # Clean text columns
    df["event_type"] = (
        df["event_type"]
        .astype(str)
        .str.strip()
    )

    df["spacecraft"] = (
        df["spacecraft"]
        .astype(str)
        .str.strip()
    )

    return df.sort_values("date_time")

df = load_data(EVENT_DATABASE)


@st.cache_data

def load_wavelet_data(wavelet_data):
    wavelet_df = pd.read_parquet(wavelet_data)
    
    wavelet_df["date_time"] = pd.to_datetime(
        wavelet_df["date_time"],
        errors="coerce"
    )
    return wavelet_df

wavelet_df = load_wavelet_data(WAVELET_DATABASE)


@st.cache_data

def load_icme_catalog(icme_catalogue):
    catalog = pd.read_csv(icme_catalogue)
    return catalog

icme_catalog = load_icme_catalog(ICME_CATALOGUE)

SC_ALIASES = {
    "helio1": "HL1",
    "helios 1": "HL1",
    "hl1": "HL1",
    "helio2": "HL2",
    "helios 2": "HL2",
    "hl2": "HL2",
    "stereo a": "STA",
    "stereoa": "STA",
    "sta": "STA",
    "stereo b": "STB",
    "stereob": "STB",
    "stb": "STB",
    "parker solar probe": "PSP",
    "parker": "PSP",
    "psp": "PSP",
    "solar orbiter": "SOLO",
    "solo": "SOLO",
    "messenger": "MES",
    "messenger spacecraft": "MES",
    "mes": "MES",
    "wind": "WIN",
    "win": "WIN",
    "ulysses": "ULY",
    "uly": "ULY",
    "maven": "MAV",
    "mav": "MAV",
    "juno": "JNO",
    "jno": "JNO"
}

@st.cache_data
def load_silso_solar_cycle(silso_csv):
    silso = pd.read_csv(
        silso_csv,
        sep=";",
        header=None,
        comment="#",
        engine="python"
    )

    # Keep only the required first 4 columns
    silso = silso.iloc[:, :4]
    silso.columns = [
        "Year",
        "Month",
        "Decimal_Year",
        "SSN"
    ]

    silso["Year"] = pd.to_numeric(silso["Year"], errors="coerce")
    silso["Month"] = pd.to_numeric(silso["Month"], errors="coerce")
    silso["SSN"] = pd.to_numeric(silso["SSN"], errors="coerce")
    silso = silso.dropna()
    silso["Year"] = silso["Year"].astype(int)
    silso["Month"] = silso["Month"].astype(int)

    silso["Date"] = pd.to_datetime(
        dict(
            year=silso["Year"],
            month=silso["Month"],
            day=1
        )
    )

    silso = silso[
        (silso["Date"] >= "1974-01-01") &
        (silso["Date"] <= "2025-12-31")
    ]

    return silso
    
silso_df = load_silso_solar_cycle(SILSO_CSV)


@st.cache_data

def load_statistical_bundle(stats_bundle):
    with open(stats_bundle, "rb") as f:
        bundle = pickle.load(f)
    return bundle

stats_bundle = load_statistical_bundle(STATS_BUNDLE)

mother_dict_all = stats_bundle["mother_dict"]
df_silso = stats_bundle["silso_df"]
df_rgo = stats_bundle["rgo_df"]

# Predefined Spacecraft Limits
SPACECRAFT_LIMITS = {
    "Helios 1": {"r": (0.29, 0.98), "theta": (-7.0, 7.0), "phi": (0.0, 360.0)},
    "Helios 2": {"r": (0.29, 0.98), "theta": (-7.0, 7.0), "phi": (0.0, 360.0)},
    "Wind": {"r": (0.97, 1.01), "theta": (-0.01, 7.28), "phi": (0.0, 360.0)},
    "Parker Solar Probe": {"r": (0.06, 0.90), "theta": (-1.0, 4.0), "phi": (0.0, 360.0)},
    "Solar Orbiter": {"r": (0.30, 1.00), "theta": (-0.08, 8.0), "phi": (0.0, 360.0)},
    "STEREO-A": {"r": (0.95, 0.97), "theta": (-0.01, 7.35), "phi": (0.0, 360.0)},
    "STEREO-B": {"r": (0.97, 1.08), "theta": (-0.02, 7.30), "phi": (0.0, 360.0)},
    "Ulysses": {"r": (1.34, 5.40), "theta": (-80.0, 80.0), "phi": (0.0, 360.0)},
    "Juno": {"r": (1, 5.4), "theta": (-0.7, 7), "phi": (0.0, 360.0)},
    "MESSENGER": {"r": (0.3, 0.74), "theta": (-0.07, 3.4), "phi": (0.0, 360.0)},
    "MAVEN": {"r":(1.38, 1.67), "theta": (-5.0, 5.0),"phi":(0.0, 360.0)}
}

TEMPORAL_LIMITS = {
    "Helios 1" : ("1975-01-08 12:00:00", "1980-07-21 17:02:00"),
    "Helios 2" : ("1976-02-28 00:00:00", "1980-02-14 22:04:00"),
    "Ulysses" : ("1990-12-11 04:00:00", "2007-09-07 02:00:00"),
    "Wind" : ("1995-02-07 19:11:00", "2025-07-26 16:44:00"),
    "STEREO-A" : ("2007-01-14 12:11:00", "2025-08-10 05:17:00"),
    "STEREO-B" : ("2007-01-14 12:24:00", "2014-09-25 14:09:00"),
    "MESSENGER" : ("2007-05-04 13:34:00", "2015-04-22 00:49:00"),
    "Juno" : ("2011-09-09 18:01:00", "2016-05-20 10:56:00"),
    "MAVEN" : ("2014-12-18 14:24:00", "2018-09-24 18:00:00"),
    "Parker Solar Probe" : ("2018-10-30 20:25:00", "2025-04-26 01:43:00"),
    "Solar Orbiter" : ("2020-04-19 05:06:00", "2025-05-27 01:20:00")
}


# 1. THE GROUPS (Strictly raw dataframe column names)
magnetic_groups = {
    "Magnetic Field": ["B", "rmsB_over_B", "deltaB"]
}

plasma_groups = {
    "Plasma / Thermodynamic": ["Np", "Vsw", "Tp", "Va", "M_A", "beta", "P_dyn", "P_T", "T_r", "S", "Na/Np"]
}

wavelet_groups = {
    "Helicity": ["Hm_t"],
    "Magnetic Energy": ["Em_t"],
    "Kinetic Energy": ["Ekin_t"]
}

# 2. THE LABELS (Strictly for GUI LaTeX translation)
units = {
    "B": "Magnetic Field Intensity $(B) (nT)$",
    "rmsB_over_B": "Normalized magnetic field fluctuation amplitude $(rmsBoB)$",
    "deltaB": r"Fluctuation in B $(\delta B)$",
    "Np": "Proton Density $(N_p) (cm^{-3})$",
    "Vsw": "Bulk velocity $(V_{SW}) (km/s)$",
    "Tp": "Proton Temperature $(T_p) (K)$",
    "Va": r"Alfvén Speed $(V_{\alpha}) (km/s)$",
    "M_A": "Alfvén Mach Number $(M_A)$",
    "beta": r"Plasma Beta $(\beta)$",
    "P_dyn": "Dynamic Pressure $(P_{dyn}) (nPa)$",
    "P_T": "Total Pressure $(P_T) (nPa)$",
    "T_r": "Temperature ratio $(T_r)$",
    "S": "Entropy $(S) (eV cm^2)$",
    "Na/Np": r"Alpha to proton ratio $(N_{\alpha, p})$"
}

units_wavelet = {
    "Hm_t": r"Helicity $(H_m(t)) (nT^2sHz^{\frac{8}{3}})$",
    "Em_t": r"Magnetic Energy $(E_m(t)) (nT^2sHz^{\frac{5}{3}})$",
    "Ekin_t": r"Kinetic Energy $(E_{kin}(t)) (km^2s^{-2}Hz^{\frac{5}{3}})$"
}

# 3. SHORT NOTATIONS (Strictly for the Selection Summary table)
short_notations = {
    "B": "$B$",
    "rmsB_over_B": "$rmsBoB$",
    "deltaB": r"$\delta B$",
    "Np": "$N_p$",
    "Vsw": "$V_{SW}$",
    "Tp": "$T_p$",
    "Va": r"$V_{\alpha}$",
    "M_A": "$M_A$",
    "beta": r"$\beta$",
    "P_dyn": "$P_{dyn}$",
    "P_T": "$P_T$",
    "T_r": "$T_r$",
    "S": "$S$",
    "Na/Np": r"$N_{\alpha, p}$",
    "Hm_t": "$H_m(t)$",
    "Em_t": "$E_m(t)$",
    "Ekin_t": "$E_{kin}(t)$"
}

DISPLAY_COLUMN_NAMES = {
    "Epoch_ID": "Epoch ID",
    "date_time": "Time (UTC)",
    "event_type": "Event Type",
    "spacecraft": "Spacecraft",
    "Location": "Location (AU, deg, deg)",
    "r": "Radial Distance (AU)",
    "theta": "Latitude θ (deg)",
    "phi": "Longitude φ (deg)",
    "B": " B (nT)",
    "rmsB_over_B": "rmsB_o_B (δB/B)",
    "deltaB": "Delta_B(δB) (nT)",
    "Np": "Np (cm⁻³)",
    "Na/Np": "Alpha/Proton Density Ratio (Na/Np)",
    "Vsw": "Vsw (km/s)",
    "Vth": "Vth (km/s)",
    "Tp": "Tp (K)",
    "Va": " Va (km/s)",
    "M_A": " Mₐ",
    "beta": "β",
    "P_dyn": " P_dyn (nPa)",
    "P_T": "P_T (nPa)",
    "T_r": "Tr",
    "S": "S(eV/K)",
}

# Dynamic range detection directly from your parquet file
min_date = df["date_time"].min().date()
max_date = df["date_time"].max().date()


# MAIN APP VIEW
if "event_type" in df.columns:
    event_counts = df["event_type"].value_counts()
else:
    event_counts = pd.Series(dtype=int)

st.sidebar.markdown("")
st.sidebar.write(f"ME : {event_counts.get('ME',0):,}")
st.sidebar.write(f"Sheath : {event_counts.get('Sheath',0):,}") 
# ADDED PRE AND POST EVENT COUNTERS
st.sidebar.write(f"Pre-Event : {event_counts.get('pre_event',0):,}") 
st.sidebar.write(f"Post-Event : {event_counts.get('post_event',0):,}") 

if st.session_state.page_view == "main":
    if st.sidebar.button("Open ICME Catalog", key="btn_catalog"):
        st.session_state.page_view = "catalog"
        st.rerun()

    if st.sidebar.button("Open Statistics", key="btn_statistics"):
        st.session_state.page_view = "statistics"
        st.rerun()
    st.sidebar.markdown("## Date Range")
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    st.sidebar.markdown("## Time Range")
    start_time_str = st.sidebar.text_input("Start Time", "00:00:00")
    end_time_str = st.sidebar.text_input("End Time", "23:59:59")

# RADIAL DISTANCE R
    st.sidebar.markdown("## Spatial Ranges")
    st.sidebar.markdown("### Radial Distance R (AU)")

    # Initialize
    if "r_range" not in st.session_state:
        st.session_state.r_range = (0.0, 6.0)
    if "rmin_input" not in st.session_state:
        st.session_state.rmin_input = 0.0
    if "rmax_input" not in st.session_state:
        st.session_state.rmax_input = 6.0

    def update_r_from_input():
        st.session_state.r_range = (st.session_state.rmin_input, st.session_state.rmax_input)
        st.session_state.r_slider = (st.session_state.rmin_input, st.session_state.rmax_input)

    def update_r_from_slider():
        st.session_state.r_range = st.session_state.r_slider
        st.session_state.rmin_input = st.session_state.r_slider[0]
        st.session_state.rmax_input = st.session_state.r_slider[1]

    r_col1, r_col2 = st.sidebar.columns([1.3,1.3])

    with r_col1:
        st.number_input("R min", min_value=0.0, max_value=6.0, step=0.1, key="rmin_input", on_change=update_r_from_input)

    with r_col2:
        st.number_input("R max", min_value=0.0, max_value=6.0, step=0.1, key="rmax_input", on_change=update_r_from_input)

    r_range = st.sidebar.slider(
        "Select R range",
        min_value=0.0, max_value=6.0, value=st.session_state.r_range, step=0.1, key="r_slider", on_change=update_r_from_slider
    )

# LATITUDE θ
    st.sidebar.markdown("### Latitude θ (deg)")

    # Initialize
    if "theta_range" not in st.session_state:
        st.session_state.theta_range = (-90.0, 90.0)
    if "theta_min_input" not in st.session_state:
        st.session_state.theta_min_input = -90.0
    if "theta_max_input" not in st.session_state:
        st.session_state.theta_max_input = 90.0

    def update_theta_from_input():
        st.session_state.theta_range = (st.session_state.theta_min_input, st.session_state.theta_max_input)
        st.session_state.theta_slider = (st.session_state.theta_min_input, st.session_state.theta_max_input)

    def update_theta_from_slider():
        st.session_state.theta_range = st.session_state.theta_slider
        st.session_state.theta_min_input = st.session_state.theta_slider[0]
        st.session_state.theta_max_input = st.session_state.theta_slider[1]

    theta_col1, theta_col2 = st.sidebar.columns([1,1])

    with theta_col1:
        st.number_input("θ min", min_value=-90.0, max_value=90.0, step=0.1, key="theta_min_input", on_change=update_theta_from_input)

    with theta_col2:
        st.number_input("θ max", min_value=-90.0, max_value=90.0, step=0.1, key="theta_max_input", on_change=update_theta_from_input)

    theta_range = st.sidebar.slider(
        "Select θ range",
        min_value=-90.0, max_value=90.0, value=st.session_state.theta_range, step=0.1, key="theta_slider", on_change=update_theta_from_slider
    )

# LONGITUDE φ
    st.sidebar.markdown("### Longitude φ (deg)")

    # Initialize
    if "phi_range" not in st.session_state:
        st.session_state.phi_range = (0.0, 360.0)
    if "phi_min_input" not in st.session_state:
        st.session_state.phi_min_input = 0.0
    if "phi_max_input" not in st.session_state:
        st.session_state.phi_max_input = 360.0

    def update_phi_from_input():
        st.session_state.phi_range = (st.session_state.phi_min_input, st.session_state.phi_max_input)
        st.session_state.phi_slider = (st.session_state.phi_min_input, st.session_state.phi_max_input)

    def update_phi_from_slider():
        st.session_state.phi_range = st.session_state.phi_slider
        st.session_state.phi_min_input = st.session_state.phi_slider[0]
        st.session_state.phi_max_input = st.session_state.phi_slider[1]

    phi_col1, phi_col2 = st.sidebar.columns([1,1])

    with phi_col1:
        st.number_input("φ min", min_value=0.0, max_value=360.0, step=1.0, key="phi_min_input", on_change=update_phi_from_input)

    with phi_col2:
        st.number_input("φ max", min_value=0.0, max_value=360.0, step=1.0, key="phi_max_input", on_change=update_phi_from_input)

    phi_range = st.sidebar.slider(
        "Select φ range",
        min_value=0.0, max_value=360.0, value=st.session_state.phi_range, step=1.0, key="phi_slider", on_change=update_phi_from_slider
    )


# EVENT TYPE (MULTI-SELECT)

    st.sidebar.markdown("## Event Types")
    
    event_type_list = st.sidebar.multiselect(
        "Select one or more event types:",
        options=["ME", "Sheath", "pre_event", "post_event"],
        default=["ME", "Sheath"], # Sensible default so the UI doesn't crash on load
        help="Select multiple regions to extract their combined telemetry."
    )
          
# MAIN SCREEN

    main_col1, main_col2 = st.columns([3,2])



# TOP : SOLAR CYCLE
    st.subheader("Solar Cycle")
    import plotly.graph_objects as go
    fig = go.Figure()

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # ==========================================
    # LAYER 1: CONTINUOUS LINE (No Fill)
    # ==========================================
    fig.add_trace(
        go.Scatter(
            x=silso_df["Date"],
            y=silso_df["SSN"],
            mode="lines",
            name="",
            line=dict(color="#FFB020", width=1.5),
            fill=None, # Strictly no fill, just the mathematical curve
            showlegend=False
        )
    )

    # ==========================================
    # LAYER 2: LEFT OUTSIDE FILL
    # ==========================================
    left_df = silso_df[silso_df["Date"] <= start_dt]
    if not left_df.empty:
        fig.add_trace(
            go.Scatter(
                x=left_df["Date"],
                y=left_df["SSN"],
                mode="lines",
                line=dict(width=0), # Invisible line
                fill="tozeroy",
                fillcolor="rgba(255, 176, 32, 0.1)",
                fillpattern=dict(
                    shape="", # <--- CHANGE OUTSIDE PATTERN HERE "-", "|", "+", "/", "\", "x", ".", or ""
                    fillmode="overlay",
                    fgcolor="rgba(255, 176, 32, 0.5)",
                    size=2
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # ==========================================
    # LAYER 3: INSIDE SELECTED FILL
    # ==========================================
    center_df = silso_df[(silso_df["Date"] >= start_dt) & (silso_df["Date"] <= end_dt)]
    if not center_df.empty:
        fig.add_trace(
            go.Scatter(
                x=center_df["Date"],
                y=center_df["SSN"],
                mode="lines",
                line=dict(width=0), # Invisible line
                fill="tozeroy",
                fillcolor="rgba(255, 176, 32, 0.4)", # Darker base fill for emphasis
                fillpattern=dict(
                    shape="", # <--- CHANGE INSIDE PATTERN HERE "-", "|", "+", "/", "\", "x", ".", or ""
                    fillmode="overlay",
                    fgcolor="rgba(255, 176, 32, 0.9)",
                    size=5
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # ==========================================
    # LAYER 4: RIGHT OUTSIDE FILL
    # ==========================================
    right_df = silso_df[silso_df["Date"] >= end_dt]
    if not right_df.empty:
        fig.add_trace(
            go.Scatter(
                x=right_df["Date"],
                y=right_df["SSN"],
                mode="lines",
                line=dict(width=0), # Invisible line
                fill="tozeroy",
                fillcolor="rgba(255, 176, 32, 0.1)",
                fillpattern=dict(
                    shape="", # <--- CHANGE OUTSIDE PATTERN HERE "-", "|", "+", "/", "\", "x", ".", or ""
                    fillmode="overlay",
                    fgcolor="rgba(255, 176, 32, 0.5)",
                    size=2
                ),
                showlegend=False,
                hoverinfo="skip"
            )
        )

    # Selected date interval (Background shading)
    fig.add_vrect(
        x0=start_dt,
        x1=end_dt,
        fillcolor="rgba(80, 80, 220, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Selected Interval",
        annotation_position="top left",
        annotation_font_color="#FFE7B3"
    )

    # Start marker
    fig.add_vline(
        x=start_dt,
        line_dash="dot",
        line_color="#4F8BF9",
        line_width=2
    )

    # End marker
    fig.add_vline(
        x=end_dt,
        line_dash="dot",
        line_color="#4F8BF9",
        line_width=2
    )

    # Scientific aesthetic layout
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Sunspot Number",
        height=280,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",  
        font=dict(color="#F8EAD8", size=12), 
        margin=dict(l=50, r=30, t=30, b=40),
        xaxis=dict(
            range=[pd.Timestamp("1974-01-01"), pd.Timestamp("2025-12-31")], 
            tickformat="%Y",
            showgrid=False,   
            zeroline=False,
            showline=True,
            linecolor="rgba(255, 255, 255, 0.3)",
            mirror=True,      
            ticks="inside"    
        ),
        yaxis=dict(
            rangemode="tozero",
            showgrid=False,   
            zeroline=False,
            showline=True,
            linecolor="rgba(255, 255, 255, 0.3)",
            mirror=True,
            ticks="inside"
        )
    )
    
    st.plotly_chart(fig, width="stretch")



# MIDDLE : SPATIAL EVENT DISTRIBUTION
    # ==========================================
    st.markdown("<hr style='border: 1px solid rgba(255, 176, 32, 0.2);'>", unsafe_allow_html=True)
    
    # 1. Dynamic Data Extractor (Filters by all sidebar parameters)
    current_events_df = df[
        (df["date_time"].dt.date >= start_date) &
        (df["date_time"].dt.date <= end_date) &
        (df["r"] >= r_range[0]) & (df["r"] <= r_range[1]) &
        (df["theta"] >= theta_range[0]) & (df["theta"] <= theta_range[1]) &
        (df["phi"] >= phi_range[0]) & (df["phi"] <= phi_range[1]) &
        (df["event_type"].isin(event_type_list))
    ].copy()

    # 2. Isolate unique event IDs (Extracts 'PSP0011' from 'PSP0011_2020_042.078750')
    if not current_events_df.empty:
        current_events_df['Event_ID'] = current_events_df['Epoch_ID'].astype(str).str.split('_').str[0]
        # Keep only the first telemetry point of each event to anchor its geometric position
        unique_events_df = current_events_df.drop_duplicates(subset=['Event_ID'])
    else:
        unique_events_df = pd.DataFrame(columns=df.columns)

    st.subheader(f"Spatial Distribution: {len(unique_events_df)} Unique Events Captured")

    # 3. Create Map Columns
    map_col1, map_col2 = st.columns(2)

    # ------------------------------------------
    # MAP 1: MERIDIONAL PLANE (R, Theta) - Side View
    # ------------------------------------------
    with map_col1:
        fig_rt = go.Figure()
        
        # Draw the Sun
        fig_rt.add_trace(go.Scatter(
            x=[0], y=[0], 
            mode='markers', 
            marker=dict(size=18, color='#FFD700', symbol='circle', line=dict(color='#FFA500', width=2)), 
            name='Sun',
            hoverinfo='name'
        ))
        
        # Plot filtered events, colored by spacecraft
        if not unique_events_df.empty:
            for sc in unique_events_df['spacecraft'].unique():
                sc_data = unique_events_df[unique_events_df['spacecraft'] == sc]
                fig_rt.add_trace(go.Scatter(
                    x=sc_data['r'], y=sc_data['theta'],
                    mode='markers',
                    marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color='white')),
                    name=sc,
                    text=sc_data['Event_ID'],
                    hovertemplate="<b>%{text}</b><br>R: %{x:.2f} AU<br>θ: %{y:.1f}°<extra></extra>"
                ))
                
        fig_rt.update_layout(
            title=dict(text="(R, θ) plane", font=dict(color="#FFB020", size=16)),
            xaxis_title="Radial Distance (AU)",
            yaxis_title="Latitude θ (°)",
            xaxis=dict(range=[0, 6.2], showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False),
            yaxis=dict(range=[-90, 90], showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=True, zerolinecolor="rgba(255, 176, 32, 0.4)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 35, 60, 0.4)", # Matches your glassmorphic cards
            font=dict(color="#F8EAD8"),
            margin=dict(l=40, r=20, t=40, b=10),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_rt, width="stretch")

    # ------------------------------------------
    # MAP 2: ECLIPTIC PLANE (R, Phi) - Top View
    # ------------------------------------------
    with map_col2:
        fig_rp = go.Figure()
        
        # Draw the Sun at Origin
        fig_rp.add_trace(go.Scatterpolar(
            r=[0], theta=[0],
            mode='markers',
            marker=dict(size=18, color='#FFD700', symbol='circle', line=dict(color='#FFA500', width=2)),
            name='Sun',
            hoverinfo='name'
        ))
        
        # Plot filtered events on Polar Coordinates
        if not unique_events_df.empty:
            for sc in unique_events_df['spacecraft'].unique():
                sc_data = unique_events_df[unique_events_df['spacecraft'] == sc]
                fig_rp.add_trace(go.Scatterpolar(
                    r=sc_data['r'], theta=sc_data['phi'],
                    mode='markers',
                    marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color='white')),
                    name=sc,
                    text=sc_data['Event_ID'],
                    hovertemplate="<b>%{text}</b><br>R: %{r:.2f} AU<br>φ: %{theta:.1f}°<extra></extra>"
                ))
                
        fig_rp.update_layout(
            title=dict(text="(R, φ) plane", font=dict(color="#FFB020", size=16)),
            polar=dict(
                radialaxis=dict(range=[0, 6.2], showgrid=True, gridcolor="rgba(255,255,255,0.2)", tickfont=dict(color="#F8EAD8")),
                angularaxis=dict(direction="counterclockwise", showgrid=True, gridcolor="rgba(255,255,255,0.2)", tickfont=dict(color="#F8EAD8")),
                bgcolor="rgba(15, 35, 60, 0.4)" # Matches your glassmorphic cards
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8EAD8"),
            margin=dict(l=40, r=40, t=40, b=10),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_rp, width="stretch")

    st.markdown("<hr style='border: 1px solid rgba(255, 176, 32, 0.2);'>", unsafe_allow_html=True)



# BOTTOM : SPACECRAFT DIRECTORY

    st.subheader("Available Spacecrafts for Selected Ranges")

    available_spacecraft = []
    for sc, limits in SPACECRAFT_LIMITS.items():
        # 1. Spatial Check (from Dictionary)
        r_match = not (limits["r"][1] < r_range[0] or limits["r"][0] > r_range[1])
        theta_match = not (limits["theta"][1] < theta_range[0] or limits["theta"][0] > theta_range[1])
        phi_match = not (limits["phi"][1] < phi_range[0] or limits["phi"][0] > phi_range[1])

        if r_match and theta_match and phi_match:
            # 2. Temporal Check (from DataFrame)
            # Does this spacecraft have ANY data inside the selected date bounds?
            has_time_data = df[
                (df["spacecraft"] == sc) &
                (df["date_time"].dt.date >= start_date) &
                (df["date_time"].dt.date <= end_date)
            ].shape[0] > 0

            if has_time_data:
                available_spacecraft.append(sc)


# Display available spacecraft
    if available_spacecraft:
        st.success(f"{len(available_spacecraft)} spacecraft available in this positional range")

        available_df = pd.DataFrame({
            "Spacecraft": available_spacecraft,
            "Temporal Availability of Spacecraft": [f"{TEMPORAL_LIMITS.get(sc, ('N/A', 'N/A'))[0]} \u2013 {TEMPORAL_LIMITS.get(sc, ('N/A', 'N/A'))[1]}" for sc in available_spacecraft],
            "Radial Range (AU)": [f"{SPACECRAFT_LIMITS[sc]['r'][0]} \u2013 {SPACECRAFT_LIMITS[sc]['r'][1]}" for sc in available_spacecraft],
            "Latitude (deg)": [f"{SPACECRAFT_LIMITS[sc]['theta'][0]} \u2013 {SPACECRAFT_LIMITS[sc]['theta'][1]}" for sc in available_spacecraft],
            "Longitude (deg)": [f"{SPACECRAFT_LIMITS[sc]['phi'][0]} \u2013 {SPACECRAFT_LIMITS[sc]['phi'][1]}" for sc in available_spacecraft]
        })
        
        # Apply Pandas Styler to force bold headers and match your amber GUI color
        styled_df = available_df.style.set_table_styles([
            {"selector": "th.col_heading", "props": [("font-weight", "bold"), ("color", "#FFB020"), ("background-color", "rgba(15, 35, 60, 0.4)")]},
            # Add these lines to completely remove the index headers and cells
            {"selector": "th.row_heading", "props": [("display", "none")]},
            {"selector": "th.blank", "props": [("display", "none")]}
        ])

        st.table(styled_df)
        
        
        # Using st.table() forces a beautiful, static HTML table instead of an interactive scroll box
        # st.table(available_df)

        st.markdown("### Select Spacecrafts")
        selected_sc = []

        # Create a clean, symmetrical 4-column grid for the checkboxes
        num_columns = 4
        cols = st.columns(num_columns)

        # Loop through available spacecraft and place them in the grid
        for idx, sc in enumerate(available_spacecraft):
            with cols[idx % num_columns]:
                if st.checkbox(sc, value=True, key=f"chk_{sc}"):
                    selected_sc.append(sc)

    else:
        st.warning("No spacecraft data exists for the selected time, region, and event parameters.")


          
    # SUBMIT BUTTON
          
    if st.button("Go to Parameter Selection →"):
        if not selected_sc:
            st.error("Please select at least one spacecraft.")
        else:
            st.session_state.active_query = {
                "event_type": event_type_list,
                "start_date": start_date,
                "end_date": end_date,
                "r_min": r_range[0],
                "r_max": r_range[1],
                "theta_min": theta_range[0],
                "theta_max": theta_range[1],
                "phi_min": phi_range[0],
                "phi_max": phi_range[1],
                "spacecrafts": selected_sc,
            }
            st.session_state.page_view = "parameters"
            st.rerun()

   
# PARAMETER EXTRACTOR SCREEN (Simulates clean secondary page/view processing)
elif st.session_state.page_view == "parameters":

    # Completely remove the sidebar from the document flow
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    q = st.session_state.active_query
    
    # Enhanced, centered header matching the GUI aesthetic
    st.markdown(
        "<h1 style='text-align: center; color: #FFB020; margin-bottom: 25px;'>Database Parameter Configuration Environment</h1>", 
        unsafe_allow_html=True
    )



# TWO PHASE PARAMETER SELECTION
   
    phase1_selected = []
    phase2_selected = []

    col1, col2, col3 = st.columns(3)
   
    # ==========================================
    # COLUMN 1: MAGNETIC PARAMETERS
    # ==========================================
    with col1:
        st.markdown("<h3 style='text-align: left; color: #FFB020;'>Magnetic Parameters</h3>", unsafe_allow_html=True)
        if st.button("Select All Magnetic Parameters", key="select_mag", type="primary", use_container_width=False):
            for group, items in magnetic_groups.items():
                for item in items:
                    st.session_state[f"p1_mag_{item}"] = True
            st.rerun()

        for group, items in mstagnetic_groups.items():
            for item in items:
                # Initialize state cleanly
                if f"p1_mag_{item}" not in st.session_state:
                    st.session_state[f"p1_mag_{item}"] = False
                
                label = units.get(item, item)
                if st.checkbox(label, key=f"p1_mag_{item}"):
                    phase1_selected.append(item)

    # ==========================================
    # COLUMN 2: PLASMA PARAMETERS
    # ==========================================
    with col2:
        st.markdown("<h3 style='text-align: left; color: #FFB020;'>Plasma Parameters</h3>", unsafe_allow_html=True)
        if st.button("Select All Plasma Parameters", key="select_plas", type="primary", use_container_width=False):
            for group, items in plasma_groups.items():
                for item in items:
                    st.session_state[f"p1_plas_{item}"] = True
            st.rerun()

        for group, items in plasma_groups.items():
            for item in items:
                # Initialize state cleanly
                if f"p1_plas_{item}" not in st.session_state:
                    st.session_state[f"p1_plas_{item}"] = False
                
                label = units.get(item, item)
                if st.checkbox(label, key=f"p1_plas_{item}"):
                    phase1_selected.append(item)

    # ==========================================
    # COLUMN 3: WAVELET PARAMETERS
    # ==========================================
    with col3:
        st.markdown("<h3 style='text-align: left; color: #FFB020;'>Wavelet Parameters</h3>", unsafe_allow_html=True)
        if st.button("Select All Wavelet Parameters", key="select_wavelet", type="primary", use_container_width=False):
            for group, items in wavelet_groups.items():
                for item in items:
                    st.session_state[f"p2_{item}"] = True
            st.rerun()

        for group, items in wavelet_groups.items():
            for item in items:
                # Initialize state cleanly
                if f"p2_{item}" not in st.session_state:
                    st.session_state[f"p2_{item}"] = False
                
                label = units_wavelet.get(item, item) 
                if st.checkbox(label, key=f"p2_{item}"):
                    phase2_selected.append(item)

    chosen_phase1 = phase1_selected.copy()
    chosen_phase2 = phase2_selected.copy()
                    
    st.markdown("---")
    
    # ==========================================
    # TIER 1: CENTERED SUBMIT BUTTON (Zoom-Proof)
    # ==========================================
    _, sub_col, _ = st.columns([7, 1.5, 7])
    with sub_col:
        if st.button("SUBMIT", width="stretch"):
            if not phase1_selected and not phase2_selected:
                st.warning("Please specify parameter subsets to generate localized telemetry data.")
            else:
                # Complete execution filtering pass on master frame array block
                filtered = df.copy()

                # EVENT TYPE FILTER
                phase1_filtered = df.copy()
                
                # EVENT FILTER (Phase 1)
                phase1_filtered = phase1_filtered[
                    phase1_filtered["event_type"].isin(q["event_type"])
                ]
                
                # SPACE-TIME FILTER
                phase1_filtered = phase1_filtered[
                    (phase1_filtered["spacecraft"].isin(q["spacecrafts"]))
                    & (phase1_filtered["date_time"].dt.date >= q["start_date"])
                    & (phase1_filtered["date_time"].dt.date <= q["end_date"])
                    & (phase1_filtered["r"] >= q["r_min"])
                    & (phase1_filtered["r"] <= q["r_max"])
                    & (phase1_filtered["theta"] >= q["theta_min"])
                    & (phase1_filtered["theta"] <= q["theta_max"])
                    & (phase1_filtered["phi"] >= q["phi_min"])
                    & (phase1_filtered["phi"] <= q["phi_max"])
                ]
                st.write("FINAL FILTERED ROWS:", len(phase1_filtered))
                
                # Create combined location column
                phase2_filtered = wavelet_df.copy()
                        
                # EVENT FILTER (Phase 2)
                phase2_filtered = phase2_filtered[
                    phase2_filtered["event_type"].isin(q["event_type"])
                ]
                        
                # SPACE + TIME + POSITION FILTER
                phase2_filtered = phase2_filtered[
                    (phase2_filtered["spacecraft"].isin(q["spacecrafts"]))
                    & (phase2_filtered["date_time"].dt.date >= q["start_date"])
                    & (phase2_filtered["date_time"].dt.date <= q["end_date"])
                    & (phase2_filtered["r"] >= q["r_min"])
                    & (phase2_filtered["r"] <= q["r_max"])
                    & (phase2_filtered["theta"] >= q["theta_min"])
                    & (phase2_filtered["theta"] <= q["theta_max"])
                    & (phase2_filtered["phi"] >= q["phi_min"])
                    & (phase2_filtered["phi"] <= q["phi_max"])
                ]
                        
                st.session_state.phase1_results = phase1_filtered
                st.session_state.phase2_results = phase2_filtered
                
                st.session_state.phase1_columns = [
                    "Epoch_ID", "date_time", "spacecraft", "event_type", "Location"
                ]
                
                st.session_state.phase2_columns = [
                    "Epoch_ID", "date_time", "spacecraft", "event_type", "Location"
                ]
                
                for param in chosen_phase1:
                    st.session_state.phase1_columns.append(param)
                    sigma_col = "sigma_" + param
                    if sigma_col in phase1_filtered.columns:
                        st.session_state.phase1_columns.append(sigma_col)
                
                for param in chosen_phase2:
                    st.session_state.phase2_columns.append(param)
                
                st.session_state.phase1_parameters = chosen_phase1
                st.session_state.phase2_parameters = chosen_phase2
                st.session_state.page_view = "results"
                st.rerun()

    # Small vertical gap for visual breathing room
    # st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================
    # TIER 2: CENTERED SIDE-BY-SIDE NAVIGATION (Zoom-Proof)
    # ==========================================
    _, nav_col1, nav_col2, _ = st.columns([6, 2, 2, 6])
    with nav_col1:
        if st.button("← Go Back", width="stretch"):
            st.session_state.page_view = "main"
            st.rerun()
    with nav_col2:
        if st.button("🏠 Homepage", width="stretch"):
            st.session_state.page_view = "main"
            st.rerun()


elif st.session_state.page_view == "catalog":
    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] {
        background-color:white !important;
        }
        [data-testid="stDataFrame"] * {
        color:black !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("ICME Event Catalogue")

    if st.button("← Back to Homepage"):
        st.session_state.page_view="main"
        st.session_state.selected_icme_event=None
        st.rerun()

    st.markdown("---")
            
    # CATALOG TABLE

    display_columns = [
        "event_id", "SC", "shock_start", "me_start", "me_end", "r", "theta", "phi"
    ]

    st.markdown(
    """
    <style>
    /* Catalog search box */
    div[data-testid="stTextInput"] label {
        color:#FFE7B3 !important;
        font-size:16px;
        font-weight:600;
    }
    div[data-testid="stTextInput"] input {
        background:rgba(255,255,255,0.12) !important;
        color:#FFE7B3 !important;
        border:1px solid rgba(255,176,32,0.7) !important;
        border-radius:12px !important;
        height:45px !important;
        font-size:16px !important;
        padding-left:15px !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color:rgba(255,231,179,0.6) !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border:2px solid #FFB020 !important;
        box-shadow:0 0 10px rgba(255,176,32,0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    search_sc = st.text_input(
    "🔍 Search Spacecraft / Year",
    placeholder="Examples: JUNO, JUNO 2011, MES 2013, 2015"
)

    catalog_filtered = icme_catalog.copy()

    if search_sc.strip():
        query = search_sc.lower().strip()

    # --------------------------
    # Extract year if present
    # --------------------------
        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        year = None
        if year_match:
            year = int(year_match.group())
            query = query.replace(year_match.group(), "").strip()

    # --------------------------
    # Spacecraft filtering
    # --------------------------
        if query:
            if query in SC_ALIASES:
                code = SC_ALIASES[query]
                catalog_filtered = catalog_filtered[
                catalog_filtered["SC"]
                .astype(str)
                .str.upper()
                .str.contains(code, na=False)
            ]
            else:
                catalog_filtered = catalog_filtered[
                catalog_filtered["SC"]
                .astype(str)
                .str.contains(query, case=False, na=False)
            ]

    # --------------------------
    # Year filtering
    # --------------------------
        if year is not None:
            catalog_filtered = catalog_filtered[
                pd.to_datetime(
                catalog_filtered["me_start"],
                errors="coerce"
            ).dt.year == year
        ]

    catalog_display = catalog_filtered[display_columns].copy()

    event_table = st.dataframe(
    catalog_display,
    width="stretch",
    height=650,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row"
)
    if event_table.selection.rows:
        selected_index = event_table.selection.rows[0]
        selected_event = (
        catalog_display
        .iloc[selected_index]["event_id"]
    )
        st.session_state.selected_icme_event = selected_event

    if st.session_state.selected_icme_event:
        event = st.session_state.selected_icme_event
        st.divider()
        st.subheader(
        f"{event} ICME Plot"
    )
        plot_path = os.path.join(
        PLOT_DATABASE,
        event + ".png"
    )
        
        if os.path.exists(plot_path):
            img = Image.open(plot_path)
            st.image(
            img,
            width="stretch"
        )
        else:
            st.warning(
            f"No plot available for {event}"
        )

# FINAL RESULTS CONTEXT VIEW

elif st.session_state.page_view == "results":

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.success("Extraction Completed Successfully!")

    # # BACK BUTTON
    # if st.button("← Go back"):
    #     st.session_state.page_view = "main"
    #     st.rerun()
            
    # LOAD SEPARATE RESULTS

    phase1_df = st.session_state.get(
    "phase1_results",
    pd.DataFrame()
)

    phase2_df = st.session_state.get(
    "phase2_results",
    pd.DataFrame()
)
 
    phase1_cols = st.session_state.phase1_columns
    phase2_cols = st.session_state.phase2_columns
            
    # QUERY SUMMARY
    st.markdown("## Selection Summary")

    if st.session_state.active_query is None:
        st.warning(
            "Please submit a spacecraft query first."
        )
        st.session_state.page_view = "main"
        st.rerun()

    q = st.session_state.active_query
    display_event = q["event_type"]

    if q["event_type"] == "ICME":
        display_event = "ICME (ME + Sheath)"

    w_params = st.session_state.get("phase2_parameters", [])
    p_params = st.session_state.get("phase1_parameters", [])

    # Map raw variables to their strict LaTeX mathematical notations
    p_labels = [short_notations.get(p, p) for p in p_params]
    w_labels = [short_notations.get(w, w) for w in w_params]

    query_info = {
        "Spacecraft": ", ".join(q["spacecrafts"]),
        "Event Type": display_event,
        "Start Date": str(q["start_date"]),
        "End Date": str(q["end_date"]),
        "Radial Range": f'{q["r_min"]} - {q["r_max"]} AU',
        "Latitude Range": f'{q["theta_min"]} - {q["theta_max"]} deg',
        "Longitude Range": f'{q["phi_min"]} - {q["phi_max"]} deg',
        "Plasma/magnetic Parameters": ", ".join(p_labels) if p_labels else "None",
        "Wavelet Parameters": ", ".join(w_labels) if w_labels else "None"
    }

    summary_df = pd.DataFrame(
        list(query_info.items()),
        columns=["Selection", "Description"]
    )

    st.table(summary_df)

# MERGE PHASE 1 + PHASE 2

    phase1_export = phase1_df[phase1_cols].copy()
    phase2_export = phase2_df[phase2_cols].copy()

# Ensure datetime format
    phase1_export["date_time"] = pd.to_datetime(
    phase1_export["date_time"],
    errors="coerce"
)
    phase2_export["date_time"] = pd.to_datetime(
    phase2_export["date_time"],
    errors="coerce"
)

# Merge using timestamp + spacecraft + Location to prevent duplication
    merge_keys = ["Epoch_ID", "date_time", "spacecraft", "event_type", "Location"]

    final_df = pd.merge(
    phase1_export,
    phase2_export,
    on=merge_keys,
    how="outer",
    suffixes=("", "_wavelet")
)


# Remove duplicate columns
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # # DROP DATE_TIME AS REQUESTED (Keeping only Epoch_ID)
    # if "date_time" in final_df.columns:
    #     final_df = final_df.drop(columns=["date_time"])

    st.subheader("Combined Dataset Preview")

# Columns always shown (date_time removed from this array)
    output_cols = ["Epoch_ID", "date_time", "event_type", "Location"]



# Add selected Phase 1 parameters
    for p in st.session_state.phase1_parameters:
        if p in final_df.columns and p not in output_cols:
            output_cols.append(p)
        sigma_col = "sigma_" + p
        if sigma_col in final_df.columns and sigma_col not in output_cols:
            output_cols.append(sigma_col)

# Add selected Phase 2 wavelet parameters
    for p in st.session_state.phase2_parameters:
        if p in final_df.columns and p not in output_cols:
            output_cols.append(p)

# Keep only available columns
    preview_df = final_df[[col for col in output_cols if col in final_df.columns]]

    st.dataframe(
    preview_df.head(),
    width="stretch"
)




# Download
    combined_txt = final_df.to_string(
    index=False,
    na_rep="NaN"
)
    st.download_button(
    label="⬇ DOWNLOAD DATASET",
    data=combined_txt,
    file_name=f"{q['event_type']}_combined_{datetime.now().strftime('%Y%m%d%H%M')}.txt",
    mime="text/plain"
)



    st.markdown("---") 
    _, nav_col1, nav_col2, _ = st.columns([5, 1.5, 1.5, 5])
    with nav_col1:
        if st.button("← Go Back", width="stretch"):
            st.session_state.page_view = "parameters"
            st.rerun()
    with nav_col2:
        if st.button("🏠 Homepage", width="stretch"):
            st.session_state.page_view = "main"
            st.rerun()



elif st.session_state.page_view == "statistics":

    # 1. HIDE THE SIDEBAR AND THE COLLAPSE TOGGLE BUTTON
    st.markdown(
        """
        <style>
            /* Hide the main sidebar container */
            section[data-testid="stSidebar"] {
                display: none !important;
            }
            /* Hide the floating collapse/expand chevron button */
            [data-testid="collapsedControl"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Statistical Plotting Interface")

    if st.button("← Back to Homepage"):
        st.session_state.page_view = "main"
        # Optional: Clear the active plot when leaving the page so it resets for next time
        if "active_prob_plot" in st.session_state:
            del st.session_state.active_prob_plot
        st.rerun()
            
    st.markdown("---")
    
    # ==========================================
    # STATIC PROBABILITY DISTRIBUTIONS SECTION
    # ==========================================
    st.subheader("Probability Distributions (Pre-computed)")
    
    @st.cache_data
    def get_available_prob_plots(folder_path, prefix):
        if not os.path.exists(folder_path): 
            return {}
        
        file_map = {}
        for file in os.listdir(folder_path):
            if file.startswith(prefix) and file.lower().endswith((".png", ".jpg", ".jpeg")):
                temp = file[len(prefix):] 
                range_str = temp.rsplit("_AU.", 1)[0] 
                
                parts = range_str.split('_')
                if len(parts) == 2:
                    try:
                        min_val = float(parts[0].replace('p', '.'))
                        max_val = float(parts[1].replace('p', '.'))
                        clean_label = f"{min_val:.2f} - {max_val:.2f} AU"
                        file_map[clean_label] = file
                    except ValueError:
                        continue
                        
        return dict(sorted(file_map.items(), key=lambda x: float(x[0].split()[0])))

    # Callback to clear the active plot from the screen if the user changes the dropdowns
    def clear_active_plot():
        if "active_prob_plot" in st.session_state:
            del st.session_state.active_prob_plot

    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        plot_category = st.selectbox(
            "Select Comparison Type",
            ["ME vs Sheath", "ICME vs Non-ICME"],
            on_change=clear_active_plot
        )
        
    if plot_category == "ME vs Sheath":
        target_folder = ME_SHEATH_DIR
        prefix = "ICME_radial_"
    else:
        target_folder = ICME_SW_DIR
        prefix = "ICME_vs_SolarWind_radial_"
        
    bin_mapping = get_available_prob_plots(target_folder, prefix)
    
    with prob_col2:
        if not bin_mapping:
            radial_selection = st.selectbox("Radial Bin (AU)", ["No plots found in directory"], on_change=clear_active_plot)
        else:
            radial_selection = st.selectbox("Select Radial Bin (AU)", list(bin_mapping.keys()), on_change=clear_active_plot)

    # 2. CENTERED GENERATE BUTTON
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([7, 2, 7])
    
    with btn_col:
        # Use type="primary" so it matches the amber GUI color from your CSS
        if st.button("Generate Plot", type="primary", width="stretch"):
            if bin_mapping and radial_selection in bin_mapping:
                # Save the target image to session state so it survives Streamlit reruns
                st.session_state.active_prob_plot = os.path.join(target_folder, bin_mapping[radial_selection])
                st.session_state.active_prob_name = bin_mapping[radial_selection]
            else:
                st.error("No valid plot available to generate.")
            
    # 3. DISPLAY PLOT & DOWNLOAD BUTTON
    if "active_prob_plot" in st.session_state and os.path.exists(st.session_state.active_prob_plot):
        img_path = st.session_state.active_prob_plot
        st.image(img_path, width="stretch")
        
        # Read the file bytes to power the download button
        with open(img_path, "rb") as file:
            img_bytes = file.read()
            
        _, dl_col, _ = st.columns([7, 2, 7])
        with dl_col:
            mime_type = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
            st.download_button(
                label="⬇ Download Plot",
                data=img_bytes,
                file_name=st.session_state.active_prob_name,
                mime=mime_type,
                width="stretch"
            )
    elif "active_prob_plot" in st.session_state:
        st.error(f"Image file missing or inaccessible at: {st.session_state.active_prob_plot}")

    st.markdown("---")
    
    
    
    
    # 1. PLOT DEFINITION MAP (PLACE HERE)
            
    PLOT_DEFINITIONS = {
        "Attribute vs Probability": {
            "func": sf.attr_vs_probability,
            "args": ["event_type", "radial_bin", "attribute_key"]
        },
        "Theta 2D (Year vs Latitude)": {
            "func": sf.t_theta_attr_R_2d,
            "args": ["attribute_key", "event_type","silso_csv"]
        },
        "Theta 3D": {
            "func": sf.t_theta_attr_R_3d,
            "args": ["attribute_key", "event_type","silso_csv"]
        },
        "Phi 2D": {
            "func": sf.t_phi_attr_R_2d,
            "args": ["attribute_key", "event_type","silso_csv"]
        },
        "Phi 3D": {
        "func": sf.t_phi_attr_R_3d,
        "args": ["attribute_key", "event_type", "silso_csv"]
        },
        "Attribute vs Events": {
            "func": sf.attr_events,
            "args": ["event_type", "radial_bin", "attribute_key"]
        },
        "R-phi / R-theta": {
            "func": sf.R_phi_R_theta,
            "args": ["event_type"]
        },
        "Spatial 4D": {
            "func": sf.spatial_events_4D,
            "args": ["event_type"]
        },
        "Spacecraft Events": {
            "func": sf.spacecraft_events,
            "args": ["event_type"]
        },
        "Butterfly Overlay": {
            "func": sf.events_over_butterfly,
            "args": ["event_type","df_rgo"]
        }
    }
            
    # 2. INPUT BUILDER (OK HERE, BUT IMPORTANT RULE FIXED)
            
    def build_inputs(arg_list):
        user_inputs = {}
        for arg in arg_list:
            if arg == "event_type":
                # ADDED PRE AND POST EVENT TO STATISTICS VIEW DROPDOWN
                user_inputs["event_type"] = st.selectbox(
                    "Event Type",
                    ["ICME", "ME", "Sheath", "Both", "pre_event", "post_event"]
                )

            elif arg == "attribute_key":
                user_inputs["attribute_key"] = st.selectbox(
                    "Attribute",
                    list(units.keys())
                )

            elif arg == "radial_bin":
            # initialize state if not present
                if "radial_min" not in st.session_state:
                    st.session_state.radial_min = 0.3
                if "radial_max" not in st.session_state:
                    st.session_state.radial_max = 1.5

            # sync from input boxes
                def sync_radial_from_inputs():
                    st.session_state.radial_bin = (
                    st.session_state.radial_min,
                    st.session_state.radial_max
                    )

            # sync from slider
                def sync_radial_from_slider():
                    st.session_state.radial_min = st.session_state.radial_bin[0]
                    st.session_state.radial_max = st.session_state.radial_bin[1]

                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("R min (AU)", min_value=0.0, max_value=6.0, step=0.1, key="radial_min", on_change=sync_radial_from_inputs)
                with col2:
                    st.number_input("R max (AU)", min_value=0.0, max_value=6.0, step=0.1, key="radial_max", on_change=sync_radial_from_inputs)

                st.slider("Radial Bin (AU)", min_value=0.0, max_value=6.0, value=(st.session_state.radial_min, st.session_state.radial_max), step=0.1, key="radial_bin", on_change=sync_radial_from_slider)

                user_inputs["radial_bin"] = st.session_state.radial_bin

            elif arg == "silso_csv":
                user_inputs["silso_csv"] = df_silso

            elif arg == "df_rgo":
                user_inputs["df_rgo"] = df_rgo

            else:
                st.warning(f"Unknown input: {arg}")

        return user_inputs

            
    # 3. UI: SELECT PLOT
            
    plot_name = st.selectbox(
        "Choose Plot",
        list(PLOT_DEFINITIONS.keys())
    )

    plot_config = PLOT_DEFINITIONS[plot_name]

    st.markdown("---")
    st.subheader("Input Parameters")

    # build required inputs dynamically
    inputs = build_inputs(plot_config["args"])

    st.markdown("---")
            
    # 4. EXECUTION
            
    if st.button("Generate Plot"):
        func = plot_config["func"]

        # SAFE FUNCTION CALL
        try:
            result = func(
    mother_dict=mother_dict_all,
    **inputs
)

            # If function returns matplotlib/plotly figure
            if result is not None:
                if hasattr(result, "savefig"):
                    st.pyplot(result)
                else:
                    st.plotly_chart(result)

        except Exception as e:
            st.error(f"Plot generation failed: {e}")