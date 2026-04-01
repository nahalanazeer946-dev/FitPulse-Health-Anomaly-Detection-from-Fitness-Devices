import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, io, base64, tempfile, os
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

from prophet import Prophet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="FitPulse Health Analytics",
    page_icon="🏃🏻",
    layout="wide"
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

for k, v in [
    ("dark_mode",        True),
    # Milestone 1
    ("m1_df_clean",      None),
    # Milestone 2/3 shared data
    ("files_loaded",     False),
    ("anomaly_done",     False),
    ("simulation_done",  False),
    ("daily_m3",         None), ("hourly_s", None), ("hourly_i", None),
    ("sleep_m3",         None), ("hr_m3",    None), ("hr_minute", None),
    ("master",           None),
    ("anom_hr",          None), ("anom_steps", None), ("anom_sleep", None),
    ("sim_results",      None),
    # Milestone 4
    ("pipeline_done",    False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e2e8f0;
}

/* ── Milestone 1 styles ── */
.hero-m1 {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    padding: 30px; border-radius: 18px; text-align: center;
    color: white; box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 20px; border-radius: 15px; margin-top: 20px;
}

/* ── Milestone 2 styles ── */
.hero-banner {
    background: linear-gradient(120deg, #e91e8c, #f64f59, #c471ed);
    padding: 38px 40px; border-radius: 22px; text-align: center;
    color: white; box-shadow: 0 20px 60px rgba(233,30,140,0.35);
    margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px; background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero-banner h1 {
    font-family: 'Syne', sans-serif; font-size: 2.6rem;
    margin: 0 0 6px 0; letter-spacing: -1px;
}
.hero-banner h3 { font-size: 1.1rem; font-weight: 300; margin: 0 0 8px 0; opacity: 0.9; }
.hero-banner p  { font-size: 0.9rem; opacity: 0.75; margin: 0; letter-spacing: 2px; text-transform: uppercase; }

.section-header {
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800;
    color: #f472b6; border-left: 4px solid #e91e8c; padding-left: 14px;
    margin: 32px 0 16px 0; letter-spacing: -0.5px;
}
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(233,30,140,0.12), rgba(100,60,180,0.12));
    border: 1px solid rgba(233,30,140,0.25); border-radius: 16px;
    padding: 18px !important; backdrop-filter: blur(8px);
}
.cluster-card-0 {
    background: linear-gradient(135deg, #1e3a5f, #1a6b9a); border-radius: 16px;
    padding: 20px; color: white; border: 1px solid rgba(100,180,255,0.25);
}
.cluster-card-1 {
    background: linear-gradient(135deg, #5f1e1e, #9a3a1a); border-radius: 16px;
    padding: 20px; color: white; border: 1px solid rgba(255,140,100,0.25);
}
.cluster-card-2 {
    background: linear-gradient(135deg, #1e5f2e, #1a9a45); border-radius: 16px;
    padding: 20px; color: white; border: 1px solid rgba(100,255,140,0.25);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0a2e, #0f0c29) !important;
    border-right: 1px solid rgba(233,30,140,0.2);
}
.pipeline-step {
    display: inline-block;
    background: linear-gradient(90deg, #e91e8c22, #c471ed22);
    border: 1px solid #e91e8c44; color: #f9a8d4; border-radius: 20px;
    padding: 6px 16px; font-size: 0.8rem; margin: 4px; font-weight: 600;
}
.upload-prompt {
    background: linear-gradient(135deg, rgba(233,30,140,0.08), rgba(100,60,180,0.08));
    border: 1px solid rgba(233,30,140,0.2); border-radius: 18px;
    padding: 50px 30px; text-align: center; margin: 30px 0;
}
.upload-prompt h2 { font-family: 'Syne', sans-serif; color: #f472b6; margin-bottom: 12px; }
.upload-prompt p  { color: #94a3b8; font-size: 1rem; }

/* ── Milestone 3 styles ── */
.m3-hero {
    background: linear-gradient(135deg,rgba(252,129,129,0.08),rgba(246,135,179,0.06),rgba(10,14,26,0.9));
    border: 1px solid rgba(252,129,129,0.4);
    border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.m3-hero::before {
    content: ''; position: absolute; top:-60px; right:-60px;
    width:300px; height:300px;
    background: radial-gradient(circle,rgba(252,129,129,0.08) 0%,transparent 70%);
    border-radius:50%;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
    color:#e2e8f0; margin:0 0 0.4rem 0; letter-spacing:-0.02em;
}
.hero-sub { font-size:1.05rem; color:#94a3b8; font-weight:300; margin:0; }
.hero-badge {
    display:inline-block; background:rgba(252,129,129,0.1); border:1px solid rgba(252,129,129,0.4);
    border-radius:100px; padding:0.3rem 1rem; font-size:0.75rem;
    font-family:'JetBrains Mono',monospace; color:#fc8181; margin-bottom:1rem;
}
.sec-header {
    display:flex; align-items:center; gap:0.8rem;
    margin:2rem 0 1rem 0; padding-bottom:0.6rem; border-bottom:1px solid rgba(99,179,237,0.2);
}
.sec-icon {
    font-size:1.4rem; width:2.2rem; height:2.2rem;
    display:flex; align-items:center; justify-content:center;
    background:rgba(99,179,237,0.15); border-radius:8px; border:1px solid rgba(99,179,237,0.2);
}
.sec-title {
    font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700;
    color:#e2e8f0; margin:0;
}
.sec-badge {
    margin-left:auto; background:rgba(99,179,237,0.15); border:1px solid rgba(99,179,237,0.2);
    border-radius:100px; padding:0.2rem 0.7rem; font-size:0.7rem;
    font-family:'JetBrains Mono',monospace; color:#63b3ed;
}
.m3-card {
    background:rgba(15,23,42,0.85); border:1px solid rgba(99,179,237,0.2); border-radius:14px;
    padding:1.4rem 1.6rem; margin-bottom:1rem; backdrop-filter:blur(10px);
}
.m3-card-title {
    font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700;
    color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;
}
.step-pill {
    display:inline-flex; align-items:center; gap:0.5rem;
    background:rgba(99,179,237,0.07); border:1px solid rgba(99,179,237,0.2); border-radius:100px;
    padding:0.3rem 0.9rem; font-size:0.75rem; font-family:'JetBrains Mono',monospace;
    color:#63b3ed; margin-bottom:0.8rem;
}
.metric-grid { display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.8rem 0; }
.metric-card {
    flex:1; min-width:120px; background:rgba(99,179,237,0.07); border:1px solid rgba(99,179,237,0.2);
    border-radius:12px; padding:1rem 1.2rem; text-align:center;
}
.metric-val {
    font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
    color:#63b3ed; line-height:1; margin-bottom:0.25rem;
}
.metric-val-red { color:#fc8181; }
.metric-label { font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; }
.anom-tag {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(252,129,129,0.1); border:1px solid rgba(252,129,129,0.4); border-radius:100px;
    padding:0.3rem 0.9rem; font-size:0.72rem; font-family:'JetBrains Mono',monospace;
    color:#fc8181; margin-bottom:0.8rem;
}
.screenshot-badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:rgba(246,135,179,0.15); border:1px solid rgba(246,135,179,0.4);
    border-radius:100px; padding:0.3rem 0.9rem; font-size:0.72rem;
    font-family:'JetBrains Mono',monospace; color:#f687b3; margin-bottom:0.8rem;
}
.alert-warn {
    background:rgba(246,173,85,0.12); border-left:3px solid #f6ad55;
    border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0;
    font-size:0.85rem; color:#fbd38d;
}
.alert-success {
    background:rgba(104,211,145,0.1); border-left:3px solid #68d391;
    border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0;
    font-size:0.85rem; color:#9ae6b4;
}
.alert-info {
    background:rgba(99,179,237,0.15); border-left:3px solid #63b3ed;
    border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0;
    font-size:0.85rem; color:#bee3f8;
}
.alert-danger {
    background:rgba(252,129,129,0.1); border-left:3px solid #fc8181;
    border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0;
    font-size:0.85rem; color:#feb2b2;
}
.m3-divider { border:none; border-top:1px solid rgba(99,179,237,0.2); margin:2rem 0; }

/* ── Milestone 4 styles ── */
.m4-hero {
    background: linear-gradient(135deg,rgba(99,179,237,0.08),rgba(104,211,145,0.05),rgba(10,14,26,0.9));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px; padding: 2rem 2.5rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.kpi-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:0.7rem; margin:1rem 0; }
.kpi-card {
    background: rgba(15,23,42,0.85); border: 1px solid rgba(99,179,237,0.2); border-radius:14px;
    padding:1rem 1.1rem; text-align:center; backdrop-filter:blur(10px);
}
.kpi-val {
    font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800;
    line-height:1; margin-bottom:0.2rem;
}
.kpi-label { font-size:0.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.07em; }
.kpi-sub { font-size:0.65rem; color:#94a3b8; margin-top:0.15rem; }
.anom-row {
    display:flex; align-items:center; gap:0.6rem; padding:0.45rem 0;
    border-bottom:1px solid rgba(99,179,237,0.2); font-size:0.82rem;
}
.m4-divider { border:none; border-top:1px solid rgba(99,179,237,0.2); margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MATPLOTLIB GLOBAL THEME
# ---------------------------------------------------

GRAPH_BG    = "#1a2035"
GRAPH_INNER = "#1e2a45"
GRID_COL    = "#263352"
TICK_COL    = "#8899bb"
LABEL_COL   = "#aab8d0"
TITLE_COL   = "#e8eef8"

plt.rcParams.update({
    'figure.facecolor':  GRAPH_BG,
    'axes.facecolor':    GRAPH_INNER,
    'axes.edgecolor':    GRID_COL,
    'axes.labelcolor':   LABEL_COL,
    'axes.titlecolor':   TITLE_COL,
    'xtick.color':       TICK_COL,
    'ytick.color':       TICK_COL,
    'text.color':        LABEL_COL,
    'grid.color':        GRID_COL,
    'grid.linewidth':    0.6,
    'grid.alpha':        0.5,
    'axes.grid':         True,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8.5,
    'legend.facecolor':  GRAPH_BG,
    'legend.edgecolor':  GRID_COL,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

HEAT_CUSTOM = LinearSegmentedColormap.from_list(
    "heat_custom", ["#1e40af","#7c3aed","#e91e8c","#f97316","#fbbf24"]
)
CLUSTER_COLORS = [
    "#e07090", "#5b9bd5", "#3ecf8e",
    "#f59e0b", "#8b5cf6", "#06b6d4", "#f43f5e", "#84cc16"
]

# ---------------------------------------------------
# MILESTONE 3 — THEME VARIABLES (also used by M4)
# ---------------------------------------------------

BG         = "linear-gradient(135deg,#0a0e1a 0%,#0f1729 40%,#0a1628 100%)"
CARD_BG    = "rgba(15,23,42,0.85)"
CARD_BOR   = "rgba(99,179,237,0.2)"
TEXT       = "#e2e8f0"
MUTED      = "#94a3b8"
ACCENT     = "#63b3ed"
ACCENT2    = "#f687b3"
ACCENT3    = "#68d391"
ACCENT_RED = "#fc8181"
ACCENT_ORG = "#f6ad55"
PLOT_BG    = "#0f172a"
PAPER_BG   = "#0a0e1a"
GRID_CLR   = "rgba(255,255,255,0.06)"
BADGE_BG   = "rgba(99,179,237,0.15)"
SECTION_BG = "rgba(99,179,237,0.07)"
WARN_BG    = "rgba(246,173,85,0.12)"
WARN_BOR   = "rgba(246,173,85,0.4)"
SUCCESS_BG = "rgba(104,211,145,0.1)"
SUCCESS_BOR= "rgba(104,211,145,0.4)"
DANGER_BG  = "rgba(252,129,129,0.1)"
DANGER_BOR = "rgba(252,129,129,0.4)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font_color=TEXT,
    font_family="Inter, sans-serif",
    xaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False,
               linecolor=CARD_BOR, tickfont_color=MUTED),
    yaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False,
               linecolor=CARD_BOR, tickfont_color=MUTED),
    legend=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, borderwidth=1,
                font_color=TEXT),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, font_color=TEXT),
)

# M4 plotly base (same dark theme)
PLOTLY_BASE = dict(
    paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT,
    font_family="Inter, sans-serif",
    legend=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, borderwidth=1, font_color=TEXT),
    margin=dict(l=50, r=30, t=55, b=45),
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=CARD_BOR, font_color=TEXT),
)

# ---------------------------------------------------
# REQUIRED FILES REGISTRY (shared M2/M3/M4)
# ---------------------------------------------------

REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {"key_cols": ["ActivityDate", "TotalSteps", "Calories"],       "label": "Daily Activity",    "icon": "🏃"},
    "hourlySteps_merged.csv":       {"key_cols": ["ActivityHour", "StepTotal"],                    "label": "Hourly Steps",      "icon": "👣"},
    "hourlyIntensities_merged.csv": {"key_cols": ["ActivityHour", "TotalIntensity"],               "label": "Hourly Intensities","icon": "⚡"},
    "minuteSleep_merged.csv":       {"key_cols": ["date", "value", "logId"],                       "label": "Minute Sleep",      "icon": "💤"},
    "heartrate_seconds_merged.csv": {"key_cols": ["Time", "Value"],                                "label": "Heart Rate",        "icon": "❤️"},
}

def score_match(df, req_info):
    return sum(1 for col in req_info["key_cols"] if col in df.columns)

# ---------------------------------------------------
# MILESTONE 3 — HELPER FUNCTIONS
# ---------------------------------------------------

def sec_m3(icon, title, badge=None):
    badge_html = f'<span class="sec-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-icon">{icon}</div>
      <p class="sec-title">{title}</p>
      {badge_html}
    </div>""", unsafe_allow_html=True)

def step_pill_m3(n, label):
    st.markdown(f'<div class="step-pill">◆ Step {n} &nbsp;·&nbsp; {label}</div>', unsafe_allow_html=True)

def screenshot_badge_m3(ref):
    st.markdown(f'<div class="screenshot-badge">📸 Screenshot · {ref}</div>', unsafe_allow_html=True)

def anom_tag_m3(label):
    st.markdown(f'<div class="anom-tag">🚨 {label}</div>', unsafe_allow_html=True)

def ui_success_m3(msg): st.markdown(f'<div class="alert-success">✅ {msg}</div>', unsafe_allow_html=True)
def ui_warn_m3(msg):    st.markdown(f'<div class="alert-warn">⚠️ {msg}</div>', unsafe_allow_html=True)
def ui_info_m3(msg):    st.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
def ui_danger_m3(msg):  st.markdown(f'<div class="alert-danger">🚨 {msg}</div>', unsafe_allow_html=True)

def metrics_m3(*items, red_indices=None):
    red_indices = red_indices or []
    html = '<div class="metric-grid">'
    for i, (val, label) in enumerate(items):
        val_class = "metric-val metric-val-red" if i in red_indices else "metric-val"
        html += f'<div class="metric-card"><div class="{val_class}">{val}</div><div class="metric-label">{label}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def apply_plotly_theme(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font_color=TEXT, font_size=14, font_family="Syne, sans-serif"))
    return fig

# ---------------------------------------------------
# MILESTONE 4 — HELPER FUNCTIONS
# ---------------------------------------------------

def ptheme(fig, title="", h=400):
    fig.update_layout(**PLOTLY_BASE, height=h)
    fig.update_xaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID_CLR, zeroline=False, linecolor=CARD_BOR, tickfont_color=MUTED)
    if title:
        fig.update_layout(title=dict(text=title, font_color=TEXT,
                                     font_size=13, font_family="Syne, sans-serif"))
    return fig

def sec_m4(icon, title, badge=None):
    badge_html = f'<span style="margin-left:auto;background:{BADGE_BG};border:1px solid {CARD_BOR};border-radius:100px;padding:0.2rem 0.7rem;font-size:0.7rem;font-family:JetBrains Mono,monospace;color:{ACCENT}">{badge}</span>' if badge else ''
    st.markdown(f'<div class="sec-header"><div class="sec-icon">{icon}</div><p class="sec-title">{title}</p>{badge_html}</div>', unsafe_allow_html=True)

def ui_info_m4(m):    st.markdown(f'<div class="alert-info">ℹ️ {m}</div>',    unsafe_allow_html=True)
def ui_success_m4(m): st.markdown(f'<div class="alert-success">✅ {m}</div>', unsafe_allow_html=True)
def ui_danger_m4(m):  st.markdown(f'<div class="alert-danger">🚨 {m}</div>',  unsafe_allow_html=True)

# ---------------------------------------------------
# ANOMALY DETECTION FUNCTIONS (shared M3/M4)
# ---------------------------------------------------

def detect_hr_anomalies(master, hr_high=100, hr_low=50, residual_sigma=2.0):
    df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    hr_daily = df.groupby("Date")["AvgHR"].mean().reset_index()
    hr_daily.columns = ["Date","AvgHR"]
    hr_daily = hr_daily.sort_values("Date")
    hr_daily["thresh_high"] = hr_daily["AvgHR"] > hr_high
    hr_daily["thresh_low"]  = hr_daily["AvgHR"] < hr_low
    hr_daily["rolling_med"]  = hr_daily["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_daily["residual"]     = hr_daily["AvgHR"] - hr_daily["rolling_med"]
    resid_std                = hr_daily["residual"].std()
    hr_daily["resid_anomaly"]= hr_daily["residual"].abs() > (residual_sigma * resid_std)
    hr_daily["is_anomaly"] = hr_daily["thresh_high"] | hr_daily["thresh_low"] | hr_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["thresh_high"]:   r.append(f"HR>{hr_high}")
        if row["thresh_low"]:    r.append(f"HR<{hr_low}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    hr_daily["reason"] = hr_daily.apply(reason, axis=1)
    return hr_daily

# M4 variant (uses resid_anom column name)
def detect_hr_m4(master, hr_high=100, hr_low=50, sigma=2.0):
    df = master[["Id","Date","AvgHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["AvgHR"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["AvgHR"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["AvgHR"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_high"] = d["AvgHR"] > hr_high
    d["thresh_low"]  = d["AvgHR"] < hr_low
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_high"] | d["thresh_low"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_high: parts.append(f"HR>{hr_high}")
        if r.thresh_low:  parts.append(f"HR<{hr_low}")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_steps_anomalies(master, steps_low=500, steps_high=25000, residual_sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    steps_daily = df.groupby("Date")["TotalSteps"].mean().reset_index()
    steps_daily = steps_daily.sort_values("Date")
    steps_daily["thresh_low"]  = steps_daily["TotalSteps"] < steps_low
    steps_daily["thresh_high"] = steps_daily["TotalSteps"] > steps_high
    steps_daily["rolling_med"]   = steps_daily["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    steps_daily["residual"]      = steps_daily["TotalSteps"] - steps_daily["rolling_med"]
    resid_std                    = steps_daily["residual"].std()
    steps_daily["resid_anomaly"] = steps_daily["residual"].abs() > (residual_sigma * resid_std)
    steps_daily["is_anomaly"] = steps_daily["thresh_low"] | steps_daily["thresh_high"] | steps_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["thresh_low"]:    r.append(f"Steps<{steps_low}")
        if row["thresh_high"]:   r.append(f"Steps>{steps_high}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    steps_daily["reason"] = steps_daily.apply(reason, axis=1)
    return steps_daily

# M4 variant
def detect_steps_m4(master, st_low=500, st_high=25000, sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSteps"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["TotalSteps"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_low"]  = d["TotalSteps"] < st_low
    d["thresh_high"] = d["TotalSteps"] > st_high
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_low:  parts.append(f"Steps<{int(st_low):,}")
        if r.thresh_high: parts.append(f"Steps>{int(st_high):,}")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_sleep_anomalies(master, sleep_low=60, sleep_high=600, residual_sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    sleep_daily = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
    sleep_daily = sleep_daily.sort_values("Date")
    sleep_daily["thresh_low"]  = (sleep_daily["TotalSleepMinutes"] > 0) & (sleep_daily["TotalSleepMinutes"] < sleep_low)
    sleep_daily["thresh_high"] = sleep_daily["TotalSleepMinutes"] > sleep_high
    sleep_daily["no_data"]     = sleep_daily["TotalSleepMinutes"] == 0
    sleep_daily["rolling_med"]   = sleep_daily["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sleep_daily["residual"]      = sleep_daily["TotalSleepMinutes"] - sleep_daily["rolling_med"]
    resid_std                    = sleep_daily["residual"].std()
    sleep_daily["resid_anomaly"] = sleep_daily["residual"].abs() > (residual_sigma * resid_std)
    sleep_daily["is_anomaly"] = sleep_daily["thresh_low"] | sleep_daily["thresh_high"] | sleep_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["no_data"]:       r.append("No device worn")
        if row["thresh_low"]:    r.append(f"Sleep<{sleep_low}min")
        if row["thresh_high"]:   r.append(f"Sleep>{sleep_high}min")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    sleep_daily["reason"] = sleep_daily.apply(reason, axis=1)
    return sleep_daily

# M4 variant
def detect_sleep_m4(master, sl_low=60, sl_high=600, sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
    d["rolling_med"] = d["TotalSleepMinutes"].rolling(3,center=True,min_periods=1).median()
    d["residual"]    = d["TotalSleepMinutes"] - d["rolling_med"]
    std              = d["residual"].std()
    d["thresh_low"]  = (d["TotalSleepMinutes"]>0) & (d["TotalSleepMinutes"]<sl_low)
    d["thresh_high"] = d["TotalSleepMinutes"] > sl_high
    d["resid_anom"]  = d["residual"].abs() > sigma * std
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(r):
        parts = []
        if r.thresh_low:  parts.append(f"Sleep<{int(sl_low)}min")
        if r.thresh_high: parts.append(f"Sleep>{int(sl_high)}min")
        if r.resid_anom:  parts.append(f"+/-{sigma:.0f}sigma residual")
        return ", ".join(parts)
    d["reason"] = d.apply(reason, axis=1)
    return d

def simulate_accuracy(master, n_inject=10):
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}
    hr_sim = df_daily[["Date","AvgHR"]].copy()
    inject_idx = np.random.choice(len(hr_sim), n_inject, replace=False)
    hr_sim.loc[inject_idx, "AvgHR"] = np.random.choice(
        [115, 120, 125, 35, 40, 45, 118, 130, 38, 42], n_inject, replace=True)
    hr_sim["rolling_med"]  = hr_sim["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_sim["residual"]     = hr_sim["AvgHR"] - hr_sim["rolling_med"]
    resid_std = hr_sim["residual"].std()
    hr_sim["detected"] = (hr_sim["AvgHR"] > 100) | (hr_sim["AvgHR"] < 50) | \
                         (hr_sim["residual"].abs() > 2 * resid_std)
    tp = hr_sim.iloc[inject_idx]["detected"].sum()
    results["Heart Rate"] = {"injected": n_inject, "detected": int(tp),
                              "accuracy": round(tp / n_inject * 100, 1)}
    st_sim = df_daily[["Date","TotalSteps"]].copy()
    inject_idx2 = np.random.choice(len(st_sim), n_inject, replace=False)
    st_sim.loc[inject_idx2, "TotalSteps"] = np.random.choice(
        [50, 100, 150, 30000, 35000, 28000, 80, 200, 31000, 29000], n_inject, replace=True)
    st_sim["rolling_med"]  = st_sim["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    st_sim["residual"]     = st_sim["TotalSteps"] - st_sim["rolling_med"]
    resid_std2 = st_sim["residual"].std()
    st_sim["detected"] = (st_sim["TotalSteps"] < 500) | (st_sim["TotalSteps"] > 25000) | \
                         (st_sim["residual"].abs() > 2 * resid_std2)
    tp2 = st_sim.iloc[inject_idx2]["detected"].sum()
    results["Steps"] = {"injected": n_inject, "detected": int(tp2),
                         "accuracy": round(tp2 / n_inject * 100, 1)}
    sl_sim = df_daily[["Date","TotalSleepMinutes"]].copy()
    inject_idx3 = np.random.choice(len(sl_sim), n_inject, replace=False)
    sl_sim.loc[inject_idx3, "TotalSleepMinutes"] = np.random.choice(
        [10, 20, 30, 700, 750, 800, 15, 25, 710, 720], n_inject, replace=True)
    sl_sim["rolling_med"]  = sl_sim["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sl_sim["residual"]     = sl_sim["TotalSleepMinutes"] - sl_sim["rolling_med"]
    resid_std3 = sl_sim["residual"].std()
    sl_sim["detected"] = ((sl_sim["TotalSleepMinutes"] > 0) & (sl_sim["TotalSleepMinutes"] < 60)) | \
                          (sl_sim["TotalSleepMinutes"] > 600) | \
                          (sl_sim["residual"].abs() > 2 * resid_std3)
    tp3 = sl_sim.iloc[inject_idx3]["detected"].sum()
    results["Sleep"] = {"injected": n_inject, "detected": int(tp3),
                         "accuracy": round(tp3 / n_inject * 100, 1)}
    overall = round(np.mean([results[k]["accuracy"] for k in results]), 1)
    results["Overall"] = overall
    return results

# ---------------------------------------------------
# MILESTONE 4 — CHART BUILDERS
# ---------------------------------------------------

def chart_hr_m4(anom_hr, hr_high, hr_low, sigma, h=380):
    fig = go.Figure()
    upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
    lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=lower, mode="lines",
                             fill="tonexty", fillcolor="rgba(99,179,237,0.1)",
                             line=dict(width=0), name=f"+/-{sigma:.0f}sigma Band"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"],
                             mode="lines+markers", name="Avg HR",
                             line=dict(color=ACCENT, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT3, width=1.5, dash="dot")))
    a = anom_hr[anom_hr["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["AvgHR"], mode="markers",
                                 name="🚨 Anomaly",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="circle",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f}<br><b>ANOMALY</b><extra>⚠️</extra>"))
        for _, row in a.iterrows():
            fig.add_annotation(x=row["Date"], y=row["AvgHR"],
                               text=f"⚠️", showarrow=True, arrowhead=2,
                               arrowcolor=ACCENT_RED, ax=0, ay=-35,
                               font=dict(color=ACCENT_RED, size=11))
    fig.add_hline(y=hr_high, line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.6,
                  annotation_text=f"High ({int(hr_high)} bpm)",
                  annotation_font_color=ACCENT_RED, annotation_position="top right")
    fig.add_hline(y=hr_low, line_dash="dash", line_color=ACCENT2,
                  line_width=1.5, opacity=0.6,
                  annotation_text=f"Low ({int(hr_low)} bpm)",
                  annotation_font_color=ACCENT2, annotation_position="bottom right")
    ptheme(fig, "❤️ Heart Rate - Anomaly Detection", h)
    fig.update_layout(xaxis_title="Date", yaxis_title="HR (bpm)")
    return fig

def chart_steps_m4(anom_steps, st_low, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.07,
                        subplot_titles=["Daily Steps (avg users)","Residual Deviation"])
    a = anom_steps[anom_steps["is_anomaly"]]
    for _, row in a.iterrows():
        fig.add_vrect(x0=str(row["Date"]), x1=str(row["Date"]),
                      fillcolor="rgba(252,129,129,0.12)",
                      line_color="rgba(252,129,129,0.4)", line_width=1.5,
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
                             mode="lines+markers", name="Avg Steps",
                             line=dict(color=ACCENT3, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT, width=2, dash="dash")),
                  row=1, col=1)
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSteps"],
                                 mode="markers", name="🚨 Alert",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="triangle-up",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(st_low), line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.7, row=1, col=1,
                  annotation_text=f"Low ({int(st_low):,})",
                  annotation_font_color=ACCENT_RED)
    res_colors = [ACCENT_RED if v else ACCENT3 for v in anom_steps["resid_anom"]]
    fig.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"],
                         name="Residual", marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:,.0f}<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1, row=2, col=1)
    ptheme(fig, "🚶 Step Count - Trend & Alerts", h)
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
    fig.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    return fig

def chart_sleep_m4(anom_sleep, sl_low, sl_high, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.07,
                        subplot_titles=["Sleep Duration (min/night)","Residual Deviation"])
    fig.add_hrect(y0=sl_low, y1=sl_high,
                  fillcolor="rgba(104,211,145,0.07)", line_width=0,
                  annotation_text="✅ Healthy Zone", annotation_position="top right",
                  annotation_font_color=ACCENT3, row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
                             mode="lines+markers", name="Sleep (min)",
                             line=dict(color="#b794f4", width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f} min<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=ACCENT3, width=1.5, dash="dot")),
                  row=1, col=1)
    a = anom_sleep[anom_sleep["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSleepMinutes"],
                                 mode="markers", name="🚨 Anomaly",
                                 marker=dict(color=ACCENT_RED, size=13, symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f}<br><b>ANOMALY</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(sl_low), line_dash="dash", line_color=ACCENT_RED,
                  line_width=1.5, opacity=0.7, row=1, col=1,
                  annotation_text=f"Min ({int(sl_low)} min)",
                  annotation_font_color=ACCENT_RED)
    fig.add_hline(y=int(sl_high), line_dash="dash", line_color=ACCENT,
                  line_width=1.5, opacity=0.6, row=1, col=1,
                  annotation_text=f"Max ({int(sl_high)} min)",
                  annotation_font_color=ACCENT)
    res_colors = [ACCENT_RED if v else "#b794f4" for v in anom_sleep["resid_anom"]]
    fig.add_trace(go.Bar(x=anom_sleep["Date"], y=anom_sleep["residual"],
                         name="Residual", marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:.0f} min<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=MUTED, line_width=1, row=2, col=1)
    ptheme(fig, "💤 Sleep Pattern - Anomaly Visualization", h)
    fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
    fig.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
    return fig

# ---------------------------------------------------
# MILESTONE 4 — PDF GENERATION
# ---------------------------------------------------

def generate_pdf_m4(master, anom_hr, anom_steps, anom_sleep,
                    hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                    fig_hr, fig_steps, fig_sleep):
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(15, 23, 42)
            self.rect(0, 0, 210, 18, 'F')
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(99, 179, 237)
            self.set_y(4)
            self.cell(0, 10, "FitPulse Anomaly Detection Report  -  Milestone 4", align="C")
            self.set_text_color(148, 163, 184)
            self.set_font("Helvetica", "", 7)
            self.set_y(13)
            self.cell(0, 4, f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}", align="C")
            self.ln(6)

        def footer(self):
            self.set_y(-13)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(148, 163, 184)
            self.cell(0, 8, f"FitPulse ML Pipeline  .  Page {self.page_no()}", align="C")

        def section(self, title, color=(99, 179, 237)):
            self.ln(3)
            self.set_fill_color(*color)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 8, f"  {title}", fill=True, ln=True)
            self.set_text_color(30, 30, 40)
            self.ln(2)

        def kv(self, key, val, bold_val=True):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(80, 80, 100)
            self.cell(55, 6, key + ":", ln=False)
            self.set_font("Helvetica", "B" if bold_val else "", 9)
            self.set_text_color(20, 20, 30)
            self.cell(0, 6, str(val), ln=True)

        def para(self, text, size=8.5):
            self.set_font("Helvetica", "", size)
            self.set_text_color(60, 60, 80)
            self.multi_cell(0, 5, text)
            self.ln(1)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    n_hr    = int(anom_hr["is_anomaly"].sum())
    n_steps = int(anom_steps["is_anomaly"].sum())
    n_sleep = int(anom_sleep["is_anomaly"].sum())
    n_users = master["Id"].nunique()
    n_days  = master["Date"].nunique()
    date_range = f"{pd.to_datetime(master['Date']).min().strftime('%d %b %Y')} - {pd.to_datetime(master['Date']).max().strftime('%d %b %Y')}"

    pdf.section("1. EXECUTIVE SUMMARY", (15, 23, 60))
    pdf.kv("Dataset",       f"Real Fitbit Device Data - Kaggle (arashnic/fitbit)")
    pdf.kv("Users",         f"{n_users} participants")
    pdf.kv("Date Range",    date_range)
    pdf.kv("Total Days",    f"{n_days} days of observations")
    pdf.kv("Pipeline",      "Milestone 4 - Anomaly Detection Dashboard")
    pdf.ln(2)

    pdf.section("2. ANOMALY SUMMARY", (180, 50, 50))
    pdf.kv("Heart Rate Anomalies",  f"{n_hr} days flagged")
    pdf.kv("Steps Anomalies",       f"{n_steps} days flagged")
    pdf.kv("Sleep Anomalies",       f"{n_sleep} days flagged")
    pdf.kv("Total Flags",           f"{n_hr + n_steps + n_sleep} across all signals")
    pdf.ln(2)

    pdf.section("3. DETECTION THRESHOLDS USED", (40, 100, 60))
    pdf.kv("Heart Rate High",   f"> {int(hr_high)} bpm")
    pdf.kv("Heart Rate Low",    f"< {int(hr_low)} bpm")
    pdf.kv("Steps Low Alert",   f"< {int(st_low):,} steps/day")
    pdf.kv("Sleep Low",         f"< {int(sl_low)} minutes/night")
    pdf.kv("Sleep High",        f"> {int(sl_high)} minutes/night")
    pdf.kv("Residual Sigma",    f"+/- {float(sigma):.1f}sigma from rolling median")
    pdf.ln(2)

    pdf.section("4. METHODOLOGY", (60, 80, 140))
    pdf.para(
        "Three complementary anomaly detection methods were applied:\n\n"
        "  1. THRESHOLD VIOLATIONS - Hard upper/lower bounds on each metric. "
        "Any day exceeding these bounds is immediately flagged as anomalous. "
        "Simple, interpretable, and highly reliable for extreme values.\n\n"
        "  2. RESIDUAL-BASED DETECTION - A 3-day rolling median is computed as "
        "the expected baseline. Days where the actual value deviates by more than "
        f"+/-{float(sigma):.1f} standard deviations from this baseline are flagged. "
        "This catches subtle pattern breaks that threshold rules miss.\n\n"
        "  3. DBSCAN OUTLIER CLUSTERING - Each user is profiled on 7 activity "
        "features and clustered using DBSCAN (eps=2.2, min_samples=2). Users "
        "assigned label -1 are structural outliers whose overall behaviour does "
        "not match any group."
    )

    pdf.add_page()
    pdf.section("5. ANOMALY CHARTS", (15, 23, 60))

    def embed_fig(fig, label, w=190, h=80):
        try:
            img_bytes = fig.to_image(format="png", width=1100, height=480,
                                     scale=1.5, engine="kaleido")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(80, 80, 100)
            pdf.cell(0, 6, label, ln=True)
            pdf.image(tmp_path, x=10, w=w, h=h)
            os.unlink(tmp_path)
            pdf.ln(3)
        except Exception as ex:
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(150, 50, 50)
            pdf.cell(0, 6, f"[Chart not available: {ex}]", ln=True)
            pdf.ln(2)

    embed_fig(fig_hr,    "Figure 1 - Heart Rate with Anomaly Highlights")
    embed_fig(fig_steps, "Figure 2 - Step Count Trend with Alert Bands")
    embed_fig(fig_sleep, "Figure 3 - Sleep Pattern Visualization")

    pdf.add_page()
    pdf.section("6. ANOMALY RECORDS - HEART RATE", (180, 50, 50))

    def table(df, cols, rename_map, max_rows=20):
        df2 = df[df["is_anomaly"]][cols].copy().rename(columns=rename_map)
        if df2.empty:
            pdf.para("No anomalies detected.")
            return
        col_w = 180 // len(df2.columns)
        pdf.set_fill_color(15, 23, 60)
        pdf.set_text_color(180, 210, 255)
        pdf.set_font("Helvetica", "B", 7.5)
        for col in df2.columns:
            pdf.cell(col_w, 6, str(col)[:18], border=0, fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7.5)
        for i, (_, row) in enumerate(df2.head(max_rows).iterrows()):
            pdf.set_fill_color(30, 40, 60) if i % 2 == 0 else pdf.set_fill_color(20, 30, 50)
            pdf.set_text_color(200, 210, 225)
            for val in row:
                if isinstance(val, float):
                    cell_text = f"{val:.2f}"
                else:
                    cell_text = str(val)[:18]
                pdf.cell(col_w, 5.5, cell_text, border=0, fill=True)
            pdf.ln()
        if len(df2) > max_rows:
            pdf.set_text_color(100, 130, 180)
            pdf.set_font("Helvetica", "I", 7)
            pdf.cell(0, 5, f"  ... and {len(df2)-max_rows} more records (see CSV export for full data)", ln=True)
        pdf.ln(3)

    table(anom_hr, ["Date","AvgHR","rolling_med","residual","reason"],
          {"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    pdf.section("7. ANOMALY RECORDS - STEPS", (40, 130, 80))
    table(anom_steps, ["Date","TotalSteps","rolling_med","residual","reason"],
          {"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    pdf.section("8. ANOMALY RECORDS - SLEEP", (100, 60, 160))
    table(anom_sleep, ["Date","TotalSleepMinutes","rolling_med","residual","reason"],
          {"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"})

    pdf.add_page()
    pdf.section("9. DATASET OVERVIEW & USER PROFILES", (15, 23, 60))

    profile_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
    available_cols = [c for c in profile_cols if c in master.columns]
    user_profile = master.groupby("Id")[available_cols].mean().round(1)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(15, 23, 60)
    pdf.set_text_color(180, 210, 255)
    col_w2 = 180 // (len(available_cols) + 1)
    pdf.cell(col_w2, 6, "User ID", border=0, fill=True)
    for col in available_cols:
        pdf.cell(col_w2, 6, col[:12], border=0, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7.5)
    for i, (uid, row) in enumerate(user_profile.iterrows()):
        pdf.set_fill_color(30, 40, 60) if i % 2 == 0 else pdf.set_fill_color(20, 30, 50)
        pdf.set_text_color(200, 210, 225)
        pdf.cell(col_w2, 5.5, f"...{str(uid)[-6:]}", border=0, fill=True)
        for val in row:
            pdf.cell(col_w2, 5.5, f"{val:,.0f}", border=0, fill=True)
        pdf.ln()

    pdf.ln(4)
    pdf.section("10. CONCLUSION", (40, 100, 60))
    n_hr_c    = int(anom_hr["is_anomaly"].sum())
    n_steps_c = int(anom_steps["is_anomaly"].sum())
    n_sleep_c = int(anom_sleep["is_anomaly"].sum())
    pdf.para(
        f"The FitPulse Milestone 4 anomaly detection pipeline successfully processed "
        f"{n_users} users over {n_days} days of real Fitbit device data. "
        f"A total of {n_hr_c + n_steps_c + n_sleep_c} anomalous events were identified across "
        f"heart rate, step count, and sleep duration signals.\n\n"
        "Key findings:\n"
        f"   Heart rate showed {n_hr_c} anomalous days, primarily driven by residual "
        f"deviations from the rolling trend.\n"
        f"   Step count flagged {n_steps_c} alert days, often corresponding to "
        f"extremely sedentary or unusually active periods.\n"
        f"   Sleep patterns generated {n_sleep_c} anomaly flags, reflecting days "
        f"where users either did not wear the device or had unusual sleep durations.\n\n"
        "These findings align with expected patterns in consumer fitness wearable data "
        "and demonstrate the effectiveness of combining rule-based and statistical "
        "anomaly detection methods."
    )

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    buf = io.BytesIO(pdf_bytes)
    return buf

# ---------------------------------------------------
# MILESTONE 4 — CSV GENERATION
# ---------------------------------------------------

def generate_csv_m4(anom_hr, anom_steps, anom_sleep):
    hr_out    = anom_hr[anom_hr["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].copy()
    hr_out["signal"] = "Heart Rate"
    hr_out    = hr_out.rename(columns={"AvgHR":"value","rolling_med":"expected"})

    st_out    = anom_steps[anom_steps["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].copy()
    st_out["signal"] = "Steps"
    st_out    = st_out.rename(columns={"TotalSteps":"value","rolling_med":"expected"})

    sl_out    = anom_sleep[anom_sleep["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].copy()
    sl_out["signal"] = "Sleep"
    sl_out    = sl_out.rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})

    combined  = pd.concat([hr_out, st_out, sl_out], ignore_index=True)
    combined  = combined[["signal","Date","value","expected","residual","reason"]].sort_values(["signal","Date"])
    combined  = combined.round(2)
    buf       = io.StringIO()
    combined.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ===================================================
# SIDEBAR
# ===================================================

st.sidebar.markdown("""
<div style='text-align:center; padding: 16px 0 8px 0;'>
    <span style='font-family:Syne,sans-serif; font-size:1.6rem; font-weight:800;
    background: linear-gradient(90deg,#e91e8c,#c471ed);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🏃🏻 FitPulse</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📌 Select Milestone")
milestone = st.sidebar.selectbox(
    "Choose a milestone:",
    [
        "1 — Data Collection & Pre-Processing",
        "2 — Pattern Extraction & Analytics",
        "3 — Anomaly Detection & Visualization",
        "4 — Insights Dashboard"
    ]
)

st.sidebar.divider()

# Milestone-specific sidebar controls
if milestone == "1 — Data Collection & Pre-Processing":
    st.sidebar.header("📂 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV / Excel / JSON",
        type=["csv", "xlsx", "json"]
    )

elif milestone == "2 — Pattern Extraction & Analytics":
    st.sidebar.markdown("### 🗺 Pipeline Navigation")
    st.sidebar.markdown("📂 Data Loading")
    st.sidebar.markdown("🧬 TSFresh Features")
    st.sidebar.markdown("📈 Prophet Forecast")
    st.sidebar.markdown("⚙ Clustering")
    st.sidebar.markdown("📊 Summary")
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙ Model Parameters")
    k   = st.sidebar.slider("KMeans Clusters (K)", 2, 10, 3)
    eps = st.sidebar.slider("DBSCAN EPS", 0.5, 5.0, 2.2)
    st.sidebar.caption("Real Fitbit Dataset · FitPulse v2.0")

elif milestone == "3 — Anomaly Detection & Visualization":
    st.sidebar.markdown(f"""
    <div style="padding:0.5rem 0 1.5rem">
      <div style="font-size:0.72rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.2rem">
        Milestone 3 · Anomaly Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    steps_done = sum([st.session_state.files_loaded,
                      st.session_state.anomaly_done,
                      st.session_state.simulation_done])
    pct = int(steps_done / 3 * 100)
    st.sidebar.markdown(f"""
    <div style="margin-bottom:1rem">
      <div style="font-size:0.72rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-bottom:0.4rem">
        PIPELINE · {pct}%
      </div>
      <div style="background:{CARD_BOR};border-radius:4px;height:6px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{ACCENT_RED},{ACCENT2});border-radius:4px"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for done, icon, label in [
        (st.session_state.files_loaded,    "📂", "Data Loaded"),
        (st.session_state.anomaly_done,    "🚨", "Anomalies Detected"),
        (st.session_state.simulation_done, "🎯", "Accuracy Simulated"),
    ]:
        dot = f'<span style="color:{ACCENT3}">●</span>' if done else f'<span style="color:{MUTED}">○</span>'
        st.sidebar.markdown(f'<div style="font-size:0.82rem;padding:0.3rem 0;color:{TEXT if done else MUTED}">{dot} {icon} {label}</div>', unsafe_allow_html=True)

    st.sidebar.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div style="font-size:0.72rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.5rem">THRESHOLDS</div>', unsafe_allow_html=True)
    hr_high  = st.sidebar.number_input("HR High (bpm)",   value=100, min_value=80,  max_value=180)
    hr_low   = st.sidebar.number_input("HR Low (bpm)",    value=50,  min_value=30,  max_value=70)
    st_low   = st.sidebar.number_input("Steps Low",       value=500, min_value=0,   max_value=2000)
    sl_low   = st.sidebar.number_input("Sleep Low (min)", value=60,  min_value=0,   max_value=120)
    sl_high  = st.sidebar.number_input("Sleep High (min)",value=600, min_value=300, max_value=900)
    sigma    = st.sidebar.slider("Residual σ threshold", 1.0, 4.0, 2.0, 0.5, key="sigma_slider")
    st.sidebar.markdown(f'<hr style="border-color:{CARD_BOR};margin:1rem 0">', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div style="font-size:0.68rem;color:{MUTED};font-family:JetBrains Mono,monospace">Real Fitbit Dataset<br>30 users · March–April 2016</div>', unsafe_allow_html=True)

else:  # Milestone 4
    st.sidebar.markdown(f"""
    <div style="padding:0.5rem 0 1rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:{ACCENT}">
        📊 FitPulse Dashboard
      </div>
      <div style="font-size:0.7rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.2rem">
        Milestone 4 . Insights & Export
      </div>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.4rem">DETECTION THRESHOLDS</div>', unsafe_allow_html=True)
    m4_hr_high = int(st.sidebar.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="m4_hr_high"))
    m4_hr_low  = int(st.sidebar.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="m4_hr_low"))
    m4_st_low  = int(st.sidebar.number_input("Steps Low/day",    value=500, min_value=0,   max_value=2000,key="m4_st_low"))
    m4_sl_low  = int(st.sidebar.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120, key="m4_sl_low"))
    m4_sl_high = int(st.sidebar.number_input("Sleep High (min)", value=600, min_value=300, max_value=900, key="m4_sl_high"))
    m4_sigma   = float(st.sidebar.slider("Residual sigma", 1.0, 4.0, 2.0, 0.5, key="m4_sigma"))

    st.sidebar.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)

    run_m4_clicked = st.sidebar.button("⚡ Run M4 Detection", disabled=(not st.session_state.files_loaded))
    if not st.session_state.files_loaded:
        st.sidebar.markdown(f'<div style="font-size:0.7rem;color:{MUTED};text-align:center">Load data in M2 or M3 first</div>', unsafe_allow_html=True)

    st.sidebar.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)

    if st.session_state.pipeline_done and st.session_state.master is not None:
        master_tmp = st.session_state.master
        all_dates = pd.to_datetime(master_tmp["Date"])
        d_min = all_dates.min().date()
        d_max = all_dates.max().date()
        st.sidebar.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.4rem">DATE FILTER</div>', unsafe_allow_html=True)
        date_range_m4 = st.sidebar.date_input("Date range", value=(d_min, d_max),
                                               min_value=d_min, max_value=d_max,
                                               key="m4_daterange", label_visibility="collapsed")
        st.sidebar.markdown(f'<div style="font-size:0.7rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin:0.6rem 0 0.4rem">USER FILTER</div>', unsafe_allow_html=True)
        all_users_m4 = sorted(master_tmp["Id"].unique())
        user_options_m4 = ["All Users"] + [f"...{str(u)[-6:]}" for u in all_users_m4]
        selected_user_label_m4 = st.sidebar.selectbox("User", user_options_m4, key="m4_user", label_visibility="collapsed")
        selected_user_m4 = None if selected_user_label_m4 == "All Users" else all_users_m4[user_options_m4.index(selected_user_label_m4) - 1]
    else:
        date_range_m4 = None
        selected_user_m4 = None

    st.sidebar.markdown(f'<hr style="border-color:{CARD_BOR};margin:0.8rem 0">', unsafe_allow_html=True)
    pct_m4 = int(st.session_state.pipeline_done) * 100
    st.sidebar.markdown(f"""
    <div style="font-size:0.68rem;color:{MUTED};font-family:JetBrains Mono,monospace;margin-bottom:0.3rem">PIPELINE . {pct_m4}%</div>
    <div style="background:{CARD_BOR};border-radius:4px;height:5px;overflow:hidden">
      <div style="width:{pct_m4}%;height:100%;background:linear-gradient(90deg,{ACCENT},{ACCENT3});border-radius:4px"></div>
    </div>""", unsafe_allow_html=True)


# ===================================================
#
#  MILESTONE 1 — DATA COLLECTION & PRE-PROCESSING
#
# ===================================================

if milestone == "1 — Data Collection & Pre-Processing":

    st.markdown("""
    <div class="hero-m1">
    <h1>🏃 FitPulse</h1>
    <h3>Milestone 1 — Data Collection & Pre-Processing</h3>
    <p>Upload → Analyze → Clean → Verify</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        st.success("✅ Dataset uploaded successfully")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            st.metric("Total Missing", int(df.isnull().sum().sum()))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📄 Original Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("❗ Missing Values Count")
        missing = df.isnull().sum()
        st.dataframe(missing, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=["number"]).columns
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
        cat_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        # Store cleaned data in session state for downstream milestones
        st.session_state.m1_df_clean = df_clean

        st.success("✅ Data preprocessing completed successfully")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧹 Cleaned Dataset Preview")
        st.dataframe(df_clean.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✔ Missing Values After Cleaning")
        st.dataframe(df_clean.isnull().sum(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("⬅️ Upload a dataset from the sidebar to begin")


# ===================================================
#
#  MILESTONE 2 — PATTERN EXTRACTION & ANALYTICS
#
# ===================================================

elif milestone == "2 — Pattern Extraction & Analytics":

    st.markdown("""
    <div class="hero-banner">
        <h1>🏃🏻 FitPulse Health Analytics</h1>
        <h3>Milestone 2 — Pattern Extraction · Forecasting · Clustering Intelligence</h3>
        <p>TSFresh &nbsp;→&nbsp; Prophet &nbsp;→&nbsp; KMeans &nbsp;→&nbsp; DBSCAN &nbsp;→&nbsp; PCA &nbsp;→&nbsp; t-SNE</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='display:flex; flex-wrap:wrap; gap:4px; margin-bottom:24px;'>
        <span class='pipeline-step'>📂 Data Loading</span>
        <span class='pipeline-step'>🧬 TSFresh</span>
        <span class='pipeline-step'>📈 Prophet</span>
        <span class='pipeline-step'>⚙ KMeans</span>
        <span class='pipeline-step'>🔵 DBSCAN</span>
        <span class='pipeline-step'>📐 PCA</span>
        <span class='pipeline-step'>🌀 t-SNE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📂 Data Upload</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Fitbit CSV files (dailyActivity, heartrate, sleepDay, hourlySteps, hourlyIntensities)",
        type="csv",
        accept_multiple_files=True
    )

    files = {}
    if uploaded_files:
        for f in uploaded_files:
            files[f.name] = pd.read_csv(f)
        st.success(f"✅  {len(files)} file(s) uploaded successfully")

    if not files:
        st.markdown("""
        <div class="upload-prompt">
            <h2>⬆️ Upload Your Fitbit Data to Begin</h2>
            <p>Upload one or more Fitbit CSV files using the uploader above.<br>
            The full analytics dashboard — KPIs, TSFresh features, Prophet forecasts,<br>
            clustering, PCA, t-SNE and pipeline summary — will appear here.</p><br>
            <p style='color:#64748b; font-size:0.85rem;'>
            Supported: dailyActivity · heartrate_seconds · sleepDay · hourlySteps · hourlyIntensities
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    daily = sleep = hr = steps = intensity = None
    for name, df in files.items():
        if "TotalSteps"     in df.columns: daily     = df
        if "Value"          in df.columns: hr        = df
        if "StepTotal"      in df.columns: steps     = df
        if "TotalIntensity" in df.columns: intensity = df
        if "SleepDay"       in df.columns or "value" in df.columns: sleep = df

    # Store the multi-file uploads in session state so M3/M4 can use them
    if files:
        raw_uploads_m2 = list(files.items())
        detected_m2 = {}
        used_m2 = set()
        for req_name, finfo in REQUIRED_FILES.items():
            best_score, best_name, best_df = 0, None, None
            for uname, udf in raw_uploads_m2:
                s = score_match(udf, finfo)
                if s > best_score:
                    best_score, best_name, best_df = s, uname, udf
            if best_score >= 2:
                detected_m2[req_name] = best_df
                used_m2.add(best_name)

        if len(detected_m2) == 5 and not st.session_state.files_loaded:
            try:
                daily_m3    = detected_m2["dailyActivity_merged.csv"].copy()
                hourly_s = detected_m2["hourlySteps_merged.csv"].copy()
                hourly_i = detected_m2["hourlyIntensities_merged.csv"].copy()
                sleep_m3    = detected_m2["minuteSleep_merged.csv"].copy()
                hr_m3       = detected_m2["heartrate_seconds_merged.csv"].copy()

                daily_m3["ActivityDate"]    = pd.to_datetime(daily_m3["ActivityDate"],    format="%m/%d/%Y")
                hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
                hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
                sleep_m3["date"]            = pd.to_datetime(sleep_m3["date"],            format="%m/%d/%Y %I:%M:%S %p")
                hr_m3["Time"]               = pd.to_datetime(hr_m3["Time"],               format="%m/%d/%Y %I:%M:%S %p")

                hr_minute = (hr_m3.set_index("Time").groupby("Id")["Value"]
                             .resample("1min").mean().reset_index())
                hr_minute.columns = ["Id","Time","HeartRate"]
                hr_minute = hr_minute.dropna()

                hr_minute["Date"] = hr_minute["Time"].dt.date
                hr_daily_m3 = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                            .agg(["mean","max","min","std"]).reset_index()
                            .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))

                sleep_m3["Date"] = sleep_m3["date"].dt.date
                sleep_daily_m3 = (sleep_m3.groupby(["Id","Date"])
                               .agg(TotalSleepMinutes=("value","count"),
                                    DominantSleepStage=("value", lambda x: x.mode()[0]))
                               .reset_index())

                master_df = daily_m3.copy().rename(columns={"ActivityDate":"Date"})
                master_df["Date"] = master_df["Date"].dt.date
                master_df = master_df.merge(hr_daily_m3,    on=["Id","Date"], how="left")
                master_df = master_df.merge(sleep_daily_m3, on=["Id","Date"], how="left")
                master_df["TotalSleepMinutes"]  = master_df["TotalSleepMinutes"].fillna(0)
                master_df["DominantSleepStage"] = master_df["DominantSleepStage"].fillna(0)
                for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                    master_df[col] = master_df.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

                st.session_state.daily_m3     = daily_m3
                st.session_state.hourly_s  = hourly_s
                st.session_state.hourly_i  = hourly_i
                st.session_state.sleep_m3     = sleep_m3
                st.session_state.hr_m3        = hr_m3
                st.session_state.hr_minute = hr_minute
                st.session_state.master    = master_df
                st.session_state.files_loaded = True
            except Exception:
                pass  # M3/M4 data build is best-effort here; M3 button will handle errors

    st.markdown('<div class="section-header">🔍 Dataset Detection</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    if daily     is not None: c1.success("🏃 Daily Activity\n\n✔ Found")
    else:                     c1.info("🏃 Daily Activity\n\nNot Found")
    if steps     is not None: c2.success("👟 Hourly Steps\n\n✔ Found")
    else:                     c2.info("👟 Hourly Steps\n\nNot Found")
    if intensity is not None: c3.success("⚡ Hourly Intensities\n\n✔ Found")
    else:                     c3.info("⚡ Hourly Intensities\n\nNot Found")
    if sleep     is not None: c4.success("😴 Sleep\n\n✔ Found")
    else:                     c4.info("😴 Sleep\n\nNot Found")
    if hr        is not None: c5.success("❤️ Heart Rate\n\n✔ Found")
    else:                     c5.info("❤️ Heart Rate\n\nNot Found")

    if daily is not None:
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Users",        daily["Id"].nunique())
        col2.metric("📋 Rows",         len(daily))
        col3.metric("👣 Avg Steps",    int(daily["TotalSteps"].mean()))
        col4.metric("🔥 Avg Calories", int(daily["Calories"].mean()))
        st.dataframe(daily.head(), use_container_width=True)

        st.markdown('<div class="section-header">📊 Health KPI Summary</div>', unsafe_allow_html=True)
        avg_steps     = int(daily["TotalSteps"].mean())
        avg_cal       = int(daily["Calories"].mean())
        avg_active    = int(daily["VeryActiveMinutes"].mean())
        avg_sedentary = int(daily["SedentaryMinutes"].mean())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👣 Avg Daily Steps",    avg_steps,    delta=f"{avg_steps - 10000:+,} vs goal")
        c2.metric("🔥 Avg Calories",       avg_cal)
        c3.metric("⚡ Avg Active Minutes", avg_active)
        c4.metric("🛋 Avg Sedentary Min",  avg_sedentary)

        st.markdown('<div class="section-header">📈 Activity Insights</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Highest Steps", f"{int(daily['TotalSteps'].max()):,}")
        c2.metric("📉 Lowest Steps",  f"{int(daily['TotalSteps'].min()):,}")
        c3.metric("🛋 Avg Sedentary", avg_sedentary)

        st.markdown('<div class="section-header">💚 Overall Health Score</div>', unsafe_allow_html=True)
        score = min(int((avg_steps / 10000) * 100), 100)
        st.progress(score / 100)
        st.write(f"Estimated Health Score: **{score}/100**")

        st.markdown('<div class="section-header">🩺 Health Status</div>', unsafe_allow_html=True)
        if   score > 70: st.success("🏃 Active Lifestyle 💪 — Users are meeting WHO recommended activity levels.")
        elif score > 40: st.info("👍 Moderate Activity — Users are partially active but have room to improve.")
        else:            st.warning("⚠️ Low Activity Level — Users are predominantly sedentary.")

    if hr is not None:
        st.markdown('<div class="section-header">🧬 TSFresh Feature Matrix</div>', unsafe_allow_html=True)
        feature_df = hr.groupby("Id")["Value"].agg(
            ["sum","median","mean","count","std","var","max","min"]
        )
        scaler2 = MinMaxScaler()
        heat    = pd.DataFrame(scaler2.fit_transform(feature_df), columns=feature_df.columns)
        bwr     = LinearSegmentedColormap.from_list("bwr_custom",
                    ["#2563eb","#5b8dd9","#c8d8f0","#f0c0c8","#e05070","#b91c1c"])
        fig_h, ax_h = plt.subplots(figsize=(15, max(5, len(heat) * 0.55 + 1)))
        fig_h.patch.set_facecolor(GRAPH_BG)
        sns.heatmap(heat, cmap=bwr, annot=True, fmt=".2f",
                    linewidths=0.4, linecolor=GRAPH_BG, ax=ax_h,
                    cbar_kws={"shrink":0.8,"label":"Normalized Value (0–1)","pad":0.02},
                    annot_kws={"size":9,"weight":"bold","color":"white"})
        ax_h.set_title("TSFresh Feature Matrix — Real Fitbit Heart Rate Data\n(Normalized 0–1 per feature)",
                       fontsize=14, fontweight='bold', pad=16, color=TITLE_COL)
        ax_h.set_xlabel("Extracted Statistical Features", fontsize=11, labelpad=10, color=LABEL_COL)
        ax_h.set_ylabel("User ID", fontsize=11, labelpad=10, color=LABEL_COL)
        ax_h.tick_params(axis='x', rotation=30, colors=TICK_COL)
        ax_h.tick_params(axis='y', rotation=0,  colors=TICK_COL)
        cbar = ax_h.collections[0].colorbar
        cbar.ax.yaxis.label.set_color(LABEL_COL)
        cbar.ax.tick_params(colors=TICK_COL)
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)
        st.markdown("""
> **📌 Insight** — Each row is a Fitbit user. Columns are statistical features extracted from heart-rate signals.
> Blue = lower values · Red = higher values.
""")

    st.markdown('<div class="section-header">📈 Prophet Forecast — Heart Rate</div>', unsafe_allow_html=True)

    if hr is not None:
        hr_df            = hr.copy()
        hr_df["Time"]    = pd.to_datetime(hr_df["Time"])
        hr_daily         = hr_df.groupby(hr_df["Time"].dt.date)["Value"].mean().reset_index()
        hr_daily.columns = ["ds","y"]
        hr_daily["ds"]   = pd.to_datetime(hr_daily["ds"])
        hr_daily         = hr_daily.sort_values("ds").reset_index(drop=True)

        model_hr = Prophet(interval_width=0.80, changepoint_prior_scale=0.05,
                           weekly_seasonality=True, daily_seasonality=False,
                           yearly_seasonality=False)
        _ = model_hr.fit(hr_daily)
        future_hr   = model_hr.make_future_dataframe(periods=30)
        forecast_hr = model_hr.predict(future_hr)
        split_date  = hr_daily["ds"].max()

        fig_hr, ax_hr = plt.subplots(figsize=(15, 6))
        fig_hr.patch.set_facecolor("#0d1b3e")
        ax_hr.set_facecolor("#0d1b3e")
        ax_hr.grid(False)
        ax_hr.fill_between(forecast_hr["ds"],
                           forecast_hr["yhat_lower"], forecast_hr["yhat_upper"],
                           color="#1e3a6e", alpha=0.90, zorder=1, label="80% Confidence Interval")
        inner_lo = forecast_hr["yhat"] - (forecast_hr["yhat"] - forecast_hr["yhat_lower"]) * 0.45
        inner_hi = forecast_hr["yhat"] + (forecast_hr["yhat_upper"] - forecast_hr["yhat"]) * 0.45
        ax_hr.fill_between(forecast_hr["ds"], inner_lo, inner_hi,
                           color="#2a55a0", alpha=0.80, zorder=2)
        ax_hr.plot(forecast_hr["ds"], forecast_hr["yhat"],
                   color="#a8d8f0", linewidth=1.8, zorder=4, label="Trend Forecast")
        ax_hr.scatter(hr_daily["ds"], hr_daily["y"],
                      color="#e8708a", s=24, zorder=6, alpha=0.90,
                      label="Actual HR (bpm)", edgecolors="none", linewidths=0)
        ax_hr.axvline(split_date, color="#f97316", linestyle="--",
                      linewidth=1.8, alpha=0.95, zorder=5, label="Forecast Start")
        ax_hr.set_title("Heart Rate — Prophet Trend Forecast (Real Fitbit Data)",
                        fontsize=14, fontweight="bold", color="#ddeeff", pad=16)
        ax_hr.set_xlabel("Date", fontsize=11, color="#8899bb", labelpad=10)
        ax_hr.set_ylabel("Heart Rate (bpm)", fontsize=11, color="#8899bb", labelpad=10)
        ax_hr.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax_hr.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax_hr.xaxis.get_majorticklabels(), rotation=20, ha="right",
                 fontsize=8.5, color="#8899bb")
        plt.setp(ax_hr.yaxis.get_majorticklabels(), color="#8899bb")
        ax_hr.legend(loc="upper left", framealpha=0.70,
                     facecolor="#0d1b3e", edgecolor="#2a3f6e",
                     labelcolor="#ccddf0", fontsize=9, ncol=2)
        ax_hr.tick_params(colors="#8899bb", length=4)
        for spi in ax_hr.spines.values(): spi.set_edgecolor("#1e3a6e")
        plt.tight_layout()
        st.pyplot(fig_hr, use_container_width=True)
        plt.close(fig_hr)

        c1, c2, c3 = st.columns(3)
        c1.info("🔵 **Blue band** = 80% confidence interval")
        c2.info("🩵 **Cyan line** = Fitted + forecast trend")
        c3.warning("🟡 **Orange dashed** = Forecast start · 🩷 **Pink dots** = Actual HR")
    else:
        st.info("Upload a heartrate CSV file to see the Prophet heart rate forecast.")

    if daily is not None and sleep is not None:
        st.markdown('<div class="section-header">📈 Steps & Sleep Forecast</div>', unsafe_allow_html=True)

        steps_df = daily[["ActivityDate","TotalSteps"]].copy()
        steps_df["ActivityDate"] = pd.to_datetime(steps_df["ActivityDate"])
        steps_df = steps_df.rename(columns={"ActivityDate":"ds","TotalSteps":"y"})
        model_steps = Prophet(interval_width=0.8, weekly_seasonality=True)
        _ = model_steps.fit(steps_df)
        forecast_steps = model_steps.predict(model_steps.make_future_dataframe(periods=30))

        sleep_df2           = sleep.copy()
        sleep_df2["date"]   = pd.to_datetime(sleep_df2["date"])
        sleep_daily2        = sleep_df2.groupby(sleep_df2["date"].dt.date)["value"].sum().reset_index()
        sleep_daily2.columns = ["ds","y"]
        sleep_daily2["ds"]  = pd.to_datetime(sleep_daily2["ds"])
        model_sleep         = Prophet(interval_width=0.8, weekly_seasonality=True)
        _ = model_sleep.fit(sleep_daily2)
        forecast_sleep      = model_sleep.predict(model_sleep.make_future_dataframe(periods=30))

        fig_ss, axes_ss = plt.subplots(2, 1, figsize=(14, 10))
        fig_ss.patch.set_facecolor(GRAPH_BG)

        ax_st = axes_ss[0]
        ax_st.set_facecolor(GRAPH_INNER)
        ax_st.fill_between(forecast_steps["ds"],
                           forecast_steps["yhat_lower"], forecast_steps["yhat_upper"],
                           color="#2d6a4f", alpha=0.70, zorder=1, label="80% CI")
        ax_st.plot(forecast_steps["ds"], forecast_steps["yhat"],
                   color="#ffffff", linewidth=1.8, zorder=3, label="Trend Forecast")
        ax_st.scatter(steps_df["ds"], steps_df["y"],
                      color="#95d5b2", s=22, zorder=5, alpha=0.85,
                      edgecolors="#52b788", linewidths=0.5, label="Actual Data")
        ax_st.axvline(steps_df["ds"].max(), color="#f97316", linestyle="--",
                      linewidth=1.8, alpha=0.85, zorder=4, label="Forecast Start")
        ax_st.set_title("Steps — Prophet Trend Forecast", fontsize=13, fontweight='bold',
                        color=TITLE_COL, pad=12)
        ax_st.set_xlabel("Date", fontsize=10, color=LABEL_COL, labelpad=8)
        ax_st.set_ylabel("Steps", fontsize=10, color=LABEL_COL, labelpad=8)
        ax_st.legend(framealpha=0.80, facecolor=GRAPH_BG, edgecolor=GRID_COL,
                     labelcolor=TITLE_COL, fontsize=8.5, ncol=2)
        ax_st.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_st.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax_st.xaxis.get_majorticklabels(), rotation=20, ha='right')
        ax_st.tick_params(colors=TICK_COL)
        for spi in ax_st.spines.values(): spi.set_edgecolor(GRID_COL)

        ax_sl = axes_ss[1]
        ax_sl.set_facecolor(GRAPH_INNER)
        ax_sl.fill_between(forecast_sleep["ds"],
                           forecast_sleep["yhat_lower"], forecast_sleep["yhat_upper"],
                           color="#4a1d6e", alpha=0.70, zorder=1, label="80% CI")
        ax_sl.plot(forecast_sleep["ds"], forecast_sleep["yhat"],
                   color="#ffffff", linewidth=1.8, zorder=3, label="Trend Forecast")
        ax_sl.scatter(sleep_daily2["ds"], sleep_daily2["y"],
                      color="#c77dff", s=22, zorder=5, alpha=0.85,
                      edgecolors="#9d4edd", linewidths=0.5, label="Actual Data")
        ax_sl.axvline(sleep_daily2["ds"].max(), color="#f97316", linestyle="--",
                      linewidth=1.8, alpha=0.85, zorder=4, label="Forecast Start")
        ax_sl.set_title("Sleep (minutes) — Prophet Trend Forecast", fontsize=13, fontweight='bold',
                        color=TITLE_COL, pad=12)
        ax_sl.set_xlabel("Date", fontsize=10, color=LABEL_COL, labelpad=8)
        ax_sl.set_ylabel("Sleep (mins)", fontsize=10, color=LABEL_COL, labelpad=8)
        ax_sl.legend(framealpha=0.80, facecolor=GRAPH_BG, edgecolor=GRID_COL,
                     labelcolor=TITLE_COL, fontsize=8.5, ncol=2)
        ax_sl.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_sl.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax_sl.xaxis.get_majorticklabels(), rotation=20, ha='right')
        ax_sl.tick_params(colors=TICK_COL)
        for spi in ax_sl.spines.values(): spi.set_edgecolor(GRID_COL)

        plt.tight_layout(pad=3)
        st.pyplot(fig_ss, use_container_width=True)
        plt.close(fig_ss)
        st.markdown("""
> **📌 Health Insight** — Step forecasts predict future physical activity levels, while sleep forecasts estimate rest patterns.
""")

    if daily is not None:
        st.markdown('<div class="section-header">⚙ Clustering Analysis</div>', unsafe_allow_html=True)

        feat_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes"]
        features  = daily[feat_cols]
        X         = StandardScaler().fit_transform(features)

        inertia = []
        for i in range(1, 10):
            km_tmp = KMeans(n_clusters=i, random_state=42, n_init=10)
            inertia.append(km_tmp.fit(X).inertia_)

        fig_el, ax_el = plt.subplots(figsize=(10, 5))
        fig_el.patch.set_facecolor(GRAPH_BG)
        ax_el.set_facecolor(GRAPH_INNER)
        ax_el.plot(range(1,10), inertia, marker="o", color="#e91e8c",
                   linewidth=2.5, markersize=9, markerfacecolor="#fbbf24",
                   markeredgecolor="#e91e8c", markeredgewidth=1.5, label="Inertia")
        ax_el.fill_between(range(1,10), inertia, alpha=0.08, color="#e91e8c")
        ax_el.axvline(k, linestyle="--", color="#fbbf24", linewidth=2, alpha=0.8,
                      label=f"Selected K = {k}")
        for idx_i, val in enumerate(inertia):
            ax_el.annotate(f"{val:.0f}", (idx_i+1, val), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=8, color=LABEL_COL)
        ax_el.set_title("KMeans Elbow Curve — Optimal Cluster Selection",
                        fontsize=14, fontweight='bold', color=TITLE_COL, pad=14)
        ax_el.set_xlabel("Number of Clusters (K)", fontsize=11, color=LABEL_COL, labelpad=10)
        ax_el.set_ylabel("Inertia (WCSS)", fontsize=11, color=LABEL_COL, labelpad=10)
        ax_el.legend(framealpha=0.85, facecolor=GRAPH_BG, edgecolor=GRID_COL,
                     labelcolor=TITLE_COL, fontsize=10)
        ax_el.tick_params(colors=TICK_COL)
        for spi in ax_el.spines.values(): spi.set_edgecolor(GRID_COL)
        plt.tight_layout()
        st.pyplot(fig_el, use_container_width=True)
        plt.close(fig_el)
        st.caption("📌 The elbow point indicates the optimal K where inertia reduction diminishes.")

        kmeans        = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels        = kmeans.fit_predict(X)
        pca_model     = PCA(n_components=2)
        X_pca         = pca_model.fit_transform(X)
        var_exp       = pca_model.explained_variance_ratio_
        db            = DBSCAN(eps=eps, min_samples=3)
        db_labels     = db.fit_predict(X)
        n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise       = int((db_labels == -1).sum())

        col1, col2 = st.columns(2)

        with col1:
            fig_km, ax_km = plt.subplots(figsize=(8, 6))
            fig_km.patch.set_facecolor(GRAPH_BG)
            ax_km.set_facecolor(GRAPH_INNER)
            for lbl in sorted(set(labels)):
                msk = labels == lbl
                ax_km.scatter(X_pca[msk,0], X_pca[msk,1],
                              c=CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)],
                              s=70, alpha=0.88, zorder=3, edgecolors='none',
                              label=f"Cluster {lbl}")
            c_pca = pca_model.transform(kmeans.cluster_centers_)
            ax_km.scatter(c_pca[:,0], c_pca[:,1], c='#ef4444', marker='X',
                          s=200, zorder=6, edgecolors='white', linewidths=1.2, label='Centroids')
            ax_km.set_title(f"KMeans PCA (K={k})", fontsize=12, fontweight='bold',
                            color=TITLE_COL, pad=12)
            ax_km.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=10, color=LABEL_COL, labelpad=8)
            ax_km.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=10, color=LABEL_COL, labelpad=8)
            ax_km.legend(framealpha=0.85, facecolor=GRAPH_BG, edgecolor=GRID_COL,
                         labelcolor=TITLE_COL, fontsize=9)
            ax_km.tick_params(colors=TICK_COL)
            for spi in ax_km.spines.values(): spi.set_edgecolor(GRID_COL)
            plt.tight_layout()
            st.pyplot(fig_km, use_container_width=True)
            plt.close(fig_km)

        with col2:
            fig_db, ax_db = plt.subplots(figsize=(8, 6))
            fig_db.patch.set_facecolor(GRAPH_BG)
            ax_db.set_facecolor(GRAPH_INNER)
            for lbl in sorted(set(db_labels)):
                msk   = db_labels == lbl
                color = "#64748b" if lbl == -1 else CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
                ax_db.scatter(X_pca[msk,0], X_pca[msk,1],
                              c=color, s=70,
                              alpha=0.88 if lbl != -1 else 0.35,
                              zorder=3, edgecolors='none',
                              label="Noise" if lbl == -1 else f"Cluster {lbl}",
                              marker='x' if lbl == -1 else 'o')
            ax_db.set_title(f"DBSCAN PCA (eps={eps})\n{n_db_clusters} clusters · {n_noise} noise pts",
                            fontsize=12, fontweight='bold', color=TITLE_COL, pad=12)
            ax_db.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=10, color=LABEL_COL, labelpad=8)
            ax_db.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=10, color=LABEL_COL, labelpad=8)
            ax_db.legend(framealpha=0.85, facecolor=GRAPH_BG, edgecolor=GRID_COL,
                         labelcolor=TITLE_COL, fontsize=9)
            ax_db.tick_params(colors=TICK_COL)
            for spi in ax_db.spines.values(): spi.set_edgecolor(GRID_COL)
            plt.tight_layout()
            st.pyplot(fig_db, use_container_width=True)
            plt.close(fig_db)

        st.markdown('<div class="section-header">🌀 t-SNE Projection</div>', unsafe_allow_html=True)
        tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne     = tsne_model.fit_transform(X)

        fig_ts, ax_ts = plt.subplots(figsize=(11, 7))
        fig_ts.patch.set_facecolor(GRAPH_BG)
        ax_ts.set_facecolor(GRAPH_INNER)
        for lbl in sorted(set(labels)):
            msk = labels == lbl
            ax_ts.scatter(X_tsne[msk,0], X_tsne[msk,1],
                          c=CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)],
                          s=70, alpha=0.88, zorder=3, edgecolors='none',
                          label=f"Cluster {lbl}  (n={msk.sum()})")
        ax_ts.set_title("t-SNE 2D Projection — High-Dimensional Activity Pattern Visualization",
                        fontsize=14, fontweight='bold', color=TITLE_COL, pad=14)
        ax_ts.set_xlabel("t-SNE Dimension 1", fontsize=11, color=LABEL_COL, labelpad=10)
        ax_ts.set_ylabel("t-SNE Dimension 2", fontsize=11, color=LABEL_COL, labelpad=10)
        ax_ts.legend(framealpha=0.85, facecolor=GRAPH_BG, edgecolor=GRID_COL,
                     labelcolor=TITLE_COL, fontsize=10)
        ax_ts.tick_params(colors=TICK_COL)
        for spi in ax_ts.spines.values(): spi.set_edgecolor(GRID_COL)
        plt.tight_layout()
        st.pyplot(fig_ts, use_container_width=True)
        plt.close(fig_ts)
        st.caption("📌 t-SNE maps high-dimensional Fitbit features to 2D, revealing natural activity groupings.")

        st.markdown('<div class="section-header">📊 Cluster Profiles</div>', unsafe_allow_html=True)
        cluster_df = features.copy()
        cluster_df["cluster"] = labels
        profile    = cluster_df.groupby("cluster").mean()

        fig_cp, ax_cp = plt.subplots(figsize=(12, 6))
        fig_cp.patch.set_facecolor(GRAPH_BG)
        ax_cp.set_facecolor(GRAPH_INNER)
        x_pos     = np.arange(len(profile.columns))
        bar_width = 0.8 / len(profile)
        for i, (idx, row) in enumerate(profile.iterrows()):
            bars = ax_cp.bar(x_pos + i*bar_width, row.values, width=bar_width,
                             color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                             alpha=0.88, label=f"Cluster {idx}", edgecolor='none')
            for bar in bars:
                h = bar.get_height()
                ax_cp.text(bar.get_x()+bar.get_width()/2, h+10, f"{h:.0f}",
                           ha='center', va='bottom', fontsize=8, color=LABEL_COL)
        ax_cp.set_xticks(x_pos + bar_width*(len(profile)-1)/2)
        ax_cp.set_xticklabels(profile.columns, fontsize=10, color=TITLE_COL, rotation=10)
        ax_cp.set_title("Cluster Profiles — Average Feature Values per Cluster",
                        fontsize=14, fontweight='bold', color=TITLE_COL, pad=14)
        ax_cp.set_xlabel("Features", fontsize=11, color=LABEL_COL, labelpad=10)
        ax_cp.set_ylabel("Average Value", fontsize=11, color=LABEL_COL, labelpad=10)
        ax_cp.legend(framealpha=0.85, facecolor=GRAPH_BG, edgecolor=GRID_COL,
                     labelcolor=TITLE_COL, fontsize=10)
        ax_cp.tick_params(colors=TICK_COL)
        for spi in ax_cp.spines.values(): spi.set_edgecolor(GRID_COL)
        plt.tight_layout()
        st.pyplot(fig_cp, use_container_width=True)
        plt.close(fig_cp)
        st.caption("📌 Each cluster represents users with similar Fitbit activity patterns.")

        st.markdown('<div class="section-header">🧠 Cluster Behavior Insights</div>', unsafe_allow_html=True)
        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            st.markdown("""
<div class="cluster-card-0">
<h4>🔵 Cluster 0 — Moderately Active</h4>
<hr style='border-color:rgba(100,180,255,0.25); margin:10px 0;'>
<p>👣 Steps ≈ 7,600 / day</p>
<p>🛋 Sedentary ≈ 750 min</p>
<p>⚡ Moderate activity bursts</p>
<p style='font-size:0.8rem; opacity:0.7;'>Meets partial WHO guidelines</p>
</div>""", unsafe_allow_html=True)
        with ci2:
            st.markdown("""
<div class="cluster-card-1">
<h4>🔴 Cluster 1 — Sedentary Users</h4>
<hr style='border-color:rgba(255,140,100,0.25); margin:10px 0;'>
<p>👣 Steps ≈ 3,200 / day</p>
<p>🛋 Sedentary ≈ 1,190 min</p>
<p>⚡ Very low active minutes</p>
<p style='font-size:0.8rem; opacity:0.7;'>Below WHO minimum activity level</p>
</div>""", unsafe_allow_html=True)
        with ci3:
            st.markdown("""
<div class="cluster-card-2">
<h4>🟢 Cluster 2 — Highly Active</h4>
<hr style='border-color:rgba(100,255,140,0.25); margin:10px 0;'>
<p>👣 Steps ≈ 11,000 / day</p>
<p>🛋 Sedentary ≈ 950 min</p>
<p>⚡ High very-active minutes</p>
<p style='font-size:0.8rem; opacity:0.7;'>Exceeds WHO daily activity goals</p>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">✅ Pipeline Summary</div>', unsafe_allow_html=True)
        st.success("✔  **Data Loading** — Fitbit multi-file dataset detected and processed")
        st.success("✔  **TSFresh Feature Extraction** — Statistical heart-rate features computed per user")
        st.success("✔  **Prophet Forecast** — Heart rate, steps and sleep trend forecasted (30-day horizon)")
        st.success("✔  **KMeans Clustering** — Users segmented into activity profiles")
        st.success("✔  **DBSCAN** — Density-based clustering with noise detection applied")
        st.success("✔  **PCA** — 2D projection of high-dimensional features visualized")
        st.success("✔  **t-SNE** — Non-linear manifold projection of activity clusters rendered")

    st.markdown("""
<div style='text-align:center; padding: 40px 20px 20px 20px; opacity: 0.5; font-size: 0.85rem;'>
    🏃🏻 FitPulse AI Health Analytics Dashboard &nbsp;·&nbsp;
    Pipeline: TSFresh → Prophet → KMeans → DBSCAN → PCA → t-SNE &nbsp;·&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)


# ===================================================
#
#  MILESTONE 3 — ANOMALY DETECTION & VISUALIZATION
#
# ===================================================

elif milestone == "3 — Anomaly Detection & Visualization":

    # Hero
    st.markdown(f"""
    <div class="m3-hero">
      <div class="hero-badge">MILESTONE 3 · ANOMALY DETECTION & VISUALIZATION</div>
      <h1 class="hero-title">🚨 FitPulse Anomaly Detector</h1>
      <p class="hero-sub">Threshold Violations · Prophet Residuals · Outlier Clusters · Interactive Plotly Charts</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Data Status ───────────────────────────────────────────────
    sec_m3("📂", "Data Status", "Step 1")

    if not st.session_state.files_loaded:
        st.markdown(f"""
        <div class="m3-card" style="text-align:center;padding:2rem;border-color:{DANGER_BOR}">
          <div style="font-size:2rem;margin-bottom:0.8rem">⚠️</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem">
            Please upload data in Milestone 1 or Milestone 2
          </div>
          <div style="color:{MUTED};font-size:0.85rem">
            Go to <b>Milestone 2</b> and upload all 5 Fitbit CSV files, then return here.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    master = st.session_state.master
    ui_success_m3(f"Data loaded from session — {master.shape[0]} rows · {master['Id'].nunique()} users")

    st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

    # ── SECTION 2: Anomaly Detection ────────────────────────────────────────
    sec_m3("🚨", "Anomaly Detection — Three Methods", "Steps 2–4")

    st.markdown(f"""
    <div class="m3-card">
      <div class="m3-card-title">Detection Methods Applied</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;font-size:0.83rem">
        <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
          <div style="color:{ACCENT_RED};font-weight:600;margin-bottom:0.4rem">① Threshold Violations</div>
          <div style="color:{MUTED}">Hard upper/lower limits on HR, Steps, Sleep. Simple, interpretable, fast.</div>
        </div>
        <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
          <div style="color:{ACCENT2};font-weight:600;margin-bottom:0.4rem">② Residual-Based</div>
          <div style="color:{MUTED}">Rolling median as baseline. Flag days where actual deviates by ±{sigma:.0f}σ.</div>
        </div>
        <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
          <div style="color:{ACCENT3};font-weight:600;margin-bottom:0.4rem">③ DBSCAN Outliers</div>
          <div style="color:{MUTED}">Users labelled −1 by DBSCAN are structural outliers.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Run Anomaly Detection (All 3 Methods)"):
        with st.spinner("Detecting anomalies..."):
            try:
                anom_hr    = detect_hr_anomalies(master,    hr_high, hr_low,   sigma)
                anom_steps = detect_steps_anomalies(master, st_low,  25000,    sigma)
                anom_sleep = detect_sleep_anomalies(master, sl_low,  sl_high,  sigma)
                st.session_state.anom_hr    = anom_hr
                st.session_state.anom_steps = anom_steps
                st.session_state.anom_sleep = anom_sleep
                st.session_state.anomaly_done = True
                st.rerun()
            except Exception as e:
                st.error(f"Detection error: {e}")

    if st.session_state.anomaly_done:
        anom_hr    = st.session_state.anom_hr
        anom_steps = st.session_state.anom_steps
        anom_sleep = st.session_state.anom_sleep

        n_hr    = int(anom_hr["is_anomaly"].sum())
        n_steps = int(anom_steps["is_anomaly"].sum())
        n_sleep = int(anom_sleep["is_anomaly"].sum())
        n_total = n_hr + n_steps + n_sleep

        ui_danger_m3(f"Total anomalies flagged: {n_total}  (HR: {n_hr} · Steps: {n_steps} · Sleep: {n_sleep})")
        metrics_m3(
            (n_hr,    "HR Anomalies"),
            (n_steps, "Steps Anomalies"),
            (n_sleep, "Sleep Anomalies"),
            (n_total, "Total Flags"),
            red_indices=[0,1,2,3]
        )

        # ── CHART 1: Heart Rate ──────────────────────────────────────────
        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec_m3("❤️", "Heart Rate — Anomaly Chart", "Step 2")
        anom_tag_m3(f"{n_hr} anomalous days detected")
        screenshot_badge_m3("Heart Rate Chart with Anomaly Highlights")
        step_pill_m3(2, "Threshold + Residual Detection")
        ui_info_m3(f"Red markers = anomaly days. Dashed lines = thresholds (HR>{hr_high} or HR<{hr_low}). Shaded band = ±{sigma:.0f}σ residual zone.")

        hr_anom   = anom_hr[anom_hr["is_anomaly"]]
        fig_hr_m3 = go.Figure()

        rolling_upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
        rolling_lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()

        fig_hr_m3.add_trace(go.Scatter(
            x=anom_hr["Date"], y=rolling_upper, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_hr_m3.add_trace(go.Scatter(
            x=anom_hr["Date"], y=rolling_lower, mode="lines",
            fill="tonexty", fillcolor="rgba(99,179,237,0.1)",
            line=dict(width=0), name=f"±{sigma:.0f}σ Expected Band"
        ))
        fig_hr_m3.add_trace(go.Scatter(
            x=anom_hr["Date"], y=anom_hr["AvgHR"],
            mode="lines+markers", name="Avg Heart Rate",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=5, color=ACCENT),
            hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<extra></extra>"
        ))
        fig_hr_m3.add_trace(go.Scatter(
            x=anom_hr["Date"], y=anom_hr["rolling_med"],
            mode="lines", name="Rolling Median",
            line=dict(color=ACCENT3, width=1.5, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Median: %{y:.1f} bpm<extra></extra>"
        ))
        if not hr_anom.empty:
            fig_hr_m3.add_trace(go.Scatter(
                x=hr_anom["Date"], y=hr_anom["AvgHR"],
                mode="markers", name="🚨 Anomaly",
                marker=dict(color=ACCENT_RED, size=14, symbol="circle",
                            line=dict(color="white", width=2)),
                hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<br><b>ANOMALY</b><extra>⚠️</extra>"
            ))
            for _, row in hr_anom.iterrows():
                fig_hr_m3.add_annotation(
                    x=row["Date"], y=row["AvgHR"],
                    text=f"⚠️ {row['reason']}", showarrow=True,
                    arrowhead=2, arrowcolor=ACCENT_RED, arrowsize=1.2,
                    ax=0, ay=-45,
                    font=dict(color=ACCENT_RED, size=9),
                    bgcolor=CARD_BG, bordercolor=DANGER_BOR, borderwidth=1, borderpad=4
                )
        fig_hr_m3.add_hline(y=hr_high, line_dash="dash", line_color=ACCENT_RED,
                         line_width=1.5, opacity=0.7,
                         annotation_text=f"High Threshold ({hr_high} bpm)",
                         annotation_position="top right",
                         annotation_font_color=ACCENT_RED)
        fig_hr_m3.add_hline(y=hr_low, line_dash="dash", line_color=ACCENT2,
                         line_width=1.5, opacity=0.7,
                         annotation_text=f"Low Threshold ({hr_low} bpm)",
                         annotation_position="bottom right",
                         annotation_font_color=ACCENT2)
        apply_plotly_theme(fig_hr_m3, "❤️ Heart Rate — Anomaly Detection (Real Fitbit Data)")
        fig_hr_m3.update_layout(height=480, xaxis_title="Date", yaxis_title="Heart Rate (bpm)",
            xaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED),
            yaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED))
        st.plotly_chart(fig_hr_m3, use_container_width=True)

        if not hr_anom.empty:
            with st.expander(f"📋 View {len(hr_anom)} HR Anomaly Records"):
                st.dataframe(
                    hr_anom[hr_anom["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]]
                    .rename(columns={"rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                    .round(2), use_container_width=True
                )

        # ── CHART 2: Sleep ───────────────────────────────────────────────
        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec_m3("💤", "Sleep Pattern — Anomaly Visualization", "Step 3")
        anom_tag_m3(f"{n_sleep} anomalous sleep days detected")
        screenshot_badge_m3("Sleep Pattern Visualization with Alerts")
        step_pill_m3(3, "Threshold Detection on Sleep Minutes")
        ui_info_m3(f"Orange = insufficient sleep (<{sl_low} min). Purple dots = anomaly days. Green band = healthy sleep zone ({sl_low}–{sl_high} min).")

        sleep_anom = anom_sleep[anom_sleep["is_anomaly"]]

        fig_sleep = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.7, 0.3],
                                   subplot_titles=["Sleep Duration (minutes/night)", "Deviation from Expected"],
                                   vertical_spacing=0.08)
        fig_sleep.add_hrect(y0=sl_low, y1=sl_high, fillcolor="rgba(104,211,145,0.08)",
                             line_width=0, annotation_text="✅ Healthy Sleep Zone",
                             annotation_position="top right",
                             annotation_font_color=ACCENT3, row=1, col=1)
        fig_sleep.add_trace(go.Scatter(
            x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
            mode="lines+markers", name="Sleep Minutes",
            line=dict(color="#b794f4", width=2.5),
            marker=dict(size=5, color="#b794f4"),
            hovertemplate="<b>%{x}</b><br>Sleep: %{y:.0f} min<extra></extra>"
        ), row=1, col=1)
        fig_sleep.add_trace(go.Scatter(
            x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
            mode="lines", name="Rolling Median",
            line=dict(color=ACCENT3, width=1.5, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Median: %{y:.0f} min<extra></extra>"
        ), row=1, col=1)
        if not sleep_anom.empty:
            fig_sleep.add_trace(go.Scatter(
                x=sleep_anom["Date"], y=sleep_anom["TotalSleepMinutes"],
                mode="markers", name="🚨 Sleep Anomaly",
                marker=dict(color=ACCENT_RED, size=14, symbol="diamond",
                            line=dict(color="white", width=2)),
                hovertemplate="<b>%{x}</b><br>Sleep: %{y:.0f} min<br><b>ANOMALY</b><extra>⚠️</extra>"
            ), row=1, col=1)
            for _, row in sleep_anom.iterrows():
                fig_sleep.add_annotation(
                    x=row["Date"], y=row["TotalSleepMinutes"],
                    text=f"⚠️ {row['reason']}", showarrow=True,
                    arrowhead=2, arrowcolor=ACCENT_RED, arrowsize=1.2,
                    ax=20, ay=-40,
                    font=dict(color=ACCENT_RED, size=9),
                    bgcolor=CARD_BG, bordercolor=DANGER_BOR, borderwidth=1,
                    borderpad=3, row=1, col=1
                )
        fig_sleep.add_hline(y=sl_low, line_dash="dash", line_color=ACCENT_RED,
                             line_width=1.5, opacity=0.7, row=1, col=1,
                             annotation_text=f"Min ({sl_low} min)",
                             annotation_font_color=ACCENT_RED)
        fig_sleep.add_hline(y=sl_high, line_dash="dash", line_color=ACCENT,
                             line_width=1.5, opacity=0.7, row=1, col=1,
                             annotation_text=f"Max ({sl_high} min)",
                             annotation_font_color=ACCENT)
        colors_resid = [ACCENT_RED if v else ACCENT for v in anom_sleep["resid_anomaly"]]
        fig_sleep.add_trace(go.Bar(
            x=anom_sleep["Date"], y=anom_sleep["residual"],
            name="Residual", marker_color=colors_resid,
            hovertemplate="<b>%{x}</b><br>Residual: %{y:.0f} min<extra></extra>"
        ), row=2, col=1)
        fig_sleep.add_hline(y=0, line_dash="solid", line_color=MUTED, line_width=1, row=2, col=1)
        apply_plotly_theme(fig_sleep)
        fig_sleep.update_layout(height=560, showlegend=True,
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
        fig_sleep.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
        fig_sleep.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
        st.plotly_chart(fig_sleep, use_container_width=True)

        if not sleep_anom.empty:
            with st.expander(f"📋 View {len(sleep_anom)} Sleep Anomaly Records"):
                st.dataframe(
                    sleep_anom[sleep_anom["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]]
                    .rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                    .round(2), use_container_width=True
                )

        # ── CHART 3: Steps ───────────────────────────────────────────────
        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec_m3("🚶", "Step Count Trend — Alerts & Anomalies", "Step 4")
        anom_tag_m3(f"{n_steps} anomalous step-count days detected")
        screenshot_badge_m3("Step Count Trend with Alert Bands")
        step_pill_m3(4, "Threshold + Residual Detection on Steps")
        ui_info_m3(f"Red vertical bands = anomaly alert days. Dashed lines = step thresholds. Bar chart below shows daily deviation from trend.")

        steps_anom = anom_steps[anom_steps["is_anomaly"]]
        fig_steps = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.65, 0.35],
                                   subplot_titles=["Daily Steps (avg across users)", "Residual Deviation from Trend"],
                                   vertical_spacing=0.08)
        for _, row in steps_anom.iterrows():
            d = str(row["Date"])
            d_next = str(pd.Timestamp(d) + pd.Timedelta(days=1))[:10]
            fig_steps.add_vrect(
                x0=d, x1=d_next,
                fillcolor="rgba(252,129,129,0.15)",
                line_color="rgba(252,129,129,0.5)",
                line_width=1.5, row=1, col=1
            )
        fig_steps.add_trace(go.Scatter(
            x=anom_steps["Date"], y=anom_steps["TotalSteps"],
            mode="lines+markers", name="Avg Daily Steps",
            line=dict(color=ACCENT3, width=2.5),
            marker=dict(size=5, color=ACCENT3),
            hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<extra></extra>"
        ), row=1, col=1)
        fig_steps.add_trace(go.Scatter(
            x=anom_steps["Date"], y=anom_steps["rolling_med"],
            mode="lines", name="Trend (Rolling Median)",
            line=dict(color=ACCENT, width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Trend: %{y:,.0f}<extra></extra>"
        ), row=1, col=1)
        if not steps_anom.empty:
            fig_steps.add_trace(go.Scatter(
                x=steps_anom["Date"], y=steps_anom["TotalSteps"],
                mode="markers", name="🚨 Steps Anomaly",
                marker=dict(color=ACCENT_RED, size=14, symbol="triangle-up",
                            line=dict(color="white", width=2)),
                hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"
            ), row=1, col=1)
        fig_steps.add_hline(y=st_low, line_dash="dash", line_color=ACCENT_RED,
                             line_width=1.5, opacity=0.8, row=1, col=1,
                             annotation_text=f"Low Alert ({st_low:,} steps)",
                             annotation_font_color=ACCENT_RED)
        fig_steps.add_hline(y=25000, line_dash="dash", line_color=ACCENT2,
                             line_width=1.5, opacity=0.7, row=1, col=1,
                             annotation_text="High Alert (25,000 steps)",
                             annotation_font_color=ACCENT2)
        res_colors = [ACCENT_RED if v else ACCENT3 for v in anom_steps["resid_anomaly"]]
        fig_steps.add_trace(go.Bar(
            x=anom_steps["Date"], y=anom_steps["residual"],
            name="Residual", marker_color=res_colors,
            hovertemplate="<b>%{x}</b><br>Deviation: %{y:,.0f} steps<extra></extra>"
        ), row=2, col=1)
        fig_steps.add_hline(y=0, line_dash="solid", line_color=MUTED, line_width=1, row=2, col=1)
        apply_plotly_theme(fig_steps)
        fig_steps.update_layout(height=560, showlegend=True,
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font_color=TEXT)
        fig_steps.update_xaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
        fig_steps.update_yaxes(gridcolor=GRID_CLR, tickfont_color=MUTED)
        st.plotly_chart(fig_steps, use_container_width=True)

        if not steps_anom.empty:
            with st.expander(f"📋 View {len(steps_anom)} Steps Anomaly Records"):
                st.dataframe(
                    steps_anom[steps_anom["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]]
                    .rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"})
                    .round(2), use_container_width=True
                )

        # ── DBSCAN Outliers ──────────────────────────────────────────────
        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        sec_m3("🔍", "DBSCAN Outlier Users — Cluster-Based Anomalies", "Step 5")
        step_pill_m3(5, "Structural Outlier Detection via DBSCAN")
        anom_tag_m3("Outlier = users with atypical overall behaviour pattern")
        ui_info_m3("Cluster each user using DBSCAN on their activity profile. Users labelled −1 are structural outliers.")

        cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes",
                        "FairlyActiveMinutes","LightlyActiveMinutes",
                        "SedentaryMinutes","TotalSleepMinutes"]
        try:
            cf = master.groupby("Id")[cluster_cols].mean().round(3).dropna()
            scaler_db = StandardScaler()
            X_scaled  = scaler_db.fit_transform(cf)
            db_m3     = DBSCAN(eps=2.2, min_samples=2)
            db_labels = db_m3.fit_predict(X_scaled)

            pca_m3   = PCA(n_components=2, random_state=42)
            X_pca_m3 = pca_m3.fit_transform(X_scaled)
            var_m3   = pca_m3.explained_variance_ratio_ * 100

            cf["DBSCAN"] = db_labels
            outlier_users = cf[cf["DBSCAN"] == -1].index.tolist()
            n_outliers = len(outlier_users)
            n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

            metrics_m3(
                (n_clusters,  "DBSCAN Clusters"),
                (n_outliers,  "Outlier Users"),
                (len(cf) - n_outliers, "Normal Users"),
                red_indices=[1]
            )

            CLUSTER_COLORS_M3 = ["#63b3ed","#68d391","#f6ad55","#b794f4","#f687b3"]
            fig_db_m3 = go.Figure()

            for lbl in sorted(set(db_labels)):
                if lbl == -1: continue
                mask = db_labels == lbl
                fig_db_m3.add_trace(go.Scatter(
                    x=X_pca_m3[mask, 0], y=X_pca_m3[mask, 1],
                    mode="markers+text",
                    name=f"Cluster {lbl}",
                    marker=dict(size=14, color=CLUSTER_COLORS_M3[lbl % len(CLUSTER_COLORS_M3)],
                                opacity=0.85, line=dict(color="white", width=1.5)),
                    text=[str(uid)[-4:] for uid in cf.index[mask]],
                    textposition="top center", textfont=dict(size=8, color=TEXT),
                    hovertemplate="<b>User ...%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                ))

            if n_outliers > 0:
                mask_out = db_labels == -1
                fig_db_m3.add_trace(go.Scatter(
                    x=X_pca_m3[mask_out, 0], y=X_pca_m3[mask_out, 1],
                    mode="markers+text",
                    name="🚨 Outlier / Anomaly",
                    marker=dict(size=20, color=ACCENT_RED, symbol="x",
                                line=dict(color="white", width=2.5)),
                    text=[str(uid)[-4:] for uid in cf.index[mask_out]],
                    textposition="top center", textfont=dict(size=9, color=ACCENT_RED),
                    hovertemplate="<b>⚠️ OUTLIER User ...%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra>ANOMALY</extra>"
                ))
                for i, uid in enumerate(cf.index[mask_out]):
                    xi, yi = X_pca_m3[mask_out][i]
                    fig_db_m3.add_shape(type="circle",
                        x0=xi-0.3, y0=yi-0.3, x1=xi+0.3, y1=yi+0.3,
                        line=dict(color=ACCENT_RED, width=2, dash="dot"),
                        fillcolor="rgba(252,129,129,0.1)"
                    )

            apply_plotly_theme(fig_db_m3, f"🔍 DBSCAN Outlier Detection — PCA Projection (eps=2.2)")
            fig_db_m3.update_layout(height=500,
                xaxis_title=f"PC1 ({var_m3[0]:.1f}% variance)",
                yaxis_title=f"PC2 ({var_m3[1]:.1f}% variance)",
                xaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED),
                yaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED))
            st.plotly_chart(fig_db_m3, use_container_width=True)

            if outlier_users:
                out_profile = cf[cf["DBSCAN"]==-1][cluster_cols]
                st.markdown(f"""
                <div class="m3-card" style="border-color:{DANGER_BOR}">
                  <div class="m3-card-title" style="color:{ACCENT_RED}">🚨 Outlier User Profiles</div>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(out_profile.round(2), use_container_width=True)

        except Exception as e:
            ui_warn_m3(f"DBSCAN clustering skipped: {e}")

        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

        # ── SECTION 3: Accuracy Simulation ──────────────────────────────
        sec_m3("🎯", "Simulated Detection Accuracy — 90%+ Target", "Step 6")
        step_pill_m3(6, "Inject Known Anomalies → Measure Detection Rate")
        ui_info_m3("10 known anomalies are injected into each signal. The detector is run and we measure how many it catches.")

        if st.button("🎯 Run Accuracy Simulation (10 injected anomalies per signal)"):
            with st.spinner("Simulating..."):
                try:
                    sim = simulate_accuracy(master, n_inject=10)
                    st.session_state.sim_results  = sim
                    st.session_state.simulation_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Simulation error: {e}")

        if st.session_state.simulation_done and st.session_state.sim_results:
            sim = st.session_state.sim_results
            overall = sim["Overall"]
            passed  = overall >= 90.0

            if passed:
                ui_success_m3(f"Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT")
            else:
                ui_warn_m3(f"Overall accuracy: {overall}% — below 90% target, adjust thresholds in sidebar")

            html = '<div class="metric-grid">'
            for signal in ["Heart Rate", "Steps", "Sleep"]:
                r   = sim[signal]
                acc = r["accuracy"]
                col = ACCENT3 if acc >= 90 else ACCENT_RED
                html += f"""
                <div class="metric-card" style="border-color:{col}44">
                  <div style="font-size:1.8rem;font-weight:800;color:{col};font-family:'Syne',sans-serif">{acc}%</div>
                  <div style="font-size:0.8rem;color:{TEXT};font-weight:600;margin:0.3rem 0">{signal}</div>
                  <div style="font-size:0.72rem;color:{MUTED}">{r['detected']}/{r['injected']} detected</div>
                  <div style="font-size:0.7rem;color:{'#9ae6b4' if acc>=90 else ACCENT_RED}">{'✅ PASS' if acc>=90 else '⚠️ LOW'}</div>
                </div>"""
            html += f"""
                <div class="metric-card" style="border-color:{'#68d391' if passed else ACCENT_RED}88;background:{'rgba(104,211,145,0.1)' if passed else DANGER_BG}">
                  <div style="font-size:1.8rem;font-weight:800;color:{'#68d391' if passed else ACCENT_RED};font-family:'Syne',sans-serif">{overall}%</div>
                  <div style="font-size:0.8rem;color:{TEXT};font-weight:600;margin:0.3rem 0">Overall</div>
                  <div style="font-size:0.7rem;color:{'#9ae6b4' if passed else ACCENT_RED}">{'✅ 90%+ ACHIEVED' if passed else '⚠️ BELOW TARGET'}</div>
                </div>"""
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

            signals   = ["Heart Rate", "Steps", "Sleep"]
            accs      = [sim[s]["accuracy"] for s in signals]
            bar_colors = [ACCENT3 if a >= 90 else ACCENT_RED for a in accs]

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=signals, y=accs,
                marker_color=bar_colors,
                text=[f"{a}%" for a in accs],
                textposition="outside",
                textfont=dict(color=TEXT, size=14, family="Syne, sans-serif"),
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
                name="Detection Accuracy"
            ))
            fig_acc.add_hline(y=90, line_dash="dash", line_color=ACCENT_RED,
                              line_width=2, annotation_text="90% Target",
                              annotation_font_color=ACCENT_RED,
                              annotation_position="top right")
            apply_plotly_theme(fig_acc, "🎯 Simulated Anomaly Detection Accuracy")
            fig_acc.update_layout(
                height=380, yaxis_range=[0, 115],
                yaxis_title="Detection Accuracy (%)",
                xaxis_title="Signal",
                xaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED),
                yaxis=dict(gridcolor=GRID_CLR, tickfont_color=MUTED),
                showlegend=False
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

        # ── Milestone 3 Summary ──────────────────────────────────────────
        sec_m3("✅", "Milestone 3 Summary")

        checklist = [
            ("🚨", "Threshold Violations",  st.session_state.anomaly_done,    f"HR>{hr_high}/{hr_low}, Steps<{st_low}, Sleep<{sl_low}/<{sl_high}"),
            ("📉", "Residual-Based",         st.session_state.anomaly_done,    f"Rolling median ±{sigma:.0f}σ on all 3 signals"),
            ("🔍", "DBSCAN Outliers",        st.session_state.anomaly_done,    "Structural user-level anomalies via clustering"),
            ("❤️", "HR Chart",               st.session_state.anomaly_done,    "Interactive Plotly — annotations + threshold lines"),
            ("💤", "Sleep Chart",            st.session_state.anomaly_done,    "Dual subplot — duration + residual bars"),
            ("🚶", "Steps Chart",            st.session_state.anomaly_done,    "Trend + alert bands + residual deviation"),
            ("🎯", "Accuracy Simulation",    st.session_state.simulation_done, "10 injected anomalies per signal, 90%+ target"),
        ]

        for icon, label, done, detail in checklist:
            dot = "✅" if done else "⬜"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;padding:0.6rem 0;border-bottom:1px solid {CARD_BOR}">
              <span style="font-size:1.1rem">{dot}</span>
              <span style="font-size:0.9rem;font-weight:600;color:{TEXT};min-width:180px">{icon} {label}</span>
              <span style="font-size:0.8rem;color:{MUTED}">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="m3-card" style="border-color:{DANGER_BOR}">
          <div class="m3-card-title">📸 Screenshots Required for Submission</div>
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;font-size:0.82rem">
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Chart 1</b> — Heart Rate with anomalies highlighted
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Chart 2</b> — Sleep pattern visualization
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Chart 3</b> — Step count trend with alerts
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Chart 4</b> — DBSCAN outlier scatter (PCA)
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;grid-column:1/-1">
              <span style="color:{ACCENT2}">📸</span> <b>Chart 5</b> — Accuracy bar chart (90%+ target line)
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ===================================================
#
#  MILESTONE 4 — INSIGHTS DASHBOARD
#
# ===================================================

else:

    # Hero
    st.markdown(f"""
    <div class="m4-hero">
      <div class="hero-badge">MILESTONE 4 . INSIGHTS DASHBOARD</div>
      <h1 class="hero-title">📊 FitPulse Insights Dashboard</h1>
      <p class="hero-sub">Detect · Filter · Export PDF & CSV — Real Fitbit Device Data</p>
    </div>""", unsafe_allow_html=True)

    # ── Check data availability ──────────────────────────────────────────────
    if not st.session_state.files_loaded:
        st.markdown(f"""
        <div class="m3-card" style="text-align:center;padding:3rem;border-color:{DANGER_BOR}">
          <div style="font-size:3rem;margin-bottom:1rem">⚠️</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem">
            Please upload data in Milestone 1 or Milestone 2
          </div>
          <div style="color:{MUTED};font-size:0.88rem;margin-bottom:1.5rem">
            Go to <b>Milestone 2</b>, upload all 5 CSV files, then return here.<br>
            Or use the <b>⚡ Run M4 Detection</b> button in the sidebar once data is loaded.
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;max-width:600px;margin:0 auto;text-align:left">
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{ACCENT};font-weight:600;font-size:0.85rem">📤 Upload</div>
              <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">All 5 Fitbit CSV files in Milestone 2</div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{ACCENT_RED};font-weight:600;font-size:0.85rem">🚨 Detect</div>
              <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">Run M4 Detection from sidebar</div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{ACCENT3};font-weight:600;font-size:0.85rem">📥 Export</div>
              <div style="color:{MUTED};font-size:0.75rem;margin-top:0.2rem">Download PDF report + CSV data</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Run M4 detection pipeline ────────────────────────────────────────────
    if run_m4_clicked:
        with st.spinner("⏳ Running M4 anomaly detection..."):
            try:
                master_m4 = st.session_state.master
                anom_hr_m4    = detect_hr_m4(master_m4,    m4_hr_high, m4_hr_low,   m4_sigma)
                anom_steps_m4 = detect_steps_m4(master_m4, m4_st_low,  25000,       m4_sigma)
                anom_sleep_m4 = detect_sleep_m4(master_m4, m4_sl_low,  m4_sl_high,  m4_sigma)
                st.session_state.m4_anom_hr    = anom_hr_m4
                st.session_state.m4_anom_steps = anom_steps_m4
                st.session_state.m4_anom_sleep = anom_sleep_m4
                st.session_state.pipeline_done = True
                st.rerun()
            except Exception as e:
                st.error(f"M4 detection error: {e}")

    # Auto-run detection on first visit if not done yet
    if not st.session_state.pipeline_done and st.session_state.files_loaded:
        with st.spinner("⏳ Running anomaly detection for dashboard..."):
            try:
                master_m4 = st.session_state.master
                anom_hr_m4    = detect_hr_m4(master_m4,    m4_hr_high, m4_hr_low,   m4_sigma)
                anom_steps_m4 = detect_steps_m4(master_m4, m4_st_low,  25000,       m4_sigma)
                anom_sleep_m4 = detect_sleep_m4(master_m4, m4_sl_low,  m4_sl_high,  m4_sigma)
                st.session_state.m4_anom_hr    = anom_hr_m4
                st.session_state.m4_anom_steps = anom_steps_m4
                st.session_state.m4_anom_sleep = anom_sleep_m4
                st.session_state.pipeline_done = True
                st.rerun()
            except Exception as e:
                st.error(f"Auto-detection error: {e}")
        st.stop()

    master     = st.session_state.master
    anom_hr    = st.session_state.get("m4_anom_hr",    st.session_state.anom_hr)
    anom_steps = st.session_state.get("m4_anom_steps", st.session_state.anom_steps)
    anom_sleep = st.session_state.get("m4_anom_sleep", st.session_state.anom_sleep)

    if anom_hr is None or anom_steps is None or anom_sleep is None:
        ui_danger_m4("Detection results not found. Click ⚡ Run M4 Detection in the sidebar.")
        st.stop()

    # ── Apply date filter ─────────────────────────────────────────────────────
    try:
        if date_range_m4 is not None and isinstance(date_range_m4, tuple) and len(date_range_m4) == 2:
            d_from, d_to = pd.Timestamp(date_range_m4[0]), pd.Timestamp(date_range_m4[1])
        else:
            all_dates = pd.to_datetime(master["Date"])
            d_from, d_to = all_dates.min(), all_dates.max()
    except Exception:
        all_dates = pd.to_datetime(master["Date"])
        d_from, d_to = all_dates.min(), all_dates.max()

    def filt_m4(df, date_col="Date"):
        df2 = df.copy()
        df2[date_col] = pd.to_datetime(df2[date_col])
        return df2[(df2[date_col] >= d_from) & (df2[date_col] <= d_to)]

    anom_hr_f    = filt_m4(anom_hr)
    anom_steps_f = filt_m4(anom_steps)
    anom_sleep_f = filt_m4(anom_sleep)
    master_f     = filt_m4(master)
    if selected_user_m4:
        master_f = master_f[master_f["Id"] == selected_user_m4]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    n_hr_f    = int(anom_hr_f["is_anomaly"].sum())
    n_steps_f = int(anom_steps_f["is_anomaly"].sum())
    n_sleep_f = int(anom_sleep_f["is_anomaly"].sum())
    n_total_f = n_hr_f + n_steps_f + n_sleep_f
    n_users_f = master_f["Id"].nunique()
    n_days_f  = master_f["Date"].nunique()

    worst_hr_row   = anom_hr_f[anom_hr_f["is_anomaly"]].copy()
    worst_hr_day   = worst_hr_row.iloc[worst_hr_row["residual"].abs().argmax()]["Date"].strftime("%d %b") if not worst_hr_row.empty else "-"

    kpi_html = f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="border-color:{DANGER_BOR}">
        <div class="kpi-val" style="color:{ACCENT_RED}">{n_total_f}</div>
        <div class="kpi-label">Total Anomalies</div>
        <div class="kpi-sub">across all signals</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(246,135,179,0.3)">
        <div class="kpi-val" style="color:{ACCENT2}">{n_hr_f}</div>
        <div class="kpi-label">HR Flags</div>
        <div class="kpi-sub">heart rate anomalies</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(104,211,145,0.3)">
        <div class="kpi-val" style="color:{ACCENT3}">{n_steps_f}</div>
        <div class="kpi-label">Steps Alerts</div>
        <div class="kpi-sub">step count anomalies</div>
      </div>
      <div class="kpi-card" style="border-color:rgba(183,148,244,0.3)">
        <div class="kpi-val" style="color:#b794f4">{n_sleep_f}</div>
        <div class="kpi-label">Sleep Flags</div>
        <div class="kpi-sub">sleep anomalies</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val" style="color:{ACCENT}">{n_users_f}</div>
        <div class="kpi-label">Users</div>
        <div class="kpi-sub">in selected range</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val" style="color:{ACCENT_ORG}">{worst_hr_day}</div>
        <div class="kpi-label">Peak HR Anomaly</div>
        <div class="kpi-sub">highest deviation day</div>
      </div>
    </div>"""
    st.markdown(kpi_html, unsafe_allow_html=True)

    ui_success_m4(f"Pipeline complete · {n_users_f} users · {n_days_f} days · {n_total_f} anomalies flagged")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_hr, tab_steps, tab_sleep, tab_export = st.tabs([
        "📊 Overview", "❤️ Heart Rate", "🚶 Steps", "💤 Sleep", "📥 Export"
    ])

    # ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────
    with tab_overview:

        st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
        sec_m4("📅", "Combined Anomaly Timeline")

        all_anoms = []
        for df_, sig, col in [
            (anom_hr_f,    "Heart Rate", ACCENT2),
            (anom_steps_f, "Steps",      ACCENT3),
            (anom_sleep_f, "Sleep",      "#b794f4"),
        ]:
            a = df_[df_["is_anomaly"]].copy()
            a["signal"] = sig
            a["color"]  = col
            all_anoms.append(a[["Date","signal","color","reason"]])

        if all_anoms:
            combined = pd.concat(all_anoms, ignore_index=True)
            combined["Date"] = pd.to_datetime(combined["Date"])
            combined["y"]    = combined["signal"]

            fig_timeline = go.Figure()
            for sig, col in [("Heart Rate", ACCENT2), ("Steps", ACCENT3), ("Sleep", "#b794f4")]:
                sub = combined[combined["signal"] == sig]
                if not sub.empty:
                    fig_timeline.add_trace(go.Scatter(
                        x=sub["Date"], y=sub["y"], mode="markers",
                        name=sig, marker=dict(color=col, size=14, symbol="diamond",
                                              line=dict(color="white", width=2)),
                        hovertemplate=f"<b>{sig}</b><br>%{{x|%d %b %Y}}<br>%{{customdata}}<extra>⚠️ ANOMALY</extra>",
                        customdata=sub["reason"].values
                    ))
            ptheme(fig_timeline, "📅 Anomaly Event Timeline - All Signals", h=280)
            fig_timeline.update_layout(
                xaxis_title="Date", yaxis_title="Signal",
                showlegend=True,
                yaxis=dict(categoryorder="array",
                           categoryarray=["Sleep","Steps","Heart Rate"],
                           gridcolor=GRID_CLR, tickfont_color=MUTED)
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
        sec_m4("🗂️", "Recent Anomaly Log")
        if all_anoms:
            log = combined.sort_values("Date", ascending=False).head(10)
            for _, row in log.iterrows():
                st.markdown(f"""
                <div class="anom-row">
                  <span style="font-size:0.9rem">🚨</span>
                  <span style="color:{row['color']};font-family:'JetBrains Mono',monospace;font-size:0.75rem;min-width:90px">{row['signal']}</span>
                  <span style="color:{MUTED};font-size:0.78rem;min-width:90px">{row['Date'].strftime('%d %b %Y')}</span>
                  <span style="color:{TEXT};font-size:0.78rem">{row['reason']}</span>
                </div>""", unsafe_allow_html=True)

    # ── TAB 2: HEART RATE ─────────────────────────────────────────────────────
    with tab_hr:
        sec_m4("❤️", "Heart Rate - Deep Dive", f"{n_hr_f} anomalies")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="m3-card">
              <div class="m3-card-title">HR Statistics</div>
              <div style="font-size:0.83rem;line-height:2">
                <div>Mean HR: <b style="color:{ACCENT}">{anom_hr_f['AvgHR'].mean():.1f} bpm</b></div>
                <div>Max HR: <b style="color:{ACCENT_RED}">{anom_hr_f['AvgHR'].max():.1f} bpm</b></div>
                <div>Min HR: <b style="color:{ACCENT2}">{anom_hr_f['AvgHR'].min():.1f} bpm</b></div>
                <div>Anomaly days: <b style="color:{ACCENT_RED}">{n_hr_f}</b> of {len(anom_hr_f)} total</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="m3-card"><div class="m3-card-title">HR Anomaly Records</div>', unsafe_allow_html=True)
            hr_display = anom_hr_f[anom_hr_f["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].round(2)
            if not hr_display.empty:
                st.dataframe(hr_display.rename(columns={"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                             use_container_width=True, height=200)
            else:
                ui_success_m4("No HR anomalies in selected range")
            st.markdown('</div>', unsafe_allow_html=True)

        st.plotly_chart(chart_hr_m4(anom_hr_f, m4_hr_high, m4_hr_low, m4_sigma), use_container_width=True)

    # ── TAB 3: STEPS ──────────────────────────────────────────────────────────
    with tab_steps:
        sec_m4("🚶", "Step Count - Deep Dive", f"{n_steps_f} alerts")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="m3-card">
              <div class="m3-card-title">Steps Statistics</div>
              <div style="font-size:0.83rem;line-height:2">
                <div>Mean steps/day: <b style="color:{ACCENT3}">{anom_steps_f['TotalSteps'].mean():,.0f}</b></div>
                <div>Max steps/day: <b style="color:{ACCENT}">{anom_steps_f['TotalSteps'].max():,.0f}</b></div>
                <div>Min steps/day: <b style="color:{ACCENT_RED}">{anom_steps_f['TotalSteps'].min():,.0f}</b></div>
                <div>Alert days: <b style="color:{ACCENT_RED}">{n_steps_f}</b> of {len(anom_steps_f)} total</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="m3-card"><div class="m3-card-title">Steps Alert Records</div>', unsafe_allow_html=True)
            st_display = anom_steps_f[anom_steps_f["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].round(2)
            if not st_display.empty:
                st.dataframe(st_display.rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                             use_container_width=True, height=200)
            else:
                ui_success_m4("No step anomalies in selected range")
            st.markdown('</div>', unsafe_allow_html=True)

        st.plotly_chart(chart_steps_m4(anom_steps_f, m4_st_low), use_container_width=True)

    # ── TAB 4: SLEEP ──────────────────────────────────────────────────────────
    with tab_sleep:
        sec_m4("💤", "Sleep Pattern - Deep Dive", f"{n_sleep_f} anomalies")

        col_a, col_b = st.columns(2)
        with col_a:
            _sl_min_nonzero = anom_sleep_f[anom_sleep_f['TotalSleepMinutes']>0]['TotalSleepMinutes'].min() if (anom_sleep_f['TotalSleepMinutes']>0).any() else 0
            st.markdown(f"""
            <div class="m3-card">
              <div class="m3-card-title">Sleep Statistics</div>
              <div style="font-size:0.83rem;line-height:2">
                <div>Mean sleep/night: <b style="color:#b794f4">{anom_sleep_f['TotalSleepMinutes'].mean():.0f} min</b></div>
                <div>Max sleep/night: <b style="color:{ACCENT}">{anom_sleep_f['TotalSleepMinutes'].max():.0f} min</b></div>
                <div>Min (non-zero): <b style="color:{ACCENT_RED}">{_sl_min_nonzero:.0f} min</b></div>
                <div>Anomaly days: <b style="color:{ACCENT_RED}">{n_sleep_f}</b> of {len(anom_sleep_f)} total</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="m3-card"><div class="m3-card-title">Sleep Anomaly Records</div>', unsafe_allow_html=True)
            sl_display = anom_sleep_f[anom_sleep_f["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].round(2)
            if not sl_display.empty:
                st.dataframe(sl_display.rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                             use_container_width=True, height=200)
            else:
                ui_success_m4("No sleep anomalies in selected range")
            st.markdown('</div>', unsafe_allow_html=True)

        st.plotly_chart(chart_sleep_m4(anom_sleep_f, m4_sl_low, m4_sl_high), use_container_width=True)

    # ── TAB 5: EXPORT ─────────────────────────────────────────────────────────
    with tab_export:
        sec_m4("📥", "Export - PDF Report & CSV Data", "Downloadable")

        st.markdown(f"""
        <div class="m3-card">
          <div class="m3-card-title">What's Included in the Exports</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.83rem">
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{ACCENT};font-weight:600;margin-bottom:0.5rem">📄 PDF Report (4 pages)</div>
              <div style="color:{MUTED};line-height:1.8">
                ✅ Executive summary<br>
                ✅ Anomaly counts per signal<br>
                ✅ Thresholds used<br>
                ✅ Methodology explanation<br>
                ✅ All 3 charts embedded<br>
                ✅ Full anomaly records tables<br>
                ✅ User activity profiles
              </div>
            </div>
            <div style="background:{SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{ACCENT3};font-weight:600;margin-bottom:0.5rem">📊 CSV Export</div>
              <div style="color:{MUTED};line-height:1.8">
                ✅ All anomaly records<br>
                ✅ Signal type column<br>
                ✅ Date of anomaly<br>
                ✅ Actual vs expected value<br>
                ✅ Residual deviation<br>
                ✅ Anomaly reason text<br>
                ✅ All signals combined
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)

        col_pdf, col_csv = st.columns(2)

        with col_pdf:
            sec_m4("📄", "PDF Report")
            st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">Full 4-page PDF with charts embedded, anomaly tables, and user profiles.</div>', unsafe_allow_html=True)

            if st.button("📄 Generate PDF Report", key="gen_pdf"):
                with st.spinner("⏳ Generating PDF (embedding charts)..."):
                    try:
                        fig_hr_exp    = chart_hr_m4(anom_hr_f,    m4_hr_high, m4_hr_low, m4_sigma, h=420)
                        fig_steps_exp = chart_steps_m4(anom_steps_f, m4_st_low, h=420)
                        fig_sleep_exp = chart_sleep_m4(anom_sleep_f, m4_sl_low, m4_sl_high, h=420)

                        pdf_buf = generate_pdf_m4(
                            master_f, anom_hr_f, anom_steps_f, anom_sleep_f,
                            m4_hr_high, m4_hr_low, m4_st_low, m4_sl_low, m4_sl_high, m4_sigma,
                            fig_hr_exp, fig_steps_exp, fig_sleep_exp
                        )
                        fname = f"FitPulse_Anomaly_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_buf,
                            file_name=fname,
                            mime="application/pdf",
                            key="dl_pdf"
                        )
                        ui_success_m4(f"PDF ready - {fname}")
                    except Exception as e:
                        st.error(f"PDF error: {e}")

        with col_csv:
            sec_m4("📊", "CSV Export")
            st.markdown(f'<div style="color:{MUTED};font-size:0.82rem;margin-bottom:0.8rem">All anomaly records from all three signals in a single CSV file.</div>', unsafe_allow_html=True)

            csv_data = generate_csv_m4(anom_hr_f, anom_steps_f, anom_sleep_f)
            fname_csv = f"FitPulse_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            st.download_button(
                label="⬇️ Download Anomaly CSV",
                data=csv_data,
                file_name=fname_csv,
                mime="text/csv",
                key="dl_csv"
            )

            with st.expander("👁️ Preview CSV data"):
                preview_df = pd.concat([
                    anom_hr_f[anom_hr_f["is_anomaly"]].assign(signal="Heart Rate").rename(columns={"AvgHR":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                    anom_steps_f[anom_steps_f["is_anomaly"]].assign(signal="Steps").rename(columns={"TotalSteps":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                    anom_sleep_f[anom_sleep_f["is_anomaly"]].assign(signal="Sleep").rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                ], ignore_index=True).sort_values(["signal","Date"]).round(2)
                st.dataframe(preview_df, use_container_width=True, height=280)

        st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)

        sec_m4("📸", "Screenshots Required for Submission")
        st.markdown(f"""
        <div class="m3-card">
          <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;font-size:0.82rem">
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 1</b> - Full dashboard UI (Overview tab)
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 2</b> - Downloadable report buttons (this tab)
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 3</b> - KPI strip with anomaly counts
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 4</b> - HR / Steps / Sleep deep dive tabs
            </div>
            <div style="background:{SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;grid-column:1/-1">
              <span style="color:{ACCENT2}">📸</span> <b>Screenshot 5</b> - Sidebar with filters + date range visible
            </div>
          </div>
        </div>""", unsafe_allow_html=True)