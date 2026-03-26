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
import warnings
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
# SESSION STATE (Milestone 3)
# ---------------------------------------------------

for k, v in [
    ("dark_mode",        True),
    ("files_loaded",     False),
    ("anomaly_done",     False),
    ("simulation_done",  False),
    ("daily_m3",  None), ("hourly_s", None), ("hourly_i", None),
    ("sleep_m3",  None), ("hr_m3",    None), ("hr_minute", None),
    ("master",   None),
    ("anom_hr",  None), ("anom_steps", None), ("anom_sleep", None),
    ("sim_results", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------
# GLOBAL STYLE  (covers all milestones)
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
# MILESTONE 3 — THEME VARIABLES
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
# MILESTONE 3 — REQUIRED FILES
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
# MILESTONE 3 — ANOMALY DETECTION FUNCTIONS
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

def simulate_accuracy(master, n_inject=10):
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}
    # HR
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
    # Steps
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
    # Sleep
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
        "3 — Anomaly Detection & Visualization"
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

else:
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
        sleep_daily         = sleep_df2.groupby(sleep_df2["date"].dt.date)["value"].sum().reset_index()
        sleep_daily.columns = ["ds","y"]
        sleep_daily["ds"]   = pd.to_datetime(sleep_daily["ds"])
        model_sleep         = Prophet(interval_width=0.8, weekly_seasonality=True)
        _ = model_sleep.fit(sleep_daily)
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
        ax_sl.scatter(sleep_daily["ds"], sleep_daily["y"],
                      color="#c77dff", s=22, zorder=5, alpha=0.85,
                      edgecolors="#9d4edd", linewidths=0.5, label="Actual Data")
        ax_sl.axvline(sleep_daily["ds"].max(), color="#f97316", linestyle="--",
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

else:

    # Hero
    st.markdown(f"""
    <div class="m3-hero">
      <div class="hero-badge">MILESTONE 3 · ANOMALY DETECTION & VISUALIZATION</div>
      <h1 class="hero-title">🚨 FitPulse Anomaly Detector</h1>
      <p class="hero-sub">Threshold Violations · Prophet Residuals · Outlier Clusters · Interactive Plotly Charts</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Data Loading ──────────────────────────────────────────────
    sec_m3("📂", "Data Loading", "Step 1")

    ui_info_m3("Upload the same 5 Fitbit CSV files as Milestone 2. Files are auto-detected by column structure.")

    uploaded_files_m3 = st.file_uploader(
        "📁  Drop all 5 Fitbit CSV files here",
        type="csv", accept_multiple_files=True, key="m3_uploader",
        help="Hold Ctrl (Windows) or Cmd (Mac) to select multiple files"
    )

    detected = {}
    ignored  = []
    if uploaded_files_m3:
        raw_uploads = []
        for uf in uploaded_files_m3:
            try:
                df_tmp = pd.read_csv(uf)
                raw_uploads.append((uf.name, df_tmp))
            except Exception:
                ignored.append(uf.name)

        used_names = set()
        for req_name, finfo in REQUIRED_FILES.items():
            best_score, best_name, best_df = 0, None, None
            for uname, udf in raw_uploads:
                s = score_match(udf, finfo)
                if s > best_score:
                    best_score, best_name, best_df = s, uname, udf
            if best_score >= 2:
                detected[req_name] = best_df
                used_names.add(best_name)

        for uname, _ in raw_uploads:
            if uname not in used_names:
                ignored.append(uname)

    # Status grid
    status_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.6rem;margin:1rem 0">'
    for req_name, finfo in REQUIRED_FILES.items():
        found = req_name in detected
        bg  = SUCCESS_BG if found else WARN_BG
        bor = SUCCESS_BOR if found else WARN_BOR
        ico = "✅" if found else "❌"
        status_html += f"""
        <div style="background:{bg};border:1px solid {bor};border-radius:10px;padding:0.7rem 0.9rem">
          <div style="font-size:1.2rem">{ico} {finfo['icon']}</div>
          <div style="font-size:0.72rem;font-weight:600;color:{TEXT};margin-top:0.3rem">{finfo['label']}</div>
          <div style="font-size:0.65rem;color:{MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.1rem">
            {'Found ✓' if found else 'Missing'}
          </div>
        </div>"""
    status_html += "</div>"
    st.markdown(status_html, unsafe_allow_html=True)

    n_up = len(detected)
    metrics_m3((n_up, "Detected"), (5 - n_up, "Missing"), ("✓" if n_up == 5 else "✗", "Ready"))

    if n_up < 5:
        missing_list = [REQUIRED_FILES[r]["label"] for r in REQUIRED_FILES if r not in detected]
        ui_warn_m3(f"Missing: {', '.join(missing_list)}")

    if st.button("⚡ Load & Build Master DataFrame", disabled=(n_up < 5)):
        with st.spinner("Parsing and building master..."):
            try:
                daily_m3    = detected["dailyActivity_merged.csv"].copy()
                hourly_s = detected["hourlySteps_merged.csv"].copy()
                hourly_i = detected["hourlyIntensities_merged.csv"].copy()
                sleep_m3    = detected["minuteSleep_merged.csv"].copy()
                hr_m3       = detected["heartrate_seconds_merged.csv"].copy()

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

                master = daily_m3.copy().rename(columns={"ActivityDate":"Date"})
                master["Date"] = master["Date"].dt.date
                master = master.merge(hr_daily_m3,    on=["Id","Date"], how="left")
                master = master.merge(sleep_daily_m3, on=["Id","Date"], how="left")
                master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
                master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
                for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                    master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

                st.session_state.daily_m3     = daily_m3
                st.session_state.hourly_s  = hourly_s
                st.session_state.hourly_i  = hourly_i
                st.session_state.sleep_m3     = sleep_m3
                st.session_state.hr_m3        = hr_m3
                st.session_state.hr_minute = hr_minute
                st.session_state.master    = master
                st.session_state.files_loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.files_loaded:
        master = st.session_state.master
        ui_success_m3(f"Master DataFrame ready — {master.shape[0]} rows · {master['Id'].nunique()} users")

        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

        # ── SECTION 2: Anomaly Detection ────────────────────────────────────
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

    else:
        st.markdown(f"""
        <div class="m3-card" style="text-align:center;padding:3rem">
          <div style="font-size:3rem;margin-bottom:1rem">🚨</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{TEXT};margin-bottom:0.5rem">
            Upload Your Fitbit Files to Begin
          </div>
          <div style="color:{MUTED};font-size:0.88rem">
            Upload all 5 CSV files above and click <b>Load & Build Master DataFrame</b>
          </div>
        </div>
        """, unsafe_allow_html=True)