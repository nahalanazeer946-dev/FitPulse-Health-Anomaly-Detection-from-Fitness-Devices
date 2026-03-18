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
# GLOBAL STYLE
# ---------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    color: #e2e8f0;
}
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
# SIDEBAR
# ---------------------------------------------------

st.sidebar.markdown("""
<div style='text-align:center; padding: 16px 0 8px 0;'>
    <span style='font-family:Syne,sans-serif; font-size:1.6rem; font-weight:800;
    background: linear-gradient(90deg,#e91e8c,#c471ed);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🏃🏻 FitPulse</span>
</div>
""", unsafe_allow_html=True)

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

# ---------------------------------------------------
# HERO HEADER
# ---------------------------------------------------

st.markdown("""
<div class="hero-banner">
    <h1>🏃🏻 FitPulse Health Analytics</h1>
    <h3>Pattern Extraction · Forecasting · Clustering Intelligence</h3>
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

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

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

# ---------------------------------------------------
# UPLOAD GATE  — nothing below runs until files are uploaded
# ---------------------------------------------------

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
    st.stop()   # ← hard stop: NOTHING below executes until files exist

# =================================================================
#  ALL CODE BELOW IS GATED — only runs after st.stop() is cleared
# =================================================================

# --- Dataset detection ---
daily = sleep = hr = steps = intensity = None
for name, df in files.items():
    if "TotalSteps"     in df.columns: daily     = df
    if "Value"          in df.columns: hr        = df
    if "StepTotal"      in df.columns: steps     = df
    if "TotalIntensity" in df.columns: intensity = df
    if "SleepDay"       in df.columns or "value" in df.columns: sleep = df

# ---------------------------------------------------
# DATASET DETECTION DISPLAY
# ---------------------------------------------------

st.markdown('<div class="section-header">🔍 Dataset Detection</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
if daily     is not None: c1.success("🏃 Daily Activity\n\n✔ Found")
else:                         c1.info("🏃 Daily Activity\n\nNot Found")
if steps     is not None: c2.success("👟 Hourly Steps\n\n✔ Found")
else:                         c2.info("👟 Hourly Steps\n\nNot Found")
if intensity is not None: c3.success("⚡ Hourly Intensities\n\n✔ Found")
else:                         c3.info("⚡ Hourly Intensities\n\nNot Found")
if sleep     is not None: c4.success("😴 Sleep\n\n✔ Found")
else:                         c4.info("😴 Sleep\n\nNot Found")
if hr        is not None: c5.success("❤️ Heart Rate\n\n✔ Found")
else:                         c5.info("❤️ Heart Rate\n\nNot Found")

# ---------------------------------------------------
# DATASET OVERVIEW + KPIs
# ---------------------------------------------------

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
    c1.metric("🏆 Highest Steps",   f"{int(daily['TotalSteps'].max()):,}")
    c2.metric("📉 Lowest Steps",    f"{int(daily['TotalSteps'].min()):,}")
    c3.metric("🛋 Avg Sedentary",   avg_sedentary)

    st.markdown('<div class="section-header">💚 Overall Health Score</div>', unsafe_allow_html=True)
    score = min(int((avg_steps / 10000) * 100), 100)
    st.progress(score / 100)
    st.write(f"Estimated Health Score: **{score}/100**")

    st.markdown('<div class="section-header">🩺 Health Status</div>', unsafe_allow_html=True)
    if   score > 70: st.success("🏃 Active Lifestyle 💪 — Users are meeting WHO recommended activity levels.")
    elif score > 40: st.info("👍 Moderate Activity — Users are partially active but have room to improve.")
    else:            st.warning("⚠️ Low Activity Level — Users are predominantly sedentary.")

# ---------------------------------------------------
# TSFRESH HEATMAP
# ---------------------------------------------------

if hr is not None:
    st.markdown('<div class="section-header">🧬 TSFresh Feature Matrix</div>', unsafe_allow_html=True)

    feature_df = hr.groupby("Id")["Value"].agg(
        ["sum","median","mean","count","std","var","max","min"]
    )
    scaler2 = MinMaxScaler()
    heat    = pd.DataFrame(scaler2.fit_transform(feature_df), columns=feature_df.columns)

    bwr = LinearSegmentedColormap.from_list("bwr_custom",
        ["#2563eb","#5b8dd9","#c8d8f0","#f0c0c8","#e05070","#b91c1c"])

    fig_h, ax_h = plt.subplots(figsize=(15, max(5, len(heat) * 0.55 + 1)))
    fig_h.patch.set_facecolor(GRAPH_BG)
    sns.heatmap(
        heat, cmap=bwr, annot=True, fmt=".2f",
        linewidths=0.4, linecolor=GRAPH_BG, ax=ax_h,
        cbar_kws={"shrink":0.8,"label":"Normalized Value (0–1)","pad":0.02},
        annot_kws={"size":9,"weight":"bold","color":"white"}
    )
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

# ---------------------------------------------------
# PROPHET — HEART RATE
# ---------------------------------------------------

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

    # ── Figure: exact match to reference screenshot ──
    fig_hr, ax_hr = plt.subplots(figsize=(15, 6))
    fig_hr.patch.set_facecolor("#0d1b3e")   # very deep navy
    ax_hr.set_facecolor("#0d1b3e")

    # Turn off ALL grid lines (clean dark bg like screenshot)
    ax_hr.grid(False)

    # ── OUTER CI band — darkest blue, very wide ──
    ax_hr.fill_between(
        forecast_hr["ds"],
        forecast_hr["yhat_lower"],
        forecast_hr["yhat_upper"],
        color="#1e3a6e",   # mid navy-blue
        alpha=0.90,
        zorder=1,
        label="80% Confidence Interval"
    )

    # ── INNER CI band — slightly lighter, tighter (visual depth) ──
    inner_lo = forecast_hr["yhat"] - (forecast_hr["yhat"] - forecast_hr["yhat_lower"]) * 0.45
    inner_hi = forecast_hr["yhat"] + (forecast_hr["yhat_upper"] - forecast_hr["yhat"]) * 0.45
    ax_hr.fill_between(
        forecast_hr["ds"], inner_lo, inner_hi,
        color="#2a55a0",
        alpha=0.80,
        zorder=2
    )

    # ── Trend line — bright cyan-white, thin and crisp ──
    ax_hr.plot(
        forecast_hr["ds"], forecast_hr["yhat"],
        color="#a8d8f0",   # bright light blue / near-white cyan
        linewidth=1.8,
        zorder=4,
        label="Trend Forecast"
    )

    # ── Actual data points — small pink dots, left of divider only ──
    ax_hr.scatter(
        hr_daily["ds"], hr_daily["y"],
        color="#e8708a",
        s=24, zorder=6, alpha=0.90,
        label="Actual HR (bpm)",
        edgecolors="none",
        linewidths=0
    )

    # ── Orange dashed vertical divider ──
    ax_hr.axvline(
        split_date,
        color="#f97316",
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
        zorder=5,
        label="Forecast Start"
    )

    ax_hr.set_title(
        "Heart Rate — Prophet Trend Forecast (Real Fitbit Data)",
        fontsize=14, fontweight="bold", color="#ddeeff", pad=16
    )
    ax_hr.set_xlabel("Date", fontsize=11, color="#8899bb", labelpad=10)
    ax_hr.set_ylabel("Heart Rate (bpm)", fontsize=11, color="#8899bb", labelpad=10)
    ax_hr.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_hr.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax_hr.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=8.5,
             color="#8899bb")
    plt.setp(ax_hr.yaxis.get_majorticklabels(), color="#8899bb")
    ax_hr.legend(
        loc="upper left", framealpha=0.70,
        facecolor="#0d1b3e", edgecolor="#2a3f6e",
        labelcolor="#ccddf0", fontsize=9, ncol=2
    )
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

# ---------------------------------------------------
# PROPHET — STEPS & SLEEP
# ---------------------------------------------------

if daily is not None and sleep is not None:
    st.markdown('<div class="section-header">📈 Steps & Sleep Forecast</div>', unsafe_allow_html=True)

    # --- Steps ---
    steps_df = daily[["ActivityDate","TotalSteps"]].copy()
    steps_df["ActivityDate"] = pd.to_datetime(steps_df["ActivityDate"])
    steps_df = steps_df.rename(columns={"ActivityDate":"ds","TotalSteps":"y"})
    model_steps    = Prophet(interval_width=0.8, weekly_seasonality=True)
    _ = model_steps.fit(steps_df)
    forecast_steps = model_steps.predict(model_steps.make_future_dataframe(periods=30))

    # --- Sleep ---
    sleep_df2             = sleep.copy()
    sleep_df2["date"]     = pd.to_datetime(sleep_df2["date"])
    sleep_daily           = sleep_df2.groupby(sleep_df2["date"].dt.date)["value"].sum().reset_index()
    sleep_daily.columns   = ["ds","y"]
    sleep_daily["ds"]     = pd.to_datetime(sleep_daily["ds"])
    model_sleep           = Prophet(interval_width=0.8, weekly_seasonality=True)
    _ = model_sleep.fit(sleep_daily)
    forecast_sleep        = model_sleep.predict(model_sleep.make_future_dataframe(periods=30))

    fig_ss, axes_ss = plt.subplots(2, 1, figsize=(14, 10))
    fig_ss.patch.set_facecolor(GRAPH_BG)

    # --- plot each panel separately (no loop, no variable collision) ---
    # TOP: Steps
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

    # BOTTOM: Sleep
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

# ---------------------------------------------------
# CLUSTERING  (one single block — labels/features stay in scope for everything below)
# ---------------------------------------------------

if daily is not None:
    st.markdown('<div class="section-header">⚙ Clustering Analysis</div>', unsafe_allow_html=True)

    feat_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes"]
    features  = daily[feat_cols]
    X         = StandardScaler().fit_transform(features)

    # Elbow curve
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

    # Fit final models
    kmeans        = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels        = kmeans.fit_predict(X)
    pca_model     = PCA(n_components=2)
    X_pca         = pca_model.fit_transform(X)
    var_exp       = pca_model.explained_variance_ratio_
    db            = DBSCAN(eps=eps, min_samples=3)
    db_labels     = db.fit_predict(X)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise       = int((db_labels == -1).sum())

    # KMeans PCA  +  DBSCAN PCA side by side
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

    # t-SNE
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

    # Cluster Profiles
    st.markdown('<div class="section-header">📊 Cluster Profiles</div>', unsafe_allow_html=True)
    cluster_df = features.copy()
    cluster_df["cluster"] = labels
    profile    = cluster_df.groupby("cluster").mean()

    fig_cp, ax_cp = plt.subplots(figsize=(12, 6))
    fig_cp.patch.set_facecolor(GRAPH_BG)
    ax_cp.set_facecolor(GRAPH_INNER)
    x_pos      = np.arange(len(profile.columns))
    bar_width  = 0.8 / len(profile)
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

    # Cluster Behavior Insights  ← INSIDE the if daily block
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

    # Pipeline Summary  ← INSIDE the if daily block
    st.markdown('<div class="section-header">✅ Pipeline Summary</div>', unsafe_allow_html=True)
    st.success("✔  **Data Loading** — Fitbit multi-file dataset detected and processed")
    st.success("✔  **TSFresh Feature Extraction** — Statistical heart-rate features computed per user")
    st.success("✔  **Prophet Forecast** — Heart rate, steps and sleep trend forecasted (30-day horizon)")
    st.success("✔  **KMeans Clustering** — Users segmented into activity profiles")
    st.success("✔  **DBSCAN** — Density-based clustering with noise detection applied")
    st.success("✔  **PCA** — 2D projection of high-dimensional features visualized")
    st.success("✔  **t-SNE** — Non-linear manifold projection of activity clusters rendered")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("""
<div style='text-align:center; padding: 40px 20px 20px 20px; opacity: 0.5; font-size: 0.85rem;'>
    🏃🏻 FitPulse AI Health Analytics Dashboard &nbsp;·&nbsp;
    Pipeline: TSFresh → Prophet → KMeans → DBSCAN → PCA → t-SNE &nbsp;·&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)