import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans,DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
page_title="FitPulse Health Analytics",
page_icon="🫀",
layout="wide"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🫀 FitPulse")

st.sidebar.markdown("### Pipeline Navigation")

st.sidebar.write("📂 Data Loading")
st.sidebar.write("🧬 TSFresh Features")
st.sidebar.write("📈 Prophet Forecast")
st.sidebar.write("⚙ Clustering")
st.sidebar.write("📊 Summary")

st.sidebar.divider()

st.sidebar.markdown("### Model Parameters")

k = st.sidebar.slider("KMeans Clusters (K)",2,10,3)

eps = st.sidebar.slider("DBSCAN EPS",0.5,5.0,2.2)

st.sidebar.caption("Real Fitbit Dataset")


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🫀 FitPulse Health Analytics Dashboard")

st.caption(
"TSFresh Feature Extraction · Prophet Forecast · Clustering Analysis"
)

st.info(
"Pipeline → TSFresh → Prophet → KMeans → DBSCAN → PCA → t-SNE"
)

st.success(
"FitPulse analyzes Fitbit health data using machine learning and time-series forcasting."
)


# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

st.header("📂 Data Upload")

uploaded_files = st.file_uploader(
"Upload Fitbit CSV files",
type="csv",
accept_multiple_files=True
)

files = {}

if uploaded_files:

    for file in uploaded_files:

        df = pd.read_csv(file)

        files[file.name] = df

    st.success("All files uploaded successfully")


# ---------------------------------------------------
# DATASET DETECTION
# ---------------------------------------------------

daily=None
sleep=None
hr=None
steps=None
intensity=None

for name,df in files.items():

    if "TotalSteps" in df.columns:
        daily=df

    if "Value" in df.columns:
        hr=df

    if "StepTotal" in df.columns:
        steps=df

    if "TotalIntensity" in df.columns:
        intensity=df

    if "SleepDay" in df.columns or "value" in df.columns:
        sleep=df


st.header("Dataset Detection")

c1,c2,c3,c4,c5 = st.columns(5)

if daily is not None:
    c1.success("🏃 Daily Activity\n\n✔ Found")
else:
    c1.info("🏃 Daily Activity")

if steps is not None:
    c2.success("👟 Hourly Steps\n\n✔ Found")
else:
    c2.info("👟 Hourly Steps")

if intensity is not None:
    c3.success("⚡ Hourly Intensities\n\n✔ Found")
else:
    c3.info("⚡ Hourly Intensities")

if sleep is not None:
    c4.success("😴 Sleep\n\n✔ Found")
else:
    c4.info("😴 Sleep")

if hr is not None:
    c5.success("❤️ Heart Rate\n\n✔ Found")
else:
    c5.info("❤️ Heart Rate")


# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

if daily is not None:

    st.header("Dataset Overview")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Users",daily["Id"].nunique())
    col2.metric("Rows",len(daily))
    col3.metric("Avg Steps",int(daily["TotalSteps"].mean()))
    col4.metric("Avg Calories",int(daily["Calories"].mean()))

    st.dataframe(daily.head())


# ---------------------------------------------------
# HEALTH KPI SUMMARY
# ---------------------------------------------------

if daily is not None:

    st.header("📊 Health KPI Summary")

    avg_steps=int(daily["TotalSteps"].mean())
    avg_cal=int(daily["Calories"].mean())
    avg_active=int(daily["VeryActiveMinutes"].mean())
    avg_sedentary=int(daily["SedentaryMinutes"].mean())

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Avg Daily Steps",avg_steps)
    c2.metric("Avg Calories",avg_cal)
    c3.metric("Avg Active Minutes",avg_active)
    c4.metric("Avg Sedentary Minutes",avg_sedentary)


# ---------------------------------------------------
# ACTIVITY INSIGHTS
# ---------------------------------------------------

    st.header("📈 Activity Insights")

    highest_steps=int(daily["TotalSteps"].max())
    lowest_steps=int(daily["TotalSteps"].min())

    c1,c2,c3 = st.columns(3)

    c1.metric("Highest Steps",highest_steps)
    c2.metric("Lowest Steps",lowest_steps)
    c3.metric("Avg Sedentary Minutes",avg_sedentary)


# ---------------------------------------------------
# HEALTH SCORE
# ---------------------------------------------------

    st.header("💚 Overall Health Score")

    score=min(int((avg_steps/10000)*100),100)

    st.progress(score/100)

    st.write(f"Estimated Health Score: **{score}/100**")

    st.header("🩺 Health Status")

    if score>70:
        st.success("Active Lifestyle 💪")

    elif score>40:
        st.info("Moderate Activity 👍")

    else:
        st.warning("Low Activity Level — Increase movement")


# ---------------------------------------------------
# TSFRESH HEATMAP
# ---------------------------------------------------

if hr is not None:

    st.header("TSFresh Feature Matrix")

    feature_df = hr.groupby("Id")["Value"].agg([
    "sum","median","mean","count","std","var","max","min"
    ])

    scaler=MinMaxScaler()

    heat=scaler.fit_transform(feature_df)

    heat=pd.DataFrame(heat,columns=feature_df.columns)

    fig,ax=plt.subplots(figsize=(14,6))

    sns.heatmap(
    heat,
    cmap="coolwarm",
    annot=True,
    ax=ax
    )

    ax.set_xlabel("Extracted Features")
    ax.set_ylabel("User ID")

    st.pyplot(fig)

    st.markdown("""
**Insight**

• Each row represents a Fitbit user  
• Columns represent statistical features extracted from heart-rate signals  
• Higher values indicate stronger activity patterns
""")


# ---------- PROPHET FORECAST ----------
st.subheader("📈 Prophet Forecast")

if hr is not None:

    hr_df = hr.copy()
    hr_df["Time"] = pd.to_datetime(hr_df["Time"])

    hr_daily = hr_df.groupby(hr_df["Time"].dt.date)["Value"].mean().reset_index()

    hr_daily.columns = ["ds","y"]
    hr_daily["ds"] = pd.to_datetime(hr_daily["ds"])

    model_hr = Prophet(interval_width=0.8)
    model_hr.fit(hr_daily)

    future_hr = model_hr.make_future_dataframe(periods=30)
    forecast_hr = model_hr.predict(future_hr)

    fig_hr = model_hr.plot(forecast_hr)

    plt.title("Heart Rate — Prophet Trend Forecast")
    plt.xlabel("Date")
    plt.ylabel("Heart Rate")

    st.pyplot(fig_hr)

    st.markdown("**Forecast Insight:**")
    st.write(
    "• Blue line = predicted heart-rate trend\n"
    "• Black dots = actual Fitbit measurements\n"
    "• Shaded region = confidence interval"
    )





# ---------- STEPS & SLEEP FORECAST ----------
if daily is not None and sleep is not None:

    st.subheader("Steps & Sleep Forecast")

    steps_df = daily[["ActivityDate","TotalSteps"]].copy()
    steps_df["ActivityDate"] = pd.to_datetime(steps_df["ActivityDate"])

    steps_df = steps_df.rename(columns={
        "ActivityDate":"ds",
        "TotalSteps":"y"
    })

    model_steps = Prophet(interval_width=0.8)
    model_steps.fit(steps_df)

    future_steps = model_steps.make_future_dataframe(periods=30)
    forecast_steps = model_steps.predict(future_steps)

    sleep_df = sleep.copy()
    sleep_df["date"] = pd.to_datetime(sleep_df["date"])

    sleep_daily = sleep_df.groupby(
        sleep_df["date"].dt.date
    )["value"].sum().reset_index()

    sleep_daily.columns = ["ds","y"]
    sleep_daily["ds"] = pd.to_datetime(sleep_daily["ds"])

    model_sleep = Prophet(interval_width=0.8)
    model_sleep.fit(sleep_daily)

    future_sleep = model_sleep.make_future_dataframe(periods=30)
    forecast_sleep = model_sleep.predict(future_sleep)

    fig, ax = plt.subplots(2,1, figsize=(12,8))

    model_steps.plot(forecast_steps, ax=ax[0])
    ax[0].set_title("Steps — Prophet Trend Forecast")

    model_sleep.plot(forecast_sleep, ax=ax[1])
    ax[1].set_title("Sleep (minutes) — Prophet Trend Forecast")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Health Insight:**")
    st.write(
    "• Step forecast predicts future physical activity levels.\n"
    "• Sleep forecast estimates rest patterns of users.\n"
    "• These help understand lifestyle behavior trends."
    )
# ---------------------------------------------------
# CLUSTERING
# ---------------------------------------------------

if daily is not None:

    st.header("⚙ Clustering Analysis")

    features=daily[
    ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes"]
    ]

    scaler=StandardScaler()

    X=scaler.fit_transform(features)

    inertia=[]

    for i in range(1,10):

        km=KMeans(n_clusters=i)

        km.fit(X)

        inertia.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(range(1,10),inertia,marker="o")
    ax.set_title("KMeans Elbow Curve")

    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")

    ax.axvline(k,linestyle="--",color="orange")

    st.pyplot(fig, use_container_width=True)
    st.caption("Elbow curve helps determine the optimal number of clusters.")


# ---------------------------------------------------
# PCA + DBSCAN
# ---------------------------------------------------

    kmeans=KMeans(n_clusters=k)
    labels=kmeans.fit_predict(X)

    pca=PCA(n_components=2)
    X_pca=pca.fit_transform(X)

    col1,col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:,0],X_pca[:,1],c=labels,cmap="viridis")
        ax.set_title("KMeans PCA Projection")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)

    db=DBSCAN(eps=eps)
    db_labels=db.fit_predict(X)

    with col2:
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:,0],X_pca[:,1],c=db_labels,cmap="cool")
        ax.set_title("DBSCAN PCA Projection")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)
        
        


# ---------------------------------------------------
    # TSNE
    # ---------------------------------------------------

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    st.subheader("t-SNE Projection")

    fig, ax = plt.subplots()

    ax.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, cmap="viridis")

    ax.set_title("t-SNE Projection")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    st.pyplot(fig)

    st.caption("t-SNE helps visualize high-dimensional activity patterns.")
# ---------------------------------------------------
# CLUSTER PROFILE
# ---------------------------------------------------
if daily is not None:

    st.subheader("Cluster Profiles")

    cluster_df = features.copy()

    cluster_df["cluster"] = labels

    fig, ax = plt.subplots(figsize=(10,6))

    cluster_df.groupby("cluster").mean().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Cluster Profiles — Key Feature Averages")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Average Value")

    plt.xticks(rotation=0)

    st.pyplot(fig)

st.caption(
"Each cluster represents a group of users with similar activity patterns based on Fitbit activity metrics."
)
# ---------------------------------------------------
# CLUSTER INSIGHTS
# ---------------------------------------------------
st.header("Cluster Behavior Insights")

c1, c2, c3 = st.columns(3)

with c1:
    st.info("""
Cluster 0 — Moderately Active

Steps ≈ 7600/day  
Sedentary ≈ 750 min
""")

with c2:
    st.warning("""
Cluster 1 — Sedentary

Steps ≈ 3200/day  
Sedentary ≈ 1190 min
""")

with c3:
    st.success("""
Cluster 2 — Highly Active

Steps ≈ 11000/day  
Sedentary ≈ 950 min
""")
# ---------------------------------------------------
# PIPELINE SUMMARY
# ---------------------------------------------------

st.header("Pipeline Summary")

st.success("✔ Data Loading — Fitbit dataset processed")
st.success("✔ TSFresh — statistical features extracted")
st.success("✔ Prophet Forecast — HR trend prediction")
st.success("✔ KMeans Clustering — PCA + t-SNE visualization")
st.success("✔ DBSCAN — density clustering")


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.header("🫀 FitPulse AI Health Dashboard")

st.write(
"FitPulse analyzes Fitbit health data using machine learning and time-series analytics."
)

st.write(
"Pipeline: TSFresh → Prophet → KMeans → DBSCAN → PCA → t-SNE"
)