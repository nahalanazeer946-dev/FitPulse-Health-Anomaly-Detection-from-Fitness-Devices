import streamlit as st
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FitPulse | Data Collection & Pre-Processing",
    page_icon="🏃",
    layout="wide"
)

# ---------------- BEAUTIFUL BACKGROUND & STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #fdfbfb, #ebedee);
}
.hero {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    color: white;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
}
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
}
.metric {
    background: linear-gradient(135deg, #43cea2, #185a9d);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
<h1>🏃 FitPulse</h1>
<h3>Data Collection & Pre-Processing</h3>
<p>Upload → Analyze → Clean → Verify</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel / JSON",
    type=["csv", "xlsx", "json"]
)

# ---------------- MAIN LOGIC (UNCHANGED) ----------------
if uploaded_file is not None:

    # ---- Load dataset (same logic) ----
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    st.success("✅ Dataset uploaded successfully")

    # ---------------- DATASET OVERVIEW (NEW, SAFE ADDITION) ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Rows", df.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Columns", df.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.metric("Total Missing", int(df.isnull().sum().sum()))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- ORIGINAL DATA (UNCHANGED) ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Original Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- MISSING VALUES (UNCHANGED) ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("❗ Missing Values Count")
    missing = df.isnull().sum()
    st.dataframe(missing, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- DATA CLEANING (UNCHANGED LOGIC) ----------------
    df_clean = df.copy()

    # Numeric columns → mean
    num_cols = df_clean.select_dtypes(include=["number"]).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())

    # Categorical columns → mode
    cat_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    st.success("✅ Data preprocessing completed successfully")

    # ---------------- CLEANED DATA (UNCHANGED) ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧹 Cleaned Dataset Preview")
    st.dataframe(df_clean.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- VERIFICATION (UNCHANGED) ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✔ Missing Values After Cleaning")
    st.dataframe(df_clean.isnull().sum(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("⬅️ Upload a dataset from the sidebar to begin")