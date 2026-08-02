"""
BSNL Telecom Customer Intelligence Dashboard
=============================================
A premium Streamlit dashboard for exploring telecom customer data,
running K-Means clustering, and evaluating cluster quality with
Silhouette Analysis.

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="BSNL Customer Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# PREMIUM CSS
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 0%, #16202c 0%, #0b0f16 55%, #05070a 100%);
}

/* Hero header */
.hero {
    padding: 2rem 2.2rem;
    border-radius: 20px;
    background: linear-gradient(120deg, rgba(0,217,192,0.16), rgba(88,101,242,0.10));
    border: 1px solid rgba(0,217,192,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00D9C0, #6EE7F9 60%, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero p {
    color: #9AA5B1;
    font-size: 0.98rem;
    margin: 0;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.045), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    transition: transform 0.15s ease, border 0.15s ease;
}
.kpi-card:hover { transform: translateY(-3px); border: 1px solid rgba(0,217,192,0.45); }
.kpi-label { color: #8A94A3; font-size: 0.78rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { color: #F0F2F6; font-size: 1.7rem; font-weight: 700; margin-top: 0.15rem; }
.kpi-delta { font-size: 0.8rem; font-weight: 500; margin-top: 0.25rem; }

/* Section title */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #E8EBF0;
    margin: 1.2rem 0 0.6rem 0;
    border-left: 4px solid #00D9C0;
    padding-left: 0.6rem;
}

/* Glass panel */
.glass-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(0,217,192,0.15);
    color: #00D9C0;
    border: 1px solid rgba(0,217,192,0.35);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d131c 0%, #090c11 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px 10px 0 0;
    padding: 0.5rem 1rem;
    color: #9AA5B1;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,217,192,0.14) !important;
    color: #00D9C0 !important;
    font-weight: 600;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# HERO
# --------------------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📡 BSNL Customer Intelligence Dashboard</h1>
    <p>Upload any telecom customer dataset to explore churn patterns, segment customers with K-Means clustering,
    and validate cluster quality with Silhouette Analysis — all in one premium view.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# SIDEBAR — DATA SOURCE
# --------------------------------------------------------------------------------------
st.sidebar.markdown("### 📂 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Upload your own customer dataset. If you skip this, the bundled BSNL sample dataset is used.",
)

use_sample = False
if uploaded_file is None:
    use_sample = st.sidebar.checkbox("Use bundled BSNL sample dataset", value=True)

@st.cache_data(show_spinner=False)
def load_data(file, is_sample_path=None):
    if is_sample_path is not None:
        return pd.read_excel(is_sample_path)
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

df = None
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.sidebar.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows × {df.shape[1]} cols")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
elif use_sample:
    df = load_data(None, is_sample_path="sample_data/BSNL_Telecom_Customer_Dataset.xlsx")
    st.sidebar.info(f"Using sample dataset — {df.shape[0]} rows × {df.shape[1]} cols")

if df is None:
    st.warning("👈 Upload a CSV/Excel file or enable the sample dataset from the sidebar to begin.")
    st.stop()

df_original = df.copy()

# --------------------------------------------------------------------------------------
# BASIC CLEANING
# --------------------------------------------------------------------------------------
# Coerce object columns that look numeric (common issue: TotalCharges as string)
for col in df.columns:
    if df[col].dtype == object:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() / max(len(df), 1) > 0.9:
            df[col] = coerced

numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()

# --------------------------------------------------------------------------------------
# KPI ROW
# --------------------------------------------------------------------------------------
churn_col = next((c for c in df.columns if c.lower() == "churn"), None)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Customers</div>
    <div class="kpi-value">{len(df):,}</div></div>""", unsafe_allow_html=True)
with k2:
    missing_pct = df.isna().mean().mean() * 100
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Missing Data</div>
    <div class="kpi-value">{missing_pct:.1f}%</div></div>""", unsafe_allow_html=True)
with k3:
    if churn_col:
        churn_rate = (df[churn_col].astype(str).str.lower() == "yes").mean() * 100
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Churn Rate</div>
        <div class="kpi-value">{churn_rate:.1f}%</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Numeric Features</div>
        <div class="kpi-value">{len(numeric_cols_all)}</div></div>""", unsafe_allow_html=True)
with k4:
    if "MonthlyCharges" in df.columns:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Avg Monthly Charges</div>
        <div class="kpi-value">₹{df['MonthlyCharges'].mean():.0f}</div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Categorical Features</div>
        <div class="kpi-value">{len(categorical_cols_all)}</div></div>""", unsafe_allow_html=True)

st.write("")

# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tab_overview, tab_eda, tab_cluster, tab_profile, tab_explore = st.tabs(
    ["🔎 Overview", "📊 EDA", "🧬 Clustering & Silhouette", "🧩 Cluster Profiles", "🗂️ Data Explorer"]
)

# --------------------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# --------------------------------------------------------------------------------------
with tab_overview:
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Column Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
        })
        st.dataframe(summary, use_container_width=True, height=380)
    with c2:
        st.markdown('<div class="section-title">Numeric Summary Statistics</div>', unsafe_allow_html=True)
        if numeric_cols_all:
            st.dataframe(df[numeric_cols_all].describe().T.round(2), use_container_width=True, height=380)
        else:
            st.info("No numeric columns detected.")

# --------------------------------------------------------------------------------------
# TAB 2 — EDA
# --------------------------------------------------------------------------------------
with tab_eda:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    plot_theme = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        if churn_col:
            churn_counts = df[churn_col].value_counts().reset_index()
            churn_counts.columns = [churn_col, "count"]
            fig = px.pie(churn_counts, names=churn_col, values="count", hole=0.55,
                         color_discrete_sequence=["#00D9C0", "#A78BFA", "#F97373"],
                         title="Churn Distribution")
            fig.update_layout(**plot_theme)
            st.plotly_chart(fig, use_container_width=True)
        elif categorical_cols_all:
            col_choice = st.selectbox("Categorical column", categorical_cols_all, key="cat_pie")
            vc = df[col_choice].value_counts().reset_index()
            vc.columns = [col_choice, "count"]
            fig = px.pie(vc, names=col_choice, values="count", hole=0.55, title=f"{col_choice} Distribution")
            fig.update_layout(**plot_theme)
            st.plotly_chart(fig, use_container_width=True)

    with row1c2:
        if numeric_cols_all:
            num_choice = st.selectbox("Numeric column", numeric_cols_all,
                                       index=numeric_cols_all.index("Tenure_Months") if "Tenure_Months" in numeric_cols_all else 0,
                                       key="num_hist")
            fig = px.histogram(df, x=num_choice, nbins=30, color=churn_col if churn_col else None,
                                color_discrete_sequence=["#00D9C0", "#F97373"],
                                title=f"Distribution of {num_choice}")
            fig.update_layout(**plot_theme)
            st.plotly_chart(fig, use_container_width=True)

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            fig = px.scatter(df, x="MonthlyCharges", y="TotalCharges",
                              color=churn_col if churn_col else None,
                              color_discrete_sequence=["#00D9C0", "#F97373"],
                              opacity=0.75, title="Monthly Charges vs Total Charges")
            fig.update_layout(**plot_theme)
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols_all) >= 2:
            xcol = st.selectbox("X axis", numeric_cols_all, index=0, key="scatter_x")
            ycol = st.selectbox("Y axis", numeric_cols_all, index=1, key="scatter_y")
            fig = px.scatter(df, x=xcol, y=ycol, color=churn_col if churn_col else None, title=f"{xcol} vs {ycol}")
            fig.update_layout(**plot_theme)
            st.plotly_chart(fig, use_container_width=True)

    with row2c2:
        cat_for_bar = next((c for c in ["Contract", "InternetService", "PaymentMethod"] if c in df.columns), None)
        if cat_for_bar is None and categorical_cols_all:
            cat_for_bar = categorical_cols_all[0]
        if cat_for_bar:
            if churn_col:
                grp = df.groupby([cat_for_bar, churn_col]).size().reset_index(name="count")
                fig = px.bar(grp, x=cat_for_bar, y="count", color=churn_col, barmode="group",
                             color_discrete_sequence=["#00D9C0", "#F97373"],
                             title=f"{cat_for_bar} vs Churn")
            else:
                grp = df[cat_for_bar].value_counts().reset_index()
                grp.columns = [cat_for_bar, "count"]
                fig = px.bar(grp, x=cat_for_bar, y="count", title=f"{cat_for_bar} Counts")
            fig.update_layout(**plot_theme)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    if len(numeric_cols_all) >= 2:
        corr = df[numeric_cols_all].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Teal", aspect="auto")
        fig.update_layout(**plot_theme)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for a correlation heatmap.")

# --------------------------------------------------------------------------------------
# TAB 3 — CLUSTERING & SILHOUETTE
# --------------------------------------------------------------------------------------
with tab_cluster:
    st.markdown('<div class="section-title">Feature Selection for Clustering</div>', unsafe_allow_html=True)

    default_feats = [c for c in ["Tenure_Months", "MonthlyCharges", "TotalCharges"] if c in numeric_cols_all]
    feature_cols = st.multiselect(
        "Select numeric features to use for clustering",
        options=numeric_cols_all,
        default=default_feats if default_feats else numeric_cols_all[:3],
    )
    include_cat = st.multiselect(
        "Optionally include encoded categorical features",
        options=[c for c in categorical_cols_all if c != churn_col and c.lower() != "customer_id"],
        default=[],
    )

    if len(feature_cols) + len(include_cat) < 2:
        st.warning("Select at least 2 features (numeric and/or categorical) to run clustering.")
        st.stop()

    cluster_df = df[feature_cols].copy()
    for c in include_cat:
        le = LabelEncoder()
        cluster_df[c] = le.fit_transform(df[c].astype(str))

    cluster_df = cluster_df.dropna()
    valid_index = cluster_df.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    st.markdown('<div class="section-title">Choosing the Right Number of Clusters</div>', unsafe_allow_html=True)
    max_k = min(10, max(3, len(cluster_df) - 1))
    k_range = list(range(2, max_k + 1))

    @st.cache_data(show_spinner=False)
    def compute_k_metrics(X, k_range):
        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X, labels))
        return inertias, sil_scores

    inertias, sil_scores = compute_k_metrics(X_scaled, k_range)

    colA, colB = st.columns(2)
    with colA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=inertias, mode="lines+markers",
                                  line=dict(color="#00D9C0", width=3), marker=dict(size=8)))
        fig.update_layout(title="Elbow Method (Inertia vs k)", xaxis_title="Number of Clusters (k)",
                           yaxis_title="Inertia", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=k_range, y=sil_scores, mode="lines+markers",
                                  line=dict(color="#A78BFA", width=3), marker=dict(size=8)))
        best_k_auto = k_range[int(np.argmax(sil_scores))]
        fig.add_vline(x=best_k_auto, line_dash="dash", line_color="#F97373",
                      annotation_text=f"Best k={best_k_auto}", annotation_font_color="#F97373")
        fig.update_layout(title="Silhouette Score vs k", xaxis_title="Number of Clusters (k)",
                           yaxis_title="Silhouette Score", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Run K-Means</div>', unsafe_allow_html=True)
    chosen_k = st.slider("Number of clusters (k)", min_value=2, max_value=max_k,
                          value=int(best_k_auto), help="Auto-suggested k is pre-selected based on the best silhouette score.")

    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    overall_sil = silhouette_score(X_scaled, cluster_labels)
    sample_sil_vals = silhouette_samples(X_scaled, cluster_labels)

    df.loc[valid_index, "Cluster"] = cluster_labels
    df["Cluster"] = df["Cluster"].astype("Int64")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Selected k</div>
        <div class="kpi-value">{chosen_k}</div></div>""", unsafe_allow_html=True)
    with m2:
        quality = "Excellent" if overall_sil > 0.5 else "Good" if overall_sil > 0.25 else "Weak"
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Silhouette Score</div>
        <div class="kpi-value">{overall_sil:.3f}</div>
        <div class="kpi-delta"><span class="badge">{quality}</span></div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Customers Clustered</div>
        <div class="kpi-value">{len(cluster_df):,}</div></div>""", unsafe_allow_html=True)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("**PCA Projection of Clusters (2D)**")
        if X_scaled.shape[1] >= 2:
            pca = PCA(n_components=2, random_state=42)
            pcs = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
            pca_df["Cluster"] = cluster_labels.astype(str)
            fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                              color_discrete_sequence=px.colors.qualitative.Set2,
                              title=f"PCA View — explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 features for a PCA scatter plot.")

    with colD:
        st.markdown("**Silhouette Plot (per-sample scores)**")
        order = np.lexsort((-sample_sil_vals, cluster_labels))
        sorted_vals = sample_sil_vals[order]
        sorted_labels = cluster_labels[order]

        fig = go.Figure()
        y_lower = 0
        palette = px.colors.qualitative.Set2
        for i in range(chosen_k):
            vals_i = sorted_vals[sorted_labels == i]
            y_upper = y_lower + len(vals_i)
            fig.add_trace(go.Bar(
                x=vals_i, y=list(range(y_lower, y_upper)), orientation="h",
                marker=dict(color=palette[i % len(palette)]), name=f"Cluster {i}",
                width=1, showlegend=True
            ))
            y_lower = y_upper + 5
        fig.add_vline(x=overall_sil, line_dash="dash", line_color="#F97373",
                      annotation_text=f"Avg = {overall_sil:.3f}")
        fig.update_layout(title="Silhouette Coefficient per Sample", xaxis_title="Silhouette Coefficient",
                           yaxis_title="Samples (grouped by cluster)", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "⬇️ Download Clustered Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="clustered_customers.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------------------
# TAB 4 — CLUSTER PROFILES
# --------------------------------------------------------------------------------------
with tab_profile:
    if "Cluster" not in df.columns or df["Cluster"].isna().all():
        st.info("Run clustering in the previous tab first to see cluster profiles.")
    else:
        st.markdown('<div class="section-title">Cluster Sizes</div>', unsafe_allow_html=True)
        size_df = df["Cluster"].value_counts().sort_index().reset_index()
        size_df.columns = ["Cluster", "Count"]
        fig = px.bar(size_df, x="Cluster", y="Count", color="Cluster",
                     color_discrete_sequence=px.colors.qualitative.Set2, title="Customers per Cluster")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Average Feature Values per Cluster</div>', unsafe_allow_html=True)
        profile_cols = [c for c in numeric_cols_all if c in df.columns]
        if profile_cols:
            profile = df.groupby("Cluster")[profile_cols].mean().round(2)
            st.dataframe(profile, use_container_width=True)

            fig = px.imshow(profile.T, text_auto=".1f", color_continuous_scale="Tealgrn", aspect="auto",
                             title="Cluster Feature Heatmap")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        if churn_col:
            st.markdown('<div class="section-title">Churn Rate by Cluster</div>', unsafe_allow_html=True)
            churn_rate_cluster = (
                df.groupby("Cluster")[churn_col]
                .apply(lambda s: (s.astype(str).str.lower() == "yes").mean() * 100)
                .reset_index(name="Churn Rate (%)")
            )
            fig = px.bar(churn_rate_cluster, x="Cluster", y="Churn Rate (%)", color="Cluster",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# TAB 5 — DATA EXPLORER
# --------------------------------------------------------------------------------------
with tab_explore:
    st.markdown('<div class="section-title">Filter & Explore</div>', unsafe_allow_html=True)
    filt_col1, filt_col2 = st.columns(2)
    filtered = df.copy()

    with filt_col1:
        if churn_col and churn_col in filtered.columns:
            churn_vals = st.multiselect(f"Filter by {churn_col}", options=sorted(filtered[churn_col].dropna().unique().tolist()))
            if churn_vals:
                filtered = filtered[filtered[churn_col].isin(churn_vals)]
    with filt_col2:
        if "Cluster" in filtered.columns and filtered["Cluster"].notna().any():
            cluster_vals = st.multiselect("Filter by Cluster", options=sorted(filtered["Cluster"].dropna().unique().tolist()))
            if cluster_vals:
                filtered = filtered[filtered["Cluster"].isin(cluster_vals)]

    st.dataframe(filtered, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Built with Streamlit · scikit-learn K-Means · Silhouette Analysis · Plotly")
