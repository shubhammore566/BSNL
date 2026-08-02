# 📡 BSNL Customer Intelligence Dashboard (Streamlit)

Premium Streamlit dashboard for telecom customer data — file upload, EDA, K-Means
clustering, Silhouette Score analysis, and cluster profiling.

## ✨ Features
- **Any file upload** — CSV / Excel (.xlsx, .xls) apni dataset upload karo, ya bundled
  BSNL sample dataset use karo.
- **Premium dark UI** — glassmorphism cards, gradient headers, custom theme.
- **EDA tab** — churn distribution, histograms, scatter plots, correlation heatmap.
- **Clustering tab**:
  - Feature selection (numeric + optional encoded categorical)
  - Elbow method (inertia vs k)
  - **Silhouette Score vs k** line chart with auto-suggested best k
  - K-Means clustering with adjustable k slider
  - PCA 2D scatter plot of clusters
  - **Per-sample Silhouette plot** (like scikit-learn's classic silhouette diagram)
- **Cluster Profiles tab** — cluster sizes, average feature values per cluster, churn
  rate per cluster.
- **Data Explorer tab** — filter by churn/cluster, download filtered/clustered CSV.

## 🚀 How to Run (Setup Steps)

1. **Extract the zip** and open a terminal inside the folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. Browser mein automatically khul jayega: `http://localhost:8501`
   Agar nahi khule to manually us URL ko open karo.

## 📁 Project Structure
```
bsnl_dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .streamlit/
│   └── config.toml           # Premium dark theme config
└── sample_data/
    └── BSNL_Telecom_Customer_Dataset.xlsx   # Bundled sample dataset
```

## 🌐 Deploy Online (Free — Streamlit Community Cloud)
1. Is folder ko GitHub repo mein push karo.
2. [share.streamlit.io](https://share.streamlit.io) par jao, GitHub se login karo.
3. "New app" → apna repo select karo → main file `app.py` select karo → Deploy.
4. Kuch minutes mein live link mil jayega jo kisi ke saath bhi share kar sakte ho.

## 🛠️ Tech Stack
- Streamlit (UI)
- Pandas / NumPy (data processing)
- scikit-learn (K-Means, PCA, Silhouette Score)
- Plotly (interactive charts)

## 📝 Notes
- Agar apni khud ki file upload karte ho, to app automatically numeric aur
  categorical columns detect kar leta hai — koi column names hardcoded nahi hain
  except churn detection ("Churn" column) jo optional hai.
- Silhouette Score range: **-1 to 1**. Jitna zyada (0.5+) utna behtar cluster
  separation hota hai.
