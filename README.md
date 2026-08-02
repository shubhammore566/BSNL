# 📊 BSNL Telecom Customer Segmentation Dashboard

A premium Streamlit dashboard for telecom customer segmentation using Machine Learning. The application enables users to upload telecom datasets, perform Exploratory Data Analysis (EDA), build customer segments using K-Means clustering, evaluate cluster quality with the Silhouette Score, and generate meaningful business insights through interactive visualizations.

---

## ✨ Features

- 📁 Upload datasets in **CSV**, **XLSX**, or **XLS** format
- 📊 Interactive Exploratory Data Analysis (EDA)
- 📈 Histograms, scatter plots, and correlation heatmaps
- 🤖 Customer segmentation using **K-Means Clustering**
- 📉 Elbow Method to determine the optimal number of clusters
- 📏 Silhouette Score analysis for cluster evaluation
- 🌐 PCA-based 2D cluster visualization
- 📋 Cluster profiling with average feature values
- 🔍 Filter customers by cluster or churn status
- 📥 Download the processed dataset
- 🎨 Modern dark-themed Streamlit interface

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- OpenPyXL

---

## 📂 Project Structure

```
BSNL/
│── app.py
│── requirements.txt
│── README.md
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│
└── sample_data/
    └── BSNL_Telecom_Customer_Dataset.xlsx
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shubhammore566/BSNL.git
cd BSNL
```

### 2. Create a Virtual Environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

## 📊 Dashboard Modules

- Home
- Dataset Upload
- Exploratory Data Analysis (EDA)
- Customer Segmentation
- Cluster Visualization
- Cluster Profiles
- Data Explorer

---

## 🤖 Machine Learning Workflow

```
Upload Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Selection
      │
      ▼
K-Means Clustering
      │
      ▼
Elbow Method
      │
      ▼
Silhouette Analysis
      │
      ▼
PCA Visualization
      │
      ▼
Cluster Profiling
```

---

## 🎯 Business Applications

This project can be used for:

- Customer Segmentation
- Churn Analysis
- Customer Behavior Analysis
- Marketing Campaigns
- Customer Retention
- Business Intelligence
- Telecom Analytics

---

## 🌐 Deployment

The application can be deployed easily using **Streamlit Community Cloud**.

1. Push the project to GitHub.
2. Connect your GitHub repository to Streamlit Community Cloud.
3. Select **app.py** as the main application file.
4. Click **Deploy**.

---

## 📌 Future Improvements

- Support for additional clustering algorithms (DBSCAN, Hierarchical Clustering)
- Advanced customer analytics
- Model performance comparison
- Export reports in PDF format
- Real-time dashboard integration

---

## 👨‍💻 Author

**Shubham More**

**B.Tech in Artificial Intelligence & Machine Learning**

GitHub: **https://github.com/shubhammore566**

---

⭐ **If you found this project useful, consider giving it a star on GitHub!**
