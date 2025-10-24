
# Libraries import

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Artifacts

@st.cache_resource
def load_artifacts():
    artifacts_dir = "model_artifacts"

    # Load results
    results_path = os.path.join(artifacts_dir, "results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError("results.pkl not found. Please run training script first.")
    results = joblib.load(results_path)

    # pick best model based on R² score
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    model_file = f"{best_model_name}_model.pkl"
    model_path = os.path.join(artifacts_dir, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_file} not found in {artifacts_dir}. Please retrain models.")

    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
    training_columns = joblib.load(os.path.join(artifacts_dir, "training_columns.pkl"))

    return model, scaler, training_columns, best_model_name, results

model, scaler, training_columns, best_model_name, results = load_artifacts()

st.set_page_config(page_title="YouTube Monetization Modeler", layout="wide")

# Styling with CSS (start)
# ==========================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #111111;
    }
    .css-18e3th9, .css-1d391kg {
        color: #FFFFFF;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1E5A78;
    }
   .main-header {
    background-color: #111111;
    padding: 0px 0px;        
    border-radius: 8px;
    text-align: center;
    color: #FF0000;
    font-size: 28px;
    font-weight: bold;
    margin-top: -10px;       
    margin-bottom: 0px;      
}

    </style>
""", unsafe_allow_html=True)

# Custom Header
st.markdown('<div class="main-header">🎬 YouTube Monetization Modeler</div>', unsafe_allow_html=True)


st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #111111;
    }
    h1, h2, h3, h4, h5, h6 {
        color:#1E5A78; /* Accent green for headers */
    }
    /*Change label text (like "Views", "Likes", etc.) */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {
        color:#1E5A78 !important;  
        font-weight: bold;
    }
    /*Change small captions (like "Model used: ...") */
    .stCaption, .stMarkdown p, .stDateInput {
        color: #FFFFFF !important;  /* White */
    }
    /*Change form text (input values) */
    input, textarea, select {
        color: #FFFFFF !important;  
        background-color: #222222 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }

    /* Sidebar (navigation bar) */
    [data-testid="stSidebar"] {
        background-color: #202020;  
        color: #FFFFFF;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #1E5A78 !important; 
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1E5A78; 
    }

    /* Input labels */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {
        color:#1E5A78 !important;
        font-weight: bold;
    }

    /* Captions & small text */
    .stCaption, .stMarkdown p, .stDateInput {
        color: #FFFFFF !important;
    }

    /* Input fields */
    input, textarea, select {
        color: #FFFFFF !important;  
        background-color: #222222 !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    /* Prediction Button Style */
    div.stButton > button:first-child {
        background-color: #1E5A78;  
        color: #000000;              
        font-weight: bold;
        height: 50px;
        width: 100%;
        border-radius: 10px;
        font-size: 16px;
    }

    div.stButton > button:first-child:hover {
        background-color: #17a444;  /* Slightly darker green on hover */
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)






# 2. Sidebar Navigation

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Revenue", "Analytics Dashboard", "Model Insights"])

# 3. Predict Revenue Page

if page == "Predict Revenue":
    st.write("Enter video performance & context details to estimate potential ad revenue (USD).")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            views = st.number_input("Views", min_value=0.000000, value=10198.000000, step=1000.000000)
            likes = st.number_input("Likes", min_value=0.000000, value=2028.000000, step=100.000000)
            comments = st.number_input("Comments", min_value=0.000000, value=375.000000, step=10.000000)
            video_length = st.number_input("Video Length (minutes)", min_value=0.000000, value=5.067021, step=1.000000)
            subscribers = st.number_input("Subscribers", min_value=0.000000, value=528766.000000, step=5000.000000)
            watch_time = st.number_input("Watch Time (minutes)", min_value=0.000000, value=61157.975673, step=5000.000000)


        with col2:
            
            from datetime import date

            # Use st.date_input to get the full date from a calendar
            selected_date = st.date_input(
                "Select a date",
                value=date.today()
            )

            # the year, month, and day of the week.
            year = selected_date.year
            month = selected_date.month
            day = selected_date.day

            st.write(f"Selected Year: {year}")
            st.write(f"Selected Month: {month}")
            st.write(f"Selected Day: {day}")
            category = st.selectbox("Category", ["Education","Technology","Entertainment","Gaming","Music"])
            device = st.selectbox("Device", ["Mobile","Desktop","Tablet","TV"])
            country = st.selectbox("Country", ["US","UK","India","Canada","Australia"])

        submitted = st.form_submit_button("Predict Revenue")

    if submitted:
        # Build input dataframe
        input_df = pd.DataFrame({
            "views": [views],
            "likes": [likes],
            "comments": [comments],
            "watch_time_minutes": [watch_time],
            "video_length_minutes": [video_length],
            "subscribers": [subscribers],
            "year": [year],
            "month": [month],
            "day": [day],
            "category": [category],
            "device": [device],
            "country": [country]
        })

        # Derived features (same as ML training script)
        input_df["engagement_rate"] = (input_df["likes"] + input_df["comments"]) / (input_df["views"] if views > 0 else 1)
        input_df["average_watch_time"] = input_df["watch_time_minutes"] / (input_df["views"] if views > 0 else 1)
        input_df["subscriber_ratio"] = input_df["views"] / (input_df["subscribers"] if subscribers > 0 else 1)

        # One-hot encode and align with training
        input_encoded = pd.get_dummies(input_df, columns=["category","device","country"], drop_first=True, dtype=int)
        input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

        # Scale and predict
        input_scaled = scaler.transform(input_encoded)
        prediction_usd = float(model.predict(input_scaled)[0])

        

        st.success(f"Predicted Ad Revenue: **${prediction_usd:,.2f} USD**")
        st.caption(f"Model used: {best_model_name}")



# ==========================================================
# 4. Analytics Dashboard
# ==========================================================
elif page == "Analytics Dashboard":
    st.title("YouTube Revenue Analytics Dashboard")

    file_path = "youtube_ad_revenue_dataset.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        col1, col2 = st.columns(2)

        # ================= Revenue Distribution =================
        with col1.expander("Revenue Distribution", expanded=False):
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.histplot(df['ad_revenue_usd'], bins=40, kde=True, ax=ax, color="#1DB954")
            ax.set_xlabel("Revenue (USD)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            st.pyplot(fig)

        # ================= Category-wise Avg Revenue =================
        with col2.expander("Category-wise Avg Revenue", expanded=False):
            avg_rev = df.groupby("category")['ad_revenue_usd'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x=avg_rev.index, y=avg_rev.values, palette="viridis", ax=ax)
            ax.set_ylabel("Avg Revenue", fontsize=8)
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.tick_params(axis="y", labelsize=7)
            st.pyplot(fig)

        # ================= Correlation Heatmap =================
        with st.expander("Correlation Heatmap", expanded=False):
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f", ax=ax, cbar=False)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)
            st.pyplot(fig)

        # Arrange top 5 charts in same row
        col3, col4 = st.columns(2)

        # ================= Top 5 Highest Ad Revenue =================
        with col3.expander("Top 5 Highest Ad Revenues", expanded=False):
            top5 = df.nlargest(5, "ad_revenue_usd")[["video_id", "ad_revenue_usd", "category"]]
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x="ad_revenue_usd", y="video_id", data=top5, palette="crest", ax=ax)
            ax.set_xlabel("Revenue (USD)", fontsize=8)
            ax.set_ylabel("Video ID", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            st.pyplot(fig)

        # ================= Top 5 Lowest Ad Revenue =================
        with col4.expander("Top 5 Lowest Ad Revenues", expanded=False):
            bottom5 = df.nsmallest(5, "ad_revenue_usd")[["video_id", "ad_revenue_usd", "category"]]
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x="ad_revenue_usd", y="video_id", data=bottom5, palette="flare", ax=ax)
            ax.set_xlabel("Revenue (USD)", fontsize=8)
            ax.set_ylabel("Video ID", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            st.pyplot(fig)

    else:
        st.warning("Dataset not found. Place `youtube_ad_revenue_dataset.csv` in the project folder.")



# ==========================================================
# 5. Model Insights
# ==========================================================
elif page == "Model Insights":
    st.title("Model Insights & Performance")

    if results:
        results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

        # Ensure numeric values for formatting
        for col in ["MSE", "MAE", "R2"]:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

        # ================= Model Performance Table =================
        with st.expander("Model Performance Table", expanded=False):
            st.dataframe(
                results_df.style.format({
                    "MSE": "{:.2f}",
                    "MAE": "{:.2f}",
                    "R2": "{:.4f}"
                }),
                height=300
            )

        # ================= R² Bar Chart =================
        with st.expander("R² Scores", expanded=False):
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.bar(results_df['Model'], results_df['R2'], color='#1DB954')
            ax.set_ylabel('R²', fontsize=8)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)

        # ================= Feature Importance =================
        with st.expander("Top 5 Important Features", expanded=False):
            if hasattr(model, "feature_importances_"):
                feat_imp = pd.DataFrame({"Feature": training_columns, "Importance": model.feature_importances_})
            elif hasattr(model, "coef_"):
                feat_imp = pd.DataFrame({"Feature": training_columns, "Importance": np.abs(model.coef_)})

            if 'feat_imp' in locals():
                feat_imp = feat_imp.sort_values(by="Importance", ascending=False).head(5)
                fig, ax = plt.subplots(figsize=(4, 2))
                sns.barplot(x="Importance", y="Feature", data=feat_imp, palette="mako", ax=ax)
                ax.set_xlabel("Importance", fontsize=8)
                ax.set_ylabel("Feature", fontsize=8)
                ax.tick_params(axis="x", labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
                st.pyplot(fig)

    else:
        st.warning("No results found. Please run training first.")
