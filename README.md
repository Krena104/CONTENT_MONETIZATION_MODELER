🎬 YouTube Monetization Modeler
End-to-End Machine Learning & Streamlit Application
📌 Overview

This project builds a complete end-to-end ML pipeline for predicting YouTube video ad revenue.
It cleans, processes, models, and visualizes YouTube video data — enabling creators and analysts to estimate potential revenue and gain data-driven insights.

The project demonstrates practical skills in:

Data cleaning and feature engineering

Regression modeling

Model evaluation and selection

Building interactive web apps with Streamlit

🧩 Business Use Cases

💰 Revenue Estimation: Predict ad revenue using engagement metrics (views, likes, comments, etc.)

📈 Content Strategy: Identify which content types generate higher revenue

📊 Performance Insights: Compare predicted vs actual revenue to optimize video strategy

📉 Dashboard Analytics: Visualize trends and correlations in viewer behavior

🤖 Model Insights: Select the best regression model automatically using R² score

⚙️ Tech Stack
Category	Tools Used
Language	Python
Libraries	Pandas, NumPy, Matplotlib, Seaborn
Machine Learning	Scikit-learn (Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting)
Model Storage	Joblib
Web Framework	Streamlit
IDE	VS Code
🧹 Data Cleaning & Preprocessing

Handled missing data → mean imputation for numerical values.

Removed duplicates for consistent data quality.

Converted date columns → extracted year, month, day, day_of_week.

Engineered new features:

engagement_rate = (likes + comments) / views

average_watch_time = watch_time_minutes / views

subscriber_ratio = views / subscribers

Encoded categorical variables: category, device, country.

⚡ Models Trained

Linear Regression

Ridge Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

✅ The model with the highest R² score is automatically selected as the best model.

💾 Saved Artifacts

All trained models and preprocessing files are stored in the model_artifacts/ folder.

File Name	Description
Linear Regression_model.pkl	Best performing trained model
scaler.pkl	StandardScaler used for feature scaling
training_columns.pkl	Feature column list used for model training
results.pkl	Performance metrics (MSE, MAE, R²) for all models
📊 Streamlit App Features
1️⃣ Predict Revenue

Input key metrics like views, likes, comments, watch time, subscribers, etc.

Get predicted ad revenue (USD) instantly.

2️⃣ Analytics Dashboard

Visualize revenue distribution & correlation heatmaps

Explore category-wise revenue averages

Identify top & bottom 5 videos by revenue

3️⃣ Model Insights

Compare model metrics (MSE, MAE, R²)

Visualize feature importance

View which regression model performed best

🔄 Project Workflow
Load dataset  →  Clean & preprocess data  
→  Train ML models  →  Save artifacts  
→  Launch Streamlit app  →  Predict & visualize insights

📁 File Structure
YOUTUBE_MONETIZATION_MODELER/
│
├── app.py
├── clean_data.py
├── train.py
├── youtube_ad_revenue_dataset.csv
├── requirements.txt
│
└── model_artifacts/
    ├── Linear Regression_model.pkl
    ├── results.pkl
    ├── scaler.pkl
    └── training_columns.pkl
│
└── Screenshots/
    ├── Screenshot_1.png
    ├── Screenshot_2.png
    └── ...

▶️ How to Run
🧩 Step 1 — Install dependencies
pip install -r requirements.txt

🧠 Step 2 — Train and save model artifacts
python clean_data.py

🖥️ Step 3 — Launch Streamlit web app
streamlit run app.py


Then open the URL shown in your terminal (e.g., http://localhost:8501).

📚 Dataset Info

Source: Synthetic dataset created by [GUVI]
Description: Contains YouTube video performance metrics and corresponding ad revenue.

Column	Description
video_id	Unique video identifier
date	Upload date
category	Video category
device	Viewer’s device type
country	Viewer country
views, likes, comments	Engagement metrics
watch_time_minutes	Total watch time
video_length_minutes	Video duration
subscribers	Channel subscriber count
ad_revenue_usd	Target variable (revenue in USD)
🧭 Key Takeaways

Demonstrates a full ML lifecycle: data preprocessing → model training → deployment.

Integrates Streamlit for real-time, interactive prediction dashboards.

Useful for content creators, data analysts, and digital marketers aiming to optimize video monetization strategies.
