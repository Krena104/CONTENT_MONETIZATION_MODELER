# train.py — produce Linear Regression artifacts with exact names
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===== 1) Load your data =====
# Replace with your real CSV if you have it:
# df = pd.read_csv("youtube_ad_revenue_dataset.csv")
# X = df[["views","likes","comments"]]  # adjust columns
# y = df["revenue"]                      # adjust target

# TEMP demo data (remove when you use your CSV)
X = pd.DataFrame({
    "views":    np.random.randint(1000, 100000, 500),
    "likes":    np.random.randint(100, 10000, 500),
    "comments": np.random.randint(10, 1000, 500),
})
y = 0.5*X["views"] + 0.3*X["likes"] + 0.2*X["comments"] + np.random.randn(500)*100

# ===== 2) Split + scale =====
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# ===== 3) Train Linear Regression =====
model = LinearRegression()
model.fit(Xtr_s, ytr)
r2 = r2_score(yte, model.predict(Xte_s))

# ===== 4) Save with EXACT names =====
base = Path(__file__).resolve().parent
art = base / "model_artifacts"
art.mkdir(exist_ok=True)

model_name = "Linear Regression"  # keep the space to match your screenshot
joblib.dump({model_name: {"R2": float(r2)}}, art / "results.pkl")
joblib.dump(model,   art / f"{model_name}_model.pkl")  # -> "Linear Regression_model.pkl"
joblib.dump(scaler,  art / "scaler.pkl")
joblib.dump(list(X.columns), art / "training_columns.pkl")

print("✅ Saved artifacts to:", art)
print("Best model:", model_name, "| R²:", r2)
