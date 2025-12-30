import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Simple Linear Regression",
    page_icon="📈",
    layout="centered"
)

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h1>📈 Simple Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> based on <b>Total Bill</b> using a Linear Regression model.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# --------------------------------------------------
# Prepare Data (Simple Linear Regression)
# --------------------------------------------------
X = df[["total_bill"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.markdown(f"""
<div class="card">
    <h3>📊 Model Performance Metrics</h3>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{mae:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{rmse:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{r2:.2f}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h3>📈 Prediction vs Actual</h3>
</div>
""", unsafe_allow_html=True)

fig, ax = plt.subplots()
ax.scatter(X_test, y_test, alpha=0.6, label="Actual Data")
ax.plot(X_test, y_pred, linewidth=2, label="Regression Line")
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")
ax.legend()
st.pyplot(fig)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h3>🧮 Predict Tip Amount</h3>
    <p>Adjust the slider to estimate the tip.</p>
</div>
""", unsafe_allow_html=True)

bill_amount = st.slider(
    "Total Bill Amount ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

predicted_tip = model.predict(np.array([[bill_amount]]))[0]

st.markdown(f"""
<div class="prediction-box">
    Predicted Tip Amount: ${predicted_tip:.2f}
</div>
""", unsafe_allow_html=True)
