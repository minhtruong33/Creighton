import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
 
# Load trained model
model_path = "model_test.pkl"
try:
    stacking_model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"Error: '{model_path}' not found.")
    st.stop()
 
# Dataset path
file_path = "model_test.csv"
 
@st.cache_data
def load_data():
    df = pd.read_csv(file_path)
    if df.iloc[:, 0].dtype in [int, float]:
        df.set_index(df.columns[0], inplace=True)
    df.index = df.index.astype(int) 
    return df
 
# Load original data into session state on first run
if "data" not in st.session_state:
    st.session_state["data"] = load_data()
    st.session_state["modified_data"] = st.session_state["data"].copy()
    st.session_state["original_data"] = st.session_state["data"].copy()  # Save the original version
 
df = st.session_state["modified_data"]
 
if df.empty:
    st.error("Dataset is empty or could not be loaded.")
    st.stop()
 
# Center the title using Markdown in Streamlit & HTML
st.markdown(
    "<h2 style='text-align: center;'> US News Ranking Projector </h2>",
    unsafe_allow_html=True
)
 
# Creighton record
selected_row = 123
 
# Get Predicted Rank 2026 & Actual Rank 2025 (Overall Rank) for Creighton
predicted_rank = df.at[selected_row, "Predicted Rank 2026"]
actual_rank_2025 = df.at[selected_row, "Overall Rank"]
 
if pd.isna(predicted_rank):
    predicted_rank = -1  # Default value if missing
if pd.isna(actual_rank_2025):
    actual_rank_2025 = -1  # Default value if missing
 
# Rank Circles & Button Design
st.markdown("""
<style>
        .rank-container {
            display: flex;
            justify-content: center;
            gap: 80px; 
            align-items: center;
            margin-bottom: 20px; 
        }
        .rank-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .circle-rank {
            width: 100px;
            height: 100px;
            background-color: transparent;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            font-family: Arial, sans-serif;
            color: #00308F;
            border: 3px solid black;
            margin-top: 5px;
        }
</style>
""", unsafe_allow_html=True)
 
st.markdown(f"""
<div class='rank-container'>
<div class='rank-item'>
<h5>Actual Rank 2025</h5>
<div class='circle-rank'>{int(actual_rank_2025)}</div>
</div>
<div class='rank-item'>
<h5>Predicted Rank 2026</h5>
<div class='circle-rank'>{int(predicted_rank)}</div>
</div>
</div>
""", unsafe_allow_html=True)
 
# Reset Button Design
col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is wider
with col2:
    if st.button("🔄 Reset to Original Number(s)", use_container_width=True):
        #Use to reset every time change(s) is made
        st.session_state["modified_data"] = st.session_state["original_data"].copy()
 
        for key in list(st.session_state.keys()):
            if key.startswith("input_"):
                del st.session_state[key]  # Fully reset UI
 
        st.cache_data.clear()
        st.rerun()
 
# Define Weighted Metrics for Editing in Streamlit
metrics = [
    "Graduate rates", "First-year retention rates", "Graduate rate performance",
    "Pell graduate rates", "Pell graduation performance", "College grads earning more than a high school grad",
    "Borrower debt", "Peer assessment", "Financial resources per student",
    "Faculty salaries", "Full-time faculty", "Student-faculty ratio", "Average Standardized Tests Score",
    "Citations per publication", "Field weighted citations",
    "Citations in top 5% journals", "Citations in top 25% journals"
]
 
available_metrics = [metric for metric in df.columns if metric in metrics]
 
# Handle missing values using mean imputation (simplerimputer)
imputer = SimpleImputer(strategy="mean")
df[available_metrics] = imputer.fit_transform(df[available_metrics])  # Fill NaN with column means
 
# Ensure session state is set for all metrics to avoid warning
for metric in available_metrics:
    key = f"input_{metric}_{selected_row}"
    if key not in st.session_state:
        st.session_state[key] = float(df.at[selected_row, metric])
 
# Display Titles
st.markdown("<h3 style='text-align: center;'>US News Weighted Metrics</h3>", unsafe_allow_html=True)
 
metric_values = {}
 
for metric in available_metrics:
    metric_values[metric] = st.number_input(
        f"**{metric}**",  #Bolded Title
        value=st.session_state[f"input_{metric}_{selected_row}"],  # Uses session state value directly
        step=1.0,
        format="%.2f",
        key=f"input_{metric}_{selected_row}"  # Keep tracking changes properly
    )
 
# Detect Changes & Apply Prediction
changes_detected = False
 
for metric, value in metric_values.items():
    if value != df.at[selected_row, metric]:
        df.at[selected_row, metric] = value
        changes_detected = True
 
#Fix Scaling Issue: Use the Original Scaler, Not Dynamic Scaling
original_scaler = StandardScaler()
original_scaler.fit(st.session_state["original_data"][available_metrics])  # Fit on original data
 
# Predict Rank with Correct Scaling (Original Scaling)
if changes_detected:
    input_scaled = original_scaler.transform(
        np.array([df.at[selected_row, metric] for metric in available_metrics]).reshape(1, -1)
    )
 
    predicted_rank = np.round(stacking_model.predict(input_scaled)[0])
    df.at[selected_row, "Predicted Rank 2026"] = predicted_rank
 
    st.session_state["modified_data"] = df.copy()
 
    st.success(f"Auto-saved updates! Predicted Rank for Row {selected_row} updated.")
    st.rerun()  # Force update of predicted rank in UI
 
# Debug: Print rank in the console
print(f"🔎 Final Predicted Rank for Row {selected_row}: {predicted_rank}")