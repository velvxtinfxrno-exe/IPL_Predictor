import streamlit as st
import pandas as pd
import joblib

# Setup Page
st.set_page_config(page_title="IPL 2026 Auction Scout", page_icon="🏏")

# Load the "Brains"
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, expected_features = load_model()

# UI Design
st.title("🏏 IPL 2026 Auction Price Predictor")
st.subheader("Predicting Player Value via Machine Learning")

# Extract Names for Dropdowns
players = [f.replace('Player_', '') for f in expected_features if f.startswith('Player_')]
teams = [f.replace('Team_', '') for f in expected_features if f.startswith('Team_')]

# User Input Section
col1, col2 = st.columns(2)
with col1:
    selected_player = st.selectbox("Select Player", sorted(players))
    selected_team = st.selectbox("Select Current Team", sorted(teams))
with col2:
    mat = st.slider("Matches Played", 1, 17, 14)
    sr = st.number_input("Strike Rate", value=135.0)

# Prediction Button
if st.button("Predict Auction Price Range", use_container_width=True):
    # Create the 418-column input row
    input_row = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # Fill Numerical stats
    if 'Mat' in input_row.columns: input_row['Mat'] = mat
    if 'SR' in input_row.columns: input_row['SR'] = sr
    
    # Fill One-Hot encoded columns
    p_col, t_col = f"Player_{selected_player}", f"Team_{selected_team}"
    if p_col in input_row.columns: input_row[p_col] = 1
    if t_col in input_row.columns: input_row[t_col] = 1
    
    # Predict
    val = model.predict(input_row)[0]
    
    # Show Results
    st.markdown("---")
    st.success(f"### Analysis for {selected_player}")
    
    low, high = max(0.20, val * 0.85), val * 1.15
    st.metric("Estimated Price Range", f"₹ {low:.2f} Cr - {high:.2f} Cr")
    st.info(f"Model Base Valuation: ₹ {val:.2f} Cr")