import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="IPL 2026 Auction Predictor",
    page_icon="🏏",
    layout="centered"
)

# 2. Smart File Loading (Fixes the FileNotFoundError)
@st.cache_resource
def load_assets():
    # This identifies the folder where this specific app.py is sitting
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(current_dir, "model.pkl")
    features_path = os.path.join(current_dir, "feature_names.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error(f"Missing Files! Ensure 'model.pkl' and 'feature_names.pkl' are in: {current_dir}")
        st.stop()
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

# 3. UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #f1c40f;
        color: #1e3799;
        font-weight: bold;
        border-radius: 10px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    model, expected_features = load_assets()

    st.title("🏏 IPL 2026 Auction Predictor")
    st.write("Predicting player market value using Machine Learning")
    st.divider()

    # 4. Extracting Names for Dropdowns
    # We clean the 'Player_' and 'Team_' prefixes created during training
    players = [f.replace('Player_', '') for f in expected_features if f.startswith('Player_')]
    teams = [f.replace('Team_', '') for f in expected_features if f.startswith('Team_')]

    # 5. User Input Layout
    col1, col2 = st.columns(2)
    
    with col1:
        selected_player = st.selectbox("Select Player Name", sorted(players))
        selected_team = st.selectbox("Select Current Team", sorted(teams))
    
    with col2:
        matches = st.slider("Matches Played (Recent Season)", 1, 17, 14)
        strike_rate = st.number_input("Strike Rate", min_value=50.0, max_value=300.0, value=135.0)

    st.divider()

    # 6. Prediction Logic
    if st.button("Generate Auction Valuation"):
        # Create a single row of 418+ zeros matching the model's training data
        input_row = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Fill Numerical Stats
        if 'Mat' in input_row.columns: input_row['Mat'] = matches
        if 'SR' in input_row.columns: input_row['SR'] = strike_rate
        
        # Fill Categorical Stats (One-Hot Encoding)
        p_col = f"Player_{selected_player}"
        t_col = f"Team_{selected_team}"
        
        if p_col in input_row.columns: input_row[p_col] = 1
        if t_col in input_row.columns: input_row[t_col] = 1
        
        # Calculate Prediction
        prediction = model.predict(input_row)[0]
        
        # Price Range Logic (±15%)
        low_range = max(0.20, prediction * 0.85) # Base price floor of 20 Lakhs
        high_range = prediction * 1.15

        # 7. Display Results
        st.balloons()
        st.success(f"### Predicted Price for {selected_player}")
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Estimated Range", f"₹ {low_range:.2f} Cr - {high_range:.2f} Cr")
        res_col2.metric("Base Valuation", f"₹ {prediction:.2f} Cr")
        
        st.info("Note: Prediction is based on 2024-2025 performance metrics and historical auction trends.")

except Exception as e:
    st.error(f"App Logic Error: {e}")
