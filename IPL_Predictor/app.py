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

# 2. Smart File Loading
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.pkl")
    features_path = os.path.join(current_dir, "feature_names.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error(f"Missing Files! Ensure 'model.pkl' and 'feature_names.pkl' are in the same folder.")
        st.stop()
        
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features

# 3. UI Styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #f1c40f;
        color: #1e3799;
        font-weight: bold;
        border-radius: 10px;
    }
    .prediction-text {
        font-size: 30px;
        font-weight: bold;
        color: #f1c40f;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    model, expected_features = load_assets()

    st.title("🏏 IPL 2026 Price Predictor")
    st.divider()

    # 4. Extracting Names for Dropdowns
    players = [f.replace('Player_', '') for f in expected_features if f.startswith('Player_')]
    teams = [f.replace('Team_', '') for f in expected_features if f.startswith('Team_')]

    # 5. User Input
    col1, col2 = st.columns(2)
    with col1:
        selected_player = st.selectbox("Select Player", sorted(players))
        selected_team = st.selectbox("Select Team", sorted(teams))
    with col2:
        matches = st.slider("Matches Played", 1, 17, 14)
        strike_rate = st.number_input("Strike Rate", value=135.0)

    st.divider()

    # 6. Prediction Logic
    if st.button("Predict Auction Price"):
        # Create input row
        input_row = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Fill Numerical Stats
        if 'Mat' in input_row.columns: input_row['Mat'] = matches
        if 'SR' in input_row.columns: input_row['SR'] = strike_rate
        
        # Fill Categorical Stats
        p_col = f"Player_{selected_player}"
        t_col = f"Team_{selected_team}"
        
        if p_col in input_row.columns: input_row[p_col] = 1
        if t_col in input_row.columns: input_row[t_col] = 1
        
        # Calculate Prediction
        prediction = model.predict(input_row)[0]
        final_price = max(0.20, prediction) # Ensure minimum 20 Lakhs

        # 7. Simplified Result Display
        st.balloons()
        st.markdown(f"<div class='prediction-text'>Predicted Auction Price</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-text'>₹ {final_price:.2f} Cr</div>", unsafe_allow_html=True)
        
        st.info(f"Analysis complete for {selected_player}.")

except Exception as e:
    st.error(f"Error: {e}")
