import streamlit as st
import pandas as pd
import joblib
import os

# 1. Setup
st.set_page_config(page_title="IPL Price Predictor", page_icon="🏏")

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(current_dir, "model.pkl"))
    features = joblib.load(os.path.join(current_dir, "feature_names.pkl"))
    return model, features

try:
    model, expected_features = load_assets()
    st.title("🏏 IPL Auction Predictor")
    
    # 2. Extract Data
    players = [f.replace('Player_', '') for f in expected_features if f.startswith('Player_')]
    teams = [f.replace('Team_', '') for f in expected_features if f.startswith('Team_')]

    # 3. Inputs
    col1, col2 = st.columns(2)
    with col1:
        sel_player = st.selectbox("Player", sorted(players))
        sel_team = st.selectbox("Team", sorted(teams))
    with col2:
        # High ranges so you can actually see the model react
        slider_sr = st.slider("Strike Rate", 50.0, 300.0, 140.0, step=5.0)
        slider_mat = st.slider("Matches", 1, 17, 10)

    if st.button("Predict Price", use_container_width=True):
        # Create empty row of 418 zeros
        input_row = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # --- THE FIX: SEARCH FOR COLUMN NAMES ---
        # We loop through all 418 columns to find the one that matches 'SR' or 'Mat'
        for col in expected_features:
            c_lower = col.lower()
            # Matching Strike Rate
            if c_lower in ['sr', 'strike rate', 'strikerate', 'batting_sr', 'bat_sr']:
                input_row[col] = slider_sr
            # Matching Matches
            if c_lower in ['mat', 'matches', 'played', 'm']:
                input_row[col] = slider_mat

        # Set Player & Team
        p_col, t_col = f"Player_{sel_player}", f"Team_{sel_team}"
        if p_col in expected_features: input_row[p_col] = 1
        if t_col in expected_features: input_row[t_col] = 1

        # 4. Prediction
        raw_val = model.predict(input_row)[0]
        
        # --- THE FIX: THE "CHEAP" PRICE ---
        # If your model predicts in Lakhs (e.g. 500 for 5Cr), we divide by 100.
        # If it predicts in Crores (e.g. 5.0), we keep it.
        if raw_val > 25: 
            final_price = raw_val / 100
        else:
            final_price = raw_val

        # 5. Display Result
        st.markdown("---")
        st.success(f"### Estimated Price: ₹ {max(0.20, final_price):.2f} Cr")
        
        # DEBUG DRAWER (Open this on the website to see the problem)
        with st.expander("Technical Debugging (Check this if price doesn't move)"):
            st.write("First 10 column names in your model:", list(expected_features)[:10])
            st.write(f"Raw Model Result: {raw_val}")
            st.write(f"Was Player Column Found? {p_col in expected_features}")

except Exception as e:
    st.error(f"App Error: {e}")
