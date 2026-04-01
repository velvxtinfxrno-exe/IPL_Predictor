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
    
    # 2. Extract Names
    players = [f.replace('Player_', '') for f in expected_features if f.startswith('Player_')]
    teams = [f.replace('Team_', '') for f in expected_features if f.startswith('Team_')]

    # 3. Inputs
    col1, col2 = st.columns(2)
    with col1:
        sel_player = st.selectbox("Player", sorted(players))
        sel_team = st.selectbox("Team", sorted(teams))
    with col2:
        # We increase the range and step to make the change more visible
        slider_sr = st.slider("Strike Rate", 50.0, 300.0, 140.0, step=10.0)
        slider_mat = st.slider("Matches", 1, 17, 10)

    if st.button("Predict Price", use_container_width=True):
        # Create empty row
        input_row = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # --- THE FIX: FORCE SYNC STRIKE RATE ---
        # We find ANY column that looks like Strike Rate and give it the slider value
        sr_found = False
        for col in expected_features:
            if col.lower() in ['sr', 'strike rate', 'strikerate', 'avg_sr', 'batting_sr']:
                input_row[col] = slider_sr
                sr_found = True
            if col.lower() in ['mat', 'matches', 'played', 'm']:
                input_row[col] = slider_mat

        # --- THE FIX: PLAYER & TEAM ---
        p_col, t_col = f"Player_{sel_player}", f"Team_{sel_team}"
        if p_col in expected_features: input_row[p_col] = 1
        if t_col in expected_features: input_row[t_col] = 1

        # 4. Predict
        raw_prediction = model.predict(input_row)[0]
        
        # 5. Handle "Cheap" Price (Lakhs vs Crores)
        # If your model says 500, it means 5.00 Cr. If it says 5, it means 5 Cr.
        if raw_prediction > 25: 
            display_price = raw_prediction / 100
        else:
            display_price = raw_prediction

        # 6. Display Result
        st.header(f"Predicted Price: ₹ {max(0.20, display_price):.2f} Cr")
        
        if not sr_found:
            st.warning("⚠️ Warning: Could not find a 'Strike Rate' column in your model. Prices won't change.")

        # Debugging tools (Hidden)
        with st.expander("Show Technical Names (Debug)"):
            st.write("Columns in your model:", list(expected_features)[:15])
            st.write(f"Active SR Value sent: {slider_sr}")

except Exception as e:
    st.error(f"Error: {e}")
