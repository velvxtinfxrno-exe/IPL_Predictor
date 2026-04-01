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

    # 3. Enhanced Inputs (Adding the missing "Power" stats)
    col1, col2 = st.columns(2)
    with col1:
        sel_player = st.selectbox("Player", sorted(players))
        sel_team = st.selectbox("Team", sorted(teams))
        runs = st.number_input("Total Runs (Season)", value=300)
    with col2:
        slider_sr = st.slider("Strike Rate", 50.0, 300.0, 140.0)
        wickets = st.number_input("Wickets Taken", value=0)
        slider_mat = st.slider("Matches", 1, 17, 14)

    if st.button("Predict Price", use_container_width=True):
        input_row = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # --- THE FIX: FILLING MORE COLUMNS ---
        for col in expected_features:
            c_low = col.lower()
            if c_low in ['sr', 'strike rate', 'strikerate']: input_row[col] = slider_sr
            if c_low in ['mat', 'matches', 'played']: input_row[col] = slider_mat
            if c_low in ['runs', 'total runs', 'r']: input_row[col] = runs
            if c_low in ['wkts', 'wickets', 'w']: input_row[col] = wickets

        # Set Player & Team
        p_col, t_col = f"Player_{sel_player}", f"Team_{sel_team}"
        if p_col in expected_features: input_row[p_col] = 1
        if t_col in expected_features: input_row[t_col] = 1

        # 4. Predict
        raw_val = model.predict(input_row)[0]
        
        # 5. Handle Units (If model predicts in Lakhs)
        if raw_val > 30: 
            final_price = raw_val / 100
        else:
            final_price = raw_val

        # 6. Display Result
        st.markdown("---")
        st.markdown(f"<h1 style='text-align: center; color: #f1c40f;'>₹ {max(0.20, final_price):.2f} Cr</h1>", unsafe_allow_html=True)
        st.success(f"Market analysis for {sel_player} is complete.")

except Exception as e:
    st.error(f"App Error: {e}")
