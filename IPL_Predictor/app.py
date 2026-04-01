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

# 2. Smart Asset Loader
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model.pkl")
    features_path = os.path.join(current_dir, "feature_names.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error("Missing model.pkl or feature_names.pkl in the app folder!")
        st.stop()
        
    return joblib.load(model_path), joblib.load(features_path)

# 3. Custom CSS for IPL Theme
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .stButton>button {
        width: 100%;
        background-color: #f1c40f;
        color: #1e3799;
        font-weight: bold;
        height: 3em;
        border-radius: 10px;
    }
    .price-display {
        font-size: 50px;
        color: #f1c40f;
        text-align: center;
        font-weight: bold;
        margin-top: -20px;
    }
    </style>
    """, unsafe_allow_html=True)

try:
    model, expected_features = load_assets()

    st.title("🏏 IPL 2026 Auction Predictor")
    st.write("Performance-based Market Valuation")
    st.divider()

    # 4. Data Parsing
    players = [f.replace('Player_', '') for f in expected_features if f.startswith('Player_')]
    teams = [f.replace('Team_', '') for f in expected_features if f.startswith('Team_')]

    # 5. User Interface
    col1, col2 = st.columns(2)
    with col1:
        selected_player = st.selectbox("Select Player", sorted(players))
        selected_team = st.selectbox("Current Team", sorted(teams))
    with col2:
        matches = st.slider("Matches (Recent Season)", 1, 17, 14)
        # We use a large range for SR to see visible changes
        strike_rate = st.number_input("Strike Rate (SR)", value=135.0, step=5.0)

    st.divider()

    # 6. Prediction Logic
    if st.button("Calculate Market Value"):
        # Create an empty input row with all features set to 0
        input_row = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # --- FEATURE MAPPING ---
        # We check common variations of names to ensure the slider works
        sr_keys = ['SR', 'Strike Rate', 'strike_rate', 'StrikeRate']
        mat_keys = ['Mat', 'Matches', 'matches', 'Played']
        
        for key in sr_keys:
            if key in input_row.columns: input_row[key] = strike_rate
            
        for key in mat_keys:
            if key in input_row.columns: input_row[key] = matches
            
        # Set One-Hot encoded columns to 1
        p_col = f"Player_{selected_player}"
        t_col = f"Team_{selected_team}"
        
        if p_col in input_row.columns: input_row[p_col] = 1
        if t_col in input_row.columns: input_row[t_col] = 1
        
        # Perform Prediction
        raw_val = model.predict(input_row)[0]

        # --- UNIT CORRECTION ---
        # If the model was trained in Lakhs (e.g. 500 for 5Cr), we divide by 100.
        # If it was trained in Crores (e.g. 5.0 for 5Cr), we leave it.
        # Most IPL models trained on 'Price' columns predict in Lakhs.
        if raw_val > 30: # If prediction is higher than 30, it's likely in Lakhs
            final_price = raw_val / 100
        else:
            final_price = raw_val

        # 7. Results
        st.balloons()
        st.markdown("<p style='text-align: center; color: #636e72;'>ESTIMATED AUCTION VALUE</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='price-display'>₹ {max(0.20, final_price):.2f} Cr</div>", unsafe_allow_html=True)
        
        # Developer Debugger (Collapse this to hide)
        with st.expander("Developer Debugging Menu"):
            st.write(f"Raw Model Output: {raw_val}")
            st.write(f"Player Column Detected: {p_col in input_row.columns}")
            st.write(f"Matches Column: {[k for k in mat_keys if k in input_row.columns]}")
            st.write(f"Strike Rate Column: {[k for k in sr_keys if k in input_row.columns]}")

except Exception as e:
    st.error(f"Prediction Error: {e}")
