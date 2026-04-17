import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Streamlit config (must be first)
st.set_page_config(
    page_title="Electric Nerve Center",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electric Nerve Center - ML Inference")
st.markdown("### 6-Dimensional PCA Inference Engine")
st.markdown("Enter the localized parameters on the left to verify active grid stability.")

@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

try:
    model, feature_names = load_assets()
except Exception as e:
    st.error(f"Failed to load engine components: {str(e)}")
    st.stop()

# Layout
col1, col2 = st.columns([1, 2])

# Base configuration values to default the sliders
default_values = {
    'grid_demand': 35000.0,
    'solar': 5000.0,
    'wind': 8000.0,
    'EV_load': 1000.0,
    'demand_lag_1h': 35000.0,
    'renewable_share': 0.3
}

input_data = {}

with col1:
    st.header("⚡ Local Parameters")
    for feature in feature_names:
        # Beautiful proper names
        proper_name = feature.replace('_', ' ').title()
        
        # Renewable share logic (percentage)
        if feature == 'renewable_share':
            val = st.slider(f"{proper_name} (Ratio)", min_value=0.0, max_value=1.0, value=default_values.get(feature, 0.3), step=0.01)
        else:
            val = st.number_input(f"{proper_name} (MW)", value=default_values.get(feature, 1000.0), step=100.0)
            
        input_data[feature] = [val]

# Inference Button and Visual logic
with col2:
    st.header("🎛 System Status")
    
    # Process
    df_input = pd.DataFrame(input_data)
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    
    st.subheader("Results")
    st.metric(label="Probability of Bad Stress", value=f"{prob*100:.1f}%")
        
    if prediction == 1:
        st.error(f"🚩 DANGER: BAD STRESS DETECTED ({prob*100:.1f}% Confidence)")
        st.markdown("**Actionable Interventions Required:**")
        
        actions = []
        if input_data.get('grid_demand', [0])[0] > 50000:
            actions.append("- Issue broad public appeal for energy conservation.")
        if input_data.get('solar', [0])[0] < 1000:
            actions.append("- Dispatch battery storage or peaker plants.")
        if input_data.get('wind', [0])[0] < 5000:
            actions.append("- Insufficient baseline renewables.")
        if input_data.get('renewable_share', [0])[0] < 0.2:
            actions.append("- Critically Low Renewable Yield: Relying heavily on conventional plants.")
            
        if not actions:
            actions.append("- General network congestion. Monitor closely.")
            
        for a in actions:
            st.warning(a)
    else:
        st.success(f"✅ STABLE: NORMAL OPERATIONS ({prob*100:.1f}% Stress Probability)")
        st.info("Grid is operating normally. No intervention required.")
        
    st.divider()
    st.markdown("*Deployed autonomously on Hugging Face Spaces.*")
