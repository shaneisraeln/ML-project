import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# Load model and feature names from the current directory (Hugging Face space root)
# The user will need to upload model.pkl and feature_names.pkl along with this script
MODEL_PATH = 'model.pkl'
FEATURES_PATH = 'feature_names.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
else:
    model = None
    feature_names = []

def predict_stress(*inputs):
    if model is None:
        return "Error", "Model not found. Please upload model.pkl.", "Please upload model."
    
    # Map inputs to feature dictionary
    input_data = {feature: [val] for feature, val in zip(feature_names, inputs)}
    df_input = pd.DataFrame(input_data)
    
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    
    action_plan = []
    
    if prediction == 1:
        status = f"🔴 BAD STRESS DETECTED (Confidence: {prob*100:.1f}%)"
        
        if input_data.get('EV_load', [0])[0] > 40:
            action_plan.append("• High EV Load: Issue demand response to delay EV charging.")
        if input_data.get('solar', [0])[0] < 500 and input_data.get('hour', [0])[0] in range(9, 16):
            action_plan.append("• Low Solar during daytime: Dispatch battery storage or peaker plants.")
        if input_data.get('wind', [0])[0] < 5000:
            action_plan.append("• Low Wind Generation: Insufficient baseline renewables, increase baseload generation.")
        if input_data.get('demand_lag_24h', [0])[0] > 60000:
            action_plan.append("• Sustained High Demand (from 24h lag): Issue broad public appeal for conservation.")
            
        if not action_plan:
            action_plan.append("• General network congestion. Monitor closely.")
            
        recommendations = "\n".join(action_plan)
        
    else:
        status = f"🟢 NORMAL - No Stress (Confidence: {(1-prob)*100:.1f}%)"
        recommendations = "Grid is operating normally. No intervention required."
        
    return status, recommendations

# Create Gradio UI dynamically based on the features the model expects
if feature_names:
    input_components = [gr.Number(label=f, value=0.0) for f in feature_names]
    
    interface = gr.Interface(
        fn=predict_stress,
        inputs=input_components,
        outputs=[
            gr.Textbox(label="Prediction Status"),
            gr.Textbox(label="Actionable Recommendations")
        ],
        title="⚡ Smart Grid Stress Predictor",
        description="Predicts if the smart grid will experience 'Bad Stress' based on current variables and suggests actionable interventions.",
        theme="default"
    )
else:
    # Dummy interface if model is missing
    interface = gr.Interface(
        fn=lambda x: ("Error", "Model not found"),
        inputs=gr.Number(label="Dummy Input"),
        outputs=[gr.Textbox(label="Error"), gr.Textbox(label="Error")],
        title="⚡ Smart Grid Stress Predictor (ERROR)",
        description="Model files not found. Upload model.pkl and feature_names.pkl."
    )

if __name__ == "__main__":
    interface.launch(share=False)
