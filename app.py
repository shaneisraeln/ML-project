from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from flask import send_from_directory

app = Flask(__name__)

# Define Base Directory dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and feature names
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'feature_names.pkl')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
else:
    model = None
    feature_names = []

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', features=feature_names)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({"error": "Model not found."}), 500
        
    try:
        data = request.json
        input_data = {}
        for feature in feature_names:
            val = data.get(feature)
            input_data[feature] = [float(val)] if val else [0.0]
            
        df_input = pd.DataFrame(input_data)
        
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
        
        action_plan = []
        if prediction == 1:
            status = "BAD STRESS DETECTED"
            color = "red"
            
            if input_data.get('grid_demand', [0])[0] > 50000:
                action_plan.append("High Grid Demand: Issue broad public appeal for energy conservation.")
            if input_data.get('solar', [0])[0] < 1000:
                action_plan.append("Low Solar Generation: Dispatch battery storage or peaker plants.")
            if input_data.get('wind', [0])[0] < 5000:
                action_plan.append("Low Wind Generation: Insufficient baseline renewables.")
            if input_data.get('renewable_share', [0])[0] < 0.2:
                action_plan.append("Critically Low Renewable Yield: Relying heavily on conventional plants.")
                
            if not action_plan:
                action_plan.append("General network congestion. Monitor closely.")
        else:
            status = "NORMAL (No Stress)"
            color = "green"
            action_plan.append("Grid is operating normally.")
            
        return jsonify({
            "prediction": status,
            "probability": f"{prob*100:.1f}%",
            "actions": action_plan,
            "color": color,
            "raw_prob": prob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/analytics', methods=['GET'])
def analytics():
    # List images in data/visualizations
    vis_dir = os.path.join(BASE_DIR, 'data', 'visualizations')
    if os.path.exists(vis_dir):
        images = [f for f in os.listdir(vis_dir) if f.endswith('.png')]
    else:
        images = []
    # Sort images to maintain a consistent order
    images.sort()
    return render_template('analytics.html', images=images)

@app.route('/visualizations/<path:filename>')
def serve_visualizations(filename):
    return send_from_directory(os.path.join(BASE_DIR, 'data', 'visualizations'), filename)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', features=feature_names, error="Model not found. Train the model first.")
    
    try:
        # Collect input data
        input_data = {}
        for feature in feature_names:
            val = request.form.get(feature)
            input_data[feature] = [float(val)] if val else [0.0]
        
        df_input = pd.DataFrame(input_data)
        
        # Predict
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]
        
        # Determine actionable insights if stress is bad
        action_plan = []
        if prediction == 1:
            status = "BAD STRESS DETECTED"
            color = "red"
            
            # Actionable logic to pinpoint stress sources
            if 'grid_demand' in input_data and input_data.get('grid_demand', [0])[0] > 50000:
                action_plan.append("High Grid Demand: Issue broad public appeal for energy conservation.")
            if 'solar' in input_data and input_data.get('solar', [0])[0] < 1000:
                action_plan.append("Low Solar Generation: Dispatch battery storage or peaker plants.")
            if 'wind' in input_data and input_data.get('wind', [0])[0] < 5000:
                action_plan.append("Low Wind Generation: Insufficient baseline renewables.")
            if 'renewable_share' in input_data and input_data.get('renewable_share', [0])[0] < 0.2:
                action_plan.append("Critically Low Renewable Yield: Relying heavily on conventional plants.")
                
            if not action_plan:
                action_plan.append("General network congestion. Monitor closely.")
        else:
            status = "NORMAL (No Stress)"
            color = "green"
            action_plan.append("Grid is operating normally.")
            
        return render_template('index.html', features=feature_names, prediction=status, probability=f"{prob*100:.1f}%", actions=action_plan, color=color, submitted_data=input_data)
        
    except Exception as e:
        return render_template('index.html', features=feature_names, error=str(e))

if __name__ == '__main__':
    # Ensure templates dir exists
    os.makedirs(os.path.join(BASE_DIR, 'templates'), exist_ok=True)
    app.run(debug=True, port=5000)
