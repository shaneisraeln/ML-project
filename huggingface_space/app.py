import gradio as gr
import joblib
import pandas as pd

# ── Load model artifacts ──────────────────────────────────────
model         = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')

# ── Prediction function ───────────────────────────────────────
def predict_stress(solar, wind, renewable_share, grid_demand):
    input_df = pd.DataFrame([{
        'solar':           solar,
        'wind':            wind,
        'renewable_share': renewable_share,
        'grid_demand':     grid_demand
    }])[feature_names]

    prediction = model.predict(input_df)[0]
    prob       = model.predict_proba(input_df)[0][1]

    action_plan = []

    if prediction == 1:
        status = f"🔴 BAD STRESS DETECTED  —  Confidence: {prob*100:.1f}%"
        if solar < 1000:
            action_plan.append("☀️ Low Solar Output: Dispatch battery storage or peaker plants.")
        if wind < 5000:
            action_plan.append("💨 Low Wind Generation: Increase baseload generation.")
        if renewable_share < 0.2:
            action_plan.append("⚡ Critically Low Renewable Share: Heavy reliance on conventional plants.")
        if grid_demand > 50000:
            action_plan.append("📢 High Grid Demand: Issue public appeal for energy conservation.")
        if not action_plan:
            action_plan.append("⚠️ General network congestion detected. Monitor closely.")
        recommendations = "\n".join(action_plan)
    else:
        status          = f"🟢 NORMAL — No Stress  —  Confidence: {(1-prob)*100:.1f}%"
        recommendations = "✅ Grid is operating normally. No intervention required."

    return status, recommendations

# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="⚡ Smart Grid Stress Predictor") as demo:
    gr.Markdown("# ⚡ Smart Grid Stress Predictor\nPredicts whether the smart grid will experience **Bad Stress** based on current generation and demand values.")

    with gr.Row():
        with gr.Column():
            solar           = gr.Number(label="Solar Generation (MW)",  value=3000.0)
            wind            = gr.Number(label="Wind Generation (MW)",   value=8000.0)
            renewable_share = gr.Number(label="Renewable Share (0–1)",  value=0.35)
            grid_demand     = gr.Number(label="Grid Demand (MW)",       value=40000.0)
            predict_btn     = gr.Button("Predict", variant="primary")
        with gr.Column():
            status_out = gr.Textbox(label="Prediction Status",          lines=2)
            action_out = gr.Textbox(label="Actionable Recommendations", lines=5)

    predict_btn.click(fn=predict_stress,
                      inputs=[solar, wind, renewable_share, grid_demand],
                      outputs=[status_out, action_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
