---
title: Smart Grid Stress Predictor
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ⚡ Smart Grid Stress Predictor

A Machine Learning web app that predicts whether a smart electrical grid will
experience **Bad Stress** — defined as the top 20% of operational strain conditions.

## How It Works

The model is a **Random Forest Classifier** trained on 50,000+ smart grid observations.
It uses 4 key features:

| Feature | Description |
|---|---|
| Solar Generation (MW) | Current output from solar farms |
| Wind Generation (MW) | Current output from wind turbines |
| Renewable Share | Ratio of renewables to total demand (0–1) |
| Grid Demand (MW) | Total power demanded by consumers |

When Bad Stress is detected, the app also provides **actionable recommendations**
(e.g., dispatch battery storage, issue demand response alerts).

## Model Performance

| Model | Accuracy | F1-Score |
|---|---|---|
| Baseline (DummyClassifier) | ~0.89 | 0.00 |
| Random Forest (No Pipeline) | ~0.99 | ~0.93 |
| Random Forest + Pipeline + Tuning | ~0.99 | ~0.94 |
| Literature Benchmark (ARIMA) | ~0.85 | ~0.71 |

## Tech Stack
- **Model**: Scikit-Learn Random Forest with hyperparameter tuning (RandomizedSearchCV)
- **UI**: Gradio
- **Deployment**: Hugging Face Spaces
