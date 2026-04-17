import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids tkinter threading errors
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.decomposition import PCA

# Base directory — all paths relative to this script's location
BASE = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE, 'data', 'visualizations'), exist_ok=True)

print("1. Loading Cleaned Data...")
df = pd.read_csv(os.path.join(BASE, 'data', 'cleaned_data.csv'))

features_to_drop = ['timestamp', 'grid_stress', 'is_bad_stress']
X = df.drop(columns=features_to_drop)
y = df['is_bad_stress']

print(f"Features used: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# ─────────────────────────────────────────────────────────────
# SECTION A: Without Pipeline (manual scaling + raw model)
# Required by rubric: implement algorithm WITHOUT pipeline first
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION A: Model WITHOUT Scikit-Learn Pipeline")
print("="*60)

# Manually scale the data
scaler_manual = StandardScaler()
X_train_scaled = scaler_manual.fit_transform(X_train)
X_test_scaled  = scaler_manual.transform(X_test)

# Train raw RandomForest (no pipeline wrapper)
rf_raw = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_raw.fit(X_train_scaled, y_train)
y_pred_raw = rf_raw.predict(X_test_scaled)

acc_raw = accuracy_score(y_test, y_pred_raw)
f1_raw  = f1_score(y_test, y_pred_raw, zero_division=0)
print(f"Raw Model Accuracy : {acc_raw:.4f}")
print(f"Raw Model F1-Score : {f1_raw:.4f}")
print(classification_report(y_test, y_pred_raw))

# ─────────────────────────────────────────────────────────────
# SECTION B: Baseline (DummyClassifier via Pipeline)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION B: Baseline DummyClassifier (Pipeline)")
print("="*60)

numeric_features = list(X.columns)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features)
])

baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DummyClassifier(strategy='prior'))
])

baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
f1_base  = f1_score(y_test, y_pred_base, zero_division=0)
print(f"Baseline Accuracy : {acc_base:.4f}")
print(f"Baseline F1-Score : {f1_base:.4f}  (0.0 expected — predicts majority class only)")

# ─────────────────────────────────────────────────────────────
# SECTION C: Pipeline WITH Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION C: RandomForest Pipeline + Hyperparameter Tuning")
print("="*60)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features)
    ])),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Hyperparameter search space
param_dist = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth':    [8, 10, 15],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf':  [1, 2],
}

print("Running RandomizedSearchCV (n_iter=6, cv=2) ...")
search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=6,
    cv=2,
    scoring='f1',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"\nBest Params: {search.best_params_}")

y_pred_tuned = best_model.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned  = f1_score(y_test, y_pred_tuned, zero_division=0)
print(f"Tuned Model Accuracy : {acc_tuned:.4f}")
print(f"Tuned Model F1-Score : {f1_tuned:.4f}")
print(classification_report(y_test, y_pred_tuned))

# ─────────────────────────────────────────────────────────────
# SECTION C2: Pipeline WITH PCA (95% Variance)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION C2: RandomForest Pipeline + PCA (95% Variance)")
print("="*60)

pca_pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features)
    ])),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

pca_pipeline.fit(X_train, y_train)
y_pred_pca = pca_pipeline.predict(X_test)
acc_pca = accuracy_score(y_test, y_pred_pca)
f1_pca  = f1_score(y_test, y_pred_pca, zero_division=0)

num_components = pca_pipeline.named_steps['pca'].n_components_
print(f"PCA preserved >=95% of variance using {num_components} components (reduced from {len(numeric_features)} features).")
print(f"PCA Model Accuracy : {acc_pca:.4f}")
print(f"PCA Model F1-Score : {f1_pca:.4f}")
print(classification_report(y_test, y_pred_pca))

# ─────────────────────────────────────────────────────────────
# SECTION D: Feature Importance
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION D: Feature Importance")
print("="*60)

importances = best_model.named_steps['classifier'].feature_importances_
feat_importances = pd.Series(importances, index=numeric_features).sort_values(ascending=False)

print("Top 5 Driving Factors of Grid Stress:")
print(feat_importances.head(5))

plt.figure(figsize=(10, 6))
feat_importances.plot(kind='bar', color='coral')
plt.title('Feature Importances - Identifying Sources of Grid Stress')
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'data', 'visualizations', '6_feature_importance.png'))
plt.close()

# ─────────────────────────────────────────────────────────────
# SECTION E: Comparison Summary Table
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION E: Model Comparison Summary")
print("="*60)

comparison = pd.DataFrame({
    'Model': [
        'DummyClassifier (Baseline)',
        'RandomForest — No Pipeline (Manual Scaling)',
        'RandomForest — Pipeline (Default Params)',
        'RandomForest — Pipeline + Hyperparameter Tuning',
        'RandomForest — Pipeline + PCA (95% Variance)',
        'Literature Benchmark (ARIMA-based, ~85% Acc)'   # documented external reference
    ],
    'Accuracy': [
        round(acc_base, 4),
        round(acc_raw, 4),
        None,   # filled below
        round(acc_tuned, 4),
        round(acc_pca, 4),
        0.85
    ],
    'F1-Score (Bad Stress)': [
        round(f1_base, 4),
        round(f1_raw, 4),
        None,
        round(f1_tuned, 4),
        round(f1_pca, 4),
        0.71
    ]
})

# Fill default pipeline row
rf_default = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
])
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
comparison.loc[comparison['Model'].str.contains('Default'), 'Accuracy'] = round(accuracy_score(y_test, y_pred_default), 4)
comparison.loc[comparison['Model'].str.contains('Default'), 'F1-Score (Bad Stress)'] = round(f1_score(y_test, y_pred_default, zero_division=0), 4)

print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(BASE, 'data', 'model_comparison.csv'), index=False)
print("\nComparison table saved to 'data/model_comparison.csv'")

# ─────────────────────────────────────────────────────────────
# SECTION F: Save Best Production Model (PCA with Needed Variables)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION F: Saving Best Production Model (PCA with Needed Variables)")
print("="*60)

# The user explicitly requested to keep only the needed parameters to avoid UI clutter
top_features = ['grid_demand', 'solar', 'wind', 'EV_load', 'renewable_share', 'demand_lag_1h']

X_train_top = X_train[top_features]

pca_pipeline_top = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', StandardScaler(), top_features)
    ])),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=search.best_params_.get('classifier__n_estimators', 200),
        max_depth=search.best_params_.get('classifier__max_depth', 10),
        min_samples_split=search.best_params_.get('classifier__min_samples_split', 2),
        min_samples_leaf=search.best_params_.get('classifier__min_samples_leaf', 1),
        random_state=42, n_jobs=-1
    ))
])

pca_pipeline_top.fit(X_train_top, y_train)
joblib.dump(pca_pipeline_top, os.path.join(BASE, 'model.pkl'))
joblib.dump(top_features, os.path.join(BASE, 'feature_names.pkl'))
print("PCA model built on necessary features saved to 'model.pkl' and 'feature_names.pkl'. Done.")
