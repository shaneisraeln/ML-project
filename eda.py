import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create data directory for saving outputs if it doesn't already exist
os.makedirs('e:/ML project/data/visualizations', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('e:/ML project/data/smart_grid_master_dataset.csv')

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("\n--- Summary Statistics of Grid Stress ---")
print(df['grid_stress'].describe())

# Define "Bad Stress" as the top 20% of grid stress values
stress_threshold = df['grid_stress'].quantile(0.80)
print(f"\nDefining 'Bad Stress' threshold at 80th percentile: {stress_threshold}")

df['is_bad_stress'] = (df['grid_stress'] >= stress_threshold).astype(int)
print(f"Bad Stress cases: {df['is_bad_stress'].sum()} out of {len(df)}")

# Generate 5 Visualizations
print("\nGenerating Visualizations...")

# 1. Distribution of Grid Stress highlighting the Bad Stress threshold
plt.figure(figsize=(10, 6))
sns.histplot(df['grid_stress'], bins=50, kde=True, color='skyblue')
plt.axvline(stress_threshold, color='red', linestyle='--', label=f'Bad Stress Threshold ({stress_threshold:.1f})')
plt.title('Distribution of Grid Stress')
plt.xlabel('Grid Stress')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('e:/ML project/data/visualizations/1_grid_stress_distribution.png')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(14, 10))
# Exclude timestamp for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('e:/ML project/data/visualizations/2_correlation_heatmap.png')
plt.close()

# 3. Bad Stress by Hour of the Day
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='hour', y='is_bad_stress', ci=None, palette='viridis')
plt.title('Probability of Bad Grid Stress by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Probability of Bad Stress')
plt.savefig('e:/ML project/data/visualizations/3_stress_by_hour.png')
plt.close()

# 4. EV Load vs Grid Demand colored by Bad Stress
plt.figure(figsize=(10, 6))
# Sample data for scatter plot to avoid overplotting
sample_df = df.sample(n=min(5000, len(df)), random_state=42)
sns.scatterplot(data=sample_df, x='EV_load', y='grid_demand', hue='is_bad_stress', alpha=0.6, palette='Set1')
plt.title('EV Load vs Grid Demand (Sampled)')
plt.xlabel('EV Load')
plt.ylabel('Grid Demand')
plt.savefig('e:/ML project/data/visualizations/4_ev_load_vs_demand.png')
plt.close()

# 5. Solar & Wind Generation vs Grid Stress
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(data=sample_df, x='solar', y='grid_stress', alpha=0.5, ax=axes[0])
axes[0].set_title('Solar Generation vs Grid Stress')
sns.scatterplot(data=sample_df, x='wind', y='grid_stress', alpha=0.5, color='green', ax=axes[1])
axes[1].set_title('Wind Generation vs Grid Stress')
plt.tight_layout()
plt.savefig('e:/ML project/data/visualizations/5_renewables_vs_stress.png')
plt.close()

print("\nVisualizations saved to 'e:/ML project/data/visualizations'.")

# Check for missing values
missing = df.isnull().sum().sum()
print(f"\nMissing values in dataset: {missing}")

# Save Cleaned Data
cleaned_path = 'e:/ML project/data/cleaned_data.csv'
df.to_csv(cleaned_path, index=False)
print(f"\nData augmented with 'is_bad_stress' and saved to {cleaned_path}")
