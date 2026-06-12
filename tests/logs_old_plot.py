import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load data from the JSON file
with open("logs_old.json", "r") as f:
    data = json.load(f)

# 2. Parse and flatten JSON records into a unified DataFrame
all_records = []
for model_name, execution_runs in data.items():
    df = pd.DataFrame(execution_runs)

    # Harmonize column names tracking execution latency
    if 'walltime' in df.columns:
        df['execution_latency'] = df['walltime']
    elif 'execution_time' in df.columns:
        df['execution_latency'] = df['execution_time']
    elif 'wall_time' in df.columns:
        df['execution_latency'] = df['wall_time']
    else:
        df['execution_latency'] = None

    df['model'] = model_name
    all_records.append(
        df[['model', 'context_size', 'F1-Score', 'execution_latency']])

df_unified = pd.concat(all_records, ignore_index=True)

# ==========================================
# 3. WEED OUT OUTLIERS AND TRUNCATE CONTEXT WINDOWS
# ==========================================
# Remove the poorly performing lfm2 outlier model
df_filtered = df_unified[df_unified['model'] != 'ollama/lfm2:24b']

# Restrict context sizes strictly below 131072 tokens
df_filtered = df_filtered[df_filtered['context_size'] < 131072]

# Aggregate multiple entries using the statistical mean
summary_metrics = df_filtered.groupby(
    ['model', 'context_size']).mean().reset_index()

# Clean up model display names for the plot legend
summary_metrics['model'] = summary_metrics['model'].str.replace('ollama/', '')

# Set clean, professional scientific plot styles
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

# ==========================================
# PLOT 1: F1-SCORE COMPARISON
# ==========================================
plt.figure(figsize=(8, 4.5))
sns.lineplot(
    data=summary_metrics,
    x="context_size",
    y="F1-Score",
    hue="model",
    marker="o",
    linewidth=2.5,
    markersize=8
)
plt.xscale("log", base=2)
plt.title("Model Accuracy ($F_1$-Score) vs. Context Size (< 131,072 Tokens)",
          fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Context Window Size (Tokens)", fontsize=11)
plt.ylabel("Mean $F_1$-Score", fontsize=11)
plt.ylim(-0.05, 1.05)
plt.xticks(sorted(summary_metrics['context_size'].unique()), sorted(
    summary_metrics['context_size'].unique()))
plt.legend(title="Model", loc='lower left', frameon=True)
plt.tight_layout()
plt.savefig("filtered_model_f1.png", dpi=300)
plt.show()
plt.close()

# ==========================================
# PLOT 2: WALLTIME LATENCY COMPARISON
# ==========================================
plt.figure(figsize=(8, 4.5))
sns.lineplot(
    data=summary_metrics,
    x="context_size",
    y="execution_latency",
    hue="model",
    marker="s",
    linewidth=2.5,
    markersize=8
)
plt.xscale("log", base=2)
plt.yscale("log")  # Using log scale to handle large differences between models
plt.title("Execution Latency (Walltime) vs. Context Size (< 131,072 Tokens)",
          fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Context Window Size (Tokens)", fontsize=11)
plt.ylabel("Walltime Delay (Seconds, Log Scale)", fontsize=11)
plt.xticks(sorted(summary_metrics['context_size'].unique()), sorted(
    summary_metrics['context_size'].unique()))
plt.legend(title="Model", loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig("filtered_model_latency.png", dpi=300)
plt.show()
plt.close()

print("Cleaned visualizations generated and exported successfully.")
