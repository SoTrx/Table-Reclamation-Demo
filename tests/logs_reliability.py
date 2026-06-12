import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load data from the JSON file
with open("logs.json", "r") as f:
    logs = json.load(f)

# 2. Define non-power-of-two context sizes to exclude for Gemma
excluded_sizes = {512, 1536, 2560, 3072, 3584, 4608}

# Filter Gemma: remove complete failures (F1 == 0) and non-power-of-two sizes
filtered_gemma = [
    run for run in logs['gemma4:31b']
    if run.get("F1-Score") != 0 and run.get("F1-Score") != 0.0 and run.get("context_size") not in excluded_sizes
]

# Build clean DataFrames for both models
df_gemma = pd.DataFrame(filtered_gemma)
df_qwen = pd.DataFrame(logs['qwen3.6:35b'])

df_gemma['model'] = 'Gemma4:31b'
df_qwen['model'] = 'Qwen3.6:35b'

# Combine into a unified dataset for plotting
df_combined = pd.concat([df_gemma, df_qwen], ignore_index=True)
df_combined = df_combined.sort_values(by="context_size")

# 3. Apply professional scientific plot styles
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

# =======================================================
# PLOT 1: MERGED WHISKER PLOT FOR F1-SCORE
# =======================================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=df_combined,
    x='context_size',
    y='F1-Score',
    hue='model',
    ax=ax,
    palette=['#1f77b4', '#ff7f0e']  # Match colors across both diagrams
)

# Set y-limits to zoom into the high-performance distribution region
ax.set_ylim(0.85, 1.02)

ax.set_title("Model $F_1$-Score Distribution across Power-of-Two Windows",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Context Window Size (Tokens)", fontsize=11)
ax.set_ylabel("$F_1$-Score", fontsize=11)
ax.legend(title="Model Architecture", loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig("merged_models_f1_whisker.png", dpi=300)
plt.close()

# =======================================================
# PLOT 2: MERGED WHISKER PLOT FOR WALLTIME
# =======================================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=df_combined,
    x='context_size',
    y='walltime',
    hue='model',
    ax=ax,
    palette=['#1f77b4', '#ff7f0e']
)

# Apply logarithmic scale to account for wide delay variations
ax.set_yscale('log')

ax.set_title("Execution Latency (Walltime) Distribution across Power-of-Two Windows",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Context Window Size (Tokens)", fontsize=11)
ax.set_ylabel("Walltime Delay (Seconds, Log Scale)", fontsize=11)
ax.legend(title="Model Architecture", loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("merged_models_walltime_whisker.png", dpi=300)
plt.close()

print("Both merged whisker plots successfully generated and saved.")
