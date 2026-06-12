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

# Initialize the plot using subplots (avoiding .figure() to prevent clipping errors)
fig, ax = plt.subplots(figsize=(10, 6))

# Generate the side-by-side whisker plot for Walltime
sns.boxplot(
    data=df_combined,
    x='context_size',
    y='walltime',
    hue='model',
    ax=ax,
    palette=['#1f77b4', '#ff7f0e']  # Clean blue and orange color coding
)

# Apply log scale since times vary dramatically (from 40s to 1200s+)
ax.set_yscale('log')

# Title and labels adjustment
ax.set_title("Execution Latency (Walltime) Distribution across Power-of-Two Windows",
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Context Window Size (Tokens)", fontsize=11)
ax.set_ylabel("Walltime Delay (Seconds, Log Scale)", fontsize=11)

# Format the legend cleanly
ax.legend(title="Model Architecture", loc='upper right', frameon=True)

# Ensure no labels overlap or truncate
plt.tight_layout()

# Save the plot configuration directly to a file
plt.savefig("models_walltime_whisker_comparison.png", dpi=300)
plt.close()

print("Combined Walltime whisker chart successfully generated and saved.")
