import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL = "gemma4:31b"

# Assuming your JSON is stored in a file named 'data.json'
with open('logs_prejoined_queries.json', 'r') as f:
    data = json.load(f)

# Extract the list of dictionaries for the specific model
df = pd.DataFrame(data[MODEL])

# Ensure required columns are numerical
df['context_size'] = df['context_size'].astype(int)
df['walltime'] = df['walltime'].astype(float)
df['Precision'] = df['Precision'].astype(float)
df['Recall'] = df['Recall'].astype(float)
df['TP'] = df['TP'].astype(int)  # Added TP

# Set the visual style
sns.set_theme(style="whitegrid")

# Create a figure with 1 row and 4 columns, expanding the width to 24
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# 1. Boxplot for Precision
sns.boxplot(x='context_size', y='Precision',
            data=df, ax=axes[0], color="skyblue")
axes[0].set_title('Precision Distribution by Context Size')
axes[0].set_xlabel('Context Size (Tokens)')
axes[0].set_ylabel('Precision')
axes[0].tick_params(axis='x', rotation=45)

# 2. Boxplot for Recall
sns.boxplot(x='context_size', y='Recall', data=df,
            ax=axes[1], color="lightgreen")
axes[1].set_title('Recall Distribution by Context Size')
axes[1].set_xlabel('Context Size (Tokens)')
axes[1].set_ylabel('Recall')
axes[1].tick_params(axis='x', rotation=45)

# 3. Boxplot for True Positives (TP) - NEW
sns.boxplot(x='context_size', y='TP', data=df,
            ax=axes[2], color="mediumpurple")
axes[2].set_title('True Positives (TP) by Context Size')
axes[2].set_xlabel('Context Size (Tokens)')
axes[2].set_ylabel('True Positives (Count)')
axes[2].tick_params(axis='x', rotation=45)

# 4. Boxplot for Execution Time
sns.boxplot(x='context_size', y='walltime',
            data=df, ax=axes[3], color="salmon")
axes[3].set_title('Execution Time by Context Size')
axes[3].set_xlabel('Context Size (Tokens)')
axes[3].set_ylabel('Execution Time (s)')
axes[3].tick_params(axis='x', rotation=45)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the plot and show
plt.savefig(f'whisker_diagrams_with_tp_{MODEL}_prejoined.png')
plt.show()

# Brief summary statistics including TP
summary = df.groupby('context_size')[
    ['Precision', 'Recall', 'TP', 'walltime']].agg(['mean', 'std']).round(3)
print("\n--- Summary Statistics ---")
print(summary)
