import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load the Data
with open("logs_old.json", "r") as f:
    data = json.load(f)

# 2. Extract metrics into a flat list
rows = []
for model, runs in data.items():
    for run in runs:
        # Handle time mapping (some iterations used 'walltime' instead of 'execution_time')
        time_taken = run.get('execution_time', run.get('walltime', 0))
        rows.append({
            'Model': model,
            'Context Size': run.get('context_size'),
            'F1-Score': run.get('F1-Score'),
            'Recall': run.get('Recall'),
            'Precision': run.get('Precision'),
            'Accuracy': run.get('Accuracy'),
            'Time (s)': time_taken
        })

# 3. Build and save the DataFrame
df = pd.DataFrame(rows)
df.to_csv("model_comparison.csv", index=False)

# 4. Plot 1: F1-Score vs Context Size for each model
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Context Size', y='F1-Score', hue='Model', marker='o')
plt.title('F1-Score vs Context Size across Models')
plt.xscale('log', base=2)
plt.ylabel('F1-Score')
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("f1_score_comparison.png")

# 5. Calculate Average Metrics per Model
df_avg = df.groupby('Model')[
    ['F1-Score', 'Recall', 'Precision', 'Accuracy', 'Time (s)']].mean().reset_index()
df_avg = df_avg.sort_values('F1-Score', ascending=False)

# 6. Plot 2: Bar chart of Average F1, Recall, Precision
df_melted = df_avg.melt(
    id_vars='Model',
    value_vars=['F1-Score', 'Recall', 'Precision'],
    var_name='Metric',
    value_name='Score'
)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
plt.title('Average Classification Metrics per Model')
plt.ylim(0, 1.1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("average_metrics_comparison.png")
