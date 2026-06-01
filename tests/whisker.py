import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL = "gemma4:31b"
# MODEL = "qwen3.6:35b"


def create_whisker_plots(json_filepath):
    # 1. Load the JSON data
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # Extract the relevant list of records
    records = data[MODEL]

    # 2. Convert to a Pandas DataFrame
    df = pd.DataFrame(records)

    # Ensure context_size is treated as a categorical variable for evenly spaced ticks
    df["context_size"] = df["context_size"].astype(str)

    # Set the visual style
    sns.set_theme(style="whitegrid")

    # 3. Create a figure with 3 side-by-side subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Performance Metrics by Context Size (" + MODEL + ")",
                 fontsize=16, fontweight='bold')

    # Plot 1: Precision
    sns.boxplot(data=df, x="context_size", y="Precision",
                ax=axes[0], color="skyblue")
    axes[0].set_title("Precision vs Context Size")
    axes[0].set_xlabel("Context Size")
    axes[0].set_ylabel("Precision")
    axes[0].set_ylim(-0.05, 1.05)  # Keep the 0-1 scale uniform

    # Plot 2: Recall
    sns.boxplot(data=df, x="context_size", y="Recall",
                ax=axes[1], color="lightgreen")
    axes[1].set_title("Recall vs Context Size")
    axes[1].set_xlabel("Context Size")
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(-0.05, 1.05)

    # Plot 3: WallTime
    sns.boxplot(data=df, x="context_size", y="walltime",
                ax=axes[2], color="salmon")
    axes[2].set_title("WallTime vs Context Size")
    axes[2].set_xlabel("Context Size")
    axes[2].set_ylabel("WallTime (seconds)")

    # Rotate x-axis labels if they overlap
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    # Adjust layout and display
    plt.tight_layout()

    # Save the plot as a high-resolution PNG instead of showing it
    output_filename = f"whisker_plots_{MODEL}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved as {output_filename}")


if __name__ == "__main__":
    # Ensure you have saved your JSON data into a file named 'data.json'
    create_whisker_plots('logs.json')
