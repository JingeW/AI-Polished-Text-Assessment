import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot, load_csv

# ------------------------------
# General Plot Variables
# ------------------------------
PLOT_STYLE = {
    "font.family": "Arial",  # Use Arial font
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.figsize": (10, 6),
}
plt.rcParams.update(PLOT_STYLE)
plt.rcParams['svg.fonttype'] = 'none'  # Ensure text is preserved in SVG

# ------------------------------
# Plot Customizations
# ------------------------------
def customize_plot(ax, grid_axis="y"):
    """Customize plot appearance: Remove top/right spines, keep bottom/left, and set grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Add dotted horizontal grid
    if grid_axis == "y":
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
    elif grid_axis == "x":
        ax.grid(axis="x", linestyle="--", linewidth=0.5)

# ------------------------------
# Figure 1: Boxplot for AI Detection (2020 vs 2024) within Asia and USA
# ------------------------------
def plot_ai_score_by_year_and_location(df, output_dir):
    # Filter for original articles and years 2020 and 2024
    filtered_df = df[(df["version"] == "original") & (df["year"].isin([2020, 2024]))]

    # Explicitly set the order of locations to ensure Asia is first
    filtered_df["location"] = pd.Categorical(filtered_df["location"], categories=["Asian", "USA"], ordered=True)

    # Create a custom palette with two distinct colors
    custom_palette = ["#3594cc", "#ea801c"]  # Example colors for 2020 and 2024

    # Plot boxplot
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        data=filtered_df,
        x="location",  # X-axis groups (Asian and USA)
        y="completely_generated_prob",
        hue="year",  # Hue for 2020 vs 2024
        dodge=True,
        palette=custom_palette
    )

    # Customizations
    customize_plot(ax)
    plt.ylabel("AI-Generated Probability")
    plt.xlabel("")  # Remove x-axis label
    plt.legend(title="Year", loc="upper right", frameon=False)  # No legend frame
    plt.tight_layout()

    # Save the plot
    save_plot("fig1", output_dir)

# ------------------------------
# Figure 2: Boxplot Across Versions by Location
# ------------------------------
def plot_ai_score_by_location_and_reps(df, output_dir):
    # Define custom color palette
    color_palette = ["#3594cc", "#ea801c", "#54a1a1", "#1f6f6f"]

    filtered_df = df[df["version"].isin(["original", "rep1", "rep2", "rep3"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    locations = ["Asian", "USA"]

    for i, loc in enumerate(locations):
        loc_df = filtered_df[filtered_df["location"] == loc]
        ax = sns.boxplot(
            data=loc_df,
            x="version",
            y="completely_generated_prob",
            ax=axes[i],
            palette=color_palette
        )
        customize_plot(ax)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("AI-Generated Probability" if i == 0 else "")
        axes[i].set_title(loc, fontsize=16)

    plt.tight_layout()
    save_plot("fig2", output_dir)


if __name__ == "__main__":
    results_csv_path = "results/ai_detection_results.csv"
    plots_output_dir = "plots"

    ai_results_df = load_csv(results_csv_path)
    plot_ai_score_by_year_and_location(ai_results_df, plots_output_dir)
    plot_ai_score_by_location_and_reps(ai_results_df, plots_output_dir)

    print("All AI Detection boxplots have been generated and saved.")
