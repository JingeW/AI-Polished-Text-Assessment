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
    custom_palette = ["#3594cc", "#228B3B"]  # Example colors for 2020 and 2024

    # Plot boxplot
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        data=filtered_df,
        x="location",  # X-axis groups (Asian and USA)
        y="completely_generated_prob",
        hue="year",  # Hue for 2020 vs 2024
        dodge=True,
        palette=custom_palette,
        showfliers=False  # Remove outliers
    )

    # Overlay data points
    sns.stripplot(
        data=filtered_df,
        x="location",
        y="completely_generated_prob",
        hue="year",
        dodge=True,
        marker='x',
        color='black',
        alpha=0.8,
        ax=ax,
        linewidth=1,
        zorder=3
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
    color_palette = ["#228B3B", "#40AD5A", "#6CBA7D", "#3594cc"]

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
            palette=color_palette,
            showfliers=False  # Remove outliers
        )
        sns.stripplot(
            data=loc_df,
            x="version",
            y="completely_generated_prob",
            color='black',
            marker='x',
            alpha=0.8,
            linewidth=1,
            zorder=3,
            ax=axes[i]
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
