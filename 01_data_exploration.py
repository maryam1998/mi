"""
01_data_exploration.py
======================
Exploratory Data Analysis (EDA) for the project:
"Prediction of Mechanical Properties of Forged Steels using Machine Learning"

What this script does:
    1. Loads the CSV dataset
    2. Prints basic info (shape, columns, data types)
    3. Checks for missing values (and duplicates)
    4. Prints descriptive statistics
    5. Plots a correlation heatmap of the numeric features
    6. Saves all plots into the `figures/` folder

Usage:
    python 01_data_exploration.py                  # uses DATA_PATH below
    python 01_data_exploration.py path/to/file.csv # custom path
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows (works on servers too)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------
DATA_PATH = Path("data/forged_steel.csv")   # <-- change if your file is elsewhere
FIGURES_DIR = Path("figures")

# Keywords used to *guess* which columns are target properties.
# Check the printed list and adjust if your column names differ.
TARGET_KEYWORDS = ["uts", "tensile", "yield", "ys", "hardness", "hv", "hb", "hrc"]

sns.set_theme(style="whitegrid")


def print_section(title: str) -> None:
    """Print a visible section header in the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # Allow the CSV path to be passed on the command line
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH
    FIGURES_DIR.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------
    # 2. Load data
    # ----------------------------------------------------------------------
    print_section("1. LOADING DATA")
    if not data_path.exists():
        sys.exit(f"File not found: {data_path}\n"
                 "Download the dataset and place it there, or pass the path "
                 "as an argument.")
    df = pd.read_csv(data_path)
    print(f"Loaded: {data_path}")

    # ----------------------------------------------------------------------
    # 3. Basic information
    # ----------------------------------------------------------------------
    print_section("2. BASIC INFORMATION")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("Columns and data types:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string())

    # ----------------------------------------------------------------------
    # 4. Missing values and duplicates
    # ----------------------------------------------------------------------
    print_section("3. MISSING VALUES AND DUPLICATES")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"missing": missing, "percent": missing_pct})
    missing_table = missing_table[missing_table["missing"] > 0]

    if missing_table.empty:
        print("No missing values found.")
    else:
        print(missing_table.sort_values("percent", ascending=False).to_string())

        # Bar plot of missing-value percentages
        fig, ax = plt.subplots(figsize=(8, 5))
        missing_table["percent"].sort_values().plot.barh(ax=ax, color="tomato")
        ax.set_xlabel("Missing values (%)")
        ax.set_title("Missing values per column")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "missing_values.png", dpi=300)
        plt.close(fig)

    n_dup = df.duplicated().sum()
    print(f"\nFully duplicated rows: {n_dup}")

    # ----------------------------------------------------------------------
    # 5. Descriptive statistics
    # ----------------------------------------------------------------------
    print_section("4. DESCRIPTIVE STATISTICS (numeric columns)")
    numeric_df = df.select_dtypes(include="number")
    print(numeric_df.describe().T.to_string())
    # Tip for a metallurgist: check that min/max values are physically
    # reasonable (e.g. carbon content, negative values, hardness range).

    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        print(f"\nNon-numeric columns (not in heatmap): {non_numeric}")

    # ----------------------------------------------------------------------
    # 6. Distributions of every numeric column
    # ----------------------------------------------------------------------
    print_section("5. DISTRIBUTIONS")
    n_cols = 4
    n_rows = -(-numeric_df.shape[1] // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    for ax, col in zip(axes.ravel(), numeric_df.columns):
        ax.hist(numeric_df[col].dropna(), bins=40, color="steelblue")
        ax.set_title(col, fontsize=10)
    for ax in axes.ravel()[numeric_df.shape[1]:]:
        ax.axis("off")  # hide unused subplots
    fig.suptitle("Distribution of numeric features", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=300)
    plt.close(fig)
    print("Saved: figures/distributions.png")

    # ----------------------------------------------------------------------
    # 7. Correlation heatmap
    # ----------------------------------------------------------------------
    print_section("6. CORRELATION HEATMAP")
    corr = numeric_df.corr()  # Pearson correlation (linear relationships)

    # Guess target columns from their names (just for information)
    targets = [c for c in numeric_df.columns
               if any(k in c.lower() for k in TARGET_KEYWORDS)]
    print(f"Columns that look like target properties: {targets}")

    size = max(8, 0.6 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=len(corr) <= 25, fmt=".2f", annot_kws={"size": 7},
                square=True, linewidths=0.3, cbar_kws={"label": "Pearson r"},
                ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close(fig)
    print("Saved: figures/correlation_heatmap.png")

    # Strongest correlations with each guessed target
    for t in targets:
        print(f"\nTop correlations with '{t}':")
        print(corr[t].drop(t).sort_values(key=abs, ascending=False)
              .head(5).round(3).to_string())

    print_section("DONE")
    print(f"All figures saved in: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
