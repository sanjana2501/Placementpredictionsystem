# ============================================================
# EXPLORATORY DATA ANALYSIS - PLACEMENT PREDICTION SYSTEM
# ============================================================

import sys
from pathlib import Path

# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. IMPORTS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_data import load_data


# ============================================================
# 3. PATHS
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. MAIN EDA FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("PLACEMENT PREDICTION SYSTEM - EDA")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")


    # --------------------------------------------------------
    # Basic Information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


    # --------------------------------------------------------
    # Statistical Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)

    print(df.describe())


    # --------------------------------------------------------
    # Placement Status Distribution
    # --------------------------------------------------------

    if "PlacementStatus" in df.columns:

        print("\nPlacement Status:")
        print(df["PlacementStatus"].value_counts())

        plt.figure(figsize=(7, 5))

        df["PlacementStatus"].value_counts().sort_index().plot(
            kind="bar"
        )

        plt.title("Placement Status Distribution")
        plt.xlabel("Placement Status")
        plt.ylabel("Number of Students")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "placement_status_distribution.png"
        )

        plt.close()


    # --------------------------------------------------------
    # Gender Distribution
    # --------------------------------------------------------

    if "Gender" in df.columns:

        plt.figure(figsize=(7, 5))

        df["Gender"].value_counts().plot(
            kind="bar"
        )

        plt.title("Gender Distribution")
        plt.xlabel("Gender")
        plt.ylabel("Number of Students")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "gender_distribution.png"
        )

        plt.close()


    # --------------------------------------------------------
    # Stream Distribution
    # --------------------------------------------------------

    if "Stream" in df.columns:

        plt.figure(figsize=(8, 5))

        df["Stream"].value_counts().plot(
            kind="bar"
        )

        plt.title("Stream Distribution")
        plt.xlabel("Stream")
        plt.ylabel("Number of Students")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "stream_distribution.png"
        )

        plt.close()


    # --------------------------------------------------------
    # College Tier Distribution
    # --------------------------------------------------------

    if "CollegeTier" in df.columns:

        plt.figure(figsize=(7, 5))

        df["CollegeTier"].value_counts().sort_index().plot(
            kind="bar"
        )

        plt.title("College Tier Distribution")
        plt.xlabel("College Tier")
        plt.ylabel("Number of Students")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "college_tier_distribution.png"
        )

        plt.close()


    # --------------------------------------------------------
    # CGPA Tier Distribution
    # --------------------------------------------------------

    if "CGPA_Tier" in df.columns:

        plt.figure(figsize=(7, 5))

        df["CGPA_Tier"].value_counts().plot(
            kind="bar"
        )

        plt.title("CGPA Tier Distribution")
        plt.xlabel("CGPA Tier")
        plt.ylabel("Number of Students")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "cgpa_tier_distribution.png"
        )

        plt.close()


    # --------------------------------------------------------
    # Salary Package Distribution
    # --------------------------------------------------------

    if "Salary Package" in df.columns:

        plt.figure(figsize=(8, 5))

        df["Salary Package"].plot(
            kind="hist",
            bins=30
        )

        plt.title("Salary Package Distribution")
        plt.xlabel("Salary Package")
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "salary_package_distribution.png"
        )

        plt.close()


    # --------------------------------------------------------
    # CGPA Distribution
    # --------------------------------------------------------

    if "CGPA" in df.columns:

        plt.figure(figsize=(8, 5))

        df["CGPA"].plot(
            kind="hist",
            bins=30
        )

        plt.title("CGPA Distribution")
        plt.xlabel("CGPA")
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "cgpa_distribution.png"
        )

        plt.close()


    # --------------------------------------------------------
    # Placement vs CGPA
    # --------------------------------------------------------

    if "CGPA" in df.columns and "PlacementStatus" in df.columns:

        plt.figure(figsize=(8, 5))

        df.boxplot(
            column="CGPA",
            by="PlacementStatus"
        )

        plt.title("CGPA vs Placement Status")
        plt.suptitle("")
        plt.xlabel("Placement Status")
        plt.ylabel("CGPA")
        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "cgpa_vs_placement.png"
        )

        plt.close()


    # --------------------------------------------------------
    # Placement vs Salary Package
    # --------------------------------------------------------

    if (
        "Salary Package" in df.columns
        and "PlacementStatus" in df.columns
    ):

        placed = df[
            df["PlacementStatus"] == 1
        ]["Salary Package"]

        if not placed.empty:

            plt.figure(figsize=(8, 5))

            placed.plot(
                kind="hist",
                bins=30
            )

            plt.title(
                "Salary Package Distribution of Placed Students"
            )

            plt.xlabel("Salary Package")
            plt.ylabel("Frequency")
            plt.tight_layout()

            plt.savefig(
                CHARTS_DIR / "placed_salary_distribution.png"
            )

            plt.close()


    # --------------------------------------------------------
    # Save Dataset Information
    # --------------------------------------------------------

    summary_path = RESULTS_DIR / "eda_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:

        file.write("PLACEMENT PREDICTION SYSTEM - EDA SUMMARY\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Rows: {df.shape[0]}\n")
        file.write(f"Columns: {df.shape[1]}\n\n")

        file.write("Missing Values:\n")
        file.write(
            df.isnull().sum().to_string()
        )

        file.write("\n\nDuplicate Rows:\n")
        file.write(
            str(df.duplicated().sum())
        )

        file.write("\n\nStatistical Summary:\n")
        file.write(
            df.describe().to_string()
        )


    # --------------------------------------------------------
    # Completion Message
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nCharts saved to:")
    print(CHARTS_DIR)

    print(f"\nSummary saved to:")
    print(summary_path)

    print("\n" + "=" * 60)


# ============================================================
# 5. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()