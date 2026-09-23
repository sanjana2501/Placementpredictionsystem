from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# Placementpredictionsystem/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "placement_data.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the original placement dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Make sure data/placement_data.csv exists."
        )

    return pd.read_csv(DATASET_PATH)


# ============================================================
# DATASET SUMMARY
# ============================================================

def get_summary(df):
    """
    Return basic dataset information.
    """

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "PlacementStatus"
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_data()

    print("First 5 Rows of the Dataset:")
    print(df.head())

    print("\nDataset Summary:")
    print(get_summary(df))