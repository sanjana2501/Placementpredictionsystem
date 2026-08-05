import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print("BASE_DIR:", BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "placement_data.csv")
print("DATA_PATH:", DATA_PATH)
print("File exists:", os.path.exists(DATA_PATH))


def load_data():
    return pd.read_csv(DATA_PATH)


def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "PlacementStatus"
    }