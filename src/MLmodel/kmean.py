import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_data


def main():

    print("=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)

    # 2. Select numerical features
    drop_columns = [
        "StudentID",
        "PlacementStatus",
        "Salary Package",
        "IsAnomaly"
    ]

    X = df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    X = X.select_dtypes(include=["number"])

    # Handle missing values
    X = X.fillna(X.median())

    print("\nFeatures selected for clustering:")
    print(X.columns.tolist())

    print("\nFeature shape:", X.shape)

    # 3. Scale data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\nFeature scaling completed.")

    # 4. Create K-Means model
    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    # 5. Train
    print("\nTraining K-Means...")

    clusters = model.fit_predict(X_scaled)

    print("K-Means training completed.")

    # 6. Add cluster labels
    result = df.copy()
    result["Cluster"] = clusters

    print("\nCluster distribution:")
    print(result["Cluster"].value_counts().sort_index())

    # 7. Save results
    results_folder = PROJECT_ROOT / "results"
    results_folder.mkdir(exist_ok=True)

    output_path = results_folder / "kmeans_results.csv"

    result.to_csv(
        output_path,
        index=False
    )

    # 8. Save model
    models_folder = PROJECT_ROOT / "models"
    models_folder.mkdir(exist_ok=True)

    model_path = models_folder / "kmeans.pkl"

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": X.columns.tolist()
        },
        model_path
    )

    print("\nResults saved to:")
    print(output_path)

    print("\nModel saved to:")
    print(model_path)

    print("\n" + "=" * 60)
    print("K-MEANS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()