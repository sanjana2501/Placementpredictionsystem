import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data.load_data import load_data
from src.data.preprocess import split_data, preprocess_train_test


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GRADIENT BOOSTING CLASSIFICATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)

    # --------------------------------------------------------
    # 2. SPLIT DATA
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_column=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ],
        test_size=0.2,
        random_state=42
    )

    print("\nData split completed.")
    print("Training data:", X_train.shape)
    print("Testing data :", X_test.shape)

    # --------------------------------------------------------
    # 3. PREPROCESS DATA
    # --------------------------------------------------------

    result = preprocess_train_test(
        X_train,
        X_test
    )

    X_train_processed = result[0]
    X_test_processed = result[1]
    preprocessor = result[4]

    print("\nPreprocessing completed.")
    print("Processed training data:", X_train_processed.shape)
    print("Processed testing data :", X_test_processed.shape)

    # --------------------------------------------------------
    # 4. CREATE GRADIENT BOOSTING MODEL
    # --------------------------------------------------------

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # --------------------------------------------------------
    # 5. TRAIN MODEL
    # --------------------------------------------------------

    print("\nTraining Gradient Boosting...")

    model.fit(
        X_train_processed,
        y_train
    )

    print("Training completed successfully.")

    # --------------------------------------------------------
    # 6. MAKE PREDICTIONS
    # --------------------------------------------------------

    print("\nMaking predictions...")

    y_pred = model.predict(
        X_test_processed
    )

    print("Prediction completed.")

    # --------------------------------------------------------
    # 7. MODEL RESULTS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # --------------------------------------------------------
    # 8. CREATE CHART
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.scatter(
        range(len(y_test)),
        y_test,
        label="Actual"
    )

    plt.scatter(
        range(len(y_pred)),
        y_pred,
        label="Predicted"
    )

    plt.title(
        "Gradient Boosting - Actual vs Predicted"
    )

    plt.xlabel("Test Sample")

    plt.ylabel("Placement Status")

    plt.legend()

    plt.grid(True)

    # --------------------------------------------------------
    # 9. SAVE CHART
    # --------------------------------------------------------

    results_folder = PROJECT_ROOT / "results"

    results_folder.mkdir(
        exist_ok=True
    )

    chart_path = (
        results_folder /
        "gradient_boosting_chart.png"
    )

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nChart saved to:")
    print(chart_path)

    # --------------------------------------------------------
    # 10. SAVE MODEL
    # --------------------------------------------------------

    models_folder = PROJECT_ROOT / "models"

    models_folder.mkdir(
        exist_ok=True
    )

    model_path = (
        models_folder /
        "gradient_boosting.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor
        },
        model_path
    )

    print("\nModel saved successfully:")
    print(model_path)

    # --------------------------------------------------------
    # 11. COMPLETED
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()