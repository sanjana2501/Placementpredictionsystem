import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data.load_data import load_data
from src.data.preprocess import split_data, preprocess_train_test


def main():

    print("=" * 60)
    print("RANDOM FOREST CLASSIFICATION")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    print("\nDataset loaded successfully.")
    print("Shape:", df.shape)

    # 2. Split data
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

    # 3. Preprocess
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

    # 4. Create Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    # 5. Train
    print("\nTraining Random Forest...")

    model.fit(
        X_train_processed,
        y_train
    )

    print("Training completed successfully.")

    # 6. Predict
    print("\nMaking predictions...")

    y_pred = model.predict(X_test_processed)

    print("Prediction completed.")

    # 7. Evaluate
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

    # 8. Save model
    models_folder = PROJECT_ROOT / "models"
    models_folder.mkdir(exist_ok=True)

    model_path = models_folder / "random_forest.pkl"

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor
        },
        model_path
    )

    print("\nModel saved successfully:")
    print(model_path)

    print("\n" + "=" * 60)
    print("RANDOM FOREST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()