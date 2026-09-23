# ============================================================
# REGRESSION MODELS - PLACEMENT PREDICTION SYSTEM
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

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.data.load_data import load_data
from src.data.preprocess import split_data, preprocess_train_test


# ============================================================
# 3. MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, name):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return model


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PLACEMENT PREDICTION SYSTEM - REGRESSION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")


    # --------------------------------------------------------
    # Split data
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="Salary Package",
        drop_column=[
            "StudentID",
            "PlacementStatus",
            "IsAnomaly"
        ],
        test_size=0.20,
        random_state=42,
        stratify=False
    )

    print(f"Training data: {X_train.shape}")
    print(f"Testing data : {X_test.shape}")


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print("\nPreprocessing data...")

    (
        X_train_processed,
        X_test_processed,
        numeric_columns,
        one_hot_columns,
        ordinal_columns
    ) = preprocess_train_test(
        X_train,
        X_test
    )

    print(
        f"Processed training features: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed testing features : "
        f"{X_test_processed.shape}"
    )


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(
            alpha=1.0
        ),

        "Lasso Regression": Lasso(
            alpha=0.01,
            max_iter=10000
        ),

        "Elastic Net Regression": ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000
        )
    }


    # --------------------------------------------------------
    # Train and evaluate models
    # --------------------------------------------------------

    trained_models = {}

    for name, model in models.items():

        trained_models[name] = evaluate_model(
            model,
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            name
        )


    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL REGRESSION MODELS COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# 5. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()