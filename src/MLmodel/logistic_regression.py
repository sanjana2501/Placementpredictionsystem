from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

MODEL_DIR = PROJECT_ROOT / "models"


TRAIN_PATH = (
    DATA_DIR /
    "preprocessed_train.csv"
)

TEST_PATH = (
    DATA_DIR /
    "preprocessed_test.csv"
)

MODEL_PATH = (
    MODEL_DIR /
    "logistic_regression.pkl"
)


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

def load_preprocessed_data():

    if not TRAIN_PATH.exists():

        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_PATH}\n"
            "Run preprocess.py first."
        )


    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"Testing file not found:\n{TEST_PATH}\n"
            "Run preprocess.py first."
        )


    train_data = pd.read_csv(
        TRAIN_PATH
    )

    test_data = pd.read_csv(
        TEST_PATH
    )

    return (
        train_data,
        test_data
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    return LogisticRegression(
        max_iter=1000,
        random_state=42
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train
):

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nLogistic Regression model trained successfully!"
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    print(
        "\n========================================"
    )

    print(
        "LOGISTIC REGRESSION EVALUATION"
    )

    print(
        "========================================"
    )


    print(
        f"\nAccuracy: {accuracy:.4f}"
    )


    print(
        "\nClassification Report:"
    )


    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Placed",
                "Placed"
            ],
            zero_division=0
        )
    )


    return y_pred


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    joblib.dump(
        model,
        MODEL_PATH
    )


    print(
        "\nModel saved successfully!"
    )

    print(
        "Location:",
        MODEL_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "RUNNING LOGISTIC REGRESSION"
    )

    print(
        "========================================"
    )


    # LOAD
    train_data, test_data = (
        load_preprocessed_data()
    )


    print(
        "\nTraining Data Shape:",
        train_data.shape
    )

    print(
        "Testing Data Shape:",
        test_data.shape
    )


    # FEATURES + TARGET
    X_train = train_data.drop(
        columns=["PlacementStatus"]
    )

    y_train = train_data[
        "PlacementStatus"
    ]


    X_test = test_data.drop(
        columns=["PlacementStatus"]
    )

    y_test = test_data[
        "PlacementStatus"
    ]


    print(
        "\nFeatures:",
        X_train.shape[1]
    )


    # MODEL
    model = create_model()


    # TRAIN
    model = train_model(
        model,
        X_train,
        y_train
    )


    # EVALUATE
    evaluate_model(
        model,
        X_test,
        y_test
    )


    # SAVE
    save_model(
        model
    )


    print(
        "\n========================================"
    )

    print(
        "LOGISTIC REGRESSION FINISHED"
    )

    print(
        "========================================"
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()