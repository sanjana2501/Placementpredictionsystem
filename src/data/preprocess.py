# ============================================================
# PREPROCESSING - PLACEMENT PREDICTION SYSTEM
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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from src.data.load_data import load_data


# ============================================================
# 3. FEATURE GROUPS
# ============================================================

# Categorical features that need One-Hot Encoding
ONE_HOT_FEATURES = [
    "Gender",
    "City",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs"
]

# Ordered categorical features
ORDINAL_FEATURES = [
    "CollegeTier",
    "CGPA_Tier"
]


# ============================================================
# 4. SPLIT DATA
# ============================================================

def split_data(
    df,
    target_column,
    drop_column=None,
    test_size=0.20,
    random_state=42,
    stratify=False
):
    """
    Splits the dataset into training and testing data.

    Parameters:
        df             : Input DataFrame
        target_column  : Target column
        drop_column    : Columns to remove before training
        test_size      : Test data percentage
        random_state   : Random seed
        stratify       : Whether to maintain target distribution

    Returns:
        X_train, X_test, y_train, y_test
    """

    from sklearn.model_selection import train_test_split

    data = df.copy()

    if drop_column is None:
        drop_column = []

    if isinstance(drop_column, str):
        drop_column = [drop_column]

    # Remove unwanted columns
    columns_to_drop = [
        column
        for column in drop_column
        if column in data.columns
    ]

    data = data.drop(columns=columns_to_drop)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Stratified split if requested
    stratify_value = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_value
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 5. PREPROCESS TRAINING AND TESTING DATA
# ============================================================

def preprocess_train_test(X_train, X_test):
    """
    Preprocesses training and testing datasets.

    Steps:
        1. Handle missing numeric values
        2. Standardize numeric features
        3. One-Hot Encode categorical features
        4. Ordinal Encode ordered categorical features

    Returns:
        X_train_processed
        X_test_processed
        numeric_columns
        one_hot_columns
        ordinal_columns
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    # --------------------------------------------------------
    # Identify available columns
    # --------------------------------------------------------

    one_hot_columns = [
        column
        for column in ONE_HOT_FEATURES
        if column in X_train.columns
    ]

    ordinal_columns = [
        column
        for column in ORDINAL_FEATURES
        if column in X_train.columns
    ]

    # Numeric columns
    numeric_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


    # --------------------------------------------------------
    # Numeric Imputation
    # --------------------------------------------------------

    if numeric_columns:

        numeric_imputer = SimpleImputer(
            strategy="median"
        )

        X_train_numeric = numeric_imputer.fit_transform(
            X_train[numeric_columns]
        )

        X_test_numeric = numeric_imputer.transform(
            X_test[numeric_columns]
        )

        # ----------------------------------------------------
        # Numeric Scaling
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_numeric = scaler.fit_transform(
            X_train_numeric
        )

        X_test_numeric = scaler.transform(
            X_test_numeric
        )

        X_train_numeric = pd.DataFrame(
            X_train_numeric,
            columns=numeric_columns,
            index=X_train.index
        )

        X_test_numeric = pd.DataFrame(
            X_test_numeric,
            columns=numeric_columns,
            index=X_test.index
        )

    else:

        X_train_numeric = pd.DataFrame(index=X_train.index)
        X_test_numeric = pd.DataFrame(index=X_test.index)


    # --------------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------------

    if one_hot_columns:

        try:
            one_hot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

        except TypeError:
            # For older versions of scikit-learn
            one_hot_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        X_train_one_hot = one_hot_encoder.fit_transform(
            X_train[one_hot_columns]
        )

        X_test_one_hot = one_hot_encoder.transform(
            X_test[one_hot_columns]
        )

        one_hot_feature_names = (
            one_hot_encoder.get_feature_names_out(
                one_hot_columns
            )
        )

        X_train_one_hot = pd.DataFrame(
            X_train_one_hot,
            columns=one_hot_feature_names,
            index=X_train.index
        )

        X_test_one_hot = pd.DataFrame(
            X_test_one_hot,
            columns=one_hot_feature_names,
            index=X_test.index
        )

    else:

        X_train_one_hot = pd.DataFrame(index=X_train.index)
        X_test_one_hot = pd.DataFrame(index=X_test.index)


    # --------------------------------------------------------
    # Ordinal Encoding
    # --------------------------------------------------------

    if ordinal_columns:

        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X_train_ordinal = ordinal_encoder.fit_transform(
            X_train[ordinal_columns]
        )

        X_test_ordinal = ordinal_encoder.transform(
            X_test[ordinal_columns]
        )

        X_train_ordinal = pd.DataFrame(
            X_train_ordinal,
            columns=ordinal_columns,
            index=X_train.index
        )

        X_test_ordinal = pd.DataFrame(
            X_test_ordinal,
            columns=ordinal_columns,
            index=X_test.index
        )

    else:

        X_train_ordinal = pd.DataFrame(index=X_train.index)
        X_test_ordinal = pd.DataFrame(index=X_test.index)


    # --------------------------------------------------------
    # Combine All Features
    # --------------------------------------------------------

    X_train_processed = pd.concat(
        [
            X_train_numeric,
            X_train_one_hot,
            X_train_ordinal
        ],
        axis=1
    )

    X_test_processed = pd.concat(
        [
            X_test_numeric,
            X_test_one_hot,
            X_test_ordinal
        ],
        axis=1
    )

    # Make sure column order is identical
    X_test_processed = X_test_processed[
        X_train_processed.columns
    ]

    return (
        X_train_processed,
        X_test_processed,
        numeric_columns,
        one_hot_columns,
        ordinal_columns
    )


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PLACEMENT PREDICTION SYSTEM - PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")


    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_column=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ],
        test_size=0.20,
        random_state=42,
        stratify=True
    )

    print(f"Training data: {X_train.shape}")
    print(f"Testing data : {X_test.shape}")


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print("\nApplying preprocessing...")

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


    # --------------------------------------------------------
    # Add target column
    # --------------------------------------------------------

    train_processed = X_train_processed.copy()
    test_processed = X_test_processed.copy()

    train_processed["PlacementStatus"] = y_train.values
    test_processed["PlacementStatus"] = y_test.values


    # --------------------------------------------------------
    # Create data directory
    # --------------------------------------------------------

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------------
    # Save processed datasets
    # --------------------------------------------------------

    train_path = data_dir / "preprocessed_train.csv"
    test_path = data_dir / "preprocessed_test.csv"

    train_processed.to_csv(
        train_path,
        index=False
    )

    test_processed.to_csv(
        test_path,
        index=False
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nOriginal dataset : {df.shape}")
    print(f"Training dataset : {train_processed.shape}")
    print(f"Testing dataset  : {test_processed.shape}")

    print(f"\nNumeric features : {len(numeric_columns)}")
    print(f"One-hot features : {len(one_hot_columns)}")
    print(f"Ordinal features  : {len(ordinal_columns)}")

    print("\nSaved files:")

    print(f"1. {train_path}")
    print(f"2. {test_path}")

    print("\nFirst 5 processed rows:")
    print(train_processed.head())

    print("\n" + "=" * 60)


# ============================================================
# 7. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()