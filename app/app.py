from pathlib import Path
import sys

from flask import Flask, render_template


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


# Add project root to Python path
if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# FLASK CONFIGURATION
# ============================================================

APP_FOLDER = Path(
    __file__
).resolve().parent

TEMPLATE_FOLDER = (
    APP_FOLDER / "templates"
)

STATIC_FOLDER = (
    APP_FOLDER / "static"
)


app = Flask(

    __name__,

    template_folder=str(
        TEMPLATE_FOLDER
    ),

    static_folder=str(
        STATIC_FOLDER
    )
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# DATASET PAGE
# ============================================================

@app.route("/dataset")
def dataset():

    from src.data.load_data import (
        load_data,
        get_summary
    )

    df = load_data()

    summary = get_summary(
        df
    )

    return render_template(

        "load_dataset.html",

        summary=summary,

        first_rows=df.head(
            10
        ).to_html(
            index=False,
            classes="table table-striped"
        )
    )


# ============================================================
# EDA PAGE
# ============================================================

@app.route("/eda")
def eda():

    return render_template(
        "eda.html"
    )


# ============================================================
# PREPROCESSING PAGE
# ============================================================

@app.route("/preprocessing")
def preprocessing():

    return render_template(
        "preprocessing.html"
    )


# ============================================================
# LOGISTIC REGRESSION PAGE
# ============================================================

@app.route("/logistic-regression")
def logistic_regression():

    return render_template(
        "logistic_regression.html"
    )


# ============================================================
# LINEAR REGRESSION PAGE
# ============================================================

@app.route("/linear-regression")
def linear_regression():

    return render_template(
        "linear_regression.html"
    )


# ============================================================
# DECISION TREE PAGE
# ============================================================

@app.route("/decision-tree")
def decision_tree():

    return render_template(
        "decision_tree.html"
    )


# ============================================================
# RANDOM TREE PAGE
# ============================================================

@app.route("/random-tree")
def random_tree():

    return render_template(
        "random_tree.html"
    )


# ============================================================
# RANDOM FOREST PAGE
# ============================================================

@app.route("/random-forest")
def random_forest():

    return render_template(
        "random_forest.html"
    )


# ============================================================
# K-MEANS PAGE
# ============================================================

@app.route("/kmeans")
def kmeans():

    return render_template(
        "kmeans.html"
    )


# ============================================================
# GRADIENT BOOSTING PAGE
# ============================================================

@app.route("/gradient-boosting")
def gradient_boosting():

    return render_template(
        "gradientboosting.html"
    )


# ============================================================
# APPLICATION INFORMATION
# ============================================================

@app.context_processor
def project_information():

    return {
        "project_name":
            "Placement Prediction System",

        "developer":
            "Sanjana",

        "github_username":
            "sanjana2501"
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )