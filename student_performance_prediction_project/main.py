"""
Student Performance Prediction System
Synthetic-data demonstration using Linear Regression and Random Forest Regression.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FEATURES = ["Study_Hours", "Attendance", "Previous_Marks"]


def generate_dataset(n=500, random_state=42):
    """Generate the synthetic dataset described in the project report."""
    rng = np.random.RandomState(random_state)
    study_hours = rng.uniform(1, 10, n)
    attendance = rng.uniform(40, 100, n)
    previous_marks = rng.uniform(30, 100, n)
    noise = rng.normal(0, 5, n)
    final_marks = (
        2.5 * study_hours + 0.25 * attendance
        + 0.45 * previous_marks + noise
    )
    final_marks = np.clip(final_marks, 0, 100)
    return pd.DataFrame({
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Previous_Marks": previous_marks,
        "Final_Marks": final_marks,
    })


def evaluate_model(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(actual, predicted),
    }


def save_eda_plots(data):
    for feature in FEATURES:
        plt.figure(figsize=(8, 5))
        plt.scatter(data[feature], data["Final_Marks"], alpha=0.65)
        plt.xlabel(feature.replace("_", " "))
        plt.ylabel("Final Marks")
        plt.title(f"{feature.replace('_', ' ')} vs Final Marks")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{feature.lower()}_vs_final_marks.png", dpi=150)
        plt.close()


def run_project(study_hours=6, attendance=85, previous_marks=75, show_report=True):
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=" * 62)
    print("          STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("=" * 62)

    data = generate_dataset()
    data.to_csv(DATA_DIR / "student_performance.csv", index=False)
    print("\nDATASET CREATED SUCCESSFULLY")
    print("First five records:\n", data.head().round(2).to_string(index=False))
    print("\nDataset shape:", data.shape)
    print("\nMissing values:\n", data.isnull().sum().to_string())
    print("\nStatistical summary:\n", data.describe().round(2).to_string())

    save_eda_plots(data)

    X = data[FEATURES]
    y = data["Final_Marks"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print("\nTRAINING AND TESTING")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)

    forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_model.fit(X_train, y_train)
    forest_predictions = forest_model.predict(X_test)

    results = [
        evaluate_model("Linear Regression", y_test, linear_predictions),
        evaluate_model("Random Forest Regression", y_test, forest_predictions),
    ]
    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON")
    print(comparison.round(4).to_string(index=False))

    if results[0]["MAE"] <= results[1]["MAE"]:
        best_model, best_name = linear_model, "Linear Regression"
        predictions = linear_predictions
    else:
        best_model, best_name = forest_model, "Random Forest Regression"
        predictions = forest_predictions
    print("\nSelected model (lower test MAE):", best_name)

    # Actual versus predicted marks
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([0, 100], [0, 100], linestyle="--")
    plt.xlim(0, 100); plt.ylim(0, 100)
    plt.xlabel("Actual Marks"); plt.ylabel("Predicted Marks")
    plt.title("Actual vs Predicted Marks")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Model MAE comparison
    plt.figure(figsize=(7, 5))
    plt.bar(comparison["Model"], comparison["MAE"])
    plt.ylabel("Mean Absolute Error")
    plt.title("Model Comparison - Test MAE")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_mae_comparison.png", dpi=150)
    plt.close()

    # Random forest feature importance
    plt.figure(figsize=(7, 5))
    plt.bar(FEATURES, forest_model.feature_importances_)
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
    })
    predicted_marks = float(np.clip(best_model.predict(new_student)[0], 0, 100))
    if predicted_marks >= 75:
        category = "Excellent"
    elif predicted_marks >= 60:
        category = "Good"
    elif predicted_marks >= 40:
        category = "Average"
    else:
        category = "Needs Improvement"

    prediction_output = new_student.copy()
    prediction_output["Predicted_Marks"] = predicted_marks
    prediction_output["Performance_Category"] = category
    prediction_output.to_csv(OUTPUT_DIR / "sample_prediction.csv", index=False)

    print("\nNEW STUDENT DETAILS")
    print(new_student.to_string(index=False))
    print(f"Predicted marks: {predicted_marks:.2f}")
    print("Performance category:", category)
    print("\nCharts and result files saved in the outputs folder.")
    print("\nNOTE: This is a demonstration trained on synthetic data.")
    print("Do not use its predictions for real academic decisions.")

    if show_report:
        print("\nOUTPUT FILES")
        for f in sorted(OUTPUT_DIR.iterdir()):
            print("-", f.name)
    return comparison, predicted_marks, category


def main():
    parser = argparse.ArgumentParser(description="Predict marks using a demonstration ML model.")
    parser.add_argument("--study-hours", type=float, default=6, help="Study hours (1 to 10)")
    parser.add_argument("--attendance", type=float, default=85, help="Attendance percentage (40 to 100)")
    parser.add_argument("--previous-marks", type=float, default=75, help="Previous marks (30 to 100)")
    args = parser.parse_args()

    if not 1 <= args.study_hours <= 10:
        parser.error("--study-hours must be between 1 and 10")
    if not 40 <= args.attendance <= 100:
        parser.error("--attendance must be between 40 and 100")
    if not 30 <= args.previous_marks <= 100:
        parser.error("--previous-marks must be between 30 and 100")

    run_project(args.study_hours, args.attendance, args.previous_marks)


if __name__ == "__main__":
    main()
