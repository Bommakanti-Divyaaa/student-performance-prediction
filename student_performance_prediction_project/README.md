# Student Performance Prediction System

## Overview
A Python machine-learning mini project that estimates final marks from:
- Study hours
- Attendance percentage
- Previous marks

It creates 500 **synthetic** records, inspects and visualizes the data, trains Linear Regression and Random Forest Regression models, evaluates both, selects the model with the lower test MAE, and predicts marks for a new student.

## Requirements
- Python 3.9 or later
- pip

## Setup (Windows)
1. Extract the ZIP file.
2. Open the extracted project folder in VS Code.
3. Open Terminal > New Terminal.
4. Create an environment (recommended):
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Run
```bash
python main.py
```
Default example inputs: 6 study hours, 85% attendance, and 75 previous marks.

To enter different values:
```bash
python main.py --study-hours 7 --attendance 90 --previous-marks 80
```

Allowed input ranges:
- Study hours: 1 to 10
- Attendance: 40 to 100
- Previous marks: 30 to 100

## Files generated
- `data/student_performance.csv` — 500 synthetic records
- `outputs/model_comparison.csv` — MAE, MSE, RMSE and R² for both models
- `outputs/sample_prediction.csv` — example prediction and category
- `outputs/*.png` — feature scatter plots, actual-vs-predicted plot, MAE comparison and feature importance

## Run in Jupyter
Install dependencies, then run:
```bash
jupyter notebook
```
Open `student_performance_prediction.ipynb` and run the cells from top to bottom.

## Important limitation
The dataset is randomly generated for learning and demonstration. Metrics from synthetic data do not establish accuracy on real student records. Do not use the predictions to grade, rank, or make decisions about real students. Real-world use requires authorized real data, independent validation, privacy safeguards, fairness checks and human oversight.
