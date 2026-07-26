import os
import sys
import io

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression

# ==========================================
# Set MLflow Local Tracking URI (localhost if not set via env)
# ==========================================
if "MLFLOW_TRACKING_URI" not in os.environ:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ==========================================
# MLflow AUTOLOG ONLY (No Manual Logging)
# ==========================================
mlflow.sklearn.autolog()

# ==========================================
# Load Dataset
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "student_performance_preprocessing")

X_train = pd.read_csv(os.path.join(dataset_dir, "X_train.csv"))
y_train = pd.read_csv(
    os.path.join(dataset_dir, "y_train.csv")
).values.ravel()

X_test = pd.read_csv(os.path.join(dataset_dir, "X_test.csv"))
y_test = pd.read_csv(
    os.path.join(dataset_dir, "y_test.csv")
).values.ravel()

# ==========================================
# Training Function (Autolog Only)
# ==========================================
def train_and_log():
    print("Memulai pelatihan model dasar LinearRegression dengan MLflow Autolog...")

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    print("\nPelatihan selesai! Autolog telah merekam parameter, metrik, dan model artifact.")

# ==========================================
# Execution Control
# ==========================================
if __name__ == "__main__":
    mlflow.set_experiment("Student_Performance_Basic_Model")
    with mlflow.start_run(run_name="LinearRegression_Basic_Model"):
        train_and_log()
