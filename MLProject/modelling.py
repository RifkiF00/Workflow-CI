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
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

# ==========================================
# Set MLflow Local Tracking URI (localhost)
# ==========================================
if "MLFLOW_TRACKING_URI" not in os.environ:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ==========================================
# MLflow AUTOLOG
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
# Training Function
# ==========================================
def train_and_log():
    print("Memulai pelatihan model dasar (LinearRegression)...")

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    mlflow.log_param("model_type", "LinearRegression")

    mlflow.log_metric("custom_mse", mse)
    mlflow.log_metric("custom_rmse", rmse)
    mlflow.log_metric("custom_mae", mae)
    mlflow.log_metric("custom_r2", r2)
    mlflow.log_metric("custom_mape", mape)

    mlflow.sklearn.log_model(
        sk_model=lr_model,
        artifact_path="model"
    )

    print("\nPelatihan selesai!")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.4f} ({mape*100:.2f}%)")

# ==========================================
# Execution Control
# ==========================================
if __name__ == "__main__":
    mlflow.set_experiment("Student_Performance_Basic_Model")
    with mlflow.start_run(run_name="LinearRegression_Basic_Model"):
        train_and_log()
