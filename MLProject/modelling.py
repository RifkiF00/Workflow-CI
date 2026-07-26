import os
import sys
import io

# Force UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression

# Enable MLflow Autologging exclusively
mlflow.sklearn.autolog()

# Load Dataset
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

# Train Model (Autolog automatically records params, metrics & artifacts)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
