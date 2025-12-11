import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import sys
import warnings

# Tambahkan fungsi pembantu untuk menghitung metrik
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    # R2 akan kita hitung menggunakan model.score()
    return rmse, mae

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # 1. Mengambil parameter dari command line (sys.argv)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    file_path = sys.argv[3] if len(sys.argv) > 3 else "auto-mpg_preprocessed.csv"
    
    # Menghindari masalah path saat dijalankan oleh MLflow
    # MLflow akan menjalankan dari direktori MLProject
    data = pd.read_csv(file_path)

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("mpg", axis=1),
        data["mpg"],
        random_state=42,
        test_size=0.2
    )
    input_example = X_train[0:5]

    with mlflow.start_run():
        # 3. Training dan Logging
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi dan Metrik
        predictions = model.predict(X_test)
        (rmse, mae) = eval_metrics(y_test, predictions)
        r2_score = model.score(X_test, y_test)
        
        # Log parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Log metrik
        mlflow.log_metric("r2_score", r2_score)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"RandomForestRegressor (n_estimators={n_estimators}, max_depth={max_depth}):")
        print(f"  R2 Score: {r2_score:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")