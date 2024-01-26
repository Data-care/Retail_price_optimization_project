import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv(r"C:\Users\hp\Downloads\retail\retail_price.csv")

    label_encoder = LabelEncoder()
    data['comp_price_diff'] = data['unit_price'] - data['comp_1'] 
    data['product_category_name'] = label_encoder.fit_transform(data['product_category_name'])
    
    x = data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
    y = data['total_price']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    with mlflow.start_run():
        best_hyperparameters = {
            'learning_rate': 0.1,
            'max_depth': 4,
            'min_samples_leaf': 2,
            'min_samples_split': 5,
            'n_estimators': 200
        }

        best_gb_regressor = GradientBoostingRegressor(**best_hyperparameters)
        best_gb_regressor.fit(xtrain, ytrain)
        prediction = best_gb_regressor.predict(xtest)
        (rmse, mae, r2) = eval_metrics(ytest, prediction)

        print(f"Gradient Boosting Regressor:")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Log hyperparameters
        for param, value in best_hyperparameters.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        signature = infer_signature(xtrain, prediction)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_gb_regressor, "model", registered_model_name="GBRModel", signature=signature)
        else:
            mlflow.sklearn.log_model(best_gb_regressor, "model", signature=signature)
