from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import sys 
import joblib
import mlflow
import yaml
import os

def load_params():
    with open("params.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

params = load_params()

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})
        # Log the model
        model_dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        mlflow.sklearn.log_model(model,  artifact_path= os.path.join(model_dirpath,"models"))


def train_model(X_train_path,X_test_path,y_train_path,y_test_path, model_type):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
    #MLFlow
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])

    

    #Generar un modelo de DecisionTree
    params_var = params['models'][model_type]
    #params_dt = {"max_depth": params['models']['decision_tree']['max_depth'],
    #             "criterion": params['models']['decision_tree']['criterion'], 
    #             "random_state": params['models']['decision_tree']['random_state']}

    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**params_var)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**params_var)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**params_var)
    

    # Entrenar el módelo con nuestros sets de entrenamiento
    train_and_log_model(
    model=model,
    model_name=model_type,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    params=params_var
    )

    return model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_train_path = sys.argv[3]
    y_test_path = sys.argv[4]
    model_type = sys.argv[5]
    model_dir = params['data']['models']
    model_path = f"{model_dir}/{model_type}_model.pkl"

    model = train_model(X_train_path, X_test_path, y_train_path, y_test_path,model_type)
    joblib.dump(model, model_path)