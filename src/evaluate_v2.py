from sklearn.metrics import classification_report, confusion_matrix
import sys
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate_model_v1(y_true, y_pred):
    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Calcular el reporte de clasificación
    class_report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(class_report)


def evaluate_model(model_path, X_test_path, y_test_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

if __name__ == '__main__':
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]
    
    evaluate_model(model_path, X_test_path, y_test_path)