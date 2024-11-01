import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path):
    dataset = pd.read_csv(data_path)
    #data = clean_csv(dataset)
    # Separar nuestros datos
    X = dataset.drop(["Grade","Tumor_Type","IDH1", "TP53", "ATRX"], axis=1)
    y = dataset["Grade"]
    # Generar los datos para probar y para entrenar con parametros seleccionados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled,X_test_scaled, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features_scaled = sys.argv[2]
    output_test_features_scaled = sys.argv[3]
    output_train_features = sys.argv[4]
    output_test_features = sys.argv[5]
    output_train_target = sys.argv[6]
    output_test_target = sys.argv[7]

    X_train_scaled,X_test_scaled, X_train, X_test, y_train, y_test = preprocess_data(data_path)
    pd.DataFrame(X_train_scaled).to_csv(output_train_features_scaled, index=False)
    pd.DataFrame(X_test_scaled).to_csv(output_test_features_scaled, index=False)
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)