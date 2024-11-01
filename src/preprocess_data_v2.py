import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Función para convertir 'years days' a número decimal
def convertir_edad_decimal(age_str):
    try:
        if not "days" in age_str:
            return float(age_str.split(' years')[0])
        else:
            if ' years ' in age_str:
                years, days = age_str.split(' years ')
                years = int(years)
                days = int(days.split(' days')[0])
                return round(years + days / 365.25, 2)  # Redondear a 2 decimales
    except ValueError:
        # En caso de que no se tenga el valor de la edad, devolver None
        return None
def extraer_tumor_info(diagnosis):    # Función para extraer el tipo de tumor y la especificación del tumor
    if diagnosis == '--' or diagnosis not in ['Oligodendroglioma, NOS', 'Glioblastoma' , 'Mixed glioma', 'Astrocytoma, NOS', 'Astrocytoma, anaplastic', 'Oligodendroglioma, anaplastic']:
        return pd.Series([None, None])
    else:
        parts = diagnosis.split(', ')
        tumor_type = parts[0]
        tumor_specification = parts[1] if len(parts) > 1 else None
        return pd.Series([tumor_type, tumor_specification])
    
def clean_csv(Mutations_df):
    # Eliminar filas con valores nulos
    Mutations_df.dropna(inplace=True)
    # Reemplazar los valores de las columnas que  pueden ser tratados como datos binarios 
    # como las columnas que informan de mutaciones, grado y genero por 0 o 1 para facilitar el análisis 
    Mutations_df.replace({'NOT_MUTATED': 0, 'MUTATED': 1}, inplace=True)
    Mutations_df.replace({'Female': 0, 'Male': 1}, inplace=True)
    Mutations_df.replace({'LGG': 0, 'GBM': 1}, inplace=True)

    # Crear diccionario de mapeo para la columna Race
    race_mapping = {
        'not reported': 0,
        '--': 0,
        'white': 1,
        'black or african american': 2,
        'asian': 3,
        'american indian or alaska native': 4
    }

    # Aplicar el mapeo a la columna Race
    Mutations_df['Race'] = Mutations_df['Race'].map(race_mapping)
    # Aplicar la función a la columna 'Age_at_diagnosis'
    Mutations_df['Age_at_diagnosis'] = Mutations_df['Age_at_diagnosis'].apply(convertir_edad_decimal)
    #Elimnar las columnas que no se utilizarán, en estes caso no creo conveniente tener la columna de "Project" y la de "Case_ID" 
    # ya que el proyecto se puede sacar con el grade dando que el project y el case Id son indiferentes para el analisis de datos posterior
    Mutations_df.drop(columns=['Project', 'Case_ID'], inplace=True)


    # Aplicar la función a la columna 'Primary_Diagnosis'
    Mutations_df[['Tumor_Type', 'Tumor_Specification']] = Mutations_df['Primary_Diagnosis'].apply(extraer_tumor_info)
    # Eliminar filas con 'Primary_Diagnosis' como None o '--'
    Mutations_df = Mutations_df.dropna(subset=['Tumor_Type'])
    # Aplicar el mapeo a las columnas 'Tumor_Type' y 'Tumor_Specification'
    mapeo_tumor = {
        "Oligodendroglioma": 0,
        "Mixed glioma": 1,
        "Astrocytoma": 2,
        "Glioblastoma": 3
    }
    Mutations_df['Tumor_Type'] = Mutations_df['Tumor_Type'].map(mapeo_tumor)
    mapeo_tipo_tumor = {
        None: 0,
        "NOS": 1,
        "anaplastic": 2
    }
    Mutations_df['Tumor_Specification'] = Mutations_df['Tumor_Specification'].map(mapeo_tipo_tumor)
    # Eliminar la columna 'Primary_Diagnosis' ya que no se utilizará
    Mutations_df.drop(columns=['Primary_Diagnosis'], inplace=True)

    # Ordenar las columnas del dataframe para que sea más fácil de leer
    column_order = ['Grade', 'Gender', 'Age_at_diagnosis', 'Race', 'Tumor_Type', 'Tumor_Specification' , 'IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
    Mutations_df = Mutations_df[column_order]
    return Mutations_df

def preprocess_data(data_path):
    dataset = pd.read_csv(data_path)
    dataset = clean_csv(dataset)
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