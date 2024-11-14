# train_model.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data_df = pd.read_csv(r"../../data/processed/TCGA_GBM_LGG_Mutations_clean.csv") #change path when testing
X=data_df.drop(["Grade","Tumor_Type","IDH1", "TP53", "ATRX"], axis=1)
y=data_df["Grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train the model
params_dt = {"max_depth": 5, "criterion": "entropy", "random_state": 42}
model_dt = DecisionTreeClassifier(**params_dt)
model_dt.fit(X_train, y_train)

# Evaluate and save the model
accuracy = accuracy_score(y_test, model_dt.predict(X_test))
print(f"Decision Tree model trained with an accuracy of: {accuracy:.2f}")

# Save the model
with open("tumor_model.pkl", "wb") as f:
    pickle.dump(model_dt, f)
print("Model saved as 'tumor_model.pkl'")