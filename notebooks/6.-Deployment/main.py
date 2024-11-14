from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import List
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the model
with open("tumor_model.pkl", "rb") as f:
    model = pickle.load(f)


# Define the input data format for prediction
class TumorData(BaseModel):
    metric: List[str]

# Initialize FastAPI
app = FastAPI()

data_df = pd.read_csv(r"TCGA_GBM_LGG_Mutations_clean.csv") #change path when testing
X=data_df.drop(["Grade","Tumor_Type","IDH1", "TP53", "ATRX"], axis=1)
y=data_df["Grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Prediction endpoint
@app.post("/metrics")
def predict(tumor_data: TumorData):
    returnRes = {}
    for x in tumor_data.metric:
        if(x == "accuracy"):
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accMes = {"accuracy": acc}
            returnRes = {**returnRes, **accMes}
        elif(x == "precision"):
            prec = precision_score(y_test, y_pred, average='weighted')
            preMess = {"precision": prec}
            returnRes = {**returnRes, **preMess}
        elif(x == "recall"):
            rec = recall_score(y_test, y_pred, average='weighted')
            recMess = {"recall": rec}
            returnRes = {**returnRes, **recMess}
        else:
            raise HTTPException(
            status_code=400,
            detail=f"Input must contain accuracy, precision or recall"
        )
    return returnRes

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Tumor classification model API"}