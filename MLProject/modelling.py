import mlflow
import pandas as pd

mlflow.set_experiment("Credit_Card_Fraud_Detection")

df = pd.read_csv("credit_card_fraud_preprocessing.csv", sep=",")

print("Kolom:", df.columns)

target_col = "Class"

if target_col not in df.columns:
    raise ValueError(f"Kolom '{target_col}' tidak ditemukan! Kolom: {df.columns}")

X = df.drop(target_col, axis=1)
y = df[target_col]