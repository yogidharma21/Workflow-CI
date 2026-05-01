import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Credit_Card_Fraud_Detection")

df = pd.read_csv("credit_card_fraud_preprocessing.csv")

print("Kolom dataset:", df.columns)

target_col = "Class"

if target_col not in df.columns:
    raise ValueError(f"Kolom '{target_col}' tidak ditemukan!")

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

mlflow.log_param("model", "RandomForest")
mlflow.log_metric("accuracy", acc)

print(f"Accuracy: {acc}")