import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from imblearn.over_sampling import SMOTE

mlflow.set_experiment("Credit_Card_Fraud_Detection")

df = pd.read_csv("credit_card_fraud_preprocessing.csv")

X = df.drop("IsFraud", axis=1)
y = df["IsFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

with mlflow.start_run():
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)

    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_train.iloc[:5]
    )

    print(classification_report(y_test, y_pred))