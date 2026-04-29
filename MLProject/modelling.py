import os
import shutil
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# modelling.py — MLProject Entry Point
# Credit Card Fraud Detection — CI Workflow
# Author: Yogi-Dharma
# ============================================================

def run_ci_modelling():
    print("Memulai CI Workflow Modelling...")

    # 1. Load Data
    data_path = "./credit_card_fraud_dataset_preprocessing/credit_card_fraud_preprocessing.csv"
    df = pd.read_csv(data_path)
    X = df.drop(columns=["IsFraud"])
    y = df["IsFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Hyperparameter Tuning
    param_dist = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
        "max_features"     : ["sqrt", "log2"],
    }

    base_model = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20, scoring="f1", cv=3,
        random_state=42, n_jobs=-1, verbose=1
    )

    print("Menjalankan RandomizedSearchCV...")
    search.fit(X_train, y_train)
    best_model  = search.best_estimator_
    best_params = search.best_params_
    print(f"Best params: {best_params}")

    # 3. Hapus folder model lama jika ada (mencegah error saat overwrite)
    if os.path.exists("saved_model"):
        shutil.rmtree("saved_model")

    # 4. Simpan artefak model lokal
    mlflow.sklearn.save_model(
        best_model,
        "saved_model",
        extra_pip_requirements=["starlette<1.0.0"]
    )
    print("Artefak model berhasil disimpan di folder 'saved_model'!")


if __name__ == "__main__":
    run_ci_modelling()
