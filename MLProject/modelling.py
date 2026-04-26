import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, roc_curve,
    average_precision_score
)

# ============================================================
# modelling.py — MLProject Entry Point
# Credit Card Fraud Detection — CI Workflow
# Author: Yogi-Dharma
# ============================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "credit_card_fraud_dataset_preprocessing",
    "credit_card_fraud_preprocessing.csv"
)

# Konfigurasi DagsHub via environment variables
DAGSHUB_USERNAME  = os.environ.get("DAGSHUB_USERNAME", "yogidharma21")
DAGSHUB_REPO_NAME = os.environ.get("DAGSHUB_REPO_NAME", "Eksperimen_SML_Yogi-Dharma")
DAGSHUB_TOKEN     = os.environ.get("DAGSHUB_TOKEN", "")

if DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"[INFO] MLflow tracking ke DagsHub: {TRACKING_URI}")
else:
    print("[INFO] MLflow tracking lokal (mlruns)")


def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"],
                ax=ax, linewidths=0.5)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_test, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [feature_names[i] for i in indices]
    sorted_imp  = importances[indices]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_feat[::-1], sorted_imp[::-1],
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(y_test, y_pred, save_path):
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=4)


def main():
    print("=" * 60)
    print("  CI TRAINING — Credit Card Fraud Detection")
    print("  Author: Yogi-Dharma")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["IsFraud"])
    y  = df["IsFraud"]
    print(f"[INFO] Dataset dimuat: {df.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

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

    print("\n[INFO] Menjalankan RandomizedSearchCV...")
    search.fit(X_train, y_train)
    best_model  = search.best_estimator_
    best_params = search.best_params_
    print(f"[INFO] Best params: {best_params}")

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"         : accuracy_score(y_test, y_pred),
        "precision"        : precision_score(y_test, y_pred),
        "recall"           : recall_score(y_test, y_pred),
        "f1_score"         : f1_score(y_test, y_pred),
        "roc_auc"          : roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
        "cv_best_score"    : search.best_score_,
    }

    artifact_dir = os.path.join(BASE_DIR, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    cm_path     = os.path.join(artifact_dir, "confusion_matrix.png")
    roc_path    = os.path.join(artifact_dir, "roc_curve.png")
    fi_path     = os.path.join(artifact_dir, "feature_importance.png")
    report_path = os.path.join(artifact_dir, "classification_report.json")

    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc_curve(y_test, y_prob, roc_path)
    plot_feature_importance(best_model, list(X.columns), fi_path)
    save_classification_report(y_test, y_pred, report_path)

    # ── Log langsung ke active run MLflow Project ──────────────
    mlflow.log_params(best_params)
    mlflow.log_param("n_iter_search", 20)
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("class_weight", "balanced")

    for name, val in metrics.items():
        mlflow.log_metric(name, val)

    model_dir = os.path.join(artifact_dir, "random_forest_model")
    mlflow.sklearn.save_model(best_model, model_dir)
    mlflow.log_artifacts(model_dir, artifact_path="random_forest_model")

    mlflow.log_artifact(cm_path,     artifact_path="plots")
    mlflow.log_artifact(roc_path,    artifact_path="plots")
    mlflow.log_artifact(fi_path,     artifact_path="plots")
    mlflow.log_artifact(report_path, artifact_path="reports")

    print("\n[RESULT] === Metrik Evaluasi ===")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    print("\n[DONE] Training selesai dan tercatat di MLflow!")


if __name__ == "__main__":
    main()
