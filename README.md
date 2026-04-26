# Workflow-CI — Credit Card Fraud Detection

> Submission Machine Learning — CI Workflow dengan MLflow Project  
> **Nama:** Yogi-Dharma  
> **GitHub:** [yogidharma21](https://github.com/yogidharma21)  
> **Dataset:** Credit Card Fraud Detection  

---

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                          ← GitHub Actions CI (Advance)
└── MLProject/
    ├── MLProject                           ← MLflow Project definition
    ├── modelling.py                        ← Entry point training
    ├── conda.yaml                          ← Environment dependencies
    ├── DockerHub.txt                       ← Link Docker Hub image
    └── credit_card_fraud_dataset_preprocessing/
        └── credit_card_fraud_preprocessing.csv
```

---

## 🔄 CI Workflow (Advance)

Workflow otomatis berjalan ketika ada **push ke folder `MLProject/`** atau **manual trigger**.

### Tahapan CI:
1. ✅ Checkout repository
2. ✅ Setup Python 3.10
3. ✅ Install dependencies
4. ✅ Run `mlflow run .` — training + hyperparameter tuning
5. ✅ Commit artifacts ke repository (GitHub)
6. ✅ Upload artifacts ke GitHub Actions
7. ✅ Login Docker Hub
8. ✅ Build Docker image dengan `mlflow models build-docker`
9. ✅ Push Docker image ke Docker Hub
10. ✅ Update DockerHub.txt

---

## 🐳 Docker Hub

```
docker pull yogidharma21/credit-card-fraud-model:latest
```

---

## ⚙️ Setup GitHub Secrets

Tambahkan secrets berikut di **Settings → Secrets → Actions**:

| Secret | Keterangan |
|---|---|
| `DAGSHUB_USERNAME` | Username DagsHub (`yogidharma21`) |
| `DAGSHUB_REPO_NAME` | Nama repo DagsHub (`Eksperimen_SML_Yogi-Dharma`) |
| `DAGSHUB_TOKEN` | Token DagsHub |
| `yogidharma21` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Access Token Docker Hub |

---

## 🛠️ Cara Menjalankan Lokal

```bash
cd MLProject
mlflow run . --env-manager=local
```
