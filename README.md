# Workflow-CI: Credit Card Fraud Detection with MLflow

Proyek ini mengimplementasikan **Continuous Integration (CI)** untuk model *credit card fraud detection* menggunakan **MLflow Project**, **DagsHub** (MLflow Tracking), dan **GitHub Actions**.

---

## Struktur Repositori

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                              ← GitHub Actions workflow
├── MLProject/
│   ├── modelling.py                            ← Script pelatihan model
│   ├── MLProject                               ← Konfigurasi MLflow Project
│   ├── conda.yaml                              ← Dependensi environment
│   ├── Dockerfile                              ← Docker image untuk serving
│   ├── credit_card_fraud_preprocessing.csv     ← Dataset hasil preprocessing
│   └── DockerHub.txt                           ← Link Docker Hub (auto-update CI)
└── README.md
```

---

## Cara Kerja CI

Workflow CI berjalan otomatis setiap `push` / `pull_request` ke branch `main`:

| Step | Aksi |
|------|------|
| 1 | Checkout repository |
| 2 | Setup Python 3.12.7 |
| 3 | Install dependencies (mlflow, scikit-learn, dll) |
| **4** | **`mlflow run MLProject --env-manager=local`** ← inti CI |
| 5 | Build Docker image via `mlflow models build-docker` |
| 6 | Push Docker image ke Docker Hub |
| 7 | Simpan link Docker Hub ke `MLProject/DockerHub.txt` |

> Artefak & metrics disimpan ke **DagsHub MLflow Tracking**.

---

## GitHub Secrets yang Diperlukan

| Secret | Keterangan |
|--------|------------|
| `GIT_USERNAME` | Username GitHub |
| `GIT_EMAIL` | Email GitHub |
| `DAGSHUB_USERNAME` | Username DagsHub |
| `DAGSHUB_REPO_NAME` | Nama repo di DagsHub |
| `DAGSHUB_TOKEN` | Access Token DagsHub |
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Access Token Docker Hub |

### Cara mendapatkan DagsHub Token:
1. Login ke [dagshub.com](https://dagshub.com)
2. Klik avatar → **User Settings** → **Tokens**
3. Klik **Generate New Token** → salin tokennya
4. Simpan sebagai secret `DAGSHUB_TOKEN`

---

## Menjalankan Secara Lokal

```bash
# Set environment variable untuk DagsHub tracking
export MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<dagshub_username>
export MLFLOW_TRACKING_PASSWORD=<dagshub_token>

# Jalankan MLflow Project
mlflow run MLProject --env-manager=local

# Dengan parameter kustom
mlflow run MLProject --env-manager=local \
  -P n_estimators=300 \
  -P max_depth=15
```

---

## Docker Hub

Link Docker image tersedia di: `MLProject/DockerHub.txt`
