# 🐳 Dockerized ML Models — Portfolio Project

A production-ready Flask API serving 2 scikit-learn ML models inside Docker with a live web UI.

## 🧠 Models
| Model | Algorithm | Dataset | Accuracy |
|---|---|---|---|
| Iris Classifier | Random Forest | Iris | ~97% |
| Cancer Classifier | Gradient Boosting | Breast Cancer | ~96% |

## 🚀 Quick Start

### Using Docker Compose (Recommended)
```bash
docker-compose up --build
```

### Using Docker directly
```bash
docker build -t ml-server .
docker run -d -p 5000:5000 --name ml-server ml-server
```

Open **http://localhost:5000** in your browser.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List models + accuracy |
| POST | `/predict/iris` | Iris species prediction |
| POST | `/predict/cancer` | Cancer diagnosis |

### Example API calls

```bash
# Health check
curl http://localhost:5000/health

# Iris prediction
curl -X POST http://localhost:5000/predict/iris \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Cancer prediction (30 features)
curl -X POST http://localhost:5000/predict/cancer \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

## 🏗️ Architecture
```
dockerized-ml/
├── Dockerfile          # Multi-stage build
├── docker-compose.yml  # Service orchestration
├── requirements.txt    # Python dependencies
├── .dockerignore
├── models/
│   └── train.py        # Model training script
├── app/
│   └── main.py         # Flask API server
└── templates/
    └── index.html      # Web UI
```

## 🔧 Multi-Stage Docker Build
- **Stage 1 (trainer)**: Trains and saves models as `.pkl` files
- **Stage 2 (production)**: Lean Flask server, copies models from trainer stage

This keeps the final image small by not including training libraries in production.

## 📦 Tech Stack
- Python 3.11, Flask, scikit-learn
- Docker (multi-stage build)
- Docker Compose
