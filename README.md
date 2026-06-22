# AUTOMATED-PARKINSON-S-DISEASE-DETECTION-FROM-BIOMEDICAL-DATA
Detects Parkinson's disease from biomedical data using deep learning models (VGG16, ResNet50, CNN Fusion). Includes a web-based diagnostic interface built with HTML, CSS, and JavaScript, deployed on Netlify
# Automated Parkinson's Disease Detection

A deep learning project to detect Parkinson's disease from biomedical MRI data using CNN models. Built as part of B.Tech final year project (CSE - AI & Data Science).

## Live Demo
[View on Netlify](https://automatedparkinsonsdisease.netlify.app)

## About the Project
Parkinson's disease is a neurological disorder that's hard to detect early. This project automates the detection process using MRI scan data and deep learning, reducing dependency on manual diagnosis.

Three CNN models were trained and compared for accuracy:
- VGG16
- ResNet50
- CNN Fusion (custom)
- # Parkinson's Disease Detection System - Requirements
 ==============================

# Core ML/DL
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Image Processing
opencv-python-headless>=4.5.0
Pillow>=8.0.0

# Web API
flask>=2.0.0
flask-cors>=3.0.0
gunicorn>=20.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Optional: XGBoost for ML comparison
xgboost>=1.5.0

# Utilities
tqdm>=4.60.0

# Parkinson's Disease Detection - Deployment Guide

## 📁 Deployment Structure

```
deployment/
├── app.py              # Flask REST API server
├── api_client.py       # Example client code
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd deployment
pip install -r requirements.txt
```

### 2. Ensure Models are Saved

Make sure you have trained and saved the models by running the notebook. The models should be in:
```
../ImprovedModels/
├── improved_fusion_model.h5
├── improved_voice_embedding.h5
├── improved_gait_embedding.h5
├── improved_image_embedding.h5
└── optimal_threshold.txt
```

### 3. Start the API Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

### 4. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model-info
```

## 📡 API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0"
}
```

### GET /model-info
Get model information.

**Response:**
```json
{
    "name": "Multimodal Parkinson Detection CNN",
    "modalities": ["voice", "gait", "handwriting"],
    "optimal_threshold": 0.5
}
```

### POST /predict
Make a single prediction.

**Request:**
```json
{
    "voice": [119.992, 157.302, ...],  // 22 voice features
    "gait": [0.1, 0.2, ...],           // Gait signal values
    "image": "base64_encoded_string"   // Handwriting image
}
```

**Response:**
```json
{
    "prediction": "Parkinson's",
    "probability": 0.8534,
    "confidence": "High",
    "threshold_used": 0.5
}
```

### POST /batch-predict
Make predictions for multiple patients.

**Request:**
```json
{
    "patients": [
        {"id": "patient_1", "voice": [...], "gait": [...], "image": "..."},
        {"id": "patient_2", "voice": [...], "gait": [...], "image": "..."}
    ]
}
```

## 🔧 Production Deployment

### Using Gunicorn (Linux/Mac)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Waitress (Windows)

```bash
pip install waitress
waitress-serve --port=5000 app:app
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
COPY ../ImprovedModels ./ImprovedModels

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t parkinson-detection .
docker run -p 5000:5000 parkinson-detection
```

##  Input Data Format

### Voice Features (22 values)
From UCI Parkinson's Dataset:
- MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
- MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP
- MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA
- NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE

### Gait Signal
- Time-series of vertical ground reaction force (VGRF)
- Will be resampled to 512 data points

### Handwriting Image
- Grayscale image of spiral or wave drawing
- Will be resized to 128x128 pixels
- Can be provided as file path or base64 encoded string

##  Security Notes

For production:
1. Add authentication (API keys, OAuth, etc.)
2. Use HTTPS
3. Implement rate limiting
4. Add input validation
5. Set up logging and monitoring


## Features
- Automated disease detection from MRI data
- Multi-model comparison (VGG16, ResNet50, CNN Fusion)
- Web-based diagnostic interface with Sandbox workspace
- Real-time model inference

## Tech Stack
- Python, TensorFlow, Keras
- HTML5, CSS3, JavaScript
- Deployed on Netlify


## College
Sidhartha Institute of Engineering and Technology, Puttur
B.Tech CSE - AI & Data Science | JNTUA | 2026
