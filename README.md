# CIFAR-10 FastAPI Inference Service

This project provides a simple **FastAPI** server to run image classification on the **CIFAR-10** dataset using a TensorFlow model.

## Features
- REST API for single and batch image predictions.
- WebSocket endpoint for real-time streaming inference.
- Input validation (JPEG/PNG, size limits).
- Easy deployment with Uvicorn.

## Requirements
- Python 3.9+
- Virtual environment recommended

Dependencies:
- `fastapi`
- `uvicorn`
- `tensorflow`
- `pillow`
- `python-multipart`

## Setup

```bash
# Clone repository and move into project folder
git clone <repo-url>
cd ml-api-playground

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn app:app --reload
```

API will be available at:  
👉 http://127.0.0.1:8000

## Endpoints

### Health check
```bash
GET /health
```

### Predict single image
```bash
POST /predict
-F "file=@image.jpg"
```

### Predict batch of images
```bash
POST /predict_batch
-F "files=@image1.jpg" -F "files=@image2.jpg"
```

### WebSocket streaming
```bash
/ws
```
Send raw image bytes and receive predictions in real time.

## Dataset
Place your test images inside the `data/` folder.  
This folder is ignored by `.gitignore`.